#include "rpc/rpc.h"
#include "backend/backend.h"
#include "log.h"
#include "profile.h"

#include <stdio.h>

#define RPC_BACKEND_PRIORITY 1
#define RPC_FORWARD_MASK                                                                           \
	(BCAP_IS_HOST | BCAP_HOST_VISIBLE_BUFFERS | BCAP_UNIFIED_MEMORY | BCAP_KV_SHIFT)
#define SEND_VALS(self, op, ...)                                                                   \
	op_vals(self, op, (rpc_val[]){__VA_ARGS__}, sizeof((rpc_val[]){__VA_ARGS__}) / sizeof(rpc_val))
#define INSTALL_OP(name)                                                                           \
	if (present[RPC_OP_##name])                                                                    \
		self->name = stub_##name;

typedef struct {
	const float *ptr;
	int			 half;
	int			 lo;
	int			 hi;
	uint32_t	 id;
	int			 sig_row;
	uint64_t	 sig;
} rpc_table;

typedef struct {
	uint32_t type;
	uint64_t d0;
	uint64_t d1;
	uint32_t re_type;
} rpc_plan_entry;

typedef struct {
	rpc_conn		conn;
	pthread_mutex_t table_mtx;
	char			desc[RPC_IDENT_MAX];
	char			device[BACKEND_DEVICE_NAME_MAX];
	uint64_t		mem_total;
	uint64_t		mem_avail;
	uint64_t		mem_avail_us;
	bool			read_pending;
	bool			stash_valid;
	status_code		stash_err;
	rpc_reply		stash;
	int				pend_n[2];
	rpc_plan_entry	plans[RPC_PLAN_CACHE];
	int				n_plans;
	uint8_t			type_native[RPC_TYPE_MAX];
	int				sent_neox;
	float			sent_theta;
	bool			params_sent;
	rpc_table		tables[RPC_MAX_TABLES];
	int				n_tables;
	int				next_slot;
	uint32_t		next_table_id;
} rpc_priv;

static rpc_priv *priv_of(backend *self) {
	return self->priv;
}

static void sync_params(backend *self) {
	rpc_priv *p = priv_of(self);
	if (p->params_sent && p->sent_neox == self->rope_neox && p->sent_theta == self->rope_theta)
		return;
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_i32(&w, self->rope_neox);
	rpc_wr_f32(&w, self->rope_theta);
	rpc_send(&p->conn, RPC_CTL_PARAMS, w.p, w.len);
	p->sent_neox   = self->rope_neox;
	p->sent_theta  = self->rope_theta;
	p->params_sent = true;
	rpc_wr_free(&w);
}

static status_code op_send(backend *self, uint32_t op, rpc_wr *w) {
	status_code s = w->bad ? ERR_UNSUPPORTED : OK;
	if (s == OK) {
		sync_params(self);
		s = rpc_send(&priv_of(self)->conn, op, w->p, w->len);
	}
	rpc_wr_free(w);
	return s;
}

static void drain_pending_read(backend *self) {
	rpc_priv *p = priv_of(self);
	if (!p->read_pending)
		return;
	p->read_pending = false;
	p->stash_err	= rpc_call_end(&p->conn, &p->stash);
	p->stash_valid	= p->stash_err == OK;
}

static status_code op_call(backend *self, uint32_t op, rpc_wr *w, rpc_reply *r) {
	status_code s = w->bad ? ERR_UNSUPPORTED : OK;
	memset(r, 0, sizeof(*r));
	drain_pending_read(self);
	if (s == OK) {
		sync_params(self);
		s = rpc_call(&priv_of(self)->conn, op, w->p, w->len, r);
	}
	rpc_wr_free(w);
	if (s == OK && r->status != OK)
		s = (status_code)r->status;
	if (s != OK)
		rpc_reply_free(r);
	return s;
}

static status_code op_vals(backend *self, uint32_t op, const rpc_val *vals, size_t n) {
	rpc_wr w;
	rpc_wr_init(&w);
	for (size_t i = 0; i < n; i++)
		rpc_wr_val(&w, self, vals[i]);
	return op_send(self, op, &w);
}

static void fill_buffer(backend *self, buffer *out, uint32_t id, uint64_t size) {
	memset(out, 0, sizeof(*out));
	out->handle = (void *)(uintptr_t)id;
	out->size	= size;
	out->owner	= self;
}

static uint64_t row_hash(const float *row, int n) {
	uint64_t h = FNV1A_OFFSET_BASIS;
	for (int i = 0; i < n; i++) {
		uint32_t bits;
		memcpy(&bits, &row[i], sizeof(bits));
		h = (h ^ bits) * 0x100000001b3ULL;
	}
	return h;
}

static void table_upload(backend *self, rpc_table *t, int row0, int n_rows) {
	if (n_rows <= 0)
		return;
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, t->id);
	rpc_wr_u32(&w, (uint32_t)t->half);
	rpc_wr_u32(&w, (uint32_t)row0);
	rpc_wr_u32(&w, (uint32_t)n_rows);
	rpc_wr_bytes(&w, t->ptr + (size_t)row0 * t->half, (size_t)n_rows * t->half * sizeof(float));
	rpc_send(&priv_of(self)->conn, RPC_CTL_TABLE, w.p, w.len);
	rpc_wr_free(&w);
}

static rpc_table *table_find(rpc_priv *p, const float *ptr, int half) {
	for (int i = 0; i < p->n_tables; i++)
		if (p->tables[i].ptr == ptr && p->tables[i].half == half)
			return &p->tables[i];
	bool	   fresh = p->n_tables < RPC_MAX_TABLES;
	rpc_table *t	 = &p->tables[fresh ? p->n_tables++ : p->next_slot++ % RPC_MAX_TABLES];
	uint32_t   id	 = fresh ? ++p->next_table_id : t->id;
	memset(t, 0, sizeof(*t));
	t->ptr	= ptr;
	t->half = half;
	t->id	= id;
	return t;
}

static void table_ensure(backend *self, rpc_table *t, int lo, int hi) {
	if (t->hi > t->lo && row_hash(t->ptr + (size_t)t->sig_row * t->half, t->half) != t->sig)
		t->lo = t->hi = 0;
	if (t->hi <= t->lo) {
		table_upload(self, t, lo, hi - lo);
		t->lo	   = lo;
		t->hi	   = hi;
		t->sig_row = hi - 1;
		t->sig	   = row_hash(t->ptr + (size_t)t->sig_row * t->half, t->half);
		return;
	}
	int new_lo = MIN(t->lo, lo);
	int new_hi = MAX(t->hi, hi);
	table_upload(self, t, new_lo, t->lo - new_lo);
	table_upload(self, t, t->hi, new_hi - t->hi);
	t->lo = new_lo;
	t->hi = new_hi;
}

static uint32_t tbl(backend *self, const float *ptr, int half, int lo, int n_rows) {
	if (!ptr || half <= 0 || n_rows <= 0)
		return 0;
	rpc_priv *p = priv_of(self);
	pthread_mutex_lock(&p->table_mtx);
	rpc_table *t = table_find(p, ptr, half);
	table_ensure(self, t, lo, lo + n_rows);
	uint32_t id = t->id;
	pthread_mutex_unlock(&p->table_mtx);
	return id;
}

#include "backend/rpc/stubs.inc"

static status_code stub_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
							 const float *cos_base, const float *sin_base) {
	int half = head_dim / 2;
	return SEND_VALS(self, RPC_OP_rope, RPC_V(vec), RPC_V(n_heads), RPC_V(head_dim), RPC_V(pos),
					 RPC_V(tbl(self, cos_base, half, pos, 1)),
					 RPC_V(tbl(self, sin_base, half, pos, 1)));
}

static status_code stub_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
								int head_dim, int pos, const float *cos_base,
								const float *sin_base) {
	int half = head_dim / 2;
	return SEND_VALS(self, RPC_OP_rope_qk, RPC_V(q), RPC_V(k), RPC_V(n_heads), RPC_V(n_kv_heads),
					 RPC_V(head_dim), RPC_V(pos), RPC_V(tbl(self, cos_base, half, pos, 1)),
					 RPC_V(tbl(self, sin_base, half, pos, 1)));
}

static status_code stub_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
								 const float *cos_base, const float *sin_base,
								 const float *freq_factors) {
	int half = head_dim / 2;
	return SEND_VALS(self, RPC_OP_rope_ext, RPC_V(vec), RPC_V(n_heads), RPC_V(head_dim), RPC_V(pos),
					 RPC_V(tbl(self, cos_base, half, pos, 1)),
					 RPC_V(tbl(self, sin_base, half, pos, 1)),
					 RPC_V(tbl(self, freq_factors, half, 0, 1)));
}

static status_code stub_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim,
									   int pos_start, const float *cos_base, const float *sin_base,
									   const float *freq_factors, int m) {
	int half = head_dim / 2;
	return SEND_VALS(self, RPC_OP_rope_ext_batch, RPC_V(vec), RPC_V(n_heads), RPC_V(head_dim),
					 RPC_V(pos_start), RPC_V(tbl(self, cos_base, half, pos_start, m)),
					 RPC_V(tbl(self, sin_base, half, pos_start, m)),
					 RPC_V(tbl(self, freq_factors, half, 0, 1)), RPC_V(m));
}

static status_code stub_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim,
								   int pos_start, const float *cos_base, const float *sin_base,
								   int m) {
	int half = head_dim / 2;
	return SEND_VALS(self, RPC_OP_rope_batch, RPC_V(vec), RPC_V(n_heads), RPC_V(head_dim),
					 RPC_V(pos_start), RPC_V(tbl(self, cos_base, half, pos_start, m)),
					 RPC_V(tbl(self, sin_base, half, pos_start, m)), RPC_V(m));
}

static status_code stub_qk_norm_rope(backend *self, buffer *q, buffer *k, const buffer *q_w,
									 const buffer *k_w, int n_heads, int n_kv_heads, int head_dim,
									 float eps, int pos_start, const float *cos_base,
									 const float *sin_base, const float *freq_factors, int m) {
	int half = head_dim / 2;
	return SEND_VALS(self, RPC_OP_qk_norm_rope, RPC_V(q), RPC_V(k), RPC_V(q_w), RPC_V(k_w),
					 RPC_V(n_heads), RPC_V(n_kv_heads), RPC_V(head_dim), RPC_V(eps),
					 RPC_V(pos_start), RPC_V(tbl(self, cos_base, half, pos_start, m)),
					 RPC_V(tbl(self, sin_base, half, pos_start, m)),
					 RPC_V(tbl(self, freq_factors, half, 0, 1)), RPC_V(m));
}

static status_code stub_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads,
									  int n_kv_heads, int head_dim, int pos_start,
									  const float *cos_base, const float *sin_base, int m) {
	int half = head_dim / 2;
	return SEND_VALS(self, RPC_OP_rope_qk_batch, RPC_V(q), RPC_V(k), RPC_V(n_heads),
					 RPC_V(n_kv_heads), RPC_V(head_dim), RPC_V(pos_start),
					 RPC_V(tbl(self, cos_base, half, pos_start, m)),
					 RPC_V(tbl(self, sin_base, half, pos_start, m)), RPC_V(m));
}

static status_code stub_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
										int n_kv_heads, int head_dim, int rope_dim, int pos_start,
										const float *cos_base, const float *sin_base, int n_rows) {
	int half = rope_dim / 2;
	return SEND_VALS(self, RPC_OP_partial_rope_qk, RPC_V(q), RPC_V(k), RPC_V(n_heads),
					 RPC_V(n_kv_heads), RPC_V(head_dim), RPC_V(rope_dim), RPC_V(pos_start),
					 RPC_V(tbl(self, cos_base, half, pos_start, n_rows)),
					 RPC_V(tbl(self, sin_base, half, pos_start, n_rows)), RPC_V(n_rows));
}

static status_code stub_attention_mla(backend *self, const buffer *q, const buffer *kv_cache,
									  const buffer *k_b_w, const buffer *v_b_w, buffer *out,
									  int layer, int pos, int n_heads, int qk_head, int qk_rope,
									  int qk_nope, int v_head, int kv_lora, int n_ctx,
									  const float *cos_base, const float *sin_base,
									  uint32_t k_b_type, uint32_t v_b_type, float scale) {
	int half = qk_rope / 2;
	return SEND_VALS(self, RPC_OP_attention_mla, RPC_V(q), RPC_V(kv_cache), RPC_V(k_b_w),
					 RPC_V(v_b_w), RPC_V(out), RPC_V(layer), RPC_V(pos), RPC_V(n_heads),
					 RPC_V(qk_head), RPC_V(qk_rope), RPC_V(qk_nope), RPC_V(v_head), RPC_V(kv_lora),
					 RPC_V(n_ctx), RPC_V(tbl(self, cos_base, half, 0, pos + 1)),
					 RPC_V(tbl(self, sin_base, half, 0, pos + 1)), RPC_V(k_b_type), RPC_V(v_b_type),
					 RPC_V(scale));
}

static void write_multi(rpc_wr *w, backend *self, const buffer **bw, const uint32_t *types,
						const buffer *x, buffer **y, const int *n_list, int k, int n) {
	rpc_wr_i32(w, n);
	for (int i = 0; i < n; i++) {
		rpc_wr_buf(w, self, bw[i]);
		rpc_wr_u32(w, types[i]);
		rpc_wr_i32(w, n_list[i]);
		rpc_wr_buf(w, self, y[i]);
	}
	rpc_wr_buf(w, self, x);
	rpc_wr_i32(w, k);
}

static status_code stub_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
									 const buffer *x, buffer **y, const int *n_list, int k,
									 int n_matmuls) {
	rpc_wr m;
	rpc_wr_init(&m);
	write_multi(&m, self, w, w_types, x, y, n_list, k, n_matmuls);
	return op_send(self, RPC_OP_matmul_multi, &m);
}

static status_code stub_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
										   const buffer *x, buffer **y, const int *n_list, int k,
										   int n_matmuls, int m) {
	rpc_wr wr;
	rpc_wr_init(&wr);
	write_multi(&wr, self, w, w_types, x, y, n_list, k, n_matmuls);
	rpc_wr_i32(&wr, m);
	return op_send(self, RPC_OP_matmul_multi_batch, &wr);
}

static status_code call_i32(backend *self, uint32_t op, rpc_val *vals, size_t n, int32_t *out) {
	rpc_wr w;
	rpc_wr_init(&w);
	for (size_t i = 0; i < n; i++)
		rpc_wr_val(&w, self, vals[i]);
	rpc_reply	r;
	status_code s = op_call(self, op, &w, &r);
	if (s != OK)
		return s;
	rpc_rd rd;
	rpc_rd_init(&rd, r.data, r.len);
	*out = rpc_rd_i32(&rd);
	s	 = rd.bad ? ERR_FORMAT : OK;
	rpc_reply_free(&r);
	return s;
}

static status_code stub_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	return call_i32(self, RPC_OP_argmax, (rpc_val[]){RPC_V(logits), RPC_V(n)}, 2, out_idx);
}

static status_code stub_argmax_to_reg(backend *self, const buffer *logits, int n, int reg) {
	return SEND_VALS(self, RPC_OP_argmax_to_reg, RPC_V(logits), RPC_V(n), RPC_V(reg));
}

static status_code stub_read_token_reg(backend *self, int reg, int32_t *out_token) {
	return call_i32(self, RPC_OP_read_token_reg, (rpc_val[]){RPC_V(reg)}, 1, out_token);
}

static void write_sample(rpc_wr *w, backend *self, const buffer *logits, int n,
						 const device_sample_params *p) {
	rpc_wr_buf(w, self, logits);
	rpc_wr_i32(w, n);
	rpc_wr_bytes(w, p, sizeof(*p));
}

static status_code stub_sample(backend *self, const buffer *logits, int n,
							   const device_sample_params *params, int32_t *out_idx) {
	rpc_wr w;
	rpc_wr_init(&w);
	write_sample(&w, self, logits, n, params);
	rpc_reply	r;
	status_code s = op_call(self, RPC_OP_sample, &w, &r);
	if (s != OK)
		return s;
	rpc_rd rd;
	rpc_rd_init(&rd, r.data, r.len);
	*out_idx = rpc_rd_i32(&rd);
	s		 = rd.bad ? ERR_FORMAT : OK;
	rpc_reply_free(&r);
	return s;
}

static status_code stub_sample_to_reg(backend *self, const buffer *logits, int n,
									  const device_sample_params *params, int reg) {
	rpc_wr w;
	rpc_wr_init(&w);
	write_sample(&w, self, logits, n, params);
	rpc_wr_i32(&w, reg);
	return op_send(self, RPC_OP_sample_to_reg, &w);
}

static status_code stub_sample_history_set(backend *self, int slot, const int32_t *tokens, int n) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_i32(&w, slot);
	rpc_wr_i32(&w, n);
	rpc_wr_bytes(&w, tokens, (size_t)MAX(n, 0) * sizeof(int32_t));
	return op_send(self, RPC_OP_sample_history_set, &w);
}

static status_code read_floats(backend *self, rpc_wr *w, rpc_reply *r, float *dst, int n) {
	status_code s = op_call(self, RPC_OP_buffer_read_f32, w, r);
	if (s == OK && r->len < (size_t)n * sizeof(float))
		s = ERR_FORMAT;
	if (s == OK)
		memcpy(dst, r->data, (size_t)n * sizeof(float));
	return s;
}

static status_code stub_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_i32(&w, 1);
	rpc_wr_buf(&w, self, buf);
	rpc_wr_i32(&w, n);
	rpc_reply	r;
	status_code s = read_floats(self, &w, &r, host_dst, n);
	if (s == OK)
		rpc_reply_free(&r);
	return s;
}

static status_code stub_buffer_read_f32_x2(backend *self, const buffer *a, float *dst_a, int n_a,
										   const buffer *b, float *dst_b, int n_b) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_i32(&w, 2);
	rpc_wr_buf(&w, self, a);
	rpc_wr_i32(&w, n_a);
	rpc_wr_buf(&w, self, b);
	rpc_wr_i32(&w, n_b);
	rpc_reply	r;
	status_code s = op_call(self, RPC_OP_buffer_read_f32, &w, &r);
	if (s == OK && r.len < ((size_t)n_a + (size_t)n_b) * sizeof(float))
		s = ERR_FORMAT;
	if (s != OK)
		return s;
	memcpy(dst_a, r.data, (size_t)n_a * sizeof(float));
	memcpy(dst_b, r.data + (size_t)n_a * sizeof(float), (size_t)n_b * sizeof(float));
	rpc_reply_free(&r);
	return OK;
}

static status_code stub_buffer_read_f32_post(backend *self, const buffer *a, int n_a,
											 const buffer *b, int n_b) {
	rpc_priv *p = priv_of(self);
	drain_pending_read(self);
	if (p->stash_valid) {
		rpc_reply_free(&p->stash);
		p->stash_valid = false;
	}
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_i32(&w, b ? 2 : 1);
	rpc_wr_buf(&w, self, a);
	rpc_wr_i32(&w, n_a);
	if (b) {
		rpc_wr_buf(&w, self, b);
		rpc_wr_i32(&w, n_b);
	}
	status_code s = w.bad ? ERR_UNSUPPORTED : OK;
	if (s == OK) {
		sync_params(self);
		s = rpc_call_begin(&p->conn, RPC_OP_buffer_read_f32, w.p, w.len);
	}
	rpc_wr_free(&w);
	if (s == OK) {
		p->read_pending = true;
		p->pend_n[0]	= n_a;
		p->pend_n[1]	= b ? n_b : 0;
	}
	return s;
}

static status_code stub_buffer_read_f32_wait(backend *self, float *dst_a, float *dst_b) {
	rpc_priv *p = priv_of(self);
	drain_pending_read(self);
	if (!p->stash_valid)
		return p->stash_err != OK ? p->stash_err : ERR_INTERNAL;
	rpc_reply r	   = p->stash;
	p->stash_valid = false;
	memset(&p->stash, 0, sizeof(p->stash));
	status_code s  = r.status != OK ? (status_code)r.status : OK;
	size_t		na = (size_t)p->pend_n[0], nb = (size_t)p->pend_n[1];
	if (s == OK && r.len < (na + nb) * sizeof(float))
		s = ERR_FORMAT;
	if (s == OK) {
		if (dst_a)
			memcpy(dst_a, r.data, na * sizeof(float));
		if (nb && dst_b)
			memcpy(dst_b, r.data + na * sizeof(float), nb * sizeof(float));
	}
	rpc_reply_free(&r);
	return s;
}

static status_code stub_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_buf(&w, self, buf);
	rpc_wr_i32(&w, n);
	rpc_wr_bytes(&w, host_src, (size_t)MAX(n, 0) * sizeof(float));
	return op_send(self, RPC_OP_buffer_write_f32, &w);
}

static void invalidate_mem(backend *self) {
	priv_of(self)->mem_avail_us = 0;
}

static status_code call_alloc(backend *self, uint32_t op, rpc_wr *w, buffer *out) {
	rpc_reply	r;
	status_code s = op_call(self, op, w, &r);
	if (s != OK)
		return s;
	rpc_rd rd;
	rpc_rd_init(&rd, r.data, r.len);
	uint32_t id	  = rpc_rd_u32(&rd);
	uint64_t size = rpc_rd_u64(&rd);
	s			  = rd.bad ? ERR_FORMAT : OK;
	if (s == OK) {
		fill_buffer(self, out, id, size);
		invalidate_mem(self);
	}
	rpc_reply_free(&r);
	return s;
}

static status_code stub_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u64(&w, size);
	return call_alloc(self, RPC_OP_buffer_alloc_scratch, &w, out);
}

static void stub_buffer_free(backend *self, buffer *buf) {
	if (!buf->handle)
		return;
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, (uint32_t)(uintptr_t)buf->handle);
	invalidate_mem(self);
	op_send(self, RPC_OP_buffer_free, &w);
}

typedef status_code (*upload_read_fn)(const void *src, size_t off, void *dst, size_t len);

static status_code read_tensor_desc(const void *src, size_t off, void *dst, size_t len) {
	return tensor_desc_read(src, off, dst, len);
}

static status_code read_host_memory(const void *src, size_t off, void *dst, size_t len) {
	memcpy(dst, (const char *)src + off, len);
	return OK;
}

static status_code upload_bytes(backend *self, const void *src, upload_read_fn read, size_t total) {
	rpc_priv *p = priv_of(self);
	rpc_wr	  w;
	rpc_wr_init(&w);
	rpc_wr_u64(&w, total);
	status_code s = rpc_send(&p->conn, RPC_CTL_UPLOAD_BEGIN, w.p, w.len);
	rpc_wr_free(&w);
	uint8_t *chunk = total ? xmalloc(MIN(total, RPC_UPLOAD_CHUNK)) : NULL;
	for (size_t off = 0; s == OK && off < total; off += RPC_UPLOAD_CHUNK) {
		size_t n = MIN(total - off, RPC_UPLOAD_CHUNK);
		s		 = read(src, off, chunk, n);
		if (s == OK)
			s = rpc_send(&p->conn, RPC_CTL_UPLOAD_DATA, chunk, n);
	}
	free(chunk);
	return s;
}

static status_code stub_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	size_t total = rpc_tensor_bytes(desc->type, desc->n_dims, desc->dims);
	if (total == 0)
		return ERR_INVALID_ARG;
	status_code s = upload_bytes(self, desc, read_tensor_desc, total);
	if (s != OK)
		return s;
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, desc->type);
	rpc_wr_i32(&w, desc->n_dims);
	for (int i = 0; i < 4; i++)
		rpc_wr_u64(&w, desc->dims[i]);
	rpc_wr_u8(&w, desc->repack != 0);
	return call_alloc(self, RPC_OP_buffer_alloc_weight, &w, out);
}

static status_code stub_buffer_alloc_from_host(backend *self, const void *host_data, size_t size,
											   buffer *out) {
	status_code s = upload_bytes(self, host_data, read_host_memory, size);
	if (s != OK)
		return s;
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u64(&w, size);
	return call_alloc(self, RPC_OP_buffer_alloc_from_host, &w, out);
}

static status_code stub_dequant_weight(backend *self, uint32_t type, const void *src, int n_rows,
									   int k, buffer *out) {
	size_t		total = ggml_row_size(type, (size_t)k) * (size_t)n_rows;
	status_code s	  = upload_bytes(self, src, read_host_memory, total);
	if (s != OK)
		return s;
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, type);
	rpc_wr_i32(&w, n_rows);
	rpc_wr_i32(&w, k);
	return call_alloc(self, RPC_OP_dequant_weight, &w, out);
}

static void write_int_array(rpc_wr *w, const int *a, int n) {
	rpc_wr_u8(w, a != NULL);
	for (int i = 0; a && i < n; i++)
		rpc_wr_i32(w, a[i]);
}

static status_code stub_kv_alloc(backend *self, const kv_desc *d, buffer *k_out, buffer *v_out) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_i32(&w, d->n_layers);
	rpc_wr_i32(&w, d->n_kv_layers);
	rpc_wr_i32(&w, d->n_kv_heads);
	rpc_wr_i32(&w, d->head_dim);
	rpc_wr_i32(&w, d->n_ctx);
	rpc_wr_i32(&w, (int32_t)d->kv_quant);
	write_int_array(&w, d->layer_head_dim, d->n_layers);
	write_int_array(&w, d->layer_n_kv_heads, d->n_layers);
	write_int_array(&w, d->layer_pos_cap, d->n_layers);
	rpc_reply	r;
	status_code s = op_call(self, RPC_OP_kv_alloc, &w, &r);
	if (s != OK)
		return s;
	rpc_rd rd;
	rpc_rd_init(&rd, r.data, r.len);
	uint32_t kid   = rpc_rd_u32(&rd);
	uint64_t ksize = rpc_rd_u64(&rd);
	uint32_t vid   = rpc_rd_u32(&rd);
	uint64_t vsize = rpc_rd_u64(&rd);
	s			   = rd.bad ? ERR_FORMAT : OK;
	if (s == OK) {
		fill_buffer(self, k_out, kid, ksize);
		fill_buffer(self, v_out, vid, vsize);
		invalidate_mem(self);
	}
	rpc_reply_free(&r);
	return s;
}

static void stub_kv_free(backend *self, buffer *k, buffer *v) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, (uint32_t)(uintptr_t)k->handle);
	rpc_wr_u32(&w, (uint32_t)(uintptr_t)v->handle);
	op_send(self, RPC_OP_kv_free, &w);
}

static status_code stub_kv_alloc_mla(backend *self, int n_layers, int n_ctx, int kv_lora,
									 int qk_rope, buffer *kv_out) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_i32(&w, n_layers);
	rpc_wr_i32(&w, n_ctx);
	rpc_wr_i32(&w, kv_lora);
	rpc_wr_i32(&w, qk_rope);
	return call_alloc(self, RPC_OP_kv_alloc_mla, &w, kv_out);
}

static status_code send_desc(backend *self, uint32_t op, const void *desc, rpc_desc_kind kind) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_desc(&w, self, desc, kind);
	return op_send(self, op, &w);
}

static status_code stub_gated_delta_net(backend *self, const gdn_desc *d) {
	return send_desc(self, RPC_OP_gated_delta_net, d, RPC_DESC_GDN);
}

static status_code stub_hc_init(backend *self, const hc_init_desc *d) {
	return send_desc(self, RPC_OP_hc_init, d, RPC_DESC_HC_INIT);
}

static status_code stub_hc_pre(backend *self, const hc_pre_desc *d) {
	return send_desc(self, RPC_OP_hc_pre, d, RPC_DESC_HC_PRE);
}

static status_code stub_hc_post(backend *self, const hc_post_desc *d) {
	return send_desc(self, RPC_OP_hc_post, d, RPC_DESC_HC_POST);
}

static status_code stub_hc_final(backend *self, const hc_final_desc *d) {
	return send_desc(self, RPC_OP_hc_final, d, RPC_DESC_HC_FINAL);
}

static status_code stub_kda(backend *self, const kda_desc *d) {
	return send_desc(self, RPC_OP_kda, d, RPC_DESC_KDA);
}

static status_code stub_mla_absorbed(backend *self, const mla_absorbed_desc *d) {
	return send_desc(self, RPC_OP_mla_absorbed, d, RPC_DESC_MLA_ABSORBED);
}

static status_code stub_attention_dense(backend *self, const dense_attention_desc *d) {
	return send_desc(self, RPC_OP_attention_dense, d, RPC_DESC_DENSE_ATTENTION);
}

static status_code stub_im2col3x3(backend *self, const im2col_desc *d) {
	return send_desc(self, RPC_OP_im2col3x3, d, RPC_DESC_IM2COL);
}

static status_code stub_upsample_shortcut_add(backend *self, const upsample_shortcut_desc *d) {
	return send_desc(self, RPC_OP_upsample_shortcut_add, d, RPC_DESC_UPSAMPLE_SHORTCUT);
}

static status_code stub_moe_expert_ffn(backend *self, const buffer *x, buffer *out,
									   const moe_resident_expert *e, int dim, int inter) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_buf(&w, self, x);
	rpc_wr_buf(&w, self, out);
	rpc_wr_expert(&w, self, e);
	rpc_wr_i32(&w, dim);
	rpc_wr_i32(&w, inter);
	return op_send(self, RPC_OP_moe_expert_ffn, &w);
}

static status_code stub_moe_experts_batch(backend *self, const buffer *xb, buffer *out, int n_rows,
										  int dim, int inter, int use_gelu, int n_experts,
										  const moe_resident_expert *experts, const int *counts,
										  const int *rows_packed, const float *weights_packed) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_buf(&w, self, xb);
	rpc_wr_buf(&w, self, out);
	rpc_wr_i32(&w, n_rows);
	rpc_wr_i32(&w, dim);
	rpc_wr_i32(&w, inter);
	rpc_wr_i32(&w, use_gelu);
	rpc_wr_i32(&w, n_experts);
	size_t total = 0;
	for (int i = 0; i < n_experts; i++) {
		rpc_wr_i32(&w, counts[i]);
		if (counts[i] > 0)
			rpc_wr_expert(&w, self, &experts[i]);
		total += (size_t)MAX(counts[i], 0);
	}
	rpc_wr_bytes(&w, rows_packed, total * sizeof(int));
	rpc_wr_bytes(&w, weights_packed, total * sizeof(float));
	return op_send(self, RPC_OP_moe_experts_batch, &w);
}

static void send_plain(backend *self, uint32_t op, int32_t arg, bool with_arg) {
	rpc_wr w;
	rpc_wr_init(&w);
	if (with_arg)
		rpc_wr_i32(&w, arg);
	op_send(self, op, &w);
}

static void stub_synchronize(backend *self) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_reply r;
	if (op_call(self, RPC_OP_synchronize, &w, &r) == OK)
		rpc_reply_free(&r);
}

static void stub_submit(backend *self) {
	send_plain(self, RPC_OP_submit, 0, false);
}

static void stub_begin_batch(backend *self) {
	send_plain(self, RPC_OP_begin_batch, 0, false);
}

static void stub_end_batch(backend *self) {
	send_plain(self, RPC_OP_end_batch, 0, false);
}

static void stub_set_gpu_profile(backend *self, int enable) {
	send_plain(self, RPC_OP_set_gpu_profile, enable, true);
}

static int stub_matmul_type_native(backend *self, uint32_t w_type) {
	return w_type < RPC_TYPE_MAX && priv_of(self)->type_native[w_type];
}

static int stub_dequant_type_native(backend *self, uint32_t type, int k) {
	int32_t out = 0;
	if (call_i32(self, RPC_OP_dequant_type_native, (rpc_val[]){RPC_V(type), RPC_V(k)}, 2, &out) !=
		OK)
		return 0;
	return out;
}

static size_t call_size(backend *self, uint32_t op) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_reply r;
	if (op_call(self, op, &w, &r) != OK)
		return 0;
	rpc_rd rd;
	rpc_rd_init(&rd, r.data, r.len);
	size_t v = (size_t)rpc_rd_u64(&rd);
	rpc_reply_free(&r);
	return rd.bad ? 0 : v;
}

static size_t stub_mem_available(backend *self) {
	rpc_priv *p	  = priv_of(self);
	uint64_t  now = time_us();
	if (p->mem_avail_us && now - p->mem_avail_us < RPC_MEM_CACHE_US)
		return p->mem_avail;
	p->mem_avail	= call_size(self, RPC_OP_mem_available);
	p->mem_avail_us = now ? now : 1;
	return p->mem_avail;
}

static size_t stub_mem_total(backend *self) {
	rpc_priv *p = priv_of(self);
	if (!p->mem_total)
		p->mem_total = call_size(self, RPC_OP_mem_total);
	return p->mem_total;
}

static status_code query_repack_plan(backend *self, uint32_t type, uint64_t d0, uint64_t d1,
									 uint32_t *re_type) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, type);
	rpc_wr_u64(&w, d0);
	rpc_wr_u64(&w, d1);
	rpc_reply	r;
	status_code s = op_call(self, RPC_OP_repack_plan, &w, &r);
	if (s != OK)
		return s;
	rpc_rd rd;
	rpc_rd_init(&rd, r.data, r.len);
	*re_type = rpc_rd_u32(&rd);
	s		 = rd.bad ? ERR_FORMAT : OK;
	rpc_reply_free(&r);
	return s;
}

static status_code stub_repack_plan(backend *self, uint32_t type, uint64_t d0, uint64_t d1,
									uint32_t *re_type_out) {
	rpc_priv *p = priv_of(self);
	for (int i = 0; i < p->n_plans; i++) {
		const rpc_plan_entry *e = &p->plans[i];
		if (e->type == type && e->d0 == d0 && e->d1 == d1) {
			*re_type_out = e->re_type;
			return e->re_type ? OK : ERR_UNSUPPORTED;
		}
	}
	uint32_t	re_type = 0;
	status_code s		= query_repack_plan(self, type, d0, d1, &re_type);
	if (s != OK)
		return s;
	if (p->n_plans < RPC_PLAN_CACHE)
		p->plans[p->n_plans++] = (rpc_plan_entry){type, d0, d1, re_type};
	if (!re_type)
		return ERR_UNSUPPORTED;
	*re_type_out = re_type;
	return OK;
}

static status_code rpc_backend_probe(void) {
	rpc_endpoint eps[RPC_MAX_ENDPOINTS];
	return rpc_endpoints_load(eps, RPC_MAX_ENDPOINTS) > 0 ? OK : ERR_NOT_FOUND;
}

static int rpc_backend_device_count(void) {
	rpc_endpoint eps[RPC_MAX_ENDPOINTS];
	return rpc_endpoints_load(eps, RPC_MAX_ENDPOINTS);
}

static status_code rpc_backend_identity(int device_index, char *out, size_t cap) {
	rpc_endpoint eps[RPC_MAX_ENDPOINTS];
	int			 n = rpc_endpoints_load(eps, RPC_MAX_ENDPOINTS);
	if (device_index < 0 || device_index >= n)
		return ERR_NOT_FOUND;
	char addr[RPC_HOST_MAX + 16];
	rpc_endpoint_format(&eps[device_index], addr, sizeof(addr));
	rpc_info	info;
	status_code s = rpc_probe(&eps[device_index], &info);
	if (s == OK)
		snprintf(out, cap, "%s @ %s", info.identity[0] ? info.identity : info.device, addr);
	else
		snprintf(out, cap, "%s (unreachable)", addr);
	return OK;
}

static void read_hello_exts(rpc_rd *rd, rpc_priv *p) {
	if (rd->pos >= rd->len)
		return;
	uint16_t n = rpc_rd_u16(rd);
	for (uint16_t i = 0; i < n && !rd->bad; i++) {
		uint16_t	   id  = rpc_rd_u16(rd);
		uint16_t	   len = rpc_rd_u16(rd);
		const uint8_t *v   = rpc_rd_bytes(rd, len);
		if (!v)
			return;
		if (id == RPC_EXT_DEVICE)
			snprintf(p->device, sizeof(p->device), "%.*s", (int)len, (const char *)v);
		else if (id == RPC_EXT_MEM_TOTAL && len == sizeof(p->mem_total))
			memcpy(&p->mem_total, v, sizeof(p->mem_total));
	}
}

static status_code hello(backend *self, rpc_priv *p, bool *present) {
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, RPC_MAGIC);
	rpc_wr_u32(&w, RPC_VERSION);
	rpc_reply	r;
	status_code s = rpc_call(&p->conn, RPC_CTL_HELLO, w.p, w.len, &r);
	rpc_wr_free(&w);
	if (s == OK && r.status != OK)
		s = (status_code)r.status;
	if (s != OK) {
		rpc_reply_free(&r);
		return s;
	}
	rpc_rd rd;
	rpc_rd_init(&rd, r.data, r.len);
	uint64_t caps		   = rpc_rd_u64(&rd);
	self->sample_max_vocab = rpc_rd_i32(&rd);
	rpc_rd_str(&rd, p->desc, sizeof(p->desc));
	uint32_t n_ops = rpc_rd_u32(&rd);
	for (uint32_t i = 0; i < n_ops && !rd.bad; i++) {
		uint16_t id = rpc_rd_u16(&rd);
		if (id < RPC_OP_COUNT)
			present[id] = true;
	}
	const uint8_t *types = rpc_rd_bytes(&rd, RPC_TYPE_MAX);
	if (types)
		memcpy(p->type_native, types, RPC_TYPE_MAX);
	snprintf(p->device, sizeof(p->device), "%.*s", (int)sizeof(p->device) - 1, p->desc);
	read_hello_exts(&rd, p);
	self->caps = (caps & ~(uint64_t)RPC_FORWARD_MASK) | BCAP_REMOTE;
	s		   = rd.bad ? ERR_FORMAT : OK;
	rpc_reply_free(&r);
	return s;
}

static void install_ops(backend *self, const bool *present) {
	RPC_ALL_OPS(INSTALL_OP)
	if (present[RPC_OP_buffer_read_f32]) {
		self->buffer_read_f32_post = stub_buffer_read_f32_post;
		self->buffer_read_f32_wait = stub_buffer_read_f32_wait;
	}
}

static status_code rpc_backend_init(backend *self, int device_index) {
	rpc_endpoint eps[RPC_MAX_ENDPOINTS];
	int			 n = rpc_endpoints_load(eps, RPC_MAX_ENDPOINTS);
	if (device_index < 0 || device_index >= n)
		return ERR_NOT_FOUND;
	rpc_priv *p = xcalloc(1, sizeof(*p));
	pthread_mutex_init(&p->table_mtx, NULL);
	char addr[RPC_HOST_MAX + 16];
	rpc_endpoint_format(&eps[device_index], addr, sizeof(addr));
	if (rpc_connect(&p->conn, &eps[device_index], RPC_CONNECT_TIMEOUT_MS) != OK) {
		ERROR("rpc: cannot connect to worker %s", addr);
		free(p);
		return ERR_IO;
	}
	self->priv						  = p;
	bool		present[RPC_OP_COUNT] = {false};
	status_code s					  = hello(self, p, present);
	if (s != OK) {
		ERROR("rpc: handshake with worker %s failed (%s)", addr, status_str(s));
		rpc_conn_close(&p->conn);
		pthread_mutex_destroy(&p->table_mtx);
		free(p);
		self->priv = NULL;
		return s;
	}
	install_ops(self, present);
	self->desc = p->desc;
	log_tag("RPC", "Worker: %s (%s)", addr, p->device);
	return OK;
}

static void rpc_backend_free(backend *self) {
	rpc_priv *p = self->priv;
	if (!p)
		return;
	rpc_conn_close(&p->conn);
	if (p->stash_valid)
		rpc_reply_free(&p->stash);
	pthread_mutex_destroy(&p->table_mtx);
	free(p);
	self->priv = NULL;
}

static status_code rpc_backend_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name			 = "rpc";
	out->priority		 = RPC_BACKEND_PRIORITY;
	out->caps			 = BCAP_REMOTE;
	out->desc			 = "remote worker (rpc)";
	out->probe			 = rpc_backend_probe;
	out->device_count	 = rpc_backend_device_count;
	out->device_identity = rpc_backend_identity;
	out->init			 = rpc_backend_init;
	out->free			 = rpc_backend_free;
	return OK;
}

BACKEND_REGISTER("rpc", rpc_backend_ctor)
