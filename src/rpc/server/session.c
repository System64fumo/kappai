#include "rpc/server/session.h"

#include "gguf.h"
#include "log.h"

#include <pthread.h>

#define RPC_VB(n) ((buffer *)msg->vals[n].v.b)
#define RPC_CB(n) (msg->vals[n].v.b)
#define RPC_I(n) (msg->vals[n].v.i)
#define RPC_F(n) (msg->vals[n].v.f)
#define RPC_U(n) (msg->vals[n].v.u)
#define TBL(n) table_data(s, msg->vals[n].v.u)
#define MARK_OP(name)                                                                              \
	if (be->name)                                                                                  \
		present[RPC_OP_##name] = true;
#define MULTI_MAX 16
#define HELLO_SYNC_TYPES RPC_TYPE_MAX

static bool op_is_sync(uint32_t op) {
	switch (op) {
	case RPC_CTL_INFO:
	case RPC_CTL_HELLO:
	case RPC_OP_buffer_read_f32:
	case RPC_OP_argmax:
	case RPC_OP_read_token_reg:
	case RPC_OP_sample:
	case RPC_OP_buffer_alloc_weight:
	case RPC_OP_buffer_alloc_scratch:
	case RPC_OP_buffer_alloc_from_host:
	case RPC_OP_dequant_weight:
	case RPC_OP_kv_alloc:
	case RPC_OP_kv_alloc_mla:
	case RPC_OP_dequant_type_native:
	case RPC_OP_mem_available:
	case RPC_OP_mem_total:
	case RPC_OP_synchronize:
	case RPC_OP_repack_plan:
		return true;
	default:
		return false;
	}
}

static bool resolve_buffer(void *ctx, uint32_t id, uint64_t off, uint64_t size, buffer *out) {
	srv_session *s = ctx;
	if (id == 0 || id > s->n_bufs || !s->bufs[id - 1].used)
		return false;
	*out = buffer_slice(&s->bufs[id - 1].buf, off, size);
	return true;
}

static uint32_t buf_register(srv_session *s, const buffer *b, void *owned) {
	uint32_t i = 0;
	while (i < s->n_bufs && s->bufs[i].used)
		i++;
	if (i == s->n_bufs) {
		s->bufs = xrealloc(s->bufs, (size_t)(s->n_bufs + 256) * sizeof(srv_buf));
		memset(s->bufs + s->n_bufs, 0, 256 * sizeof(srv_buf));
		s->n_bufs += 256;
	}
	s->bufs[i] = (srv_buf){.buf = *b, .owned = owned, .used = true};
	return i + 1;
}

static void buf_release(srv_session *s, uint32_t id) {
	if (id == 0 || id > s->n_bufs || !s->bufs[id - 1].used)
		return;
	srv_buf *r = &s->bufs[id - 1];
	s->be->buffer_free(s->be, &r->buf);
	free(r->owned);
	memset(r, 0, sizeof(*r));
}

static float *table_data(srv_session *s, uint32_t id) {
	return id && id < ARRAY_LEN(s->tables) ? s->tables[id].data : NULL;
}

static status_code table_store(srv_session *s, rpc_rd *rd) {
	uint32_t	id	   = rpc_rd_u32(rd);
	uint32_t	half   = rpc_rd_u32(rd);
	uint32_t	row0   = rpc_rd_u32(rd);
	uint32_t	n_rows = rpc_rd_u32(rd);
	size_t		bytes  = (size_t)n_rows * half * sizeof(float);
	const void *src	   = rpc_rd_bytes(rd, bytes);
	if (!src || id == 0 || id >= ARRAY_LEN(s->tables) || half == 0)
		return ERR_FORMAT;
	srv_table *t	= &s->tables[id];
	size_t	   need = ((size_t)row0 + n_rows) * half;
	if (need > t->cap_floats) {
		t->data		  = xrealloc(t->data, need * sizeof(float));
		t->cap_floats = need;
	}
	memcpy(t->data + (size_t)row0 * half, src, bytes);
	return OK;
}

static status_code upload_begin(srv_session *s, rpc_rd *rd) {
	uint64_t total = rpc_rd_u64(rd);
	free(s->upload);
	s->upload		= NULL;
	s->upload_len	= 0;
	s->upload_total = 0;
	if (rd->bad || total > RPC_MAX_FRAME * 64)
		return ERR_FORMAT;
	s->upload		= xmalloc_aligned(total ? total : 1, 64);
	s->upload_total = total;
	return OK;
}

static status_code upload_data(srv_session *s, const uint8_t *p, size_t len) {
	if (!s->upload || len > s->upload_total - s->upload_len)
		return ERR_FORMAT;
	memcpy(s->upload + s->upload_len, p, len);
	s->upload_len += len;
	return OK;
}

static void upload_drop(srv_session *s) {
	free(s->upload);
	s->upload		= NULL;
	s->upload_len	= 0;
	s->upload_total = 0;
}

static bool upload_ready(const srv_session *s, size_t expected) {
	return s->upload && s->upload_len == s->upload_total && s->upload_total == expected;
}

static status_code reply_alloc(srv_session *s, rpc_wr *out, buffer *b, void *owned, bool keep) {
	if (!keep)
		free(owned);
	uint32_t id = buf_register(s, b, keep ? owned : NULL);
	rpc_wr_u32(out, id);
	rpc_wr_u64(out, b->size);
	return OK;
}

static bool retained(const buffer *b, const void *data) {
	return b->handle == data || b->host_ptr == data;
}

static status_code repack_upload(srv_session *s, tensor_desc *d, void **packed) {
	backend *be = s->be;
	uint32_t re = 0;
	*packed		= NULL;
	if (d->n_dims != 2 || !backend_has_cap(be, BCAP_IS_HOST) || !be->repack_plan ||
		!be->repack_weight || be->repack_plan(be, d->type, d->dims[0], d->dims[1], &re) != OK ||
		!re)
		return OK;
	void *dst = xmalloc_aligned(ggml_row_size(re, (size_t)d->dims[0]) * (size_t)d->dims[1], 64);
	status_code st =
		be->repack_weight(be, d->type, s->upload, dst, (int)d->dims[1], (int)d->dims[0]);
	if (st != OK) {
		free(dst);
		return st;
	}
	d->type		 = re;
	d->host_data = dst;
	*packed		 = dst;
	return OK;
}

static status_code op_alloc_weight(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	backend	   *be = s->be;
	tensor_desc d  = {0};
	d.type		   = rpc_rd_u32(rd);
	d.n_dims	   = rpc_rd_i32(rd);
	for (int i = 0; i < 4; i++)
		d.dims[i] = rpc_rd_u64(rd);
	d.repack = rpc_rd_u8(rd);
	if (rd->bad || d.n_dims < 1 || d.n_dims > 4)
		return ERR_FORMAT;
	size_t total = rpc_tensor_bytes(d.type, d.n_dims, d.dims);
	if (!upload_ready(s, total))
		return ERR_FORMAT;
	d.host_data		   = s->upload;
	void	   *packed = NULL;
	status_code st	   = d.repack ? repack_upload(s, &d, &packed) : OK;
	if (st != OK)
		return st;
	size_t stored = packed ? rpc_tensor_bytes(d.type, d.n_dims, d.dims) : total;
	void  *keep	  = packed ? packed : s->upload;
	buffer b	  = {0};
	st			  = be->buffer_alloc_weight(be, &d, &b);
	if (packed)
		upload_drop(s);
	else
		s->upload = NULL;
	if (st != OK) {
		free(keep);
		return st;
	}
	b.size = MAX(b.size, stored);
	return reply_alloc(s, out, &b, keep, retained(&b, keep));
}

static status_code op_repack_plan(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	uint32_t type = rpc_rd_u32(rd);
	uint64_t d0	  = rpc_rd_u64(rd);
	uint64_t d1	  = rpc_rd_u64(rd);
	if (rd->bad)
		return ERR_FORMAT;
	uint32_t re = 0;
	if (!backend_has_cap(s->be, BCAP_IS_HOST) || !s->be->repack_plan ||
		s->be->repack_plan(s->be, type, d0, d1, &re) != OK)
		re = 0;
	rpc_wr_u32(out, re);
	return OK;
}

static status_code op_alloc_from_host(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	backend *be	  = s->be;
	size_t	 size = (size_t)rpc_rd_u64(rd);
	if (rd->bad || !upload_ready(s, size))
		return ERR_FORMAT;
	buffer		b = {0};
	status_code st;
	if (be->buffer_alloc_from_host) {
		st = be->buffer_alloc_from_host(be, s->upload, size, &b);
	} else {
		tensor_desc d = {.type = GGML_TYPE_F32, .n_dims = 1, .dims = {size / 4}};
		d.host_data	  = s->upload;
		st			  = be->buffer_alloc_weight(be, &d, &b);
		b.size		  = size;
	}
	void *up  = s->upload;
	s->upload = NULL;
	upload_drop(s);
	if (st != OK) {
		free(up);
		return st;
	}
	return reply_alloc(s, out, &b, up, retained(&b, up));
}

static status_code op_dequant_weight(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	backend *be	   = s->be;
	uint32_t type  = rpc_rd_u32(rd);
	int		 rows  = rpc_rd_i32(rd);
	int		 k	   = rpc_rd_i32(rd);
	size_t	 total = ggml_row_size(type, (size_t)k) * (size_t)rows;
	if (rd->bad || !upload_ready(s, total))
		return ERR_FORMAT;
	buffer		b  = {0};
	status_code st = be->dequant_weight(be, type, s->upload, rows, k, &b);
	upload_drop(s);
	return st == OK ? reply_alloc(s, out, &b, NULL, false) : st;
}

static status_code op_alloc_scratch(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	size_t size = (size_t)rpc_rd_u64(rd);
	if (rd->bad)
		return ERR_FORMAT;
	buffer		b  = {0};
	status_code st = s->be->buffer_alloc_scratch(s->be, size, &b);
	return st == OK ? reply_alloc(s, out, &b, NULL, false) : st;
}

static const int *read_int_array(rpc_rd *rd, int n, int **storage) {
	if (!rpc_rd_u8(rd))
		return NULL;
	*storage = xmalloc((size_t)MAX(n, 1) * sizeof(int));
	for (int i = 0; i < n; i++)
		(*storage)[i] = rpc_rd_i32(rd);
	return *storage;
}

static status_code op_kv_alloc(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	kv_desc d	  = {0};
	d.n_layers	  = rpc_rd_i32(rd);
	d.n_kv_layers = rpc_rd_i32(rd);
	d.n_kv_heads  = rpc_rd_i32(rd);
	d.head_dim	  = rpc_rd_i32(rd);
	d.n_ctx		  = rpc_rd_i32(rd);
	d.kv_quant	  = (kv_quant_type)rpc_rd_i32(rd);
	if (rd->bad || d.n_layers < 0 || d.n_layers > (1 << 16))
		return ERR_FORMAT;
	int *hd = NULL, *kvh = NULL, *cap = NULL;
	d.layer_head_dim   = read_int_array(rd, d.n_layers, &hd);
	d.layer_n_kv_heads = read_int_array(rd, d.n_layers, &kvh);
	d.layer_pos_cap	   = read_int_array(rd, d.n_layers, &cap);
	buffer		k = {0}, v = {0};
	status_code st = rd->bad ? ERR_FORMAT : s->be->kv_alloc(s->be, &d, &k, &v);
	free(hd);
	free(kvh);
	free(cap);
	if (st != OK)
		return st;
	reply_alloc(s, out, &k, NULL, false);
	return reply_alloc(s, out, &v, NULL, false);
}

static status_code op_kv_alloc_mla(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	int n_layers = rpc_rd_i32(rd);
	int n_ctx	 = rpc_rd_i32(rd);
	int kv_lora	 = rpc_rd_i32(rd);
	int qk_rope	 = rpc_rd_i32(rd);
	if (rd->bad)
		return ERR_FORMAT;
	buffer		b  = {0};
	status_code st = s->be->kv_alloc_mla(s->be, n_layers, n_ctx, kv_lora, qk_rope, &b);
	return st == OK ? reply_alloc(s, out, &b, NULL, false) : st;
}

static status_code op_kv_free(srv_session *s, rpc_rd *rd) {
	uint32_t kid = rpc_rd_u32(rd);
	uint32_t vid = rpc_rd_u32(rd);
	if (rd->bad || kid > s->n_bufs || vid > s->n_bufs || !kid || !vid)
		return ERR_FORMAT;
	s->be->kv_free(s->be, &s->bufs[kid - 1].buf, &s->bufs[vid - 1].buf);
	memset(&s->bufs[kid - 1], 0, sizeof(srv_buf));
	memset(&s->bufs[vid - 1], 0, sizeof(srv_buf));
	return OK;
}

static status_code op_read_f32(srv_session *s, rpc_msg *msg, rpc_wr *out) {
	int32_t count = rpc_rd_i32(&msg->rd);
	if (count < 1 || count > 2)
		return ERR_FORMAT;
	for (int i = 0; i < count; i++) {
		const buffer *b = rpc_msg_buf(msg);
		int			  n = rpc_rd_i32(&msg->rd);
		if (msg->rd.bad || !b || n < 0)
			return ERR_FORMAT;
		float	   *tmp = xmalloc(MAX((size_t)n, 1) * sizeof(float));
		status_code st	= s->be->buffer_read_f32(s->be, b, tmp, n);
		if (st == OK)
			rpc_wr_bytes(out, tmp, (size_t)n * sizeof(float));
		free(tmp);
		if (st != OK)
			return st;
	}
	return OK;
}

static status_code op_write_f32(srv_session *s, rpc_msg *msg) {
	buffer	   *b	= (buffer *)rpc_msg_buf(msg);
	int			n	= rpc_rd_i32(&msg->rd);
	const void *src = n >= 0 ? rpc_rd_bytes(&msg->rd, (size_t)n * sizeof(float)) : NULL;
	if (msg->rd.bad || !b || (n > 0 && !src))
		return ERR_FORMAT;
	float *tmp = xmalloc(MAX((size_t)n, 1) * sizeof(float));
	if (n > 0)
		memcpy(tmp, src, (size_t)n * sizeof(float));
	status_code st = s->be->buffer_write_f32(s->be, b, tmp, n);
	free(tmp);
	return st;
}

static status_code op_multi(srv_session *s, rpc_msg *msg, bool batched) {
	int n = rpc_rd_i32(&msg->rd);
	if (n < 1 || n > MULTI_MAX)
		return ERR_FORMAT;
	const buffer *w[MULTI_MAX];
	uint32_t	  types[MULTI_MAX];
	int			  n_list[MULTI_MAX];
	buffer		 *y[MULTI_MAX];
	for (int i = 0; i < n; i++) {
		w[i]	  = rpc_msg_buf(msg);
		types[i]  = rpc_rd_u32(&msg->rd);
		n_list[i] = rpc_rd_i32(&msg->rd);
		y[i]	  = (buffer *)rpc_msg_buf(msg);
	}
	const buffer *x = rpc_msg_buf(msg);
	int			  k = rpc_rd_i32(&msg->rd);
	int			  m = batched ? rpc_rd_i32(&msg->rd) : 0;
	if (msg->rd.bad)
		return ERR_FORMAT;
	backend *be = s->be;
	return batched ? be->matmul_multi_batch(be, w, types, x, y, n_list, k, n, m)
				   : be->matmul_multi(be, w, types, x, y, n_list, k, n);
}

static status_code read_sample(rpc_msg *msg, const buffer **logits, int *n,
							   device_sample_params *params) {
	*logits			= rpc_msg_buf(msg);
	*n				= rpc_rd_i32(&msg->rd);
	const void *raw = rpc_rd_bytes(&msg->rd, sizeof(*params));
	if (msg->rd.bad || !raw)
		return ERR_FORMAT;
	memcpy(params, raw, sizeof(*params));
	return OK;
}

static status_code op_sample(srv_session *s, rpc_msg *msg, rpc_wr *out, bool to_reg) {
	const buffer		*logits;
	int					 n;
	device_sample_params params;
	status_code			 st = read_sample(msg, &logits, &n, &params);
	if (st != OK)
		return st;
	if (to_reg) {
		int reg = rpc_rd_i32(&msg->rd);
		return s->be->sample_to_reg(s->be, logits, n, &params, reg);
	}
	int32_t idx = 0;
	st			= s->be->sample(s->be, logits, n, &params, &idx);
	rpc_wr_i32(out, idx);
	return st;
}

static status_code op_history(srv_session *s, rpc_rd *rd) {
	int			slot = rpc_rd_i32(rd);
	int			n	 = rpc_rd_i32(rd);
	const void *src	 = n >= 0 ? rpc_rd_bytes(rd, (size_t)n * sizeof(int32_t)) : NULL;
	if (rd->bad || (n > 0 && !src))
		return ERR_FORMAT;
	int32_t *tmp = xmalloc(MAX((size_t)n, 1) * sizeof(int32_t));
	if (n > 0)
		memcpy(tmp, src, (size_t)n * sizeof(int32_t));
	status_code st = s->be->sample_history_set(s->be, slot, tmp, n);
	free(tmp);
	return st;
}

static int pool_threads(backend *be) {
	tpool *pool = backend_get_pool(be);
	return pool ? MAX(tpool_n_threads(pool), 1) : 1;
}

static status_code op_desc(srv_session *s, rpc_msg *msg, uint32_t op) {
	backend *be	  = s->be;
	bool	 host = backend_has_cap(be, BCAP_IS_HOST);
	union {
		gdn_desc			   gdn;
		hc_init_desc		   hc_init;
		hc_final_desc		   hc_final;
		hc_pre_desc			   hc_pre;
		hc_post_desc		   hc_post;
		kda_desc			   kda;
		mla_absorbed_desc	   mla;
		dense_attention_desc   dense;
		im2col_desc			   im2col;
		upsample_shortcut_desc up;
	} d;
	memset(&d, 0, sizeof(d));
	switch (op) {
	case RPC_OP_gated_delta_net:
		if (!rpc_msg_desc(msg, &d.gdn, RPC_DESC_GDN))
			return ERR_FORMAT;
		d.gdn.ws = host ? float_buf_ensure(&s->ws, (size_t)d.gdn.n_tokens * d.gdn.conv_dim +
													   (size_t)pool_threads(be) *
														   (2 * (size_t)d.gdn.state_size +
															2 * (size_t)d.gdn.value_head_dim))
						: NULL;
		return be->gated_delta_net(be, &d.gdn);
	case RPC_OP_hc_init:
		return rpc_msg_desc(msg, &d.hc_init, RPC_DESC_HC_INIT) ? be->hc_init(be, &d.hc_init)
															   : ERR_FORMAT;
	case RPC_OP_hc_final:
		return rpc_msg_desc(msg, &d.hc_final, RPC_DESC_HC_FINAL) ? be->hc_final(be, &d.hc_final)
																 : ERR_FORMAT;
	case RPC_OP_hc_post:
		return rpc_msg_desc(msg, &d.hc_post, RPC_DESC_HC_POST) ? be->hc_post(be, &d.hc_post)
															   : ERR_FORMAT;
	case RPC_OP_hc_pre:
		if (!rpc_msg_desc(msg, &d.hc_pre, RPC_DESC_HC_PRE))
			return ERR_FORMAT;
		d.hc_pre.ws = host ? float_buf_ensure(&s->ws, hc_pre_ws_floats(&d.hc_pre)) : NULL;
		return be->hc_pre(be, &d.hc_pre);
	case RPC_OP_kda:
		if (!rpc_msg_desc(msg, &d.kda, RPC_DESC_KDA))
			return ERR_FORMAT;
		d.kda.ws = host ? float_buf_ensure(&s->ws, kda_ws_floats(&d.kda)) : NULL;
		return be->kda(be, &d.kda);
	case RPC_OP_mla_absorbed:
		if (!rpc_msg_desc(msg, &d.mla, RPC_DESC_MLA_ABSORBED))
			return ERR_FORMAT;
		if (host) {
			d.mla.n_threads = pool_threads(be);
			d.mla.ws		= float_buf_ensure(&s->ws, mla_absorbed_ws_floats(&d.mla));
		}
		return be->mla_absorbed(be, &d.mla);
	case RPC_OP_attention_dense:
		return rpc_msg_desc(msg, &d.dense, RPC_DESC_DENSE_ATTENTION)
				   ? be->attention_dense(be, &d.dense)
				   : ERR_FORMAT;
	case RPC_OP_im2col3x3:
		return rpc_msg_desc(msg, &d.im2col, RPC_DESC_IM2COL) ? be->im2col3x3(be, &d.im2col)
															 : ERR_FORMAT;
	default:
		return rpc_msg_desc(msg, &d.up, RPC_DESC_UPSAMPLE_SHORTCUT)
				   ? be->upsample_shortcut_add(be, &d.up)
				   : ERR_FORMAT;
	}
}

static status_code op_expert_ffn(srv_session *s, rpc_msg *msg) {
	const buffer	   *x	= rpc_msg_buf(msg);
	buffer			   *out = (buffer *)rpc_msg_buf(msg);
	moe_resident_expert e;
	if (!rpc_msg_expert(msg, &e))
		return ERR_FORMAT;
	int dim	  = rpc_rd_i32(&msg->rd);
	int inter = rpc_rd_i32(&msg->rd);
	if (msg->rd.bad || !x || !out)
		return ERR_FORMAT;
	return s->be->moe_expert_ffn ? s->be->moe_expert_ffn(s->be, x, out, &e, dim, inter)
								 : resident_expert_ffn(s, x, out, &e, dim, inter);
}

static status_code op_experts_batch(srv_session *s, rpc_msg *msg) {
	const buffer *xb		= rpc_msg_buf(msg);
	buffer		 *out		= (buffer *)rpc_msg_buf(msg);
	int			  n_rows	= rpc_rd_i32(&msg->rd);
	int			  dim		= rpc_rd_i32(&msg->rd);
	int			  inter		= rpc_rd_i32(&msg->rd);
	int			  use_gelu	= rpc_rd_i32(&msg->rd);
	int			  n_experts = rpc_rd_i32(&msg->rd);
	if (msg->rd.bad || !xb || !out || n_experts < 0 || n_experts > RPC_MAX_BUFS)
		return ERR_FORMAT;
	moe_resident_expert *experts = xcalloc((size_t)MAX(n_experts, 1), sizeof(*experts));
	int					*counts	 = xcalloc((size_t)MAX(n_experts, 1), sizeof(int));
	size_t				 total	 = 0;
	for (int i = 0; i < n_experts && !msg->rd.bad; i++) {
		counts[i] = rpc_rd_i32(&msg->rd);
		if (counts[i] > 0 && rpc_msg_expert(msg, &experts[i]))
			total += (size_t)counts[i];
	}
	const int	*rows_raw = rpc_rd_bytes(&msg->rd, total * sizeof(int));
	const float *w_raw	  = rpc_rd_bytes(&msg->rd, total * sizeof(float));
	status_code	 st		  = ERR_FORMAT;
	if (!msg->rd.bad) {
		int	  *rows = xmalloc(MAX(total, 1) * sizeof(int));
		float *w	= xmalloc(MAX(total, 1) * sizeof(float));
		memcpy(rows, rows_raw, total * sizeof(int));
		memcpy(w, w_raw, total * sizeof(float));
		backend *be = s->be;
		st = be->moe_experts_batch
				 ? be->moe_experts_batch(be, xb, out, n_rows, dim, inter, use_gelu, n_experts,
										 experts, counts, rows, w)
				 : resident_experts_batch(s, xb, out, n_rows, dim, inter, use_gelu, n_experts,
										  experts, counts, rows, w);
		free(rows);
		free(w);
	}
	free(experts);
	free(counts);
	return st;
}

static status_code exec_generic(srv_session *s, rpc_msg *msg, uint32_t op) {
	backend *be = s->be;
	if (!rpc_msg_vals(msg))
		return ERR_FORMAT;
	switch (op) {
#include "rpc/dispatch.inc"
	case RPC_OP_rope:
		return be->rope(be, RPC_VB(0), RPC_I(1), RPC_I(2), RPC_I(3), TBL(4), TBL(5));
	case RPC_OP_rope_qk:
		return be->rope_qk(be, RPC_VB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_I(4), RPC_I(5), TBL(6),
						   TBL(7));
	case RPC_OP_rope_ext:
		return be->rope_ext(be, RPC_VB(0), RPC_I(1), RPC_I(2), RPC_I(3), TBL(4), TBL(5), TBL(6));
	case RPC_OP_rope_ext_batch:
		return be->rope_ext_batch(be, RPC_VB(0), RPC_I(1), RPC_I(2), RPC_I(3), TBL(4), TBL(5),
								  TBL(6), RPC_I(7));
	case RPC_OP_rope_batch:
		return be->rope_batch(be, RPC_VB(0), RPC_I(1), RPC_I(2), RPC_I(3), TBL(4), TBL(5),
							  RPC_I(6));
	case RPC_OP_qk_norm_rope:
		return be->qk_norm_rope(be, RPC_VB(0), RPC_VB(1), RPC_CB(2), RPC_CB(3), RPC_I(4), RPC_I(5),
								RPC_I(6), RPC_F(7), RPC_I(8), TBL(9), TBL(10), TBL(11), RPC_I(12));
	case RPC_OP_rope_qk_batch:
		return be->rope_qk_batch(be, RPC_VB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_I(4), RPC_I(5),
								 TBL(6), TBL(7), RPC_I(8));
	case RPC_OP_partial_rope_qk:
		return be->partial_rope_qk(be, RPC_VB(0), RPC_VB(1), RPC_I(2), RPC_I(3), RPC_I(4), RPC_I(5),
								   RPC_I(6), TBL(7), TBL(8), RPC_I(9));
	case RPC_OP_attention_mla:
		return be->attention_mla(be, RPC_CB(0), RPC_CB(1), RPC_CB(2), RPC_CB(3), RPC_VB(4),
								 RPC_I(5), RPC_I(6), RPC_I(7), RPC_I(8), RPC_I(9), RPC_I(10),
								 RPC_I(11), RPC_I(12), RPC_I(13), TBL(14), TBL(15), RPC_U(16),
								 RPC_U(17), RPC_F(18));
	case RPC_OP_argmax_to_reg:
		return be->argmax_to_reg(be, RPC_CB(0), RPC_I(1), RPC_I(2));
	case RPC_OP_set_gpu_profile:
		if (be->set_gpu_profile)
			be->set_gpu_profile(be, RPC_I(0));
		return OK;
	default:
		return ERR_UNSUPPORTED;
	}
}

static status_code exec_value(srv_session *s, rpc_msg *msg, uint32_t op, rpc_wr *out) {
	backend *be = s->be;
	switch (op) {
	case RPC_OP_argmax: {
		if (!rpc_msg_vals(msg))
			return ERR_FORMAT;
		int32_t		idx = 0;
		status_code st	= be->argmax(be, RPC_CB(0), RPC_I(1), &idx);
		rpc_wr_i32(out, idx);
		return st;
	}
	case RPC_OP_read_token_reg: {
		if (!rpc_msg_vals(msg))
			return ERR_FORMAT;
		int32_t		tok = 0;
		status_code st	= be->read_token_reg(be, RPC_I(0), &tok);
		rpc_wr_i32(out, tok);
		return st;
	}
	case RPC_OP_dequant_type_native:
		if (!rpc_msg_vals(msg))
			return ERR_FORMAT;
		rpc_wr_i32(out,
				   be->dequant_type_native ? be->dequant_type_native(be, RPC_U(0), RPC_I(1)) : 0);
		return OK;
	case RPC_OP_mem_available:
		rpc_wr_u64(out, backend_mem_available(be));
		return OK;
	default:
		rpc_wr_u64(out, backend_mem_total(be));
		return OK;
	}
}

static status_code exec_special(srv_session *s, rpc_msg *msg, uint32_t op, rpc_wr *out) {
	backend *be = s->be;
	rpc_rd	*rd = &msg->rd;
	switch (op) {
	case RPC_OP_buffer_alloc_weight:
		return op_alloc_weight(s, rd, out);
	case RPC_OP_buffer_alloc_scratch:
		return op_alloc_scratch(s, rd, out);
	case RPC_OP_buffer_alloc_from_host:
		return op_alloc_from_host(s, rd, out);
	case RPC_OP_repack_plan:
		return op_repack_plan(s, rd, out);
	case RPC_OP_dequant_weight:
		return op_dequant_weight(s, rd, out);
	case RPC_OP_buffer_free:
		buf_release(s, rpc_rd_u32(rd));
		return rd->bad ? ERR_FORMAT : OK;
	case RPC_OP_kv_alloc:
		return op_kv_alloc(s, rd, out);
	case RPC_OP_kv_alloc_mla:
		return op_kv_alloc_mla(s, rd, out);
	case RPC_OP_kv_free:
		return op_kv_free(s, rd);
	case RPC_OP_buffer_read_f32:
		return op_read_f32(s, msg, out);
	case RPC_OP_buffer_write_f32:
		return op_write_f32(s, msg);
	case RPC_OP_matmul_multi:
	case RPC_OP_matmul_multi_batch:
		return op_multi(s, msg, op == RPC_OP_matmul_multi_batch);
	case RPC_OP_sample:
		return op_sample(s, msg, out, false);
	case RPC_OP_sample_to_reg:
		return op_sample(s, msg, out, true);
	case RPC_OP_sample_history_set:
		return op_history(s, rd);
	case RPC_OP_moe_expert_ffn:
		return op_expert_ffn(s, msg);
	case RPC_OP_moe_experts_batch:
		return op_experts_batch(s, msg);
	case RPC_OP_synchronize:
		if (be->synchronize)
			be->synchronize(be);
		return OK;
	case RPC_OP_submit:
		if (be->submit)
			be->submit(be);
		return OK;
	case RPC_OP_begin_batch:
		if (be->begin_batch)
			be->begin_batch(be);
		return OK;
	case RPC_OP_end_batch:
		if (be->end_batch)
			be->end_batch(be);
		return OK;
	case RPC_OP_gated_delta_net:
	case RPC_OP_hc_init:
	case RPC_OP_hc_pre:
	case RPC_OP_hc_post:
	case RPC_OP_hc_final:
	case RPC_OP_kda:
	case RPC_OP_mla_absorbed:
	case RPC_OP_attention_dense:
	case RPC_OP_im2col3x3:
	case RPC_OP_upsample_shortcut_add:
		return op_desc(s, msg, op);
	case RPC_OP_argmax:
	case RPC_OP_read_token_reg:
	case RPC_OP_dequant_type_native:
	case RPC_OP_mem_available:
	case RPC_OP_mem_total:
		return exec_value(s, msg, op, out);
	default:
		return exec_generic(s, msg, op);
	}
}

static pthread_mutex_t g_lease_mtx = PTHREAD_MUTEX_INITIALIZER;
static backend		  *g_shared_be = NULL;
static bool			   g_leased	   = false;

static status_code ensure_backend(srv_session *s) {
	if (s->be)
		return OK;
	char name[BACKEND_DEVICE_NAME_MAX];
	int	 index = 0;
	if (backend_parse_device(s->device, name, sizeof(name), &index) != 0)
		return ERR_INVALID_ARG;
	pthread_mutex_lock(&g_lease_mtx);
	status_code st = g_leased ? ERR_BUSY : OK;
	if (st == OK && !g_shared_be)
		st = backend_create(name, index, &g_shared_be);
	if (st == OK) {
		g_leased	   = true;
		s->be		   = g_shared_be;
		s->emulate_moe = backend_has_cap(s->be, BCAP_IS_HOST) && !s->be->moe_expert_ffn;
	}
	pthread_mutex_unlock(&g_lease_mtx);
	if (st == ERR_BUSY)
		WARN("rpc: refusing session, device '%s' is in use by another client", s->device);
	else if (st == OK)
		INFO("rpc: session opened device '%s' (%s)", s->device, s->be->name);
	return st;
}

static void release_backend(srv_session *s) {
	if (!s->be)
		return;
	if (s->be->synchronize)
		s->be->synchronize(s->be);
	backend_release_thread(s->be);
	pthread_mutex_lock(&g_lease_mtx);
	g_leased = false;
	pthread_mutex_unlock(&g_lease_mtx);
	s->be = NULL;
}

static status_code reply_info(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	uint32_t magic	 = rpc_rd_u32(rd);
	uint32_t version = rpc_rd_u32(rd);
	if (rd->bad || magic != RPC_MAGIC || version != RPC_VERSION)
		return ERR_FORMAT;
	char identity[RPC_IDENT_MAX] = "";
	backend_device_identity(s->device, identity, sizeof(identity));
	rpc_wr_str(out, s->device);
	rpc_wr_str(out, identity);
	return OK;
}

static void write_ext(rpc_wr *out, uint16_t id, const void *p, size_t len) {
	rpc_wr_u16(out, id);
	rpc_wr_u16(out, (uint16_t)len);
	rpc_wr_bytes(out, p, len);
}

static status_code reply_hello(srv_session *s, rpc_rd *rd, rpc_wr *out) {
	uint32_t magic	 = rpc_rd_u32(rd);
	uint32_t version = rpc_rd_u32(rd);
	if (rd->bad || magic != RPC_MAGIC || version != RPC_VERSION)
		return ERR_FORMAT;
	status_code st = ensure_backend(s);
	if (st != OK)
		return st;
	backend *be					   = s->be;
	bool	 present[RPC_OP_COUNT] = {false};
	RPC_ALL_OPS(MARK_OP)
	uint64_t caps = be->caps;
	if (s->emulate_moe) {
		present[RPC_OP_moe_expert_ffn]		   = true;
		present[RPC_OP_moe_experts_batch]	   = true;
		present[RPC_OP_buffer_alloc_from_host] = true;
		present[RPC_OP_submit]				   = true;
		caps |= BCAP_MOE_EXPERT_RESIDENT | BCAP_IN_ORDER_QUEUE;
	}
	char identity[RPC_IDENT_MAX] = "";
	backend_identity(be, identity, sizeof(identity));
	rpc_wr_u64(out, caps);
	rpc_wr_i32(out, be->sample_max_vocab);
	rpc_wr_str(out, identity[0] ? identity : (be->desc ? be->desc : be->name));
	uint32_t n_ops = 0;
	for (int i = 0; i < RPC_OP_COUNT; i++)
		n_ops += present[i];
	rpc_wr_u32(out, n_ops);
	for (int i = 0; i < RPC_OP_COUNT; i++)
		if (present[i])
			rpc_wr_u16(out, (uint16_t)i);
	uint8_t types[RPC_TYPE_MAX];
	for (uint32_t t = 0; t < RPC_TYPE_MAX; t++)
		types[t] = !be->matmul_type_native || be->matmul_type_native(be, t);
	rpc_wr_bytes(out, types, sizeof(types));
	rpc_wr_u16(out, 2);
	write_ext(out, RPC_EXT_DEVICE, s->device, strlen(s->device));
	uint64_t mem_total = backend_mem_total(be);
	write_ext(out, RPC_EXT_MEM_TOTAL, &mem_total, sizeof(mem_total));
	return OK;
}

static status_code exec_frame(srv_session *s, uint32_t op, const uint8_t *p, size_t len,
							  rpc_wr *out) {
	rpc_rd rd;
	rpc_rd_init(&rd, p, len);
	switch (op) {
	case RPC_CTL_INFO:
		return reply_info(s, &rd, out);
	case RPC_CTL_HELLO:
		return reply_hello(s, &rd, out);
	case RPC_CTL_PARAMS:
		if (!s->be)
			return ERR_INVALID_ARG;
		s->be->rope_neox  = rpc_rd_i32(&rd);
		s->be->rope_theta = rpc_rd_f32(&rd);
		return rd.bad ? ERR_FORMAT : OK;
	case RPC_CTL_TABLE:
		return table_store(s, &rd);
	case RPC_CTL_UPLOAD_BEGIN:
		return upload_begin(s, &rd);
	case RPC_CTL_UPLOAD_DATA:
		return upload_data(s, p, len);
	default:
		break;
	}
	if (!s->be || op <= RPC_CTL_UPLOAD_DATA || op >= (uint32_t)RPC_OP_COUNT)
		return ERR_INVALID_ARG;
	rpc_msg_init(s->msg, resolve_buffer, s, p, len);
	return exec_special(s, s->msg, op, out);
}

static void session_close(srv_session *s) {
	for (uint32_t i = 0; i < s->n_bufs; i++)
		buf_release(s, i + 1);
	release_backend(s);
	for (size_t i = 0; i < ARRAY_LEN(s->tables); i++)
		free(s->tables[i].data);
	free(s->bufs);
	free(s->upload);
	free(s->ws.p);
	free(s->moe_scratch.p);
	free(s->msg);
	close(s->fd);
	free(s);
}

void session_run(int fd, const char *device) {
	srv_session *s = xcalloc(1, sizeof(*s));
	s->fd		   = fd;
	s->msg		   = xcalloc(1, sizeof(*s->msg));
	snprintf(s->device, sizeof(s->device), "%s", device);
	for (;;) {
		uint32_t op;
		uint8_t *payload;
		size_t	 len;
		if (rpc_read_frame(fd, &op, &payload, &len) != OK)
			break;
		rpc_wr out;
		rpc_wr_init(&out);
		status_code st = exec_frame(s, op, payload, len, &out);
		free(payload);
		if (op_is_sync(op)) {
			int32_t status = st != OK ? st : s->sticky;
			if (op != RPC_OP_synchronize)
				s->sticky = OK;
			bool fail = rpc_write_reply(fd, status, status == OK ? out.p : NULL,
										status == OK ? out.len : 0) != OK;
			rpc_wr_free(&out);
			if (fail)
				break;
			continue;
		}
		if (st != OK && s->sticky == OK) {
			s->sticky = st;
			WARN("rpc: async op %u failed (%s)", op, status_str(st));
		}
		rpc_wr_free(&out);
	}
	INFO("rpc: session closed");
	session_close(s);
}
