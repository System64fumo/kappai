#define _GNU_SOURCE
#include "rpc/rpc.h"

#include "gguf.h"
#include "log.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <sys/socket.h>

#define RPC_FRAME_HEADER 8
#define RPC_PROBE_TIMEOUT_MS 2000
#define RPC_CONF_FILE "rpc.conf"
#define RPC_CONF_KEY "worker"
#define FIELD(type, member, kind) {offsetof(type, member), kind}

static const rpc_field GDN_FIELDS[] = {
	FIELD(gdn_desc, n_tokens, RPC_F_I32),	 FIELD(gdn_desc, n_value_heads, RPC_F_I32),
	FIELD(gdn_desc, n_key_heads, RPC_F_I32), FIELD(gdn_desc, conv_dim, RPC_F_I32),
	FIELD(gdn_desc, conv_kernel, RPC_F_I32), FIELD(gdn_desc, key_dim, RPC_F_I32),
	FIELD(gdn_desc, state_size, RPC_F_I32),	 FIELD(gdn_desc, value_head_dim, RPC_F_I32),
	FIELD(gdn_desc, value_dim, RPC_F_I32),	 FIELD(gdn_desc, reset, RPC_F_I32),
	FIELD(gdn_desc, eps, RPC_F_F32),		 FIELD(gdn_desc, mixed, RPC_F_CBUF),
	FIELD(gdn_desc, z, RPC_F_CBUF),			 FIELD(gdn_desc, alpha, RPC_F_CBUF),
	FIELD(gdn_desc, beta, RPC_F_CBUF),		 FIELD(gdn_desc, conv_w, RPC_F_CBUF),
	FIELD(gdn_desc, dt, RPC_F_CBUF),		 FIELD(gdn_desc, a_vec, RPC_F_CBUF),
	FIELD(gdn_desc, norm_w, RPC_F_CBUF),	 FIELD(gdn_desc, conv_state, RPC_F_BUF),
	FIELD(gdn_desc, state, RPC_F_BUF),		 FIELD(gdn_desc, out, RPC_F_BUF),
};

static const rpc_field HC_INIT_FIELDS[] = {
	FIELD(hc_init_desc, dim, RPC_F_I32),
	FIELD(hc_init_desc, n_hc, RPC_F_I32),
	FIELD(hc_init_desc, x, RPC_F_CBUF),
	FIELD(hc_init_desc, hc, RPC_F_BUF),
};

static const rpc_field HC_FINAL_FIELDS[] = {
	FIELD(hc_final_desc, dim, RPC_F_I32),
	FIELD(hc_final_desc, n_hc, RPC_F_I32),
	FIELD(hc_final_desc, hc, RPC_F_CBUF),
	FIELD(hc_final_desc, x, RPC_F_BUF),
};

static const rpc_field HC_PRE_FIELDS[] = {
	FIELD(hc_pre_desc, dim, RPC_F_I32),
	FIELD(hc_pre_desc, n_hc, RPC_F_I32),
	FIELD(hc_pre_desc, sinkhorn_iters, RPC_F_I32),
	FIELD(hc_pre_desc, norm_eps, RPC_F_F32),
	FIELD(hc_pre_desc, eps, RPC_F_F32),
	FIELD(hc_pre_desc, stream, RPC_F_CBUF),
	FIELD(hc_pre_desc, fn, RPC_F_WEIGHT),
	FIELD(hc_pre_desc, base, RPC_F_CBUF),
	FIELD(hc_pre_desc, scale, RPC_F_CBUF),
	FIELD(hc_pre_desc, x, RPC_F_BUF),
	FIELD(hc_pre_desc, coef, RPC_F_BUF),
};

static const rpc_field HC_POST_FIELDS[] = {
	FIELD(hc_post_desc, dim, RPC_F_I32),  FIELD(hc_post_desc, n_hc, RPC_F_I32),
	FIELD(hc_post_desc, sub, RPC_F_CBUF), FIELD(hc_post_desc, coef, RPC_F_CBUF),
	FIELD(hc_post_desc, hc, RPC_F_BUF),
};

static const rpc_field KDA_FIELDS[] = {
	FIELD(kda_desc, dim, RPC_F_I32),	   FIELD(kda_desc, n_heads, RPC_F_I32),
	FIELD(kda_desc, head_dim, RPC_F_I32),  FIELD(kda_desc, conv_kernel, RPC_F_I32),
	FIELD(kda_desc, reset, RPC_F_I32),	   FIELD(kda_desc, gate_lower_bound, RPC_F_F32),
	FIELD(kda_desc, norm_eps, RPC_F_F32),  FIELD(kda_desc, x, RPC_F_CBUF),
	FIELD(kda_desc, wq, RPC_F_WEIGHT),	   FIELD(kda_desc, wk, RPC_F_WEIGHT),
	FIELD(kda_desc, wv, RPC_F_WEIGHT),	   FIELD(kda_desc, wo, RPC_F_WEIGHT),
	FIELD(kda_desc, f_a, RPC_F_WEIGHT),	   FIELD(kda_desc, f_b, RPC_F_WEIGHT),
	FIELD(kda_desc, g_a, RPC_F_WEIGHT),	   FIELD(kda_desc, g_b, RPC_F_WEIGHT),
	FIELD(kda_desc, beta, RPC_F_WEIGHT),   FIELD(kda_desc, conv_q, RPC_F_CBUF),
	FIELD(kda_desc, conv_k, RPC_F_CBUF),   FIELD(kda_desc, conv_v, RPC_F_CBUF),
	FIELD(kda_desc, dt_bias, RPC_F_CBUF),  FIELD(kda_desc, ssm_a, RPC_F_CBUF),
	FIELD(kda_desc, out_norm, RPC_F_CBUF), FIELD(kda_desc, conv_state, RPC_F_BUF),
	FIELD(kda_desc, state, RPC_F_BUF),	   FIELD(kda_desc, out, RPC_F_BUF),
};

static const rpc_field MLA_ABSORBED_FIELDS[] = {
	FIELD(mla_absorbed_desc, dim, RPC_F_I32),
	FIELD(mla_absorbed_desc, n_heads, RPC_F_I32),
	FIELD(mla_absorbed_desc, kv_lora, RPC_F_I32),
	FIELD(mla_absorbed_desc, qk_head, RPC_F_I32),
	FIELD(mla_absorbed_desc, v_head, RPC_F_I32),
	FIELD(mla_absorbed_desc, q_lora, RPC_F_I32),
	FIELD(mla_absorbed_desc, pos, RPC_F_I32),
	FIELD(mla_absorbed_desc, n_threads, RPC_F_I32),
	FIELD(mla_absorbed_desc, norm_eps, RPC_F_F32),
	FIELD(mla_absorbed_desc, x, RPC_F_CBUF),
	FIELD(mla_absorbed_desc, q_a, RPC_F_WEIGHT),
	FIELD(mla_absorbed_desc, q_b, RPC_F_WEIGHT),
	FIELD(mla_absorbed_desc, kv_a, RPC_F_WEIGHT),
	FIELD(mla_absorbed_desc, wo, RPC_F_WEIGHT),
	FIELD(mla_absorbed_desc, q_a_norm, RPC_F_CBUF),
	FIELD(mla_absorbed_desc, kv_a_norm, RPC_F_CBUF),
	FIELD(mla_absorbed_desc, k_b, RPC_F_CBUF),
	FIELD(mla_absorbed_desc, v_b, RPC_F_CBUF),
	FIELD(mla_absorbed_desc, latent_cache, RPC_F_BUF),
	FIELD(mla_absorbed_desc, out, RPC_F_BUF),
};

static const rpc_field DENSE_ATTENTION_FIELDS[] = {
	FIELD(dense_attention_desc, q, RPC_F_CBUF),
	FIELD(dense_attention_desc, k0, RPC_F_CBUF),
	FIELD(dense_attention_desc, v0, RPC_F_CBUF),
	FIELD(dense_attention_desc, k1, RPC_F_CBUF),
	FIELD(dense_attention_desc, v1, RPC_F_CBUF),
	FIELD(dense_attention_desc, out, RPC_F_BUF),
	FIELD(dense_attention_desc, n_q, RPC_F_I32),
	FIELD(dense_attention_desc, n_heads, RPC_F_I32),
	FIELD(dense_attention_desc, head_dim, RPC_F_I32),
	FIELD(dense_attention_desc, n0, RPC_F_I32),
	FIELD(dense_attention_desc, n1, RPC_F_I32),
	FIELD(dense_attention_desc, causal, RPC_F_I32),
};

static const rpc_field IM2COL_FIELDS[] = {
	FIELD(im2col_desc, in, RPC_F_CBUF),			FIELD(im2col_desc, col, RPC_F_BUF),
	FIELD(im2col_desc, first_pixel, RPC_F_I32), FIELD(im2col_desc, n_pixels, RPC_F_I32),
	FIELD(im2col_desc, width, RPC_F_I32),		FIELD(im2col_desc, in_height, RPC_F_I32),
	FIELD(im2col_desc, in_width, RPC_F_I32),	FIELD(im2col_desc, channels, RPC_F_I32),
	FIELD(im2col_desc, shift, RPC_F_I32),
};

static const rpc_field UPSAMPLE_SHORTCUT_FIELDS[] = {
	FIELD(upsample_shortcut_desc, in, RPC_F_CBUF),
	FIELD(upsample_shortcut_desc, out, RPC_F_BUF),
	FIELD(upsample_shortcut_desc, height, RPC_F_I32),
	FIELD(upsample_shortcut_desc, width, RPC_F_I32),
	FIELD(upsample_shortcut_desc, in_channels, RPC_F_I32),
	FIELD(upsample_shortcut_desc, out_channels, RPC_F_I32),
	FIELD(upsample_shortcut_desc, repeats, RPC_F_I32),
	FIELD(upsample_shortcut_desc, frames_up, RPC_F_I32),
	FIELD(upsample_shortcut_desc, factor, RPC_F_I32),
};

#define LAYOUT(fields) {fields, (int)ARRAY_LEN(fields)}

static const rpc_layout LAYOUTS[] = {
	[RPC_DESC_GDN]				 = LAYOUT(GDN_FIELDS),
	[RPC_DESC_HC_INIT]			 = LAYOUT(HC_INIT_FIELDS),
	[RPC_DESC_HC_FINAL]			 = LAYOUT(HC_FINAL_FIELDS),
	[RPC_DESC_HC_PRE]			 = LAYOUT(HC_PRE_FIELDS),
	[RPC_DESC_HC_POST]			 = LAYOUT(HC_POST_FIELDS),
	[RPC_DESC_KDA]				 = LAYOUT(KDA_FIELDS),
	[RPC_DESC_MLA_ABSORBED]		 = LAYOUT(MLA_ABSORBED_FIELDS),
	[RPC_DESC_DENSE_ATTENTION]	 = LAYOUT(DENSE_ATTENTION_FIELDS),
	[RPC_DESC_IM2COL]			 = LAYOUT(IM2COL_FIELDS),
	[RPC_DESC_UPSAMPLE_SHORTCUT] = LAYOUT(UPSAMPLE_SHORTCUT_FIELDS),
};

static void wr_reserve(rpc_wr *w, size_t extra) {
	if (w->len + extra <= w->cap)
		return;
	size_t cap = MAX(w->cap * 2, w->len + extra);
	if (w->p == w->inline_buf) {
		uint8_t *heap = xmalloc(cap);
		memcpy(heap, w->p, w->len);
		w->p = heap;
	} else {
		w->p = xrealloc(w->p, cap);
	}
	w->cap = cap;
}

void rpc_wr_init(rpc_wr *w) {
	w->p   = w->inline_buf;
	w->len = 0;
	w->cap = sizeof(w->inline_buf);
	w->bad = false;
}

void rpc_wr_free(rpc_wr *w) {
	if (w->p != w->inline_buf)
		free(w->p);
	rpc_wr_init(w);
}

void rpc_wr_bytes(rpc_wr *w, const void *data, size_t n) {
	wr_reserve(w, n);
	memcpy(w->p + w->len, data, n);
	w->len += n;
}

void rpc_wr_u8(rpc_wr *w, uint8_t v) {
	rpc_wr_bytes(w, &v, sizeof(v));
}

void rpc_wr_u16(rpc_wr *w, uint16_t v) {
	rpc_wr_bytes(w, &v, sizeof(v));
}

void rpc_wr_u32(rpc_wr *w, uint32_t v) {
	rpc_wr_bytes(w, &v, sizeof(v));
}

void rpc_wr_i32(rpc_wr *w, int32_t v) {
	rpc_wr_bytes(w, &v, sizeof(v));
}

void rpc_wr_u64(rpc_wr *w, uint64_t v) {
	rpc_wr_bytes(w, &v, sizeof(v));
}

void rpc_wr_f32(rpc_wr *w, float v) {
	rpc_wr_bytes(w, &v, sizeof(v));
}

void rpc_wr_str(rpc_wr *w, const char *s) {
	size_t n = s ? strnlen(s, UINT16_MAX) : 0;
	rpc_wr_u16(w, (uint16_t)n);
	rpc_wr_bytes(w, s, n);
}

void rpc_wr_buf(rpc_wr *w, const backend *owner, const buffer *b) {
	if (!b) {
		rpc_wr_u32(w, 0);
		return;
	}
	if (b->owner != owner || !b->handle) {
		w->bad = true;
		rpc_wr_u32(w, 0);
		return;
	}
	rpc_wr_u32(w, (uint32_t)(uintptr_t)b->handle);
	rpc_wr_u64(w, b->offset);
	rpc_wr_u64(w, b->size);
}

void rpc_wr_val(rpc_wr *w, const backend *owner, rpc_val v) {
	rpc_wr_u8(w, v.kind);
	switch (v.kind) {
	case RPC_VAL_BUF:
		rpc_wr_buf(w, owner, v.v.b);
		break;
	case RPC_VAL_I32:
		rpc_wr_i32(w, v.v.i);
		break;
	case RPC_VAL_U32:
		rpc_wr_u32(w, v.v.u);
		break;
	default:
		rpc_wr_f32(w, v.v.f);
		break;
	}
}

rpc_val rpc_val_buf(const buffer *b) {
	return (rpc_val){.kind = RPC_VAL_BUF, .v.b = b};
}

rpc_val rpc_val_i32(int32_t v) {
	return (rpc_val){.kind = RPC_VAL_I32, .v.i = v};
}

rpc_val rpc_val_u32(uint32_t v) {
	return (rpc_val){.kind = RPC_VAL_U32, .v.u = v};
}

rpc_val rpc_val_f32(float v) {
	return (rpc_val){.kind = RPC_VAL_F32, .v.f = v};
}

void rpc_wr_expert(rpc_wr *w, const backend *owner, const moe_resident_expert *e) {
	rpc_wr_buf(w, owner, e->gate_w);
	rpc_wr_buf(w, owner, e->up_w);
	rpc_wr_buf(w, owner, e->down_w);
	rpc_wr_u32(w, e->gate_type);
	rpc_wr_u32(w, e->up_type);
	rpc_wr_u32(w, e->down_type);
	rpc_wr_i32(w, e->gate_up_fused);
	rpc_wr_i32(w, e->use_gelu);
	rpc_wr_f32(w, e->gate_scale);
	rpc_wr_f32(w, e->up_scale);
	rpc_wr_f32(w, e->down_scale);
	rpc_wr_f32(w, e->weight);
}

static void wr_weight(rpc_wr *w, const backend *owner, const char *base) {
	rpc_wr_buf(w, owner, *(const buffer *const *)(base + offsetof(backend_weight, buf)));
	rpc_wr_u32(w, *(const uint32_t *)(base + offsetof(backend_weight, type)));
}

void rpc_wr_desc(rpc_wr *w, const backend *owner, const void *desc, rpc_desc_kind kind) {
	const rpc_layout *layout = &LAYOUTS[kind];
	for (int i = 0; i < layout->n_fields; i++) {
		const rpc_field *f	  = &layout->fields[i];
		const char		*base = (const char *)desc + f->offset;
		switch (f->kind) {
		case RPC_F_I32:
			rpc_wr_i32(w, *(const int32_t *)base);
			break;
		case RPC_F_F32:
			rpc_wr_f32(w, *(const float *)base);
			break;
		case RPC_F_WEIGHT:
			wr_weight(w, owner, base);
			break;
		default:
			rpc_wr_buf(w, owner, *(const buffer *const *)base);
			break;
		}
	}
}

void rpc_rd_init(rpc_rd *r, const void *p, size_t len) {
	r->p   = p;
	r->len = len;
	r->pos = 0;
	r->bad = false;
}

const void *rpc_rd_bytes(rpc_rd *r, size_t n) {
	if (r->bad || n > r->len - r->pos) {
		r->bad = true;
		return NULL;
	}
	const void *p = r->p + r->pos;
	r->pos += n;
	return p;
}

static void rd_scalar(rpc_rd *r, void *out, size_t n) {
	const void *p = rpc_rd_bytes(r, n);
	if (p)
		memcpy(out, p, n);
	else
		memset(out, 0, n);
}

uint8_t rpc_rd_u8(rpc_rd *r) {
	uint8_t v;
	rd_scalar(r, &v, sizeof(v));
	return v;
}

uint16_t rpc_rd_u16(rpc_rd *r) {
	uint16_t v;
	rd_scalar(r, &v, sizeof(v));
	return v;
}

uint32_t rpc_rd_u32(rpc_rd *r) {
	uint32_t v;
	rd_scalar(r, &v, sizeof(v));
	return v;
}

int32_t rpc_rd_i32(rpc_rd *r) {
	int32_t v;
	rd_scalar(r, &v, sizeof(v));
	return v;
}

uint64_t rpc_rd_u64(rpc_rd *r) {
	uint64_t v;
	rd_scalar(r, &v, sizeof(v));
	return v;
}

float rpc_rd_f32(rpc_rd *r) {
	float v;
	rd_scalar(r, &v, sizeof(v));
	return v;
}

void rpc_rd_str(rpc_rd *r, char *out, size_t cap) {
	size_t		n = rpc_rd_u16(r);
	const char *p = rpc_rd_bytes(r, n);
	size_t		c = p ? MIN(n, cap - 1) : 0;
	if (c)
		memcpy(out, p, c);
	out[c] = '\0';
}

void rpc_msg_init(rpc_msg *m, rpc_resolve_fn resolve, void *ctx, const void *p, size_t len) {
	rpc_rd_init(&m->rd, p, len);
	m->resolve = resolve;
	m->ctx	   = ctx;
	m->n_bufs  = 0;
	m->n_vals  = 0;
}

const buffer *rpc_msg_buf(rpc_msg *m) {
	uint32_t id = rpc_rd_u32(&m->rd);
	if (!id)
		return NULL;
	uint64_t off  = rpc_rd_u64(&m->rd);
	uint64_t size = rpc_rd_u64(&m->rd);
	if (m->rd.bad || m->n_bufs >= RPC_MAX_BUFS ||
		!m->resolve(m->ctx, id, off, size, &m->bufs[m->n_bufs])) {
		m->rd.bad = true;
		return NULL;
	}
	return &m->bufs[m->n_bufs++];
}

bool rpc_msg_vals(rpc_msg *m) {
	while (!m->rd.bad && m->rd.pos < m->rd.len) {
		if (m->n_vals >= RPC_MAX_VALS) {
			m->rd.bad = true;
			break;
		}
		rpc_val *v = &m->vals[m->n_vals++];
		v->kind	   = rpc_rd_u8(&m->rd);
		switch (v->kind) {
		case RPC_VAL_BUF:
			v->v.b = rpc_msg_buf(m);
			break;
		case RPC_VAL_I32:
			v->v.i = rpc_rd_i32(&m->rd);
			break;
		case RPC_VAL_U32:
			v->v.u = rpc_rd_u32(&m->rd);
			break;
		case RPC_VAL_F32:
			v->v.f = rpc_rd_f32(&m->rd);
			break;
		default:
			m->rd.bad = true;
			break;
		}
	}
	return !m->rd.bad;
}

bool rpc_msg_expert(rpc_msg *m, moe_resident_expert *e) {
	e->gate_w		 = rpc_msg_buf(m);
	e->up_w			 = rpc_msg_buf(m);
	e->down_w		 = rpc_msg_buf(m);
	e->gate_type	 = rpc_rd_u32(&m->rd);
	e->up_type		 = rpc_rd_u32(&m->rd);
	e->down_type	 = rpc_rd_u32(&m->rd);
	e->gate_up_fused = rpc_rd_i32(&m->rd);
	e->use_gelu		 = rpc_rd_i32(&m->rd);
	e->gate_scale	 = rpc_rd_f32(&m->rd);
	e->up_scale		 = rpc_rd_f32(&m->rd);
	e->down_scale	 = rpc_rd_f32(&m->rd);
	e->weight		 = rpc_rd_f32(&m->rd);
	return !m->rd.bad;
}

static void rd_weight(rpc_msg *m, char *base) {
	*(const buffer **)(base + offsetof(backend_weight, buf)) = rpc_msg_buf(m);
	*(uint32_t *)(base + offsetof(backend_weight, type))	 = rpc_rd_u32(&m->rd);
}

bool rpc_msg_desc(rpc_msg *m, void *desc, rpc_desc_kind kind) {
	const rpc_layout *layout = &LAYOUTS[kind];
	for (int i = 0; i < layout->n_fields; i++) {
		const rpc_field *f	  = &layout->fields[i];
		char			*base = (char *)desc + f->offset;
		switch (f->kind) {
		case RPC_F_I32:
			*(int32_t *)base = rpc_rd_i32(&m->rd);
			break;
		case RPC_F_F32:
			*(float *)base = rpc_rd_f32(&m->rd);
			break;
		case RPC_F_WEIGHT:
			rd_weight(m, base);
			break;
		default:
			*(const buffer **)base = rpc_msg_buf(m);
			break;
		}
	}
	return !m->rd.bad;
}

static status_code write_all(int fd, const void *data, size_t n) {
	const uint8_t *p = data;
	while (n > 0) {
		ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
		if (w < 0 && errno == EINTR)
			continue;
		if (w <= 0)
			return ERR_IO;
		p += w;
		n -= (size_t)w;
	}
	return OK;
}

static status_code read_all(int fd, void *data, size_t n) {
	uint8_t *p = data;
	while (n > 0) {
		ssize_t r = recv(fd, p, n, 0);
		if (r < 0 && errno == EINTR)
			continue;
		if (r <= 0)
			return ERR_IO;
		p += r;
		n -= (size_t)r;
	}
	return OK;
}

static int connect_with_timeout(int fd, const struct sockaddr *addr, socklen_t len,
								int timeout_ms) {
	int flags = fcntl(fd, F_GETFL, 0);
	fcntl(fd, F_SETFL, flags | O_NONBLOCK);
	int rc = connect(fd, addr, len);
	if (rc < 0 && errno == EINPROGRESS) {
		struct pollfd pfd  = {.fd = fd, .events = POLLOUT};
		int			  err  = 0;
		socklen_t	  elen = sizeof(err);
		rc = poll(&pfd, 1, timeout_ms) == 1 &&
					 getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &elen) == 0 && err == 0
				 ? 0
				 : -1;
	}
	fcntl(fd, F_SETFL, flags);
	return rc;
}

static int open_connection(const rpc_endpoint *ep, int timeout_ms) {
	char			port[16];
	struct addrinfo hints = {.ai_socktype = SOCK_STREAM}, *res = NULL;
	snprintf(port, sizeof(port), "%d", ep->port);
	if (getaddrinfo(ep->host, port, &hints, &res) != 0)
		return -1;
	int fd = -1;
	for (struct addrinfo *ai = res; ai && fd < 0; ai = ai->ai_next) {
		int s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
		if (s < 0)
			continue;
		if (connect_with_timeout(s, ai->ai_addr, ai->ai_addrlen, timeout_ms) == 0)
			fd = s;
		else
			close(s);
	}
	freeaddrinfo(res);
	return fd;
}

status_code rpc_connect(rpc_conn *c, const rpc_endpoint *ep, int timeout_ms) {
	memset(c, 0, sizeof(*c));
	c->fd = open_connection(ep, timeout_ms);
	if (c->fd < 0)
		return ERR_IO;
	int one = 1;
	setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
	setsockopt(c->fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
	pthread_mutex_init(&c->mtx, NULL);
	return OK;
}

void rpc_conn_close(rpc_conn *c) {
	if (c->fd < 0)
		return;
	shutdown(c->fd, SHUT_RDWR);
	close(c->fd);
	c->fd = -1;
	free(c->out);
	c->out	   = NULL;
	c->out_len = c->out_cap = 0;
	pthread_mutex_destroy(&c->mtx);
}

static status_code conn_fail(rpc_conn *c) {
	c->dead = true;
	return ERR_IO;
}

static status_code conn_flush_locked(rpc_conn *c) {
	if (c->out_len == 0)
		return OK;
	status_code s = write_all(c->fd, c->out, c->out_len);
	c->out_len	  = 0;
	return s == OK ? OK : conn_fail(c);
}

static void conn_append(rpc_conn *c, const void *data, size_t n) {
	if (c->out_len + n > c->out_cap) {
		c->out_cap = MAX(c->out_cap * 2, MAX(c->out_len + n, RPC_FLUSH_BYTES * 2));
		c->out	   = xrealloc(c->out, c->out_cap);
	}
	memcpy(c->out + c->out_len, data, n);
	c->out_len += n;
}

static status_code conn_queue_locked(rpc_conn *c, uint32_t op, const void *payload, size_t len) {
	if (c->dead)
		return ERR_IO;
	if (len > RPC_MAX_FRAME)
		return ERR_INVALID_ARG;
	uint32_t header[2] = {op, (uint32_t)len};
	if (len < RPC_FLUSH_BYTES) {
		conn_append(c, header, sizeof(header));
		conn_append(c, payload, len);
		return c->out_len >= RPC_FLUSH_BYTES ? conn_flush_locked(c) : OK;
	}
	status_code s = conn_flush_locked(c);
	if (s == OK)
		s = write_all(c->fd, header, sizeof(header));
	if (s == OK)
		s = write_all(c->fd, payload, len);
	return s == OK ? OK : conn_fail(c);
}

status_code rpc_send(rpc_conn *c, uint32_t op, const void *payload, size_t len) {
	pthread_mutex_lock(&c->mtx);
	status_code s = conn_queue_locked(c, op, payload, len);
	pthread_mutex_unlock(&c->mtx);
	return s;
}

status_code rpc_flush(rpc_conn *c) {
	pthread_mutex_lock(&c->mtx);
	status_code s = c->dead ? ERR_IO : conn_flush_locked(c);
	pthread_mutex_unlock(&c->mtx);
	return s;
}

static status_code conn_read_reply_locked(rpc_conn *c, rpc_reply *out) {
	int32_t header[2];
	if (read_all(c->fd, header, sizeof(header)) != OK)
		return conn_fail(c);
	size_t len = (uint32_t)header[1];
	if (len > RPC_MAX_FRAME)
		return conn_fail(c);
	out->status = header[0];
	out->len	= len;
	out->data	= len ? xmalloc(len) : NULL;
	if (len && read_all(c->fd, out->data, len) != OK) {
		free(out->data);
		out->data = NULL;
		return conn_fail(c);
	}
	return OK;
}

status_code rpc_call(rpc_conn *c, uint32_t op, const void *payload, size_t len, rpc_reply *out) {
	memset(out, 0, sizeof(*out));
	pthread_mutex_lock(&c->mtx);
	status_code s = conn_queue_locked(c, op, payload, len);
	if (s == OK)
		s = conn_flush_locked(c);
	if (s == OK)
		s = conn_read_reply_locked(c, out);
	pthread_mutex_unlock(&c->mtx);
	return s;
}

void rpc_reply_free(rpc_reply *r) {
	free(r->data);
	memset(r, 0, sizeof(*r));
}

status_code rpc_read_frame(int fd, uint32_t *op, uint8_t **payload, size_t *len) {
	uint32_t header[2];
	if (read_all(fd, header, sizeof(header)) != OK)
		return ERR_IO;
	if (header[1] > RPC_MAX_FRAME)
		return ERR_FORMAT;
	*op		 = header[0];
	*len	 = header[1];
	*payload = xmalloc(header[1] + 1);
	if (header[1] && read_all(fd, *payload, header[1]) != OK) {
		free(*payload);
		*payload = NULL;
		return ERR_IO;
	}
	return OK;
}

status_code rpc_write_reply(int fd, int32_t status, const void *payload, size_t len) {
	int32_t header[2] = {status, (int32_t)len};
	if (write_all(fd, header, sizeof(header)) != OK)
		return ERR_IO;
	return len ? write_all(fd, payload, len) : OK;
}

int rpc_listen(const char *host, int port, char *err, size_t errlen) {
	char			pstr[16];
	struct addrinfo hints = {.ai_socktype = SOCK_STREAM, .ai_flags = AI_PASSIVE}, *res = NULL;
	snprintf(pstr, sizeof(pstr), "%d", port);
	int rc = getaddrinfo(host, pstr, &hints, &res);
	if (rc != 0) {
		snprintf(err, errlen, "cannot resolve '%s': %s", host, gai_strerror(rc));
		return -1;
	}
	int fd = -1;
	for (struct addrinfo *ai = res; ai && fd < 0; ai = ai->ai_next) {
		int s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
		if (s < 0)
			continue;
		int one = 1;
		setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
		if (bind(s, ai->ai_addr, ai->ai_addrlen) == 0 && listen(s, 8) == 0)
			fd = s;
		else
			close(s);
	}
	freeaddrinfo(res);
	if (fd < 0)
		snprintf(err, errlen, "cannot listen on %s:%d: %s", host, port, strerror(errno));
	return fd;
}

int rpc_endpoint_parse(const char *text, rpc_endpoint *out, char *err, size_t errlen) {
	str_span sp = span_trim(text, strlen(text));
	if (sp.len == 0) {
		snprintf(err, errlen, "enter host or host:port");
		return -1;
	}
	char buf[RPC_HOST_MAX + 16];
	if (sp.len >= sizeof(buf)) {
		snprintf(err, errlen, "address too long");
		return -1;
	}
	memcpy(buf, sp.p, sp.len);
	buf[sp.len] = '\0';

	char *host	  = buf;
	char *colon	  = strrchr(buf, ':');
	char *bracket = buf[0] == '[' ? strchr(buf, ']') : NULL;
	if (bracket) {
		*bracket = '\0';
		host++;
		colon = bracket[1] == ':' ? bracket + 1 : NULL;
	} else if (colon && colon != strchr(buf, ':')) {
		colon = NULL;
	}
	int port = RPC_DEFAULT_PORT;
	if (colon) {
		*colon	  = '\0';
		char *end = NULL;
		long  v	  = strtol(colon + 1, &end, 10);
		if (*end || end == colon + 1 || v < 1 || v > 65535) {
			snprintf(err, errlen, "invalid port");
			return -1;
		}
		port = (int)v;
	}
	if (!*host || strlen(host) >= RPC_HOST_MAX) {
		snprintf(err, errlen, "invalid host");
		return -1;
	}
	snprintf(out->host, sizeof(out->host), "%s", host);
	out->port = port;
	return 0;
}

void rpc_endpoint_format(const rpc_endpoint *ep, char *out, size_t cap) {
	bool v6 = strchr(ep->host, ':') != NULL;
	snprintf(out, cap, v6 ? "[%s]:%d" : "%s:%d", ep->host, ep->port);
}

static int rpc_conf_path(char *out, size_t cap) {
	const char *base = getenv("XDG_CONFIG_HOME");
	const char *home = getenv("HOME");
	if (base && base[0] == '/' && base[1])
		return snprintf(out, cap, "%s/sys64/kappai/%s", base, RPC_CONF_FILE) >= (int)cap ? -1 : 0;
	if (!home || !home[0])
		return -1;
	return snprintf(out, cap, "%s/.config/sys64/kappai/%s", home, RPC_CONF_FILE) >= (int)cap ? -1
																							 : 0;
}

int rpc_endpoints_load(rpc_endpoint *out, int max) {
	char path[4096];
	if (rpc_conf_path(path, sizeof(path)) < 0)
		return 0;
	FILE *fp = fopen(path, "r");
	if (!fp)
		return 0;
	int	 n = 0;
	char line[RPC_HOST_MAX + 64];
	while (n < max && fgets(line, sizeof(line), fp)) {
		str_span	sp = span_trim(line, strlen(line));
		const char *eq = sp.len ? memchr(sp.p, '=', sp.len) : NULL;
		if (!eq || sp.p[0] == '#' || sp.p[0] == ';')
			continue;
		str_span key = span_trim(sp.p, (size_t)(eq - sp.p));
		str_span val = span_trim(eq + 1, sp.len - (size_t)(eq - sp.p) - 1);
		char	 value[RPC_HOST_MAX + 16], err[64];
		if (key.len != strlen(RPC_CONF_KEY) || strncmp(key.p, RPC_CONF_KEY, key.len) != 0 ||
			val.len >= sizeof(value))
			continue;
		snprintf(value, sizeof(value), "%.*s", (int)val.len, val.p);
		if (rpc_endpoint_parse(value, &out[n], err, sizeof(err)) == 0)
			n++;
	}
	fclose(fp);
	return n;
}

int rpc_endpoints_save(const rpc_endpoint *eps, int n, char *err, size_t errlen) {
	char path[4096], tmp[4112];
	if (rpc_conf_path(path, sizeof(path)) < 0) {
		snprintf(err, errlen, "could not determine the config directory (HOME is not set)");
		return -1;
	}
	char root[4096];
	snprintf(root, sizeof(root), "%.*s", (int)(strrchr(path, '/') - path), path);
	if (mkdir_p(root, err, errlen) < 0)
		return -1;
	snprintf(tmp, sizeof(tmp), "%s.tmp", path);
	FILE *fp = fopen(tmp, "w");
	if (!fp) {
		snprintf(err, errlen, "cannot write '%s': %s", tmp, strerror(errno));
		return -1;
	}
	fprintf(fp, "[rpc]\n");
	for (int i = 0; i < n; i++) {
		char addr[RPC_HOST_MAX + 16];
		rpc_endpoint_format(&eps[i], addr, sizeof(addr));
		fprintf(fp, "%s = %s\n", RPC_CONF_KEY, addr);
	}
	if (fclose(fp) != 0 || rename(tmp, path) != 0) {
		snprintf(err, errlen, "cannot save '%s': %s", path, strerror(errno));
		unlink(tmp);
		return -1;
	}
	return 0;
}

status_code rpc_probe(const rpc_endpoint *ep, rpc_info *out) {
	rpc_conn	c;
	status_code s = rpc_connect(&c, ep, RPC_PROBE_TIMEOUT_MS);
	if (s != OK)
		return s;
	rpc_wr w;
	rpc_wr_init(&w);
	rpc_wr_u32(&w, RPC_MAGIC);
	rpc_wr_u32(&w, RPC_VERSION);
	rpc_reply r;
	s = rpc_call(&c, RPC_CTL_INFO, w.p, w.len, &r);
	rpc_wr_free(&w);
	if (s == OK && r.status == OK) {
		rpc_rd rd;
		rpc_rd_init(&rd, r.data, r.len);
		rpc_rd_str(&rd, out->device, sizeof(out->device));
		rpc_rd_str(&rd, out->identity, sizeof(out->identity));
		s = rd.bad ? ERR_FORMAT : OK;
	} else if (s == OK) {
		s = (status_code)r.status;
	}
	rpc_reply_free(&r);
	rpc_conn_close(&c);
	return s;
}

size_t rpc_tensor_bytes(uint32_t type, int n_dims, const uint64_t *dims) {
	if (n_dims <= 0)
		return 0;
	size_t bytes = ggml_row_size(type, (size_t)dims[0]);
	for (int i = 1; i < n_dims && i < 4; i++)
		bytes *= dims[i] ? (size_t)dims[i] : 1;
	return bytes;
}
