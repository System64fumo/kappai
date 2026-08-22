#ifndef KVCACHE_H
#define KVCACHE_H

#include "backend/backend.h"
#include "common.h"
#include "model.h"

typedef struct kvcache_mla {
	buffer kv;
	buffer kv_host;
	int	   lora_dim;
	int	   qk_rope_dim;
	int	   host_alloced;
} kvcache_mla;

typedef struct kvcache_hybrid {
	float *conv_state;
	float *recurrent_state;
	size_t conv_stride;
	size_t recurrent_stride;
	size_t conv_total;
	size_t recurrent_total;
} kvcache_hybrid;

typedef struct kvcache {
	int			  n_ctx, n_pos;
	int			  n_kv_layers;
	buffer		  k, v;
	buffer		  k_host, v_host;
	int			  has_host_kv;
	int			 *kv_slot_on_host;
	int			  n_kv_slot_flags;
	float		 *kv_transfer_buf;
	size_t		  kv_transfer_cap;
	kv_quant_type kv_quant;
	backend		 *backend;

	int head_dim_max;
	int n_kv_heads_max;

	kvcache_mla	   *mla;
	kvcache_hybrid *hybrid;
} kvcache;

status_code kvcache_init(kvcache *c, const model *m, int n_ctx, kv_quant_type kv_quant);
void		kvcache_free(kvcache *c);
void		kvcache_reset(kvcache *c);

status_code kvcache_alloc_host_mirror(kvcache *c, const model *m);

static inline int kvcache_kv_heads_stride(const kvcache *c) {
	return c->n_kv_heads_max;
}

static inline int kvcache_layer_uses_host_kv(const kvcache *c, const model *m, int layer) {
	if (!c->has_host_kv || !m || !m->mixed_backend_mode)
		return 0;
	if (layer < 0 || layer >= m->n_layer_backends)
		return 0;
	if (!m->layer_backends[layer])
		return 0;
	if (m->layer_backends[layer] == c->backend)
		return 0;
	if (!backend_has_cap(m->layer_backends[layer], BCAP_IS_HOST))
		return 0;
	return 1;
}

static inline int kvcache_slot_on_host(const kvcache *c, int slot) {
	if (!c || !c->kv_slot_on_host)
		return 0;
	if (slot < 0 || slot >= c->n_kv_slot_flags)
		return 0;
	return c->kv_slot_on_host[slot];
}

static inline buffer *kvcache_k_for_layer(kvcache *c, const model *m, int layer) {
	(void)m;
	return kvcache_slot_on_host(c, layer) ? &c->k_host : &c->k;
}

static inline buffer *kvcache_v_for_layer(kvcache *c, const model *m, int layer) {
	(void)m;
	return kvcache_slot_on_host(c, layer) ? &c->v_host : &c->v;
}

static inline backend *kvcache_backend_for_layer(kvcache *c, const model *m, int layer) {
	(void)m;
	return kvcache_slot_on_host(c, layer) ? (c->k_host.owner ? c->k_host.owner : backend_host())
										  : (c->k.owner ? c->k.owner : c->backend);
}

status_code kvcache_put(kvcache *c, const model *m, int layer, int pos, const buffer *k_in,
						const buffer *v_in);

status_code kvcache_put_batch(kvcache *c, const model *m, int layer, int pos_start,
							  const buffer *k_in, const buffer *v_in, int kv_row_stride,
							  int n_rows);

#endif
