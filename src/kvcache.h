#ifndef KVCACHE_H
#define KVCACHE_H

#include "backend/backend.h"
#include "common.h"
#include "model.h"

typedef struct kvcache_mla {
	buffer kv;
	int	   lora_dim;
	int	   qk_rope_dim;
} kvcache_mla;

typedef struct kvcache {
	int			  n_ctx, n_pos;
	int			  n_kv_layers;
	buffer		  k, v;
	kv_quant_type kv_quant;
	backend		 *backend;

	int head_dim_max;
	int n_kv_heads_max;

	kvcache_mla *mla;
} kvcache;

status_code kvcache_init(kvcache *c, const model *m, int n_ctx, kv_quant_type kv_quant);
void		kvcache_free(kvcache *c);
void		kvcache_reset(kvcache *c);

static inline int kvcache_kv_heads_stride(const kvcache *c) {
	return c->n_kv_heads_max;
}

static inline status_code kvcache_put(kvcache *c, const model *m, int layer, int pos,
									  const buffer *k_in, const buffer *v_in) {
	if (pos < 0 || pos >= c->n_ctx)
		return ERR_INVALID_ARG;
	if (layer < 0 || layer >= m->n_layers)
		return ERR_INVALID_ARG;
	int		 hd			= model_layer_head_dim(m, layer);
	int		 kvh_stride = kvcache_kv_heads_stride(c);
	int		 kvh_active = model_layer_kv_heads(m, layer);
	backend *kv_backend = c->k.owner ? c->k.owner : c->backend;
	if (kv_backend != c->backend && c->backend->synchronize)
		c->backend->synchronize(c->backend);
	return kv_backend->kv_put(kv_backend, &c->k, &c->v, layer, pos, k_in, v_in, kvh_stride, hd,
							  c->n_ctx, kvh_active);
}

#endif
