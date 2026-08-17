#include "kvcache.h"
#include "log.h"

#include <stdlib.h>
#include <string.h>

status_code kvcache_init(kvcache *c, const model *m, int n_ctx, kv_quant_type kv_quant) {
	memset(c, 0, sizeof(*c));
	c->n_ctx	= n_ctx;
	c->n_pos	= 0;
	c->backend	= m->backend;
	c->kv_quant = kv_quant;

	int is_mla = m->arch_info->is_mla;

	if (kv_quant != KV_QUANT_F16 && is_mla) {
		ERROR("kvcache: quantized KV cache is not supported for MLA models");
		return ERR_UNSUPPORTED;
	}

	c->head_dim_max	  = m->head_dim;
	c->n_kv_heads_max = m->n_kv_heads;
	if (m->arch_info->has_variable_layer_dims) {
		for (int i = 0; i < m->n_layers; i++) {
			int hd = model_layer_head_dim(m, i);
			if (hd > c->head_dim_max)
				c->head_dim_max = hd;
			int kvh = model_layer_kv_heads(m, i);
			if (kvh > c->n_kv_heads_max)
				c->n_kv_heads_max = kvh;
		}
	}

	int n_kv_layers =
		m->layer_dims.n_layer_kv_from_start > 0 ? m->layer_dims.n_layer_kv_from_start : m->n_layers;
	c->n_kv_layers = n_kv_layers;

	if (is_mla) {
		if (!m->backend->kv_alloc_mla) {
			ERROR("kvcache: model is MLA but backend has no kv_alloc_mla");
			return ERR_UNSUPPORTED;
		}
		c->mla				= xcalloc(1, sizeof(kvcache_mla));
		c->mla->lora_dim	= m->mla.kv_lora;
		c->mla->qk_rope_dim = m->mla.qk_rope;
		return m->backend->kv_alloc_mla(m->backend, m->n_layers, n_ctx, c->mla->lora_dim,
										c->mla->qk_rope_dim, &c->mla->kv);
	}

	backend *kv_backend = c->backend->kv_alloc ? c->backend : backend_host();

	if (kv_quant == KV_QUANT_Q8_0 && !backend_has_cap(kv_backend, BCAP_KV_QUANT_Q8_0)) {
		ERROR("kvcache: backend '%s' does not support Q8_0 quantized KV cache; "
			  "use --kv-quant f16 or select a backend that supports it",
			  kv_backend->name);
		return ERR_UNSUPPORTED;
	}

	int *layer_head_dim	  = NULL;
	int *layer_n_kv_heads = NULL;
	if (m->arch_info->has_variable_layer_dims) {
		layer_head_dim	 = xcalloc(m->n_layers, sizeof(int));
		layer_n_kv_heads = xcalloc(m->n_layers, sizeof(int));
		for (int i = 0; i < m->n_layers; i++) {
			layer_head_dim[i]	= model_layer_head_dim(m, i);
			layer_n_kv_heads[i] = model_layer_kv_heads(m, i);
		}
	}

	kv_desc desc = {
		.n_layers		  = m->n_layers,
		.n_kv_layers	  = n_kv_layers,
		.n_kv_heads		  = c->n_kv_heads_max,
		.head_dim		  = c->head_dim_max,
		.n_ctx			  = n_ctx,
		.kv_quant		  = kv_quant,
		.layer_head_dim	  = layer_head_dim,
		.layer_n_kv_heads = layer_n_kv_heads,
	};
	status_code s = kv_backend->kv_alloc(kv_backend, &desc, &c->k, &c->v);
	free(layer_head_dim);
	free(layer_n_kv_heads);
	return s;
}

void kvcache_free(kvcache *c) {
	if (!c->backend)
		return;
	if (c->mla) {
		if (c->mla->kv.owner)
			c->mla->kv.owner->buffer_free(c->mla->kv.owner, &c->mla->kv);
		free(c->mla);
	} else {
		if (c->backend->kv_free) {
			c->backend->kv_free(c->backend, &c->k, &c->v);
		} else {
			if (c->k.owner)
				c->k.owner->buffer_free(c->k.owner, &c->k);
			if (c->v.owner)
				c->v.owner->buffer_free(c->v.owner, &c->v);
		}
	}
	memset(c, 0, sizeof(*c));
}

void kvcache_reset(kvcache *c) {
	c->n_pos = 0;
}
