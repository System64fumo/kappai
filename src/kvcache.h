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
	float *mla_latent;
	size_t mla_latent_stride;
} kvcache_hybrid;

#define KV_MAX_MIRRORS 7

typedef struct {
	backend *be;
	buffer	 k, v;
	int		*remap;
	int		 n_remap;
	int		 n_slots;
} kv_mirror;

typedef struct kvcache {
	int			  n_ctx, n_pos;
	int			  n_kv_layers;
	buffer		  k, v;
	int			  has_host_kv;
	kv_mirror	  mirror[KV_MAX_MIRRORS];
	int			  n_mirrors;
	unsigned	 *slot_devices;
	int			  n_slot_devices;
	float		 *kv_transfer_buf;
	size_t		  kv_transfer_cap;
	kv_quant_type kv_quant;
	backend		 *backend;

	int *mirror_remap;
	int	 n_mirror_remap;

	int head_dim_max;
	int n_kv_heads_max;

	kvcache_mla	   *mla;
	kvcache_hybrid *hybrid;
} kvcache;

status_code kvcache_init(kvcache *c, const model *m, int n_ctx, kv_quant_type kv_quant);
void		kvcache_free(kvcache *c);
void		kvcache_reset(kvcache *c);

status_code kvcache_alloc_mirrors(kvcache *c, const model *m);
int			kvcache_layer_device(const kvcache *c, const model *m, int layer);
void kvcache_attn_view(const kvcache *c, const model *m, int exec_layer, int kv_layer, buffer **k,
					   buffer **v, int *slot);

status_code kvcache_put(kvcache *c, const model *m, int layer, int pos, const buffer *k_in,
						const buffer *v_in);

status_code kvcache_put_batch(kvcache *c, const model *m, int layer, int pos_start,
							  const buffer *k_in, const buffer *v_in, int kv_row_stride,
							  int n_rows);

#endif
