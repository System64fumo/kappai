#ifndef MOE_STREAM_H
#define MOE_STREAM_H

#include "backend/backend.h"
#include "common.h"
#include "gguf.h"

struct model;

typedef struct moe_stream_cache moe_stream_cache;
typedef struct moe_stream_op	moe_stream_op;

#define MOE_MAX_K 512

typedef struct {
	const void		*gate_w;
	const void		*up_w;
	const void		*down_w;
	uint64_t		 last_used;
	void			*heap_buf;
	size_t			 heap_size;
	_Atomic uint64_t gen;
	uint64_t		 gate_off;
	uint64_t		 up_off;
	uint64_t		 down_off;
	uint32_t		 gate_type;
	uint32_t		 up_type;
	uint32_t		 down_type;
	buffer			 gpu_gate;
	buffer			 gpu_up;
	buffer			 gpu_down;
	int				 gpu_ready;
	int				 eid;
	int				 gate_up_fused;
	float			 gate_scale;
	float			 up_scale;
	float			 down_scale;
	int				 pinned;
	_Atomic int		 io_ready;
	_Atomic int		 inuse;
	int				 owned;
} moe_expert_slot;

typedef struct {
	size_t gate_b;
	size_t up_b;
	size_t down_b;
	size_t total;
} moe_expert_bytes;

typedef struct {
	uint64_t requests;
	uint64_t cache_hits;
	uint64_t cache_misses;
	uint64_t pin_hits;
	uint64_t lru_hits;
	uint64_t direct_io_ok;
	uint64_t direct_io_fallback;
} moe_stream_stats;

typedef struct {
	double	 hit_rate;
	double	 pin_rate;
	double	 lru_rate;
	uint64_t cache_misses;
	uint64_t direct_io_ok;
	uint64_t direct_io_fallback;
	int		 has_moe;
} moe_stats_summary;

status_code moe_stream_cache_init(struct model *m);
void		moe_stream_cache_free(moe_stream_cache *c);

moe_expert_bytes moe_calc_expert_bytes(const struct model *m, int layer, int eid);

status_code moe_stream_resolve(struct model *m, int layer, const int *expert_ids, int k,
							   moe_expert_slot *out_slots);

typedef void (*moe_stream_compute_hook_fn)(int begin, int end, int tid, void *ctx);

typedef struct {
	moe_stream_cache *cache;
	uint8_t			 *dst;
	uint64_t		  nomap_off;
	const uint8_t	 *mmap_src;
	size_t			  len;
	int				  direct_io_eligible;
	int				  precheck_aligned;
	_Atomic int		 *err_flag;
} moe_fill_chunk;

moe_stream_op *moe_stream_resolve_prep(struct model *m, int layer, const int *expert_ids, int k,
									   moe_expert_slot *out_slots);
void moe_stream_op_set_compute_hook(moe_stream_op *op, moe_stream_compute_hook_fn fn, void *ctx);
int	 moe_stream_op_n_items(const moe_stream_op *op);
int	 moe_stream_op_compute_k(const moe_stream_op *op, int item);
void moe_stream_op_fill_run(moe_stream_op *op, int item, int tid);
status_code moe_stream_op_finish(moe_stream_op *op);
void		moe_stream_op_free(moe_stream_op *op);

void moe_stream_wait_slot(const moe_expert_slot *slot);

void moe_stream_release_slot(struct model *m, int layer, const moe_expert_slot *slot);

void moe_stream_release_slots(struct model *m, int layer, const moe_expert_slot *slots, int n);

status_code moe_stream_preload_all(struct model *m);

void moe_stream_get_stats(const moe_stream_cache *c, moe_stream_stats *out);
void moe_stream_summarize(const struct model *m, moe_stats_summary *out);
void moe_stream_thread_cleanup(void);

void moe_stream_record_extra_hits(struct model *m, int layer, int n_extra_hits,
								  int n_extra_requests, int is_pinned);

#endif