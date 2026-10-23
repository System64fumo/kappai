#ifndef GRAPH_H
#define GRAPH_H

#include "recipe.h"

#include <stdio.h>

#define CG_NONE (-1)

#define CG_STATE_SLOTS 5

#define CG_ADD_ACC 0
#define CG_ADD_ADDEND 1
#define CG_RESID_INPUT 0
#define CG_RESID_ACC 1

#define CG_BATCH_SLOTS                                                                             \
	((1u << RECIPE_SLOT_X) | (1u << RECIPE_SLOT_XB) | (1u << RECIPE_SLOT_XB2) |                    \
	 (1u << RECIPE_SLOT_ATTN_OUT) | (1u << RECIPE_SLOT_Q) | (1u << RECIPE_SLOT_K) |                \
	 (1u << RECIPE_SLOT_V) | (1u << RECIPE_SLOT_FFN_GATE) | (1u << RECIPE_SLOT_FFN_UP) |           \
	 (1u << RECIPE_SLOT_FFN_ACT) | (1u << RECIPE_SLOT_FFN_GATE_UP) |                               \
	 (1u << RECIPE_SLOT_RESID_TMP) | (1u << RECIPE_SLOT_HYB_PROJ) | (1u << RECIPE_SLOT_HYB_GATE) | \
	 (1u << RECIPE_SLOT_HYB_ALPHA) | (1u << RECIPE_SLOT_HYB_BETA) | (1u << RECIPE_SLOT_LOGITS) |   \
	 (1u << RECIPE_SLOT_HC) | (1u << RECIPE_SLOT_HC_COEF))

#define CG_DUMP_OFF 0
#define CG_DUMP_TEXT 1
#define CG_DUMP_DOT 2

enum {
	BS_SIZE_NONE = 0,
	BS_SIZE_ATTN_BUF,
	BS_SIZE_Q_OUT,
	BS_SIZE_KV_OUT,
	BS_SIZE_INTER,
	BS_SIZE_INTER_X2,
	BS_SIZE_ACT,
	BS_SIZE_HC,
	BS_SIZE_HC_COEF,
};

typedef enum { CG_SECTION_PRE = 0, CG_SECTION_LAYER, CG_SECTION_POST, CG_SECTION_COUNT } cg_section;

typedef enum { CG_ROWS_BATCH = 0, CG_ROWS_OUTPUT } cg_rows;
typedef enum { CG_DTYPE_F32 = 0, CG_DTYPE_I32 } cg_dtype;
typedef enum { CG_RES_DEVICE = 0, CG_RES_HOST } cg_residency;

typedef struct {
	int		producer;
	int		alias_of;
	int		n_uses;
	uint8_t slot;
	uint8_t loop_carried;
	uint8_t port;
	uint8_t size_kind;
	uint8_t rows;
	uint8_t dtype;
	uint8_t residency;
} cg_value;

typedef struct {
	op_kind			  kind;
	cg_section		  section;
	stage			  stage;
	int				  recipe_index;
	uint8_t			  w_idx;
	uint8_t			  coalesce_run_len;
	uint8_t			  fuse_next;
	uint8_t			  impl;
	uint8_t			  state_reads;
	uint8_t			  state_writes;
	int				  n_in;
	int				  n_out;
	int				  in[OP_TOUCH_MAX];
	int				  out[OP_TOUCH_MAX];
	int				  state_dep[CG_STATE_SLOTS];
	op_params		  params;
	const weight_ref *weight;
} cg_node;

typedef struct {
	uint8_t slot;
	int		in;
	int		out;
} cg_carry;

typedef struct {
	int start;
	int end;
} cg_range;

typedef struct cg_graph {
	cg_node	 *nodes;
	cg_value *values;
	int		  n_nodes;
	int		  n_values;
	int		  section_begin[CG_SECTION_COUNT + 1];
	cg_carry  carry[RECIPE_SLOT_MAX];
	int		  n_carry;
	int		 *consumer_begin;
	int		 *consumers;
	uint8_t	  frozen;
} cg_graph;

static inline int graph_slot_valid(uint8_t slot) {
	return slot < RECIPE_SLOT_MAX && ((CG_BATCH_SLOTS >> slot) & 1u);
}

int graph_slot_size_kind(uint8_t slot);
int graph_dump_mode(void);
typedef struct {
	backend	   *be;
	op_kind		kind;
	const char *fn;
} cg_fallback;

typedef struct cg_set {
	cg_graph			 **graphs;
	int					  *first_layer;
	int					  *layer_graph;
	const weight_ref	 **layer_weights;
	backend				 **layer_backend;
	uint8_t				  *layer_impl;
	int					   weight_stride;
	const layer_ctx_entry *layer_ctx;
	int					   n_layer_ctx;
	recipe_op			  *pre_ops;
	recipe_op			  *layer_ops;
	recipe_op			  *post_ops;
	int					   n_layer_ops;
	cg_fallback			  *fallbacks;
	int					   n_fallbacks;
	int					   n_graphs;
	int					   n_layers;
} cg_set;

cg_graph			  *graph_build(const model_recipe *r);
cg_graph			  *graph_build_variant(const model_recipe *r, int variant);
cg_set				  *graph_set_build(const model_recipe *r, const struct model *m, int n_layers);
const weight_ref	  *graph_set_weight(const cg_set *s, int layer, int node);
backend				  *graph_set_backend(const cg_set *s, int layer);
uint8_t				   graph_set_impl(const cg_set *s, int layer, int node);
void				   graph_set_free(cg_set *s);
int					   graph_set_executable(const cg_set *s);
const layer_ctx_entry *graph_set_layer_ctx(const cg_set *s, int layer);
uint32_t			   graph_set_batch_slot_mask(const cg_set *s);
const recipe_op		  *graph_set_pre_ops(const cg_set *s);
const recipe_op		  *graph_set_layer_ops(const cg_set *s, int layer);
const recipe_op		  *graph_set_post_ops(const cg_set *s);
int					   graph_set_validate(const cg_set *s);
int					   graph_set_check_weights(const cg_set *s, const struct model *m);
void				   graph_set_dump(const cg_set *s, FILE *out, int dot);
void				   graph_free(cg_graph *g);
void				   graph_index_consumers(cg_graph *g);
int					   graph_consumers(const cg_graph *g, int value, const int **list);
void				   graph_freeze(cg_graph *g);
int					   graph_validate(const cg_graph *g);
cg_range			  *graph_value_ranges(const cg_graph *g);
int					   graph_peak_live_buffers(const cg_graph *g);
void				   graph_dump(const cg_graph *g, FILE *out);
void				   graph_dump_dot(const cg_graph *g, FILE *out);

typedef struct {
	const struct model *m;
	int					layer;
} cg_pass_ctx;

typedef int (*graph_pass_fn)(cg_graph *g, const cg_pass_ctx *ctx);

typedef struct {
	const char	 *name;
	graph_pass_fn run;
} graph_pass;

int	 graph_run_passes(cg_graph *g, const graph_pass *passes, int n_passes, const cg_pass_ctx *ctx,
					  int *per_pass);
int	 graph_refine_matmul_stages(cg_graph *g, const cg_pass_ctx *ctx);
int	 graph_fuse_output_scale(cg_graph *g, const cg_pass_ctx *ctx);
int	 graph_fuse_gate_up_multi(cg_graph *g, const cg_pass_ctx *ctx);
int	 graph_coalesce_matmul_runs(cg_graph *g, const cg_pass_ctx *ctx);
int	 graph_fuse_rmsnorm_add_norm(cg_graph *g, const cg_pass_ctx *ctx);
int	 graph_fuse_matmul_residual(cg_graph *g, const cg_pass_ctx *ctx);
void graph_optimize_recipe(model_recipe *r, const struct model *m);
int	 graph_lower_section(const cg_graph *g, cg_section section, const recipe_op *orig, int n_orig,
						 recipe_op *out);

#endif
