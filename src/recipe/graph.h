#ifndef RECIPE_GRAPH_H
#define RECIPE_GRAPH_H

#include "backend/backend.h"
#include "common.h"
#include "nn.h"
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

typedef enum { CG_EXT_NONE = 0, CG_EXT_HOST_INPUT, CG_EXT_LAYER } cg_external;

typedef struct {
	uint8_t scope;
	int		idx;
} cg_param_ref;

typedef struct {
	int		producer;
	int		alias_of;
	int		view_of;
	int		n_uses;
	uint8_t slot;
	uint8_t loop_carried;
	uint8_t port;
	uint8_t size_kind;
	uint8_t rows;
	uint8_t dtype;
	uint8_t residency;
	uint8_t external;
	int		bind;
	size_t	elems;
	size_t	view_offset;
	size_t	layer_stride;
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
	cg_param_ref	  param;
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
	cg_graph		 **graphs;
	int				  *first_layer;
	int				  *layer_graph;
	const weight_ref **layer_weights;
	backend			 **layer_backend;
	uint8_t			  *layer_impl;
	int				   weight_stride;
	recipe_op		  *pre_ops;
	recipe_op		  *layer_ops;
	recipe_op		  *post_ops;
	int				   n_layer_ops;
	cg_fallback		  *fallbacks;
	int				   n_fallbacks;
	int				   n_graphs;
	int				   n_layers;
} cg_set;

cg_graph		 *graph_build(const model_recipe *r);
cg_graph		 *graph_build_variant(const model_recipe *r, int variant);
cg_set			 *graph_set_build(const model_recipe *r, const struct model *m, int n_layers);
const weight_ref *graph_set_weight(const cg_set *s, int layer, int node);
backend			 *graph_set_backend(const cg_set *s, int layer);
uint8_t			  graph_set_impl(const cg_set *s, int layer, int node);
void			  graph_set_free(cg_set *s);
int				  graph_set_executable(const cg_set *s);
uint32_t		  graph_set_batch_slot_mask(const cg_set *s);
const recipe_op	 *graph_set_pre_ops(const cg_set *s);
const recipe_op	 *graph_set_layer_ops(const cg_set *s, int layer);
const recipe_op	 *graph_set_post_ops(const cg_set *s);
int				  graph_set_validate(const cg_set *s);
int				  graph_set_check_weights(const cg_set *s, const struct model *m);
void			  graph_set_dump(const cg_set *s, FILE *out, int dot);
void			  graph_free(cg_graph *g);
void			  graph_index_consumers(cg_graph *g);
int				  graph_consumers(const cg_graph *g, int value, const int **list);
void			  graph_freeze(cg_graph *g);
int				  graph_validate(const cg_graph *g);
cg_range		 *graph_value_ranges(const cg_graph *g);
int				  graph_peak_live_buffers(const cg_graph *g);
void			  graph_dump(const cg_graph *g, FILE *out);
void			  graph_dump_dot(const cg_graph *g, FILE *out);

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

/* ------------------------------------------------------------------ *
 * Raw graph builder ("device graph").
 *
 * The construction API shared by the LLM recipe path and the diffusion
 * models: model code emits nodes into a cg_builder (a cg_graph plus
 * weight/activation bindings), which is then planned (recipe/plan.h)
 * and executed (recipe/exec.h) by the same engine machinery used for
 * recipes.  Nodes use the shared op_kind enum and op_params, so both
 * paths dispatch through one capability/one naming layer.
 * ------------------------------------------------------------------ */

#define CG_NEW CG_NONE
#define CG_EMIT_MAX_IN 5
#define CG_MAX_BINDS 4

typedef enum {
	CG_SCOPE_NONE = 0,
	CG_SCOPE_GLOBAL,
	CG_SCOPE_LAYER,
} cg_scope;

typedef struct {
	const weight  *mat;
	const buffer  *vec;
	const nn_conv *conv;
} cg_param;

typedef struct {
	int value;
	int bind;
} cg_output;

typedef struct cg_builder {
	cg_graph	   *g;
	int				cap_nodes, cap_values;
	cg_param	   *globals;
	int				n_globals, cap_globals;
	cg_output		outputs[CG_MAX_BINDS];
	int				n_outputs;
	cg_section		current;
	int				n_layers;
	const cg_param *layer_params;
	int				layer_stride;
} cg_builder;

typedef df_attention_dims cg_attention_dims;
typedef df_upsample_dims  cg_upsample_dims;

cg_builder *cg_builder_new(void);
void		cg_builder_free(cg_builder *g);

void cg_builder_section(cg_builder *g, cg_section section);
void cg_builder_layers(cg_builder *g, int n_layers, const cg_param *rows, int stride);
void cg_builder_finish(cg_builder *g);

cg_param_ref cg_param_mat(cg_builder *g, const weight *w);
cg_param_ref cg_param_vec(cg_builder *g, const buffer *v);
cg_param_ref cg_param_conv(cg_builder *g, const nn_conv *c);

static inline cg_param_ref cg_layer_param(int idx) {
	return (cg_param_ref){CG_SCOPE_LAYER, idx};
}

const cg_param *cg_builder_param(const cg_builder *g, cg_param_ref ref, int layer);

int	 cg_input(cg_builder *g, size_t elems, int bind);
int	 cg_layer_external(cg_builder *g, size_t elems, size_t layer_stride, int bind);
int	 cg_view(cg_builder *g, int parent, size_t offset, size_t elems);
void cg_output_value(cg_builder *g, int value, int bind);

int cg_matmul(cg_builder *g, cg_param_ref w, int in, int out, int n, int k, int rows);
int cg_conv(cg_builder *g, cg_param_ref conv, int in, int out, int height, int width, int shift);
int cg_rmsnorm(cg_builder *g, cg_param_ref gamma, int in, int out, int dim, float eps, int rows);
int cg_rmsnorm_per_head(cg_builder *g, cg_param_ref gamma, int in, int out, int heads, int head_dim,
						float eps, int rows);
int cg_layernorm_modulate(cg_builder *g, int in, int scale, int out, int dim, float eps, int rows);
int cg_add(cg_builder *g, int x, int y, int dim, int rows);
int cg_gated_add(cg_builder *g, int x, int y, int gate, int dim, int rows);
int cg_activate(cg_builder *g, int x, int activation);
int cg_ffn_activate(cg_builder *g, int gate, int up, int out, int inter, int activation, int rows);
int cg_rope_table(cg_builder *g, int x, int table, int rows, int heads, int head_dim);
int cg_attention(cg_builder *g, int q, int k0, int v0, int k1, int v1, int out,
				 cg_attention_dims dims);
int cg_upsample_shortcut_add(cg_builder *g, int in, int acc, cg_upsample_dims dims);

int cg_value_root(const cg_graph *c, int value);
int cg_builder_validate(const cg_builder *g);

#endif
