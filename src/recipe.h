#ifndef RECIPE_H
#define RECIPE_H

#include "common.h"
#include "gguf.h"
#include "profile.h"

#define RECIPE_SLOT_X 0
#define RECIPE_SLOT_XB 1
#define RECIPE_SLOT_XB2 2
#define RECIPE_SLOT_ATTN_OUT 3
#define RECIPE_SLOT_Q 4
#define RECIPE_SLOT_K 5
#define RECIPE_SLOT_V 6
#define RECIPE_SLOT_FFN_GATE 7
#define RECIPE_SLOT_FFN_UP 8
#define RECIPE_SLOT_FFN_ACT 9
#define RECIPE_SLOT_FFN_GATE_UP 10
#define RECIPE_SLOT_RESID_TMP 11
#define RECIPE_SLOT_LOGITS 14
#define RECIPE_SLOT_ROUTER_IDS 15
#define RECIPE_SLOT_ROUTER_W 16
#define RECIPE_SLOT_HYB_PROJ 17
#define RECIPE_SLOT_HYB_GATE 18
#define RECIPE_SLOT_HYB_ALPHA 19
#define RECIPE_SLOT_HYB_BETA 20
#define RECIPE_SLOT_MAX 21

#define RECIPE_SLOT_NONE 0xFF

#define RECIPE_NO_WEIGHT WIDX_NONE

struct model;
struct kvcache;
struct compute_scratch;
struct buffer;
struct layer_weights;

struct batch_scratch;
typedef struct batch_scratch batch_scratch;

typedef enum {
	OP_NONE = 0,

	OP_EMBD_LOOKUP,
	OP_SCALE_EMBEDDINGS,
	OP_PLE_BUILD,

	OP_RMSNORM,
	OP_RMSNORM_PER_HEAD,
	OP_RMSNORM_NOWEIGHT,
	OP_RMSNORM_ADD,

	OP_MATMUL,
	OP_MATMUL_RESIDUAL,
	OP_MATMUL_MULTI,
	OP_MATMUL_FUSED_GATEUP,
	OP_MATMUL_FFN_DOWN,

	OP_ROPE,
	OP_ROPE_QK_FUSED,
	OP_ROPE_EXT,

	OP_KV_PUT,
	OP_ATTENTION,
	OP_ATTENTION_SWA,

	OP_ADD,
	OP_SWAP,
	OP_FFN_ACTIVATE,
	OP_FFN_ACTIVATE_EX,
	OP_FFN_ACTIVATE_FUSED,
	OP_SCALE,

	OP_PLE_PROJ_INJECT,

	OP_SOFTCAP,
	OP_LOGITS_READBACK,

	OP_MLA_Q_PROJ,
	OP_MLA_KV_PROJ,
	OP_MLA_QKV_PROJ_FUSED,
	OP_ATTENTION_MLA,
	OP_MOE_ROUTER,
	OP_MOE_EXPERTS,
	OP_MOE_SHARED,

	OP_SPLIT_QGATE,
	OP_PARTIAL_ROPE_QK,
	OP_ATTN_OUTPUT_GATE,
	OP_GATED_DELTA_NET,
	OP_SHORTCONV,

	OP_KIND_COUNT
} op_kind;

enum {
	ACTIVATION_SILU = 0,
	ACTIVATION_GELU = 1,
};

typedef enum {
	WIDX_NONE = 0,
	WIDX_TOK_EMBD,
	WIDX_OUTPUT_NORM,
	WIDX_OUTPUT_W,
	WIDX_ATTN_NORM,
	WIDX_WQ,
	WIDX_WK,
	WIDX_WV,
	WIDX_WO,
	WIDX_FFN_NORM,
	WIDX_GATE,
	WIDX_UP,
	WIDX_DOWN,
	WIDX_GATE_UP,
	WIDX_POST_ATTN_NORM,
	WIDX_POST_FFN_NORM,
	WIDX_ATTN_Q_NORM,
	WIDX_ATTN_K_NORM,
	WIDX_PLE_POST_NORM,
	WIDX_PLE_INP_GATE,
	WIDX_PLE_PROJ,
	WIDX_LAYER_OUT_SCALE,
	WIDX_ROPE_FREQS,
	WIDX_PER_LAYER_TOK_EMBD,
	WIDX_PER_LAYER_MODEL_PROJ,
	WIDX_PER_LAYER_PROJ_NORM,
	WIDX_FFN_GATE_INP,
	WIDX_EXP_PROBS_BIAS,
	WIDX_FFN_GATE_INP_S,
	WIDX_FFN_PRE_NORM_2,
	WIDX_FFN_POST_NORM_1,
	WIDX_FFN_POST_NORM_2,
	WIDX_SHEXP_GATE,
	WIDX_SHEXP_UP,
	WIDX_SHEXP_DOWN,
	WIDX_MLA_Q_A,
	WIDX_MLA_Q_B,
	WIDX_MLA_Q_A_NORM,
	WIDX_MLA_KV_A,
	WIDX_MLA_K_B,
	WIDX_MLA_V_B,
	WIDX_MLA_KV_A_NORM,
	WIDX_ATTN_QKV,
	WIDX_ATTN_GATE,
	WIDX_SSM_CONV1D,
	WIDX_SSM_DT,
	WIDX_SSM_A,
	WIDX_SSM_BETA,
	WIDX_SSM_ALPHA,
	WIDX_SSM_NORM,
	WIDX_SSM_OUT,
	WIDX_COUNT
} weight_idx;

typedef union {
	struct {
		float eps;
		int	  n_heads;
	} rmsnorm;
	struct {
		int n;
		int k;
	} matmul;
	struct {
		int n;
		int k;
		int n_out[3];
	} matmul_multi;
	struct {
		int n;
		int k;
		int activation;
	} matmul_ffn_down;
	struct {
		int n_heads;
		int n_kv_heads;
		int head_dim;
		int rope_neox;
	} rope;
	struct {
		int n_heads;
		int head_dim;
		int use_freq_factors;
		int rope_neox;
	} rope_ext;
	struct {
		int	  n_heads;
		int	  n_kv_heads;
		int	  head_dim;
		int	  n_ctx;
		float scale;
		int	  sliding_window;
		int	  n_kv_heads_active;
		int	  kv_layer;
	} attention;
	struct {
		int n;
		int activation;
	} ffn_act;
	struct {
		float scale;
	} scale;
	struct {
		float cap;
	} softcap;
} op_params;

typedef struct {
	op_kind	  kind;
	uint8_t	  in[3];
	uint8_t	  out;
	uint8_t	  w_idx;
	uint8_t	  coalesce_run_len;
	stage	  stage;
	op_params u;
} recipe_op;

typedef struct {
	recipe_op *ops;
	int		   n_ops;
} layer_recipe;

typedef struct {
	int		head_dim;
	int		n_kv_heads;
	int		intermediate;
	int		kv_layer;
	int		kv_row_stride;
	int		q_row_stride;
	uint8_t has_kv;
	uint8_t has_own_v;
	uint8_t is_global;
} layer_ctx_entry;

typedef struct model_recipe {
	layer_recipe layer;
	recipe_op	*pre_ops;
	recipe_op	*post_ops;
	int			 n_pre_ops;
	int			 n_post_ops;

	recipe_op *per_layer_ops;

	layer_ctx_entry *layer_ctx;

	int max_intermediate;
	int max_head_dim;
	int max_kv_heads;

	uint32_t bs_slot_mask;
} model_recipe;

typedef struct {
	const recipe_op		   *op;
	struct model		   *m;
	struct kvcache		   *cache;
	struct compute_scratch *s;
	batch_scratch		   *bs;
	int						token, pos, li, flash_attn;
	int						n_rows;
	int						pos_start;
	float				   *logits_out;
} exec_ctx;

float *recipe_slot_f32(const exec_ctx *ctx, uint8_t idx);

static inline int recipe_exec_is_batch(const exec_ctx *ctx) {
	return ctx && ctx->bs != NULL;
}

status_code op_split_qgate(exec_ctx *ctx);
status_code op_partial_rope_qk(exec_ctx *ctx);
status_code op_attn_output_gate(exec_ctx *ctx);
status_code op_gated_delta_net(exec_ctx *ctx);
status_code op_shortconv(exec_ctx *ctx);

typedef model_recipe *(*recipe_builder_fn)(const struct model *m);

void					 recipe_register(const char *arch_gguf_name, recipe_builder_fn builder);
const recipe_builder_fn *recipe_lookup(const char *arch_gguf_name);

void recipe_init(void);

model_recipe *recipe_build(const struct model *m);
void		  recipe_free(model_recipe *r);

status_code compute_forward_recipe(struct model *m, struct kvcache *cache,
								   struct compute_scratch *s, int token, int pos, int flash_attn,
								   float *logits_out);

status_code compute_forward_batch_recipe(struct model *m, struct kvcache *cache,
										 struct compute_scratch *s, const int32_t *tokens,
										 int n_tokens, int pos_start, int flash_attn,
										 float *logits_out);

void batch_scratch_free(struct batch_scratch *bs);

recipe_op mk_rmsnorm(uint8_t in, uint8_t out, uint8_t widx, float eps, stage stage);
recipe_op mk_rmsnorm_add(uint8_t in, uint8_t residual, uint8_t out, uint8_t widx, float eps,
						 stage stage);
recipe_op mk_matmul(uint8_t in, uint8_t out, uint8_t widx, int n, int k, stage stage);
recipe_op mk_add(uint8_t in0, uint8_t in1, stage stage);
recipe_op mk_swap(uint8_t in0, uint8_t in1, stage stage);

void recipe_build_post_ops(model_recipe *r, const struct model *m);
void recipe_build_pre_ops(model_recipe *r, const struct model *m);

#define RECIPE_REGISTER(id, arch_name_str, builder_fn)                                             \
	static void __attribute__((constructor)) recipe_autoreg_##id(void) {                           \
		recipe_register(arch_name_str, builder_fn);                                                \
	}

#endif