#include "recipe/internal.h"

#define RECIPE_MAX_ARCHES 32

typedef struct {
	char			  arch_gguf_name[64];
	recipe_builder_fn builder;
} recipe_reg_entry;

static recipe_reg_entry g_recipe_registry[RECIPE_MAX_ARCHES];

static int g_recipe_registry_count = 0;

void recipe_register(const char *arch_gguf_name, recipe_builder_fn builder) {
	if (!arch_gguf_name || !builder)
		return;
	if (g_recipe_registry_count >= RECIPE_MAX_ARCHES) {
		ERROR("recipe_register: registry full (%d entries)", RECIPE_MAX_ARCHES);
		return;
	}
	for (int i = 0; i < g_recipe_registry_count; i++) {
		if (strcmp(g_recipe_registry[i].arch_gguf_name, arch_gguf_name) == 0) {
			WARN("recipe_register: arch '%s' already registered, replacing", arch_gguf_name);
			g_recipe_registry[i].builder = builder;
			return;
		}
	}
	recipe_reg_entry *e = &g_recipe_registry[g_recipe_registry_count++];
	snprintf(e->arch_gguf_name, sizeof(e->arch_gguf_name), "%s", arch_gguf_name);
	e->builder = builder;
}

static const recipe_builder_fn *recipe_lookup(const char *arch_gguf_name) {
	if (!arch_gguf_name)
		return NULL;
	for (int i = 0; i < g_recipe_registry_count; i++) {
		if (strcmp(g_recipe_registry[i].arch_gguf_name, arch_gguf_name) == 0) {
			return &g_recipe_registry[i].builder;
		}
	}
	return NULL;
}

static inline int layer_has_kv(const model *m, int li) {
	if (!m->arch_info->has_variable_layer_dims)
		return 1;
	if (m->layer_dims.n_layer_kv_from_start <= 0)
		return 1;
	return li < m->layer_dims.n_layer_kv_from_start;
}

static void resolve_matmul_dims(const layer_ctx_entry *lc, const model *m, uint8_t w_idx, int *n,
								int *k) {
	int head_dim = lc->head_dim;
	switch ((weight_idx)w_idx) {
	case WIDX_WQ:
		*n = m->n_heads * head_dim;
		break;
	case WIDX_WK:
	case WIDX_WV:
		*n = lc->n_kv_heads * head_dim;
		break;
	case WIDX_WO:
		*n = m->dim;
		*k = m->n_heads * head_dim;
		break;
	case WIDX_DOWN:
		*n = m->dim;
		*k = lc->intermediate;
		break;
	case WIDX_GATE:
	case WIDX_UP:
		*n = lc->intermediate;
		break;
	default:
		break;
	}
}

const recipe_op *recipe_layer_ops(const model_recipe *r, int li) {
	return r->per_layer_ops ? &r->per_layer_ops[(size_t)li * r->layer.n_ops] : r->layer.ops;
}

static int recipe_uses_op(const model_recipe *r, const struct model *m, op_kind kind) {
	const int n_variants = r->per_layer_ops ? m->n_layers : 1;
	for (int li = 0; li < n_variants; li++) {
		const recipe_op *ops = recipe_layer_ops(r, li);
		for (int i = 0; i < r->layer.n_ops; i++)
			if (ops[i].kind == kind)
				return 1;
	}
	return 0;
}

static status_code recipe_require_capability(const struct model *m, const backend *a,
											 const char *op_name, int native, int fallback_native,
											 int allow_fallback) {
	if (native)
		return OK;
	if (allow_fallback && fallback_native) {
		WARN("backend '%s': falling back to cpu for %s", a->name, op_name);
		return OK;
	}
	ERROR("recipe_build: '%s' needs %s, unsupported on backend '%s'%s", m->arch_info->gguf_name,
		  op_name, a->name, allow_fallback ? " (no CPU fallback)" : "");
	return allow_fallback ? ERR_FORMAT : ERR_UNSUPPORTED;
}

static status_code recipe_require_native(const model_recipe *r, const struct model *m,
										 op_kind kind) {
	if (!recipe_uses_op(r, m, kind))
		return OK;
	cap_resolution res = cap_resolve(m->backend, m, kind, NULL, 1);
	return res.impl == CAP_NATIVE ? OK : recipe_require_capability(m, m->backend, res.fn, 0, 0, 0);
}

static status_code recipe_check_backend_capabilities(const model_recipe *r, const struct model *m) {
	const backend *a = m->backend;
	status_code	   s;

	if (recipe_uses_op(r, m, OP_ATTENTION_MLA)) {
		backend *host = backend_host();
		s = recipe_require_capability(m, a, "attention_mla", a->attention_mla != NULL,
									  host && host != a && host->attention_mla != NULL, 1);
		if (s != OK)
			return s;
	}
	if (model_has_sliding_layers(m) && !a->attention_swa) {
		backend *host = backend_host();
		s = recipe_require_capability(m, a, "attention_swa", 0,
									  host && host != a && host->attention_swa != NULL, 1);
		if (s != OK)
			return s;
	}
	static const op_kind NATIVE_REQUIRED_OPS[] = {OP_HC_INIT,  OP_HC_PRE, OP_HC_POST,
												  OP_HC_FINAL, OP_KDA,	  OP_MLA_ABSORBED};
	for (size_t i = 0; i < ARRAY_LEN(NATIVE_REQUIRED_OPS); i++) {
		s = recipe_require_native(r, m, NATIVE_REQUIRED_OPS[i]);
		if (s != OK)
			return s;
	}
	if (m->moe.n_experts > 0 && !a->matmul_thread_local &&
		!(backend_has_cap(a, BCAP_MOE_EXPERT_RESIDENT) && a->moe_experts_batch &&
		  a->moe_expert_ffn && a->buffer_alloc_from_host)) {
		backend *host = backend_host();
		s = recipe_require_capability(m, a, "matmul_thread_local", 0,
									  host && host != a && host->matmul_thread_local != NULL, 1);
		if (s != OK)
			return s;
	}

	return OK;
}

int matmul_type_ok(const backend *a, uint32_t t) {
	return !a->matmul_type_native || a->matmul_type_native((backend *)a, t);
}

int ple_build_dev_path_cached(const struct model *m, backend *a, int batched) {
	if (!m->mixed_backend_mode && a == m->backend && m->recipe)
		return batched ? m->recipe->ple_build_dev_ok_batch : m->recipe->ple_build_dev_ok;
	return ple_build_dev_path_ok(a, m, batched);
}

model_recipe *recipe_build(const struct model *m) {
	if (!m || !m->arch_info)
		return NULL;
	const recipe_builder_fn *fb = recipe_lookup(m->arch_info->gguf_name);
	if (!fb) {
		ERROR("recipe_build: no recipe builder registered for arch '%s'", m->arch_info->gguf_name);
		return NULL;
	}
	model_recipe *r = (*fb)(m);
	if (!r)
		return NULL;

	assert(m->moe.n_experts <= MOE_MAX_K);
	if (m->moe.n_experts > MOE_MAX_K) {
		ERROR("recipe_build: n_experts=%d exceeds supported maximum MOE_MAX_K=%d", m->moe.n_experts,
			  MOE_MAX_K);
		recipe_free(r);
		return NULL;
	}
	if (m->moe.n_experts > 0 && m->moe.n_experts_used > MOE_MAX_TOPK)
		ERROR("recipe_build: n_experts_used=%d exceeds supported top-k %d; execution will fail "
			  "rather than silently truncate",
			  m->moe.n_experts_used, MOE_MAX_TOPK);

	int has_vld = m->arch_info->has_variable_layer_dims;

	r->layer_ctx = xcalloc((size_t)m->n_layers, sizeof(layer_ctx_entry));
	for (int li = 0; li < m->n_layers; li++) {
		layer_ctx_entry *lc = &r->layer_ctx[li];
		lc->head_dim		= model_layer_head_dim(m, li);
		lc->n_kv_heads		= model_layer_kv_heads(m, li);
		lc->intermediate	= model_layer_intermediate(m, li);
		lc->has_kv			= layer_has_kv(m, li);
		lc->has_own_v		= model_layer_has_own_v(m, li);
		lc->is_global		= m->layers[li].is_global_layer;
		lc->kv_row_stride	= lc->n_kv_heads * lc->head_dim;
		lc->q_row_stride	= m->n_heads * lc->head_dim;
		lc->kv_layer		= li;
		if (has_vld && !lc->has_kv)
			lc->kv_layer =
				lc->is_global ? m->layer_dims.kv_layer_global : m->layer_dims.kv_layer_swa;
	}

	if (has_vld) {
		int n_ops		 = r->layer.n_ops;
		r->per_layer_ops = xcalloc((size_t)m->n_layers * n_ops, sizeof(recipe_op));
		for (int li = 0; li < m->n_layers; li++) {
			const layer_ctx_entry *lc  = &r->layer_ctx[li];
			recipe_op			  *dst = &r->per_layer_ops[(size_t)li * n_ops];
			memcpy(dst, r->layer.ops, (size_t)n_ops * sizeof(recipe_op));
			for (int j = 0; j < n_ops; j++) {
				recipe_op *op = &dst[j];
				if (!lc->has_kv) {
					if ((op->w_idx == WIDX_WK || op->w_idx == WIDX_WV) || op->kind == OP_KV_PUT ||
						(op->kind == OP_ROPE_EXT && !op->u.rope_ext.is_q) ||
						(op->kind == OP_RMSNORM_PER_HEAD && op->w_idx == WIDX_ATTN_K_NORM) ||
						op->kind == OP_RMSNORM_NOWEIGHT) {
						op->kind = OP_NONE;
						continue;
					}
				}
				if (op->kind == OP_MATMUL || op->kind == OP_MATMUL_RESIDUAL) {
					if (op->u.matmul.n == 0 || op->u.matmul.k == 0)
						resolve_matmul_dims(lc, m, op->w_idx, &op->u.matmul.n, &op->u.matmul.k);
				}
				if (op->kind == OP_MATMUL_FFN_DOWN) {
					if (op->u.matmul_ffn_down.n == 0)
						op->u.matmul_ffn_down.n = m->dim;
					if (op->u.matmul_ffn_down.k == 0)
						op->u.matmul_ffn_down.k = lc->intermediate;
				}
				if (op->kind == OP_MATMUL_MULTI && op->w_idx == WIDX_GATE) {
					if (op->u.matmul_multi.n_out[0] == 0)
						op->u.matmul_multi.n_out[0] = lc->intermediate;
					if (op->u.matmul_multi.n_out[1] == 0)
						op->u.matmul_multi.n_out[1] = lc->intermediate;
				}
				if (op->kind == OP_MATMUL_MULTI && op->w_idx == WIDX_WQ) {
					op->u.matmul_multi.n_out[0] = m->n_heads * lc->head_dim;
					op->u.matmul_multi.n_out[1] = lc->n_kv_heads * lc->head_dim;
					op->u.matmul_multi.n_out[2] = lc->n_kv_heads * lc->head_dim;
				}
				if (op->kind == OP_ATTENTION) {
					op->u.attention.head_dim		  = lc->head_dim;
					op->u.attention.n_kv_heads		  = lc->n_kv_heads;
					op->u.attention.n_kv_heads_active = lc->n_kv_heads;
					op->u.attention.kv_layer		  = lc->kv_layer;
				}
			}
		}
	}

	r->has_sliding_layers = model_has_sliding_layers(m);
	r->ple_build_dev_ok	  = cap_resolve(m->backend, m, OP_PLE_BUILD, NULL, 0).impl == CAP_NATIVE;
	r->ple_build_dev_ok_batch =
		cap_resolve(m->backend, m, OP_PLE_BUILD, NULL, 1).impl == CAP_NATIVE;

	graph_optimize_recipe(r, m);

	if (recipe_check_backend_capabilities(r, m) != OK) {
		recipe_free(r);
		return NULL;
	}

	r->graphs = graph_set_build(r, m, m->n_layers);
	graph_set_validate(r->graphs);
	graph_set_check_weights(r->graphs, m);
	if (exec_plan_build(r, m) != OK) {
		recipe_free(r);
		return NULL;
	}
	if (graph_dump_mode() != CG_DUMP_OFF)
		graph_set_dump(r->graphs, stderr, graph_dump_mode() == CG_DUMP_DOT);
	r->mem	= mem_plan_build(r->graphs, m);
	r->part = part_plan_build(m);
	part_plan_check_carry(r->part, r->graphs);
	backend_fallback_report();

	return r;
}

void recipe_free(model_recipe *r) {
	if (!r)
		return;
	free(r->pre_ops);
	free(r->layer.ops);
	free(r->post_ops);
	free(r->per_layer_ops);
	free(r->layer_ctx);
	exec_plan_free(r->exec);
	graph_set_free(r->graphs);
	mem_plan_free(r->mem);
	part_plan_free(r->part);
	memset(r, 0, sizeof(*r));
	free(r);
}

const weight_ref WEIGHT_REF_NONE = {0};

weight_ref *resolve_weight_ref(const model *m, int li, uint8_t w_idx) {
	if (w_idx == RECIPE_NO_WEIGHT || w_idx == WIDX_NONE)
		return (weight_ref *)&WEIGHT_REF_NONE;
	if (!m || !m->wrefs_by_layer || m->n_layers <= 0)
		return (weight_ref *)&WEIGHT_REF_NONE;
	int			row = (li >= 0 && li < m->n_layers) ? li : 0;
	weight_ref *w	= m->wrefs_by_layer[(size_t)row * WIDX_COUNT + w_idx];
	return w ? w : (weight_ref *)&WEIGHT_REF_NONE;
}

weight_ref *recipe_resolve_weight_ref(const struct model *m, int li, uint8_t w_idx) {
	return resolve_weight_ref(m, li, w_idx);
}

int model_has_sliding_layers(const model *m) {
	for (int i = 0; i < m->n_layers; i++)
		if (m->layers[i].is_sliding)
			return 1;
	return 0;
}
