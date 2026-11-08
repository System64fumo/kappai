#include "recipe/capability.h"

const char *cap_host_only_fn(op_kind kind) {
	if (kind == OP_LAYERNORM || kind == OP_ADD_BIAS || kind == OP_ADD_TYPE_EMBD ||
		kind == OP_SPLIT_QKV || kind == OP_ACTIVATE)
		return "encoder_elementwise";
	if (kind == OP_MOE_ROUTER)
		return "moe_router_topk";
	return NULL;
}

static cap_resolution host_resolution(const char *fn) {
	return (cap_resolution){.impl = CAP_HOST, .fn = fn};
}

static int weight_type_native(const backend *be, const weight_ref *w) {
	return !w || !be->matmul_type_native || be->matmul_type_native((backend *)be, w->type);
}

cap_resolution cap_resolve(backend *be, const struct model *m, op_kind kind, const weight_ref *w,
						   int batched) {
	const char *host_only = cap_host_only_fn(kind);
	if (host_only)
		return backend_has_cap(be, BCAP_IS_HOST) || (kind == OP_MOE_ROUTER && be->moe_router_topk)
				   ? (cap_resolution){.impl = CAP_NATIVE}
				   : host_resolution(host_only);
	if (kind == OP_PLE_BUILD)
		return ple_build_dev_path_ok(be, m, batched) ? (cap_resolution){.impl = CAP_NATIVE}
													 : host_resolution("ple_build");
	if (kind == OP_FFN_ACTIVATE_FUSED && !backend_has_cap(be, BCAP_IS_HOST) &&
		!be->ffn_activate_fused_batch)
		return host_resolution("ffn_activate_fused");
	const char *missing = recipe_op_missing_fn(be, kind);
	if (missing)
		return host_resolution(missing);
	if (w && recipe_op_has_matmul_substage(kind) && !weight_type_native(be, w)) {
		return host_resolution("matmul");
	}
	return (cap_resolution){.impl = CAP_NATIVE};
}
