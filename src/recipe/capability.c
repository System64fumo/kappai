#include "recipe/capability.h"

const char *cap_host_only_fn(op_kind kind) {
	if (kind == OP_LAYERNORM || kind == OP_ADD_BIAS || kind == OP_ADD_TYPE_EMBD ||
		kind == OP_SPLIT_QKV || kind == OP_ACTIVATE)
		return "encoder_elementwise";
	if (kind == OP_MOE_ROUTER)
		return "moe_router_topk";
	return NULL;
}

static cap_resolution host_resolution(const char *fn, host_fallback_reason reason,
									  const char *detail) {
	return (cap_resolution){.impl = CAP_HOST, .fn = fn, .reason = reason, .detail = detail};
}

static int weight_type_native(const backend *be, const weight_ref *w) {
	return !w || !be->matmul_type_native || be->matmul_type_native((backend *)be, w->type);
}

cap_resolution cap_resolve(backend *be, const struct model *m, op_kind kind, const weight_ref *w,
						   int batched) {
	const char *host_only = cap_host_only_fn(kind);
	if (host_only)
		return backend_has_cap(be, BCAP_IS_HOST)
				   ? (cap_resolution){.impl = CAP_NATIVE}
				   : host_resolution(host_only, HFB_OP_NOT_NATIVE,
									 "no device implementation; executes on the host");
	if (kind == OP_PLE_BUILD)
		return ple_build_dev_path_ok(be, m, batched)
				   ? (cap_resolution){.impl = CAP_NATIVE}
				   : host_resolution("ple_build", HFB_WEIGHT_TYPE, NULL);
	if (kind == OP_FFN_ACTIVATE_FUSED && !backend_has_cap(be, BCAP_IS_HOST) &&
		!be->ffn_activate_fused_batch)
		return host_resolution("ffn_activate_fused", HFB_BATCH_DESIGN,
							   "batch prefill stages gate/up activation through host memory");
	const char *missing = recipe_op_missing_fn(be, kind);
	if (missing)
		return host_resolution(missing, HFB_OP_NOT_NATIVE, NULL);
	if (w && recipe_op_has_matmul_substage(kind) && !weight_type_native(be, w)) {
		cap_resolution res = host_resolution("matmul", HFB_WEIGHT_TYPE, NULL);
		res.type		   = w->type;
		return res;
	}
	return (cap_resolution){.impl = CAP_NATIVE};
}

void cap_report(backend *be, const cap_resolution *res) {
	if (res->impl == CAP_NATIVE)
		return;
	if (res->type)
		backend_report_host_fallback(be, res->fn, res->reason, "weight type %u", res->type);
	else if (res->detail)
		backend_report_host_fallback(be, res->fn, res->reason, "%s", res->detail);
	else
		backend_report_host_fallback(be, res->fn, res->reason, NULL);
}
