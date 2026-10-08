#include "test_core.h"

#include "arch.h"
#include "memconfig.h"
#include "model.h"

static float *ref_softmax_attn(const float *q, const float *k, const float *v, int n_heads, int kvh,
							   int hd, int t0, int n_pos, float scale);

static void test_op_matmul(backend *ref, backend *tgt, op_family fam, const qtype_info *qt, int n,
						   int k) {
	char label[128];
	if (!tgt->matmul || !tgt->buffer_alloc_weight) {
		snprintf(label, sizeof(label), "%s matmul N=%d K=%d (%s)", qt->name, n, k, tgt->name);
		record_result(fam, label, V_SKIP, "backend has no native matmul");
		return;
	}
	if (k % qt->block != 0)
		return;
	if (tgt->matmul_type_native && !tgt->matmul_type_native(tgt, qt->type)) {
		snprintf(label, sizeof(label), "%s matmul N=%d K=%d (%s)", qt->name, n, k, tgt->name);
		record_result(fam, label, V_SKIP, "missing native implementation");
		return;
	}

	seed_test_rng((0xA5A5ULL * (qt->type + 1) * 1000003ULL) + ((uint64_t)n * 31) + (uint64_t)k);
	void *blocks = test_make_weight(ref, qt, n, k, NULL);
	if (!blocks) {
		record_result(fam, label, V_SKIP, "reference cannot repack weight");
		return;
	}

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);

	tensor_desc wd = {
		.host_data = blocks,
		.type	   = qt->type,
		.n_dims	   = 2,
		.dims	   = {k, n},
	};

	buffer w_ref   = {0};
	buffer x_ref   = {0};
	buffer y_ref_b = {0};
	ref->buffer_alloc_weight(ref, &wd, &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)k * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &y_ref_b);
	ref->buffer_write_f32(ref, &x_ref, x, k);
	ref->matmul(ref, &w_ref, qt->type, &x_ref, &y_ref_b, n, k);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref_b, y_ref, n);

	buffer w_tgt = {0};
	buffer x_tgt = {0};
	buffer y_tgt = {0};
	tgt->buffer_alloc_weight(tgt, &wd, &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)k * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, k);

	status_code s_tgt;
	{
		s_tgt = tgt->matmul(tgt, &w_tgt, qt->type, &x_tgt, &y_tgt, n, k);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, y_got, n);

	char detail[256];
	snprintf(label, sizeof(label), "%s matmul N=%d K=%d", qt->name, n, k);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(fam, label, v, detail);

	free(y_ref);
	free(y_got);
	free(x);
	free(blocks);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &y_ref_b);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_embd_lookup(backend *ref, backend *tgt, const qtype_info *qt, int dim,
								int vocab) {
	if (!tgt->embd_lookup) {
		char label[128];
		snprintf(label, sizeof(label), "%s embd_lookup dim=%d (%s)", qt->name, dim, tgt->name);
		record_result(OPFAM_EMBD_LOOKUP, label, V_SKIP, "backend has no native embd_lookup");
		return;
	}
	if (dim % qt->block != 0)
		return;
	if (!test_type_per_row(qt->type)) {
		char label[128];
		snprintf(label, sizeof(label), "%s embd_lookup dim=%d (skip)", qt->name, dim);
		record_result(OPFAM_EMBD_LOOKUP, label, V_SKIP,
					  "group-repacked type has no per-row lookup");
		return;
	}

	seed_test_rng((0xC0FFEEULL * (qt->type + 1)) + ((uint64_t)dim * 17));
	void *blocks = test_make_weight(ref, qt, vocab, dim, NULL);
	if (!blocks) {
		char label[128];
		snprintf(label, sizeof(label), "%s embd_lookup dim=%d", qt->name, dim);
		record_result(OPFAM_EMBD_LOOKUP, label, V_SKIP, "reference cannot repack weight");
		return;
	}
	int token = (int)(next_u32() % (uint32_t)vocab);

	tensor_desc wd = {
		.host_data = blocks,
		.type	   = qt->type,
		.n_dims	   = 2,
		.dims	   = {dim, vocab},
	};

	buffer w_ref   = {0};
	buffer out_ref = {0};
	ref->buffer_alloc_weight(ref, &wd, &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)dim * sizeof(float), &out_ref);
	status_code s_ref;
	{
		s_ref = ref->embd_lookup(ref, &w_ref, qt->type, token, dim, &out_ref);
	}
	float *y_ref = xmalloc((size_t)dim * sizeof(float));
	ref->buffer_read_f32(ref, &out_ref, y_ref, dim);

	buffer w_tgt   = {0};
	buffer out_tgt = {0};
	tgt->buffer_alloc_weight(tgt, &wd, &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)dim * sizeof(float), &out_tgt);
	status_code s_tgt;
	{
		s_tgt = tgt->embd_lookup(tgt, &w_tgt, qt->type, token, dim, &out_tgt);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)dim * sizeof(float));
	tgt->buffer_read_f32(tgt, &out_tgt, y_got, dim);

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "%s embd_lookup dim=%d tok=%d", qt->name, dim, token);
	verdict v;
	if (s_ref != OK) {
		snprintf(detail, sizeof(detail), "reference status=%d", s_ref);
		v = V_FAIL;
	} else {
		v = classify_output("loose", y_ref, y_got, dim, s_tgt, detail, sizeof(detail));
	}
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, dim);
	record_result(OPFAM_EMBD_LOOKUP, label, v, detail);

	free(y_ref);
	free(y_got);
	free(blocks);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &out_ref);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &out_tgt);
}

static void test_op_embd_lookup_f32(backend *ref, backend *tgt) {
	if (!tgt->embd_lookup) {
		record_result(OPFAM_EMBD_LOOKUP, "f32 embd_lookup (skip)", V_SKIP,
					  "backend has no native embd_lookup");
		return;
	}
	int	   dim	 = 64;
	int	   vocab = 16;
	float *tab	 = xmalloc((size_t)dim * vocab * sizeof(float));
	seed_test_rng(0xF32E);
	fill_random_f32(tab, dim * vocab, 1.0f);
	int token = 5;

	tensor_desc wd = {
		.host_data = tab,
		.type	   = GGML_TYPE_F32,
		.n_dims	   = 2,
		.dims	   = {dim, vocab},
	};

	buffer w_ref   = {0};
	buffer out_ref = {0};
	ref->buffer_alloc_weight(ref, &wd, &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)dim * sizeof(float), &out_ref);
	ref->embd_lookup(ref, &w_ref, GGML_TYPE_F32, token, dim, &out_ref);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)dim * sizeof(float));
	ref->buffer_read_f32(ref, &out_ref, y_ref, dim);

	buffer w_tgt   = {0};
	buffer out_tgt = {0};
	tgt->buffer_alloc_weight(tgt, &wd, &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)dim * sizeof(float), &out_tgt);
	status_code s_tgt;
	{
		s_tgt = tgt->embd_lookup(tgt, &w_tgt, GGML_TYPE_F32, token, dim, &out_tgt);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)dim * sizeof(float));
	tgt->buffer_read_f32(tgt, &out_tgt, y_got, dim);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "f32 embd_lookup dim=%d", dim);
	verdict v = classify_output("exact", y_ref, y_got, dim, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, dim);
	record_result(OPFAM_EMBD_LOOKUP, label, v, detail);

	free(tab);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &out_ref);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &out_tgt);
}

static void test_op_rmsnorm(backend *ref, backend *tgt, int n) {
	if (!tgt->rmsnorm) {
		char label[96];
		snprintf(label, sizeof(label), "rmsnorm N=%d (%s)", n, tgt->name);
		record_result(OPFAM_RMSNORM, label, V_SKIP, "backend has no native rmsnorm");
		return;
	}
	float *x = xmalloc((size_t)n * sizeof(float));
	float *w = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0x1234ULL + (uint64_t)n);
	fill_random_f32(x, n, 2.0f);
	fill_random_f32(w, n, 1.5f);
	for (int i = 0; i < n; i++)
		w[i] += 1.0f;

	buffer x_ref   = {0};
	buffer w_ref   = {0};
	buffer y_ref_b = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &y_ref_b);
	ref->buffer_write_f32(ref, &x_ref, x, n);
	ref->buffer_write_f32(ref, &w_ref, w, n);
	ref->rmsnorm(ref, &x_ref, &w_ref, &y_ref_b, n, 1e-5f);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref_b, y_ref, n);

	buffer x_tgt = {0};
	buffer w_tgt = {0};
	buffer y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, n);
	tgt->buffer_write_f32(tgt, &w_tgt, w, n);
	status_code s_tgt;
	{
		s_tgt = tgt->rmsnorm(tgt, &x_tgt, &w_tgt, &y_tgt, n, 1e-5f);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, y_got, n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "rmsnorm N=%d", n);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(OPFAM_RMSNORM, label, v, detail);

	free(x);
	free(w);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &y_ref_b);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rmsnorm_per_head(backend *ref, backend *tgt, int n_heads, int head_dim) {
	if (!tgt->rmsnorm_per_head) {
		char label[112];
		snprintf(label, sizeof(label), "rmsnorm_per_head h=%d d=%d (%s)", n_heads, head_dim,
				 tgt->name);
		record_result(OPFAM_RMSNORM_PER_HEAD, label, V_SKIP,
					  "backend has no native rmsnorm_per_head");
		return;
	}
	int	   N = n_heads * head_dim;
	float *x = xmalloc((size_t)N * sizeof(float));
	float *w = xmalloc((size_t)head_dim * sizeof(float));
	seed_test_rng(0x9EADULL + ((uint64_t)n_heads * 91) + ((uint64_t)head_dim * 13));
	fill_random_f32(x, N, 2.0f);
	fill_random_f32(w, head_dim, 1.0f);
	for (int i = 0; i < head_dim; i++)
		w[i] += 1.0f;

	buffer x_ref   = {0};
	buffer w_ref   = {0};
	buffer y_ref_b = {0};
	ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)head_dim * sizeof(float), &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &y_ref_b);
	ref->buffer_write_f32(ref, &x_ref, x, N);
	ref->buffer_write_f32(ref, &w_ref, w, head_dim);
	ref->rmsnorm_per_head(ref, &x_ref, &w_ref, &y_ref_b, n_heads, head_dim, 1e-5f);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref_b, y_ref, N);

	buffer x_tgt = {0};
	buffer w_tgt = {0};
	buffer y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)head_dim * sizeof(float), &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, N);
	tgt->buffer_write_f32(tgt, &w_tgt, w, head_dim);
	status_code s_tgt;
	{
		s_tgt = tgt->rmsnorm_per_head(tgt, &x_tgt, &w_tgt, &y_tgt, n_heads, head_dim, 1e-5f);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)N * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, y_got, N);

	char label[112];
	char detail[256];
	snprintf(label, sizeof(label), "rmsnorm_per_head h=%d d=%d", n_heads, head_dim);
	verdict v = classify_output("loose", y_ref, y_got, N, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, N);
	record_result(OPFAM_RMSNORM_PER_HEAD, label, v, detail);

	free(x);
	free(w);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &y_ref_b);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rmsnorm_noweight(backend *ref, backend *tgt, int n) {
	if (!tgt->rmsnorm_noweight) {
		char label[96];
		snprintf(label, sizeof(label), "rmsnorm_noweight N=%d (%s)", n, tgt->name);
		record_result(OPFAM_RMSNORM_NOWEIGHT, label, V_SKIP,
					  "backend has no native rmsnorm_noweight");
		return;
	}
	float *x = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0x50A9ULL + (uint64_t)n);
	fill_random_f32(x, n, 2.0f);

	buffer x_ref   = {0};
	buffer y_ref_b = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &y_ref_b);
	ref->buffer_write_f32(ref, &x_ref, x, n);
	ref->rmsnorm_noweight(ref, &x_ref, &y_ref_b, n, 1e-5f);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref_b, y_ref, n);

	buffer x_tgt = {0};
	buffer y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, n);
	status_code s_tgt;
	{
		s_tgt = tgt->rmsnorm_noweight(tgt, &x_tgt, &y_tgt, n, 1e-5f);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, y_got, n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "rmsnorm_noweight N=%d", n);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(OPFAM_RMSNORM_NOWEIGHT, label, v, detail);

	free(x);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &y_ref_b);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rope(backend *ref, backend *tgt, int n_heads, int head_dim, int pos) {
	if (!tgt->rope) {
		char label[112];
		snprintf(label, sizeof(label), "rope h=%d d=%d pos=%d (%s)", n_heads, head_dim, pos,
				 tgt->name);
		record_result(OPFAM_ROPE, label, V_SKIP, "backend has no native rope");
		return;
	}
	int N	  = n_heads * head_dim;
	int half  = head_dim / 2;
	int n_ctx = pos + 1;

	float *vec	 = xmalloc((size_t)N * sizeof(float));
	float *cos_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	float *sin_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	seed_test_rng(0x5EEDULL + ((uint64_t)n_heads * 131) + ((uint64_t)head_dim * 17) +
				  (uint64_t)pos);
	fill_random_f32(vec, N, 1.0f);
	for (int j = 0; j < half; j++) {
		float c = cosf(((float)j * 0.0731f) + 0.1f);
		float s = sinf(((float)j * 0.0731f) + 0.1f);
		for (int p = 0; p < n_ctx; p++) {
			cos_v[(p * half) + j] = c;
			sin_v[(p * half) + j] = s;
		}
	}

	kv_desc kvd = {.n_ctx		= n_ctx,
				   .n_kv_heads	= n_heads,
				   .head_dim	= head_dim,
				   .n_layers	= 1,
				   .n_kv_layers = 1};

	ref->rope_neox = 0;
	tgt->rope_neox = 0;

	buffer kc_ref = {0};
	buffer vc_ref = {0};
	buffer v_ref  = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &v_ref);
	ref->buffer_write_f32(ref, &v_ref, vec, N);
	ref->rope(ref, &v_ref, n_heads, head_dim, pos, cos_v, sin_v);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	ref->buffer_read_f32(ref, &v_ref, y_ref, N);

	buffer kc_tgt = {0};
	buffer vc_tgt = {0};
	buffer v_tgt  = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &v_tgt);
	tgt->buffer_write_f32(tgt, &v_tgt, vec, N);
	status_code s_tgt;
	{
		s_tgt = tgt->rope(tgt, &v_tgt, n_heads, head_dim, pos, cos_v, sin_v);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)N * sizeof(float));
	tgt->buffer_read_f32(tgt, &v_tgt, y_got, N);

	char label[112];
	char detail[256];
	snprintf(label, sizeof(label), "rope h=%d d=%d pos=%d", n_heads, head_dim, pos);
	verdict v = classify_output("loose", y_ref, y_got, N, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, N);
	record_result(OPFAM_ROPE, label, v, detail);

	free(vec);
	free(cos_v);
	free(sin_v);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &v_ref);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	tgt->buffer_free(tgt, &v_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_rope_ext(backend *ref, backend *tgt, int n_heads, int head_dim, int pos,
							 int use_freq_factors) {
	if (!tgt->rope_ext) {
		char label[128];
		snprintf(label, sizeof(label), "rope_ext h=%d d=%d pos=%d ff=%d (%s)", n_heads, head_dim,
				 pos, use_freq_factors, tgt->name);
		record_result(OPFAM_ROPE_EXT, label, V_SKIP, "backend has no native rope_ext");
		return;
	}
	int N	  = n_heads * head_dim;
	int half  = head_dim / 2;
	int n_ctx = pos + 1;

	float *vec	 = xmalloc((size_t)N * sizeof(float));
	float *cos_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	float *sin_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	float *ff	 = use_freq_factors ? xmalloc((size_t)half * sizeof(float)) : NULL;
	seed_test_rng(0xF00DULL + ((uint64_t)n_heads * 131) + ((uint64_t)head_dim * 17) +
				  (uint64_t)pos + ((uint64_t)use_freq_factors * 977));
	fill_random_f32(vec, N, 1.0f);
	for (int j = 0; j < half; j++) {
		float c = cosf(((float)j * 0.0511f) + 0.2f);
		float s = sinf(((float)j * 0.0511f) + 0.2f);
		for (int p = 0; p < n_ctx; p++) {
			cos_v[(p * half) + j] = c;
			sin_v[(p * half) + j] = s;
		}
	}
	if (ff)
		for (int j = 0; j < half; j++)
			ff[j] = (j % 5 == 0) ? 1e12f : (1.0f + (0.01f * (float)j));

	ref->rope_theta = 10000.0f;
	tgt->rope_theta = 10000.0f;
	ref->rope_neox	= 0;
	tgt->rope_neox	= 0;

	kv_desc kvd = {.n_ctx		= n_ctx,
				   .n_kv_heads	= n_heads,
				   .head_dim	= head_dim,
				   .n_layers	= 1,
				   .n_kv_layers = 1};

	buffer kc_ref = {0};
	buffer vc_ref = {0};
	buffer v_ref  = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &v_ref);
	ref->buffer_write_f32(ref, &v_ref, vec, N);
	ref->rope_ext(ref, &v_ref, n_heads, head_dim, pos, cos_v, sin_v, ff);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	ref->buffer_read_f32(ref, &v_ref, y_ref, N);

	buffer kc_tgt = {0};
	buffer vc_tgt = {0};
	buffer v_tgt  = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &v_tgt);
	tgt->buffer_write_f32(tgt, &v_tgt, vec, N);
	status_code s_tgt;
	{
		s_tgt = tgt->rope_ext(tgt, &v_tgt, n_heads, head_dim, pos, cos_v, sin_v, ff);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)N * sizeof(float));
	tgt->buffer_read_f32(tgt, &v_tgt, y_got, N);

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "rope_ext h=%d d=%d pos=%d ff=%d", n_heads, head_dim, pos,
			 use_freq_factors);
	verdict v = classify_output("loose", y_ref, y_got, N, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, N);
	record_result(OPFAM_ROPE_EXT, label, v, detail);

	free(vec);
	free(cos_v);
	free(sin_v);
	free(ff);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &v_ref);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	tgt->buffer_free(tgt, &v_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_add_inplace(backend *ref, backend *tgt, int n) {
	if (!tgt->add_inplace) {
		char label[96];
		snprintf(label, sizeof(label), "add_inplace N=%d (%s)", n, tgt->name);
		record_result(OPFAM_ADD_INPLACE, label, V_SKIP, "backend has no native add_inplace");
		return;
	}
	float *x = xmalloc((size_t)n * sizeof(float));
	float *y = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0xADD1ULL + (uint64_t)n);
	fill_random_f32(x, n, 1.0f);
	fill_random_f32(y, n, 1.0f);

	buffer x_ref = {0};
	buffer y_ref = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &y_ref);
	ref->buffer_write_f32(ref, &x_ref, x, n);
	ref->buffer_write_f32(ref, &y_ref, y, n);
	ref->add_inplace(ref, &x_ref, &y_ref, n);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *r_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &x_ref, r_ref, n);

	buffer x_tgt = {0};
	buffer y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, n);
	tgt->buffer_write_f32(tgt, &y_tgt, y, n);
	status_code s_tgt;
	{
		s_tgt = tgt->add_inplace(tgt, &x_tgt, &y_tgt, n);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *r_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &x_tgt, r_got, n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "add_inplace N=%d", n);
	verdict v = classify_output("exact", r_ref, r_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(r_ref, r_got, n);
	record_result(OPFAM_ADD_INPLACE, label, v, detail);

	free(x);
	free(y);
	free(r_ref);
	free(r_got);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &y_ref);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_ple_combine(backend *ref, backend *tgt, int n) {
	if (!tgt->ple_combine) {
		char label[96];
		snprintf(label, sizeof(label), "ple_combine N=%d (%s)", n, tgt->name);
		record_result(OPFAM_PLE_COMBINE, label, V_SKIP, "backend has no native ple_combine");
		return;
	}
	float *ple	= xmalloc((size_t)n * sizeof(float));
	float *proj = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0xFE1E0ULL + (uint64_t)n);
	fill_random_f32(ple, n, 1.0f);
	fill_random_f32(proj, n, 1.0f);
	const float combine_scale = 0.70710678118654752f;

	buffer ple_ref	= {0};
	buffer proj_ref = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &ple_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &proj_ref);
	ref->buffer_write_f32(ref, &ple_ref, ple, n);
	ref->buffer_write_f32(ref, &proj_ref, proj, n);
	ref->ple_combine(ref, &ple_ref, &proj_ref, n, combine_scale);
	if (ref->synchronize)
		ref->synchronize(ref);
	float *r_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &ple_ref, r_ref, n);

	buffer ple_tgt	= {0};
	buffer proj_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &ple_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &proj_tgt);
	tgt->buffer_write_f32(tgt, &ple_tgt, ple, n);
	tgt->buffer_write_f32(tgt, &proj_tgt, proj, n);
	status_code s_tgt = tgt->ple_combine(tgt, &ple_tgt, &proj_tgt, n, combine_scale);
	if (tgt->synchronize)
		tgt->synchronize(tgt);
	float *r_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &ple_tgt, r_got, n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "ple_combine N=%d", n);
	verdict v = classify_output("tight", r_ref, r_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(r_ref, r_got, n);
	record_result(OPFAM_PLE_COMBINE, label, v, detail);

	free(ple);
	free(proj);
	free(r_ref);
	free(r_got);
	ref->buffer_free(ref, &ple_ref);
	ref->buffer_free(ref, &proj_ref);
	tgt->buffer_free(tgt, &ple_tgt);
	tgt->buffer_free(tgt, &proj_tgt);
}

static void test_op_rmsnorm_add(backend *ref, backend *tgt, int n) {
	if (!tgt->rmsnorm_add || !backend_has_cap(tgt, BCAP_RMSNORM_ADD)) {
		char label[96];
		snprintf(label, sizeof(label), "rmsnorm_add N=%d (%s)", n, tgt->name);
		record_result(OPFAM_RMSNORM_ADD, label, V_SKIP, "backend has no native rmsnorm_add");
		return;
	}
	float *x		= xmalloc((size_t)n * sizeof(float));
	float *w		= xmalloc((size_t)n * sizeof(float));
	float *residual = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0xADD2ULL + (uint64_t)n);
	fill_random_f32(x, n, 1.0f);
	fill_random_f32(w, n, 1.0f);
	fill_random_f32(residual, n, 1.0f);
	const float eps = 1e-6f;

	buffer x_ref = {0}, w_ref = {0}, r_ref_b = {0}, y_ref_b = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &r_ref_b);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &y_ref_b);
	ref->buffer_write_f32(ref, &x_ref, x, n);
	ref->buffer_write_f32(ref, &w_ref, w, n);
	ref->buffer_write_f32(ref, &r_ref_b, residual, n);
	ref->rmsnorm_add(ref, &x_ref, &w_ref, &r_ref_b, &y_ref_b, n, eps, 1.0f);
	if (ref->synchronize)
		ref->synchronize(ref);
	float *r_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref_b, r_ref, n);

	buffer x_tgt = {0}, w_tgt = {0}, r_tgt = {0}, y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &r_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, n);
	tgt->buffer_write_f32(tgt, &w_tgt, w, n);
	tgt->buffer_write_f32(tgt, &r_tgt, residual, n);
	status_code s_tgt = tgt->rmsnorm_add(tgt, &x_tgt, &w_tgt, &r_tgt, &y_tgt, n, eps, 1.0f);
	if (tgt->synchronize)
		tgt->synchronize(tgt);
	float *r_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, r_got, n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "rmsnorm_add N=%d", n);
	verdict v = classify_output("loose", r_ref, r_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(r_ref, r_got, n);
	record_result(OPFAM_RMSNORM_ADD, label, v, detail);

	free(x);
	free(residual);
	free(r_ref);
	free(r_got);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &r_ref_b);
	ref->buffer_free(ref, &y_ref_b);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &r_tgt);
	tgt->buffer_free(tgt, &y_tgt);

	if (tgt->rmsnorm_add_batch) {
		const int	m		  = 3;
		const float scales[2] = {1.0f, 0.5f};
		for (int si = 0; si < 2; si++) {
			float  out_scale = scales[si];
			float *xb		 = xmalloc((size_t)m * (size_t)n * sizeof(float));
			float *rb		 = xmalloc((size_t)m * (size_t)n * sizeof(float));
			seed_test_rng(0xADD3ULL + (uint64_t)n + (uint64_t)si);
			fill_random_f32(xb, m * n, 1.0f);
			fill_random_f32(rb, m * n, 1.0f);

			buffer x_br = {0}, w_br = {0}, r_br = {0}, y_br = {0};
			ref->buffer_alloc_scratch(ref, (size_t)m * (size_t)n * sizeof(float), &x_br);
			ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &w_br);
			ref->buffer_alloc_scratch(ref, (size_t)m * (size_t)n * sizeof(float), &r_br);
			ref->buffer_alloc_scratch(ref, (size_t)m * (size_t)n * sizeof(float), &y_br);
			ref->buffer_write_f32(ref, &x_br, xb, m * n);
			ref->buffer_write_f32(ref, &w_br, w, n);
			ref->buffer_write_f32(ref, &r_br, rb, m * n);
			if (ref->rmsnorm_add_batch)
				ref->rmsnorm_add_batch(ref, &x_br, &w_br, &r_br, &y_br, n, eps, out_scale, m);
			else {
				for (int row = 0; row < m; row++) {
					buffer xr = x_br, rr = r_br, yr = y_br;
					xr.offset += (size_t)row * (size_t)n * sizeof(float);
					rr.offset += (size_t)row * (size_t)n * sizeof(float);
					yr.offset += (size_t)row * (size_t)n * sizeof(float);
					ref->rmsnorm_add(ref, &xr, &w_br, &rr, &yr, n, eps, out_scale);
				}
			}
			if (ref->synchronize)
				ref->synchronize(ref);
			float *r_bref = xmalloc((size_t)m * (size_t)n * sizeof(float));
			ref->buffer_read_f32(ref, &y_br, r_bref, m * n);

			buffer x_bt = {0}, w_bt = {0}, r_bt = {0}, y_bt = {0};
			tgt->buffer_alloc_scratch(tgt, (size_t)m * (size_t)n * sizeof(float), &x_bt);
			tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &w_bt);
			tgt->buffer_alloc_scratch(tgt, (size_t)m * (size_t)n * sizeof(float), &r_bt);
			tgt->buffer_alloc_scratch(tgt, (size_t)m * (size_t)n * sizeof(float), &y_bt);
			tgt->buffer_write_f32(tgt, &x_bt, xb, m * n);
			tgt->buffer_write_f32(tgt, &w_bt, w, n);
			tgt->buffer_write_f32(tgt, &r_bt, rb, m * n);
			status_code s_bt =
				tgt->rmsnorm_add_batch(tgt, &x_bt, &w_bt, &r_bt, &y_bt, n, eps, out_scale, m);
			if (tgt->synchronize)
				tgt->synchronize(tgt);
			float *r_bgot = xmalloc((size_t)m * (size_t)n * sizeof(float));
			tgt->buffer_read_f32(tgt, &y_bt, r_bgot, m * n);

			char blabel[96];
			char bdetail[256];
			snprintf(blabel, sizeof(blabel), "rmsnorm_add_batch N=%d M=%d os=%.1f", n, m,
					 out_scale);
			verdict bv =
				classify_output("loose", r_bref, r_bgot, m * n, s_bt, bdetail, sizeof(bdetail));
			if (bv != V_PASS && bv != V_SKIP)
				compute_debug(r_bref, r_bgot, m * n);
			record_result(OPFAM_RMSNORM_ADD, blabel, bv, bdetail);

			free(xb);
			free(rb);
			free(r_bref);
			free(r_bgot);
			ref->buffer_free(ref, &x_br);
			ref->buffer_free(ref, &w_br);
			ref->buffer_free(ref, &r_br);
			ref->buffer_free(ref, &y_br);
			tgt->buffer_free(tgt, &x_bt);
			tgt->buffer_free(tgt, &w_bt);
			tgt->buffer_free(tgt, &r_bt);
			tgt->buffer_free(tgt, &y_bt);
		}
	}
	free(w);
}

static void test_op_matmul_ffn_down_batch(backend *ref, backend *tgt, int n, int k) {
	if (!tgt->matmul_ffn_down_batch) {
		char label[96];
		snprintf(label, sizeof(label), "matmul_ffn_down_batch N=%d K=%d (%s)", n, k, tgt->name);
		record_result(OPFAM_MATMUL_FFN_DOWN, label, V_SKIP,
					  "backend has no native matmul_ffn_down_batch");
		return;
	}
	const int m			 = 3;
	const int activation = 1;
	float	 *w			 = xmalloc((size_t)n * (size_t)k * sizeof(float));
	float	 *g			 = xmalloc((size_t)m * (size_t)k * sizeof(float));
	float	 *u			 = xmalloc((size_t)m * (size_t)k * sizeof(float));
	seed_test_rng(0xD090ULL + (uint64_t)n * 131 + (uint64_t)k);
	fill_random_f32(w, n * k, 1.0f);
	fill_random_f32(g, m * k, 2.0f);
	fill_random_f32(u, m * k, 2.0f);

	buffer w_ref = {0}, g_ref = {0}, u_ref = {0}, y_ref = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * (size_t)k * sizeof(float), &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)m * (size_t)k * sizeof(float), &g_ref);
	ref->buffer_alloc_scratch(ref, (size_t)m * (size_t)k * sizeof(float), &u_ref);
	ref->buffer_alloc_scratch(ref, (size_t)m * (size_t)n * sizeof(float), &y_ref);
	ref->buffer_write_f32(ref, &w_ref, w, n * k);
	ref->buffer_write_f32(ref, &g_ref, g, m * k);
	ref->buffer_write_f32(ref, &u_ref, u, m * k);
	for (int row = 0; row < m; row++) {
		buffer gr = g_ref, ur = u_ref, yr = y_ref;
		gr.offset += (size_t)row * (size_t)k * sizeof(float);
		ur.offset += (size_t)row * (size_t)k * sizeof(float);
		yr.offset += (size_t)row * (size_t)n * sizeof(float);
		ref->matmul_ffn_down(ref, &w_ref, GGML_TYPE_F32, &gr, &ur, &yr, n, k, activation);
	}
	if (ref->synchronize)
		ref->synchronize(ref);
	float *r_ref = xmalloc((size_t)m * (size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref, r_ref, m * n);

	buffer w_tgt = {0}, g_tgt = {0}, u_tgt = {0}, y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * (size_t)k * sizeof(float), &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)m * (size_t)k * sizeof(float), &g_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)m * (size_t)k * sizeof(float), &u_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)m * (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &w_tgt, w, n * k);
	tgt->buffer_write_f32(tgt, &g_tgt, g, m * k);
	tgt->buffer_write_f32(tgt, &u_tgt, u, m * k);
	status_code s_tgt = tgt->matmul_ffn_down_batch(tgt, &w_tgt, GGML_TYPE_F32, &g_tgt, &u_tgt,
												   &y_tgt, n, k, activation, m);
	if (tgt->synchronize)
		tgt->synchronize(tgt);
	float *r_got = xmalloc((size_t)m * (size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, r_got, m * n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "matmul_ffn_down_batch N=%d K=%d M=%d", n, k, m);
	verdict v = classify_output("loose", r_ref, r_got, m * n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(r_ref, r_got, m * n);
	record_result(OPFAM_MATMUL_FFN_DOWN, label, v, detail);

	free(w);
	free(g);
	free(u);
	free(r_ref);
	free(r_got);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &g_ref);
	ref->buffer_free(ref, &u_ref);
	ref->buffer_free(ref, &y_ref);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &g_tgt);
	tgt->buffer_free(tgt, &u_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_ffn_activate(backend *ref, backend *tgt, int n) {
	if (!tgt->ffn_activate) {
		char label[96];
		snprintf(label, sizeof(label), "ffn_activate(SwiGLU) N=%d (%s)", n, tgt->name);
		record_result(OPFAM_FFN_ACTIVATE, label, V_SKIP, "backend has no native ffn_activate");
		return;
	}
	float *g = xmalloc((size_t)n * sizeof(float));
	float *u = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0xFFA0ULL + (uint64_t)n);
	fill_random_f32(g, n, 3.0f);
	fill_random_f32(u, n, 3.0f);

	buffer g_ref = {0};
	buffer u_ref = {0};
	buffer o_ref = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &g_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &u_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &o_ref);
	ref->buffer_write_f32(ref, &g_ref, g, n);
	ref->buffer_write_f32(ref, &u_ref, u, n);
	ref->ffn_activate(ref, &g_ref, &u_ref, &o_ref, n);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &o_ref, y_ref, n);

	buffer g_tgt = {0};
	buffer u_tgt = {0};
	buffer o_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &g_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &u_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &o_tgt);
	tgt->buffer_write_f32(tgt, &g_tgt, g, n);
	tgt->buffer_write_f32(tgt, &u_tgt, u, n);
	status_code s_tgt;
	{
		s_tgt = tgt->ffn_activate(tgt, &g_tgt, &u_tgt, &o_tgt, n);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &o_tgt, y_got, n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "ffn_activate(SwiGLU) N=%d", n);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(OPFAM_FFN_ACTIVATE, label, v, detail);

	free(g);
	free(u);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &g_ref);
	ref->buffer_free(ref, &u_ref);
	ref->buffer_free(ref, &o_ref);
	tgt->buffer_free(tgt, &g_tgt);
	tgt->buffer_free(tgt, &u_tgt);
	tgt->buffer_free(tgt, &o_tgt);
}

static void test_op_ffn_activate_ex(backend *ref, backend *tgt, int n, int activation) {
	if (!tgt->ffn_activate_ex) {
		char label[112];
		snprintf(label, sizeof(label), "ffn_activate_ex(%s) N=%d (%s)",
				 activation == 1 ? "GELU" : "SiLU", n, tgt->name);
		record_result(OPFAM_FFN_ACTIVATE_EX, label, V_SKIP,
					  "backend has no native ffn_activate_ex");
		return;
	}
	float *g = xmalloc((size_t)n * sizeof(float));
	float *u = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0x6E10ULL + ((uint64_t)n * 7) + ((uint64_t)activation * 131));
	fill_random_f32(g, n, 4.0f);
	fill_random_f32(u, n, 4.0f);

	buffer g_ref = {0};
	buffer u_ref = {0};
	buffer o_ref = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &g_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &u_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &o_ref);
	ref->buffer_write_f32(ref, &g_ref, g, n);
	ref->buffer_write_f32(ref, &u_ref, u, n);
	ref->ffn_activate_ex(ref, &g_ref, &u_ref, &o_ref, n, activation);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &o_ref, y_ref, n);

	buffer g_tgt = {0};
	buffer u_tgt = {0};
	buffer o_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &g_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &u_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &o_tgt);
	tgt->buffer_write_f32(tgt, &g_tgt, g, n);
	tgt->buffer_write_f32(tgt, &u_tgt, u, n);
	status_code s_tgt;
	{
		s_tgt = tgt->ffn_activate_ex(tgt, &g_tgt, &u_tgt, &o_tgt, n, activation);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &o_tgt, y_got, n);

	char label[112];
	char detail[256];
	snprintf(label, sizeof(label), "ffn_activate_ex(%s) N=%d", activation == 1 ? "GELU" : "SiLU",
			 n);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(OPFAM_FFN_ACTIVATE_EX, label, v, detail);

	free(g);
	free(u);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &g_ref);
	ref->buffer_free(ref, &u_ref);
	ref->buffer_free(ref, &o_ref);
	tgt->buffer_free(tgt, &g_tgt);
	tgt->buffer_free(tgt, &u_tgt);
	tgt->buffer_free(tgt, &o_tgt);
}

static void test_op_attention(backend *ref, backend *tgt, int n_heads, int n_kv_heads, int head_dim,
							  int n_ctx, int pos, int flash) {
	if (!tgt->attention) {
		char label[128];
		snprintf(label, sizeof(label), "attention h=%d/%d d=%d pos=%d flash=%d (%s)", n_heads,
				 n_kv_heads, head_dim, pos, flash, tgt->name);
		record_result(OPFAM_ATTENTION, label, V_SKIP, "backend has no native attention");
		return;
	}
	int	  n		= n_heads * head_dim;
	int	  n_kv	= n_kv_heads * head_dim;
	float scale = 1.0f / sqrtf((float)head_dim);
	int	  n_t	= pos + 1;

	seed_test_rng(0x47E47104ULL + ((uint64_t)n_heads * 97) + ((uint64_t)n_kv_heads * 13) +
				  ((uint64_t)head_dim * 7) + (uint64_t)pos + ((uint64_t)flash * 7));

	float *q = xmalloc((size_t)n * sizeof(float));
	fill_random_f32(q, n, 1.0f);
	float **k_all = xmalloc((size_t)n_t * sizeof(float *));
	float **v_all = xmalloc((size_t)n_t * sizeof(float *));
	for (int t = 0; t < n_t; t++) {
		k_all[t] = xmalloc((size_t)n_kv * sizeof(float));
		v_all[t] = xmalloc((size_t)n_kv * sizeof(float));
		fill_random_f32(k_all[t], n_kv, 1.0f);
		fill_random_f32(v_all[t], n_kv, 1.0f);
	}

	kv_desc kvd = {.n_ctx		= n_ctx,
				   .n_kv_heads	= n_kv_heads,
				   .head_dim	= head_dim,
				   .n_layers	= 1,
				   .n_kv_layers = 1};

	buffer kc_ref  = {0};
	buffer vc_ref  = {0};
	buffer ki_ref  = {0};
	buffer vi_ref  = {0};
	buffer q_ref   = {0};
	buffer out_ref = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &ki_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &vi_ref);
	for (int t = 0; t < n_t; t++) {
		ref->buffer_write_f32(ref, &ki_ref, k_all[t], n_kv);
		ref->buffer_write_f32(ref, &vi_ref, v_all[t], n_kv);
		ref->kv_put(ref, &kc_ref, &vc_ref, 0, t, &ki_ref, &vi_ref, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
	}
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &q_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &out_ref);
	ref->buffer_write_f32(ref, &q_ref, q, n);
	ref->attention(ref, &q_ref, &kc_ref, &vc_ref, &out_ref, 0, pos, n_heads, n_kv_heads, head_dim,
				   n_ctx, 0, scale, n_kv_heads);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &out_ref, y_ref, n);

	buffer kc_tgt  = {0};
	buffer vc_tgt  = {0};
	buffer ki_tgt  = {0};
	buffer vi_tgt  = {0};
	buffer q_tgt   = {0};
	buffer out_tgt = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &ki_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &vi_tgt);
	for (int t = 0; t < n_t; t++) {
		tgt->buffer_write_f32(tgt, &ki_tgt, k_all[t], n_kv);
		tgt->buffer_write_f32(tgt, &vi_tgt, v_all[t], n_kv);
		tgt->kv_put(tgt, &kc_tgt, &vc_tgt, 0, t, &ki_tgt, &vi_tgt, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
	}
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &q_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &out_tgt);
	tgt->buffer_write_f32(tgt, &q_tgt, q, n);
	status_code s_tgt;
	{
		s_tgt = tgt->attention(tgt, &q_tgt, &kc_tgt, &vc_tgt, &out_tgt, 0, pos, n_heads, n_kv_heads,
							   head_dim, n_ctx, flash, scale, n_kv_heads);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &out_tgt, y_got, n);

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "attention h=%d/%d d=%d pos=%d flash=%d", n_heads, n_kv_heads,
			 head_dim, pos, flash);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	record_result(OPFAM_ATTENTION, label, v, detail);

	for (int t = 0; t < n_t; t++) {
		free(k_all[t]);
		free(v_all[t]);
	}
	free(k_all);
	free(v_all);
	free(q);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &ki_ref);
	ref->buffer_free(ref, &vi_ref);
	ref->buffer_free(ref, &q_ref);
	ref->buffer_free(ref, &out_ref);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	tgt->buffer_free(tgt, &ki_tgt);
	tgt->buffer_free(tgt, &vi_tgt);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &out_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_attention_swa(backend *ref, backend *tgt, int n_heads, int n_kv_heads,
								  int head_dim, int n_ctx, int pos, int sliding_window, int flash) {
	if (!tgt->attention_swa) {
		char label[160];
		snprintf(label, sizeof(label), "attention_swa h=%d/%d d=%d pos=%d win=%d flash=%d (%s)",
				 n_heads, n_kv_heads, head_dim, pos, sliding_window, flash, tgt->name);
		record_result(OPFAM_ATTENTION_SWA, label, V_SKIP, "backend has no native attention_swa");
		return;
	}
	int	  n		= n_heads * head_dim;
	int	  n_kv	= n_kv_heads * head_dim;
	float scale = 1.0f / sqrtf((float)head_dim);

	seed_test_rng(0x54A9ULL + ((uint64_t)n_heads * 97) + ((uint64_t)n_kv_heads * 13) +
				  ((uint64_t)head_dim * 7) + ((uint64_t)pos * 3) + (uint64_t)sliding_window +
				  ((uint64_t)flash * 5));

	float *q = xmalloc((size_t)n * sizeof(float));
	fill_random_f32(q, n, 1.0f);

	kv_desc kvd = {.n_ctx		= n_ctx,
				   .n_kv_heads	= n_kv_heads,
				   .head_dim	= head_dim,
				   .n_layers	= 1,
				   .n_kv_layers = 1};

	buffer kc_ref  = {0};
	buffer vc_ref  = {0};
	buffer q_ref   = {0};
	buffer out_ref = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &q_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &out_ref);
	ref->buffer_write_f32(ref, &q_ref, q, n);

	buffer kc_tgt  = {0};
	buffer vc_tgt  = {0};
	buffer q_tgt   = {0};
	buffer out_tgt = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &q_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &out_tgt);
	tgt->buffer_write_f32(tgt, &q_tgt, q, n);

	float *kbuf = xmalloc((size_t)n_kv * sizeof(float));
	float *vbuf = xmalloc((size_t)n_kv * sizeof(float));
	for (int p = 0; p <= pos; p++) {
		fill_random_f32(kbuf, n_kv, 1.0f);
		fill_random_f32(vbuf, n_kv, 1.0f);
		buffer k_in_ref = {0};
		buffer v_in_ref = {0};
		ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &k_in_ref);
		ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &v_in_ref);
		ref->buffer_write_f32(ref, &k_in_ref, kbuf, n_kv);
		ref->buffer_write_f32(ref, &v_in_ref, vbuf, n_kv);
		ref->kv_put(ref, &kc_ref, &vc_ref, 0, p, &k_in_ref, &v_in_ref, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
		ref->buffer_free(ref, &k_in_ref);
		ref->buffer_free(ref, &v_in_ref);

		buffer k_in_tgt = {0};
		buffer v_in_tgt = {0};
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &k_in_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &v_in_tgt);
		tgt->buffer_write_f32(tgt, &k_in_tgt, kbuf, n_kv);
		tgt->buffer_write_f32(tgt, &v_in_tgt, vbuf, n_kv);
		tgt->kv_put(tgt, &kc_tgt, &vc_tgt, 0, p, &k_in_tgt, &v_in_tgt, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
		tgt->buffer_free(tgt, &k_in_tgt);
		tgt->buffer_free(tgt, &v_in_tgt);
	}
	free(kbuf);
	free(vbuf);
	ref->attention_swa(ref, &q_ref, &kc_ref, &vc_ref, &out_ref, 0, pos, n_heads, n_kv_heads,
					   head_dim, n_ctx, 0, scale, sliding_window, n_kv_heads);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &out_ref, y_ref, n);

	status_code s_tgt;
	{
		s_tgt =
			tgt->attention_swa(tgt, &q_tgt, &kc_tgt, &vc_tgt, &out_tgt, 0, pos, n_heads, n_kv_heads,
							   head_dim, n_ctx, flash, scale, sliding_window, n_kv_heads);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &out_tgt, y_got, n);

	char label[160];
	char detail[256];
	snprintf(label, sizeof(label), "attention_swa h=%d/%d d=%d pos=%d win=%d flash=%d", n_heads,
			 n_kv_heads, head_dim, pos, sliding_window, flash);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	record_result(OPFAM_ATTENTION_SWA, label, v, detail);

	free(q);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &q_ref);
	ref->buffer_free(ref, &out_ref);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &out_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_kv_batch_put_layers(backend *b, int compacted) {
	char label[160];
	snprintf(label, sizeof(label), "kv batch put per-layer dims cap=%s [%s]",
			 compacted ? "compact" : "full", b->name);
	if (!b->kv_put_batch || !b->kv_alloc || !b->kv_free || !b->attention) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP,
					  "backend has no kv_put_batch/attention");
		return;
	}
	if (compacted && (!b->attention_swa || !backend_has_cap(b, BCAP_KV_POS_CAP))) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP,
					  "backend lacks attention_swa or BCAP_KV_POS_CAP");
		return;
	}
	static const int LHD[3]	  = {64, 96, 128};
	static const int LK[3]	  = {1, 1, 1};
	const int		 n_layers = 3, n_ctx = 64, n_heads = 4, n_pos = 32, chunk = 16, cap = 20;
	int				 pcaps[3] = {cap, cap, cap};
	kv_desc			 d		  = {.n_layers		   = n_layers,
								 .n_kv_layers	   = n_layers,
								 .n_kv_heads	   = 1,
								 .head_dim		   = LHD[2],
								 .n_ctx			   = n_ctx,
								 .kv_quant		   = KV_QUANT_F16,
								 .layer_head_dim   = LHD,
								 .layer_n_kv_heads = LK,
								 .layer_pos_cap	   = compacted ? pcaps : NULL};
	buffer			 kc = {0}, vc = {0}, kb = {0}, vb = {0}, qb = {0}, ob = {0};
	if (b->kv_alloc(b, &d, &kc, &vc) != OK) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP, "kv_alloc failed");
		return;
	}
	b->buffer_alloc_scratch(b, (size_t)LHD[2] * chunk * sizeof(float), &kb);
	b->buffer_alloc_scratch(b, (size_t)LHD[2] * chunk * sizeof(float), &vb);
	b->buffer_alloc_scratch(b, (size_t)n_heads * LHD[2] * sizeof(float), &qb);
	b->buffer_alloc_scratch(b, (size_t)n_heads * LHD[2] * sizeof(float), &ob);

	float *kf	= xmalloc((size_t)n_pos * LHD[2] * sizeof(float));
	float *vf	= xmalloc((size_t)n_pos * LHD[2] * sizeof(float));
	float *qf	= xmalloc((size_t)n_heads * LHD[2] * sizeof(float));
	float *got	= xmalloc((size_t)n_heads * LHD[2] * sizeof(float));
	float *krow = xmalloc((size_t)LHD[2] * sizeof(float));
	float *vrow = xmalloc((size_t)LHD[2] * sizeof(float));
	seed_test_rng(0x5A17ULL);
	const int t0	 = (compacted && n_pos > cap) ? n_pos - cap : 0;
	const int window = compacted ? cap : n_pos;
	float	  worst	 = 0.0f;
	for (int l = 0; l < n_layers; l++) {
		int hd = LHD[l];
		fill_random_f32(kf, (size_t)n_pos * hd, 1.0f);
		fill_random_f32(vf, (size_t)n_pos * hd, 1.0f);
		for (int start = 0; start < n_pos; start += chunk) {
			int rows = (n_pos - start) < chunk ? (n_pos - start) : chunk;
			for (int r = 0; r < rows; r++) {
				memcpy(krow, kf + (size_t)(start + r) * hd, (size_t)hd * sizeof(float));
				memcpy(vrow, vf + (size_t)(start + r) * hd, (size_t)hd * sizeof(float));
				b->buffer_write_f32(b, &kb, krow, hd);
				b->buffer_write_f32(b, &vb, vrow, hd);
				b->kv_put_batch(b, &kc, &vc, l, start + r, &kb, &vb, hd, 1, hd, n_ctx, 1, 1);
			}
		}
		fill_random_f32(qf, (size_t)n_heads * hd, 1.0f);
		b->buffer_write_f32(b, &qb, qf, (size_t)n_heads * hd);
		const float scale = 1.0f / sqrtf((float)hd);
		if (compacted)
			b->attention_swa(b, &qb, &kc, &vc, &ob, l, n_pos - 1, n_heads, 1, hd, n_ctx, 1, scale,
							 window, 1);
		else
			b->attention(b, &qb, &kc, &vc, &ob, l, n_pos - 1, n_heads, 1, hd, n_ctx, 1, scale, 1);
		b->buffer_read_f32(b, &ob, got, (size_t)n_heads * hd);
		float *r  = ref_softmax_attn(qf, kf, vf, n_heads, 1, hd, t0, n_pos, scale);
		float  lw = 0.0f;
		for (int i = 0; i < n_heads * hd; i++) {
			float e = fabsf(r[i] - got[i]);
			if (e > lw)
				lw = e;
		}
		if (lw > worst)
			worst = lw;
		free(r);
	}
	char detail[192];
	snprintf(detail, sizeof(detail), "max_abs=%.3e over %d layers (hd %d/%d/%d, window %d)", worst,
			 n_layers, LHD[0], LHD[1], LHD[2], window);
	record_result(OPFAM_KV_QUANT_PARITY, label, worst < 2e-2f ? V_PASS : V_FAIL, detail);
	free(kf);
	free(vf);
	free(qf);
	free(got);
	free(krow);
	free(vrow);
	b->buffer_free(b, &kb);
	b->buffer_free(b, &vb);
	b->buffer_free(b, &qb);
	b->buffer_free(b, &ob);
	b->kv_free(b, &kc, &vc);
}

static void test_op_attention_swa_slide(backend *b, int n_heads, int n_kv_heads, int head_dim,
										int n_ctx, int n_steps, int sliding_window, int flash,
										int n_prefill, int compact, kv_quant_type kq,
										const float *ref_all, float **y_out, int record) {
	char label[224];
	snprintf(label, sizeof(label),
			 "attention_swa slide h=%d/%d d=%d ctx=%d steps=%d pre=%d win=%d flash=%d "
			 "cap=%s kv=%s [%s]",
			 n_heads, n_kv_heads, head_dim, n_ctx, n_steps, n_prefill, sliding_window, flash,
			 compact ? "compact" : "full", kq == KV_QUANT_Q8_0 ? "q8_0" : "f16", b->name);
	if (!b->attention_swa || !b->attention_swa_batch) {
		if (record)
			record_result(OPFAM_ATTENTION_SWA, label, V_SKIP,
						  "backend has no native attention_swa/attention_swa_batch");
		return;
	}
	if (n_steps > n_ctx)
		n_steps = n_ctx;
	if (n_prefill > n_steps)
		n_prefill = n_steps;
	int	  chunk = n_prefill < 16 ? (n_prefill > 0 ? n_prefill : 1) : 16;
	int	  n		= n_heads * head_dim;
	int	  n_kv	= n_kv_heads * head_dim;
	float scale = 1.0f / sqrtf((float)head_dim);
	seed_test_rng(0x51DEULL + ((uint64_t)n_heads * 89) + ((uint64_t)n_kv_heads * 11) +
				  ((uint64_t)head_dim * 5) + ((uint64_t)n_ctx * 3) + ((uint64_t)n_steps * 17) +
				  ((uint64_t)n_prefill * 19) + ((uint64_t)sliding_window * 7) +
				  ((uint64_t)flash * 23));

	float *kf = xmalloc((size_t)n_steps * (size_t)n_kv * sizeof(float));
	float *vf = xmalloc((size_t)n_steps * (size_t)n_kv * sizeof(float));
	float *qf = xmalloc((size_t)n_steps * (size_t)n * sizeof(float));
	for (int t = 0; t < n_steps; t++) {
		fill_random_f32(kf + (size_t)t * n_kv, n_kv, 1.0f);
		fill_random_f32(vf + (size_t)t * n_kv, n_kv, 1.0f);
		fill_random_f32(qf + (size_t)t * n, n, 1.0f);
	}

	int pos_cap = n_ctx;
	if (compact) {
		pos_cap = sliding_window + (sliding_window / 4 < 8 ? 8 : sliding_window / 4);
		if (pos_cap > n_ctx)
			pos_cap = n_ctx;
	}
	int		layer_pos_cap[1] = {pos_cap};
	kv_desc kvd				 = {.n_ctx		   = n_ctx,
								.n_kv_heads	   = n_kv_heads,
								.head_dim	   = head_dim,
								.n_layers	   = 1,
								.n_kv_layers   = 1,
								.kv_quant	   = kq,
								.layer_pos_cap = compact ? layer_pos_cap : NULL};
	buffer	kc = {0}, vc = {0}, ki = {0}, vi = {0}, qb = {0}, ob = {0};
	if (b->kv_alloc(b, &kvd, &kc, &vc) != OK ||
		b->buffer_alloc_scratch(b, (size_t)n_kv * sizeof(float), &ki) != OK ||
		b->buffer_alloc_scratch(b, (size_t)n_kv * sizeof(float), &vi) != OK ||
		b->buffer_alloc_scratch(b, (size_t)n * (size_t)chunk * sizeof(float), &qb) != OK ||
		b->buffer_alloc_scratch(b, (size_t)n * (size_t)chunk * sizeof(float), &ob) != OK) {
		if (record)
			record_result(OPFAM_ATTENTION_SWA, label, V_SKIP, "kv_alloc/scratch failed");
		goto done_alloc;
	}

	float	   *y_all	   = xmalloc((size_t)n_steps * (size_t)n * sizeof(float));
	float	   *y_ref	   = xmalloc((size_t)n * sizeof(float));
	float	   *y_worst	   = xmalloc((size_t)n * sizeof(float));
	int			worst_step = -1;
	int		   *base_at	   = xmalloc((size_t)n_steps * sizeof(int));
	float		worst_abs  = 0.0f;
	status_code s_worst	   = OK;
	for (int start = 0; start < n_prefill; start += chunk) {
		int rows = n_prefill - start;
		if (rows > chunk)
			rows = chunk;
		for (int r = 0; r < rows; r++) {
			int p = start + r;
			b->buffer_write_f32(b, &ki, kf + (size_t)p * n_kv, n_kv);
			b->buffer_write_f32(b, &vi, vf + (size_t)p * n_kv, n_kv);
			b->kv_put(b, &kc, &vc, 0, p, &ki, &vi, n_kv_heads, head_dim, n_ctx, n_kv_heads);
		}
		b->buffer_write_f32(b, &qb, qf + (size_t)start * n, (size_t)n * (size_t)rows);
		s_worst =
			b->attention_swa_batch(b, &qb, &kc, &vc, &ob, 0, start, n_heads, n_kv_heads, head_dim,
								   n_ctx, flash, scale, sliding_window, n_kv_heads, rows);
		if (b->synchronize)
			b->synchronize(b);
		b->buffer_read_f32(b, &ob, y_all + (size_t)start * n, (size_t)n * (size_t)rows);
		int cb = start + rows - pos_cap;
		if (cb < 0)
			cb = 0;
		for (int r = 0; r < rows; r++)
			base_at[start + r] = cb;
	}

	for (int p = n_prefill; p < n_steps; p++) {
		b->buffer_write_f32(b, &ki, kf + (size_t)p * n_kv, n_kv);
		b->buffer_write_f32(b, &vi, vf + (size_t)p * n_kv, n_kv);
		b->kv_put(b, &kc, &vc, 0, p, &ki, &vi, n_kv_heads, head_dim, n_ctx, n_kv_heads);
		b->buffer_write_f32(b, &qb, qf + (size_t)p * n, n);
		if (b->synchronize)
			b->synchronize(b);
		status_code st =
			b->attention_swa(b, &qb, &kc, &vc, &ob, 0, p, n_heads, n_kv_heads, head_dim, n_ctx,
							 flash, scale, sliding_window, n_kv_heads);
		if (b->synchronize)
			b->synchronize(b);
		b->buffer_read_f32(b, &ob, y_all + (size_t)p * n, n);
		base_at[p] = (p + 1 - pos_cap) > 0 ? (p + 1 - pos_cap) : 0;
		if (st != OK)
			s_worst = st;
	}

	for (int p = 0; p < n_steps; p++) {
		int t0 = (p + 1 > sliding_window) ? (p + 1 - sliding_window) : 0;
		if (t0 < base_at[p])
			t0 = base_at[p];
		float *r = xmalloc((size_t)n * sizeof(float));
		if (ref_all)
			memcpy(r, ref_all + (size_t)p * n, (size_t)n * sizeof(float));
		else
			memcpy(r,
				   ref_softmax_attn(qf + (size_t)p * n, kf, vf, n_heads, n_kv_heads, head_dim, t0,
									p + 1, scale),
				   (size_t)n * sizeof(float));
		float ma = 0.0f;
		for (int i = 0; i < n; i++) {
			float e = fabsf(r[i] - y_all[(size_t)p * n + i]);
			if (e > ma)
				ma = e;
		}
		if (ma > worst_abs) {
			worst_abs  = ma;
			worst_step = p;
			memcpy(y_ref, r, (size_t)n * sizeof(float));
			memcpy(y_worst, y_all + (size_t)p * n, (size_t)n * sizeof(float));
		}
		free(r);
	}

	if (worst_step < 0) {
		if (record)
			record_result(OPFAM_ATTENTION_SWA, label, s_worst == OK ? V_PASS : V_FAIL,
						  "bit-identical to the uncompacted run across all steps");
	} else {
		char	detail[320];
		verdict v  = classify_output(ref_all ? "kv_quant" : "loose", y_ref, y_worst, n, s_worst,
									 detail, sizeof(detail));
		int		dl = (int)strlen(detail);
		snprintf(detail + dl, sizeof(detail) - dl, " |prefill %d +%d dec, worst pos=%d", n_prefill,
				 n_steps - n_prefill, worst_step);
		if (v == V_FAIL) {
			dl = (int)strlen(detail);
			for (int i = 0; i < n && dl + 40 < (int)sizeof(detail); i++) {
				if (isfinite(y_worst[i]))
					continue;
				dl += snprintf(detail + dl, sizeof(detail) - dl, " h%d/d%d=%g", i / head_dim,
							   i % head_dim, y_worst[i]);
			}
		}
		if (record)
			record_result(OPFAM_ATTENTION_SWA, label, v, detail);
	}

	if (y_out)
		*y_out = y_all;
	else
		free(y_all);
	free(y_ref);
	free(y_worst);
	free(base_at);
	b->buffer_free(b, &ki);
	b->buffer_free(b, &vi);
	b->buffer_free(b, &qb);
	b->buffer_free(b, &ob);
	b->buffer_free(b, &kc);
	b->buffer_free(b, &vc);
done_alloc:
	free(qf);
	free(kf);
	free(vf);
}

static void test_op_attention_swa_compact_parity(backend *b, kv_quant_type kq, int n_heads,
												 int n_kv_heads, int head_dim, int n_ctx,
												 int n_steps, int sliding_window, int flash,
												 int n_prefill) {
	char label[192];
	snprintf(label, sizeof(label),
			 "attention_swa compaction parity vs full kv=%s h=%d/%d d=%d win=%d [%s]",
			 kq == KV_QUANT_Q8_0 ? "q8_0" : "f16", n_heads, n_kv_heads, head_dim, sliding_window,
			 b->name);
	if (!b->attention_swa || !b->attention_swa_batch || !b->kv_alloc) {
		record_result(OPFAM_ATTENTION_SWA, label, V_SKIP, "backend lacks swa ops or kv_alloc");
		return;
	}
	if (kq == KV_QUANT_Q8_0 && !backend_has_cap(b, BCAP_KV_QUANT_Q8_0)) {
		record_result(OPFAM_ATTENTION_SWA, label, V_SKIP,
					  "backend does not advertise BCAP_KV_QUANT_Q8_0");
		return;
	}
	if (!backend_has_cap(b, BCAP_KV_POS_CAP)) {
		record_result(OPFAM_ATTENTION_SWA, label, V_SKIP,
					  "backend does not advertise BCAP_KV_POS_CAP");
		return;
	}
	float *y_full = NULL;
	test_op_attention_swa_slide(b, n_heads, n_kv_heads, head_dim, n_ctx, n_steps, sliding_window,
								flash, n_prefill, 0, kq, NULL, &y_full, 0);
	if (!y_full)
		return;
	test_op_attention_swa_slide(b, n_heads, n_kv_heads, head_dim, n_ctx, n_steps, sliding_window,
								flash, n_prefill, 1, kq, y_full, NULL, 1);
	free(y_full);
}

static void test_op_attention_mla(backend *ref, backend *tgt, int n_heads, int qk_head, int qk_rope,
								  int qk_nope, int v_head, int kv_lora, int n_ctx, int pos) {
	if (!tgt->attention_mla || !tgt->kv_alloc_mla || !tgt->kv_put_mla) {
		char label[128];
		snprintf(label, sizeof(label), "attention_mla h=%d qk=%d/%d/%d v=%d lora=%d pos=%d (%s)",
				 n_heads, qk_head, qk_nope, qk_rope, v_head, kv_lora, pos, tgt->name);
		record_result(OPFAM_ATTENTION_MLA, label, V_SKIP, "backend has no native MLA ops");
		return;
	}

	const int	total_dim = kv_lora + qk_rope;
	const int	half_rope = qk_rope / 2;
	const int	n_t		  = pos + 1;
	const int	n_out	  = n_heads * v_head;
	const float scale	  = 1.0f / sqrtf((float)qk_head);

	seed_test_rng(0x5EED0000ULL + ((uint64_t)n_heads * 131) + ((uint64_t)qk_head * 17) +
				  ((uint64_t)kv_lora * 7) + (uint64_t)pos);

	float *q = xmalloc((size_t)n_heads * (size_t)qk_head * sizeof(float));
	fill_random_f32(q, n_heads * qk_head, 1.0f);
	float **kv_a_all = xmalloc((size_t)n_t * sizeof(float *));
	for (int t = 0; t < n_t; t++) {
		kv_a_all[t] = xmalloc((size_t)total_dim * sizeof(float));
		fill_random_f32(kv_a_all[t], total_dim, 1.0f);
	}
	float *norm_w	= xmalloc((size_t)kv_lora * sizeof(float));
	float *k_b		= xmalloc((size_t)n_heads * (size_t)qk_nope * (size_t)kv_lora * sizeof(float));
	float *v_b		= xmalloc((size_t)n_heads * (size_t)kv_lora * (size_t)v_head * sizeof(float));
	float *rope_cos = xmalloc((size_t)n_t * (size_t)half_rope * sizeof(float));
	float *rope_sin = xmalloc((size_t)n_t * (size_t)half_rope * sizeof(float));
	fill_random_f32(norm_w, kv_lora, 1.0f);
	fill_random_f32(k_b, n_heads * qk_nope * kv_lora, 0.5f);
	fill_random_f32(v_b, n_heads * kv_lora * v_head, 0.5f);
	fill_random_f32(rope_cos, n_t * half_rope, 1.0f);
	fill_random_f32(rope_sin, n_t * half_rope, 1.0f);

	float *y_ref = xmalloc((size_t)n_out * sizeof(float));
	float *y_got = xmalloc((size_t)n_out * sizeof(float));

	typedef struct {
		backend *b;
		buffer	 kc, q_b, kb_b, vb_b, kva_b, nw_b, out_b;
	} side;
	side ref_s = {.b = ref};
	side tgt_s = {.b = tgt};

	for (int which = 0; which < 2; which++) {
		side	*s = which ? &tgt_s : &ref_s;
		backend *b = s->b;
		b->kv_alloc_mla(b, 1, n_ctx, kv_lora, qk_rope, &s->kc);
		b->buffer_alloc_scratch(b, (size_t)total_dim * sizeof(float), &s->kva_b);
		b->buffer_alloc_scratch(b, (size_t)kv_lora * sizeof(float), &s->nw_b);
		b->buffer_alloc_scratch(
			b, (size_t)n_heads * (size_t)qk_nope * (size_t)kv_lora * sizeof(float), &s->kb_b);
		b->buffer_alloc_scratch(
			b, (size_t)n_heads * (size_t)kv_lora * (size_t)v_head * sizeof(float), &s->vb_b);
		b->buffer_alloc_scratch(b, (size_t)n_heads * (size_t)qk_head * sizeof(float), &s->q_b);
		b->buffer_alloc_scratch(b, (size_t)n_out * sizeof(float), &s->out_b);
		b->buffer_write_f32(b, &s->nw_b, norm_w, kv_lora);
		b->buffer_write_f32(b, &s->kb_b, k_b, n_heads * qk_nope * kv_lora);
		b->buffer_write_f32(b, &s->vb_b, v_b, n_heads * kv_lora * v_head);
		b->buffer_write_f32(b, &s->q_b, q, n_heads * qk_head);
		for (int t = 0; t < n_t; t++) {
			b->buffer_write_f32(b, &s->kva_b, kv_a_all[t], total_dim);
			b->kv_put_mla(b, &s->kc, 0, t, &s->kva_b, &s->nw_b, kv_lora, qk_rope, n_ctx, 1e-5f);
		}
		b->attention_mla(b, &s->q_b, &s->kc, &s->kb_b, &s->vb_b, &s->out_b, 0, pos, n_heads,
						 qk_head, qk_rope, qk_nope, v_head, kv_lora, n_ctx, rope_cos, rope_sin,
						 scale);
		if (b->synchronize)
			b->synchronize(b);
		b->buffer_read_f32(b, &s->out_b, which ? y_got : y_ref, n_out);
	}

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "attention_mla h=%d qk=%d/%d/%d v=%d lora=%d pos=%d", n_heads,
			 qk_head, qk_nope, qk_rope, v_head, kv_lora, pos);
	verdict v = classify_output("loose", y_ref, y_got, n_out, OK, detail, sizeof(detail));
	record_result(OPFAM_ATTENTION_MLA, label, v, detail);

	for (int t = 0; t < n_t; t++)
		free(kv_a_all[t]);
	free(kv_a_all);
	free(q);
	free(norm_w);
	free(k_b);
	free(v_b);
	free(rope_cos);
	free(rope_sin);
	free(y_ref);
	free(y_got);
	{
		backend *b = ref;
		b->buffer_free(b, &ref_s.kc);
		b->buffer_free(b, &ref_s.q_b);
		b->buffer_free(b, &ref_s.kb_b);
		b->buffer_free(b, &ref_s.vb_b);
		b->buffer_free(b, &ref_s.kva_b);
		b->buffer_free(b, &ref_s.nw_b);
		b->buffer_free(b, &ref_s.out_b);
	}
	{
		backend *b = tgt;
		b->buffer_free(b, &tgt_s.kc);
		b->buffer_free(b, &tgt_s.q_b);
		b->buffer_free(b, &tgt_s.kb_b);
		b->buffer_free(b, &tgt_s.vb_b);
		b->buffer_free(b, &tgt_s.kva_b);
		b->buffer_free(b, &tgt_s.nw_b);
		b->buffer_free(b, &tgt_s.out_b);
	}
}

static void test_op_kv_put(backend *ref, backend *tgt, int n_kv_heads, int head_dim, int n_ctx,
						   int pos) {
	if (!tgt->kv_put) {
		char label[112];
		snprintf(label, sizeof(label), "kv_put h=%d d=%d pos=%d", n_kv_heads, head_dim, pos);
		record_result(OPFAM_KV_PUT, label, V_SKIP, "backend has no native kv_put");
		return;
	}

	int		n_kv = n_kv_heads * head_dim;
	kv_desc kvd	 = {.n_ctx		 = n_ctx,
					.n_kv_heads	 = n_kv_heads,
					.head_dim	 = head_dim,
					.n_layers	 = 1,
					.n_kv_layers = 1};

	int is_host = backend_has_cap(tgt, BCAP_IS_HOST);

	if (is_host) {
		float *k = xmalloc((size_t)n_kv * sizeof(float));
		float *v = xmalloc((size_t)n_kv * sizeof(float));
		seed_test_rng(0x4B56ULL + ((uint64_t)n_kv_heads * 7) + ((uint64_t)head_dim * 3) +
					  (uint64_t)pos);
		fill_random_f32(k, n_kv, 1.0f);
		fill_random_f32(v, n_kv, 1.0f);

		buffer kc_ref = {0};
		buffer vc_ref = {0};
		buffer ki_ref = {0};
		buffer vi_ref = {0};
		ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
		ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &ki_ref);
		ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &vi_ref);
		ref->buffer_write_f32(ref, &ki_ref, k, n_kv);
		ref->buffer_write_f32(ref, &vi_ref, v, n_kv);
		ref->kv_put(ref, &kc_ref, &vc_ref, 0, pos, &ki_ref, &vi_ref, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
		if (ref && ref->synchronize)
			ref->synchronize(ref);

		buffer kc_tgt = {0};
		buffer vc_tgt = {0};
		buffer ki_tgt = {0};
		buffer vi_tgt = {0};
		tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &ki_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &vi_tgt);
		tgt->buffer_write_f32(tgt, &ki_tgt, k, n_kv);
		tgt->buffer_write_f32(tgt, &vi_tgt, v, n_kv);
		status_code s_tgt;
		{
			s_tgt = tgt->kv_put(tgt, &kc_tgt, &vc_tgt, 0, pos, &ki_tgt, &vi_tgt, n_kv_heads,
								head_dim, n_ctx, n_kv_heads);
			if (tgt->synchronize)
				tgt->synchronize(tgt);
		}

		uint16_t *k_ref = xmalloc((size_t)n_kv * sizeof(uint16_t));
		uint16_t *v_ref = xmalloc((size_t)n_kv * sizeof(uint16_t));
		uint16_t *k_got = xmalloc((size_t)n_kv * sizeof(uint16_t));
		uint16_t *v_got = xmalloc((size_t)n_kv * sizeof(uint16_t));

		uint16_t *kc_ref_p = kc_ref.handle;
		uint16_t *vc_ref_p = vc_ref.handle;
		uint16_t *kc_tgt_p = kc_tgt.handle;
		uint16_t *vc_tgt_p = vc_tgt.handle;

		size_t kvh_stride = (size_t)n_ctx * head_dim;
		size_t pos_off	  = (size_t)pos * head_dim;
		for (int h = 0; h < n_kv_heads; h++) {
			size_t base = (h * kvh_stride) + pos_off;
			memcpy(k_ref + (h * head_dim), kc_ref_p + base, (size_t)head_dim * sizeof(uint16_t));
			memcpy(v_ref + (h * head_dim), vc_ref_p + base, (size_t)head_dim * sizeof(uint16_t));
			memcpy(k_got + (h * head_dim), kc_tgt_p + base, (size_t)head_dim * sizeof(uint16_t));
			memcpy(v_got + (h * head_dim), vc_tgt_p + base, (size_t)head_dim * sizeof(uint16_t));
		}

		float *k_ref_f = xmalloc((size_t)n_kv * sizeof(float));
		float *k_got_f = xmalloc((size_t)n_kv * sizeof(float));
		float *v_ref_f = xmalloc((size_t)n_kv * sizeof(float));
		float *v_got_f = xmalloc((size_t)n_kv * sizeof(float));
		for (int i = 0; i < n_kv; i++) {
			k_ref_f[i] = test_f16_to_f32(k_ref[i]);
			k_got_f[i] = test_f16_to_f32(k_got[i]);
			v_ref_f[i] = test_f16_to_f32(v_ref[i]);
			v_got_f[i] = test_f16_to_f32(v_got[i]);
		}

		char label[112];
		char detail[256];
		snprintf(label, sizeof(label), "kv_put h=%d d=%d pos=%d K", n_kv_heads, head_dim, pos);
		verdict vk =
			classify_output("exact", k_ref_f, k_got_f, n_kv, s_tgt, detail, sizeof(detail));
		if (vk != V_PASS && vk != V_SKIP)
			compute_debug(k_ref_f, k_got_f, n_kv);
		record_result(OPFAM_KV_PUT, label, vk, detail);

		snprintf(label, sizeof(label), "kv_put h=%d d=%d pos=%d V", n_kv_heads, head_dim, pos);
		verdict vv =
			classify_output("exact", v_ref_f, v_got_f, n_kv, s_tgt, detail, sizeof(detail));
		if (vv != V_PASS && vv != V_SKIP)
			compute_debug(v_ref_f, v_got_f, n_kv);
		record_result(OPFAM_KV_PUT, label, vv, detail);

		free(k);
		free(v);
		free(k_ref);
		free(v_ref);
		free(k_got);
		free(v_got);
		free(k_ref_f);
		free(k_got_f);
		free(v_ref_f);
		free(v_got_f);
		ref->buffer_free(ref, &ki_ref);
		ref->buffer_free(ref, &vi_ref);
		ref->buffer_free(ref, &kc_ref);
		ref->buffer_free(ref, &vc_ref);
		tgt->buffer_free(tgt, &ki_tgt);
		tgt->buffer_free(tgt, &vi_tgt);
		tgt->buffer_free(tgt, &kc_tgt);
		tgt->buffer_free(tgt, &vc_tgt);
		return;
	}

	if (!tgt->attention) {
		char label[112];
		snprintf(label, sizeof(label), "kv_put h=%d d=%d pos=%d (indirect)", n_kv_heads, head_dim,
				 pos);
		record_result(OPFAM_KV_PUT, label, V_SKIP,
					  "non-host backend has no attention for indirect kv_put test");
		return;
	}

	int	  n		= n_kv_heads * head_dim;
	float scale = 1.0f / sqrtf((float)head_dim);
	int	  n_t	= pos + 1;

	seed_test_rng(0x4B56ULL + ((uint64_t)n_kv_heads * 7) + ((uint64_t)head_dim * 3) +
				  (uint64_t)pos);

	float *q = xmalloc((size_t)n * sizeof(float));
	fill_random_f32(q, n, 1.0f);
	float **k_all = xmalloc((size_t)n_t * sizeof(float *));
	float **v_all = xmalloc((size_t)n_t * sizeof(float *));
	for (int t = 0; t < n_t; t++) {
		k_all[t] = xmalloc((size_t)n_kv * sizeof(float));
		v_all[t] = xmalloc((size_t)n_kv * sizeof(float));
		fill_random_f32(k_all[t], n_kv, 1.0f);
		fill_random_f32(v_all[t], n_kv, 1.0f);
	}

	buffer kc_ref  = {0};
	buffer vc_ref  = {0};
	buffer ki_ref  = {0};
	buffer vi_ref  = {0};
	buffer q_ref   = {0};
	buffer out_ref = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &ki_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &vi_ref);
	for (int t = 0; t < n_t; t++) {
		ref->buffer_write_f32(ref, &ki_ref, k_all[t], n_kv);
		ref->buffer_write_f32(ref, &vi_ref, v_all[t], n_kv);
		ref->kv_put(ref, &kc_ref, &vc_ref, 0, t, &ki_ref, &vi_ref, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
	}
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &q_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &out_ref);
	ref->buffer_write_f32(ref, &q_ref, q, n);
	ref->attention(ref, &q_ref, &kc_ref, &vc_ref, &out_ref, 0, pos, n_kv_heads, n_kv_heads,
				   head_dim, n_ctx, 0, scale, n_kv_heads);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &out_ref, y_ref, n);

	buffer kc_tgt  = {0};
	buffer vc_tgt  = {0};
	buffer ki_tgt  = {0};
	buffer vi_tgt  = {0};
	buffer q_tgt   = {0};
	buffer out_tgt = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &ki_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &vi_tgt);
	for (int t = 0; t < n_t; t++) {
		tgt->buffer_write_f32(tgt, &ki_tgt, k_all[t], n_kv);
		tgt->buffer_write_f32(tgt, &vi_tgt, v_all[t], n_kv);
		tgt->kv_put(tgt, &kc_tgt, &vc_tgt, 0, t, &ki_tgt, &vi_tgt, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &q_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &out_tgt);
	tgt->buffer_write_f32(tgt, &q_tgt, q, n);
	status_code s_tgt;
	{
		s_tgt = tgt->attention(tgt, &q_tgt, &kc_tgt, &vc_tgt, &out_tgt, 0, pos, n_kv_heads,
							   n_kv_heads, head_dim, n_ctx, 0, scale, n_kv_heads);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &out_tgt, y_got, n);

	char label[112];
	char detail[256];
	snprintf(label, sizeof(label), "kv_put+attn h=%d d=%d pos=%d (indirect)", n_kv_heads, head_dim,
			 pos);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	int dl = (int)strlen(detail);
	snprintf(detail + dl, sizeof(detail) - dl,
			 " | indirect: validates kv_put via attention output");
	record_result(OPFAM_KV_PUT, label, v, detail);

	for (int t = 0; t < n_t; t++) {
		free(k_all[t]);
		free(v_all[t]);
	}
	free(k_all);
	free(v_all);
	free(q);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &ki_ref);
	ref->buffer_free(ref, &vi_ref);
	ref->buffer_free(ref, &q_ref);
	ref->buffer_free(ref, &out_ref);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	tgt->buffer_free(tgt, &ki_tgt);
	tgt->buffer_free(tgt, &vi_tgt);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &out_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_kv_put_batch(backend *ref, backend *tgt, int n_kv_heads, int head_dim,
								 int n_ctx, int pos_start, int m) {
	char label[112];
	snprintf(label, sizeof(label), "kv_put_batch h=%d d=%d pos=%d m=%d", n_kv_heads, head_dim,
			 pos_start, m);
	if (!tgt->kv_put_batch) {
		record_result(OPFAM_KV_PUT, label, V_SKIP, "backend has no native kv_put_batch");
		return;
	}

	int		is_host = backend_has_cap(tgt, BCAP_IS_HOST);
	int		n_kv	= n_kv_heads * head_dim;
	size_t	total	= (size_t)m * (size_t)n_kv;
	kv_desc kvd		= {.n_ctx		= n_ctx,
					   .n_kv_heads	= n_kv_heads,
					   .head_dim	= head_dim,
					   .n_layers	= 1,
					   .n_kv_layers = 1};

	float *k_all = xmalloc(total * sizeof(float));
	float *v_all = xmalloc(total * sizeof(float));
	seed_test_rng(0x4B57ULL + ((uint64_t)n_kv_heads * 11) + ((uint64_t)head_dim * 5) +
				  (uint64_t)pos_start + (uint64_t)m);
	fill_random_f32(k_all, (int)total, 1.0f);
	fill_random_f32(v_all, (int)total, 1.0f);

	buffer kc_ref = {0}, vc_ref = {0}, ki_ref = {0}, vi_ref = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &ki_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &vi_ref);
	for (int r = 0; r < m; r++) {
		ref->buffer_write_f32(ref, &ki_ref, k_all + (size_t)r * n_kv, n_kv);
		ref->buffer_write_f32(ref, &vi_ref, v_all + (size_t)r * n_kv, n_kv);
		ref->kv_put(ref, &kc_ref, &vc_ref, 0, pos_start + r, &ki_ref, &vi_ref, n_kv_heads, head_dim,
					n_ctx, n_kv_heads);
	}
	if (ref->synchronize)
		ref->synchronize(ref);

	buffer kc_tgt = {0}, vc_tgt = {0}, ki_tgt = {0}, vi_tgt = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, total * sizeof(float), &ki_tgt);
	tgt->buffer_alloc_scratch(tgt, total * sizeof(float), &vi_tgt);
	tgt->buffer_write_f32(tgt, &ki_tgt, k_all, (int)total);
	tgt->buffer_write_f32(tgt, &vi_tgt, v_all, (int)total);
	status_code s_tgt = tgt->kv_put_batch(tgt, &kc_tgt, &vc_tgt, 0, pos_start, &ki_tgt, &vi_tgt,
										  n_kv, n_kv_heads, head_dim, n_ctx, n_kv_heads, m);
	if (tgt->synchronize)
		tgt->synchronize(tgt);

	verdict v;
	char	detail[256];
	if (is_host || !tgt->attention) {
		if (!is_host) {
			v = V_SKIP;
			snprintf(detail, sizeof(detail),
					 "non-host backend has no attention for indirect check");
			goto record;
		}
		uint16_t *kc_ref_p	 = kc_ref.handle;
		uint16_t *vc_ref_p	 = vc_ref.handle;
		uint16_t *kc_tgt_p	 = kc_tgt.handle;
		uint16_t *vc_tgt_p	 = vc_tgt.handle;
		size_t	  kvh_stride = (size_t)n_ctx * head_dim;
		int		  fail		 = 0;
		for (int r = 0; r < m && !fail; r++) {
			for (int h = 0; h < n_kv_heads && !fail; h++) {
				size_t base = ((size_t)h * kvh_stride) + (size_t)(pos_start + r) * head_dim;
				for (int d = 0; d < head_dim; d++) {
					float kr = test_f16_to_f32(kc_ref_p[base + d]);
					float kt = test_f16_to_f32(kc_tgt_p[base + d]);
					float vr = test_f16_to_f32(vc_ref_p[base + d]);
					float vt = test_f16_to_f32(vc_tgt_p[base + d]);
					if (kr != kt || vr != vt) {
						fail = 1;
						break;
					}
				}
			}
		}
		v = (s_tgt == OK && !fail) ? V_PASS : V_FAIL;
		snprintf(detail, sizeof(detail), "%s",
				 (s_tgt == OK && !fail) ? "batched write matches per-row writes"
										: "batched KV write mismatch");
	} else {
		int	  n_att = n_kv_heads * head_dim;
		float scale = 1.0f / sqrtf((float)head_dim);
		int	  last	= pos_start + m - 1;

		float *q = xmalloc((size_t)n_att * sizeof(float));
		fill_random_f32(q, n_att, 1.0f);

		buffer kc_ref = {0}, vc_ref = {0};
		tgt->kv_alloc(tgt, &kvd, &kc_ref, &vc_ref);
		buffer row_k = {0}, row_v = {0};
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &row_k);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &row_v);
		for (int r = 0; r < m; r++) {
			tgt->buffer_write_f32(tgt, &row_k, k_all + (size_t)r * n_kv, n_kv);
			tgt->buffer_write_f32(tgt, &row_v, v_all + (size_t)r * n_kv, n_kv);
			status_code prs = tgt->kv_put(tgt, &kc_ref, &vc_ref, 0, pos_start + r, &row_k, &row_v,
										  n_kv_heads, head_dim, n_ctx, n_kv_heads);
			if (prs != OK) {
				v = V_FAIL;
				snprintf(detail, sizeof(detail), "reference kv_put failed at row %d: %d", r, prs);
				goto record;
			}
		}
		if (tgt->synchronize)
			tgt->synchronize(tgt);
		tgt->buffer_free(tgt, &row_k);
		tgt->buffer_free(tgt, &row_v);

		buffer q_tgt = {0}, out_a = {0}, out_b = {0};
		tgt->buffer_alloc_scratch(tgt, (size_t)n_att * sizeof(float), &q_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_att * sizeof(float), &out_a);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_att * sizeof(float), &out_b);
		tgt->buffer_write_f32(tgt, &q_tgt, q, n_att);
		status_code st_batch =
			tgt->attention(tgt, &q_tgt, &kc_tgt, &vc_tgt, &out_a, 0, last, n_kv_heads, n_kv_heads,
						   head_dim, n_ctx, 0, scale, n_kv_heads);
		status_code st_rows =
			tgt->attention(tgt, &q_tgt, &kc_ref, &vc_ref, &out_b, 0, last, n_kv_heads, n_kv_heads,
						   head_dim, n_ctx, 0, scale, n_kv_heads);
		if (tgt->synchronize)
			tgt->synchronize(tgt);

		float *y_batch = xmalloc((size_t)n_att * sizeof(float));
		float *y_rows  = xmalloc((size_t)n_att * sizeof(float));
		tgt->buffer_read_f32(tgt, &out_a, y_batch, n_att);
		tgt->buffer_read_f32(tgt, &out_b, y_rows, n_att);

		v = classify_output("exact", y_rows, y_batch, n_att, (st_batch == OK ? st_rows : st_batch),
							detail, sizeof(detail));
		int dl = (int)strlen(detail);
		snprintf(detail + dl, sizeof(detail) - dl,
				 " | indirect: batch-written cache vs per-row cache, same backend");
		free(q);
		free(y_batch);
		free(y_rows);
		tgt->buffer_free(tgt, &q_tgt);
		tgt->buffer_free(tgt, &out_a);
		tgt->buffer_free(tgt, &out_b);
		tgt->buffer_free(tgt, &kc_ref);
		tgt->buffer_free(tgt, &vc_ref);
	}

record:
	record_result(OPFAM_KV_PUT, label, v, detail);

	free(k_all);
	free(v_all);
	ref->buffer_free(ref, &ki_ref);
	ref->buffer_free(ref, &vi_ref);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	tgt->buffer_free(tgt, &ki_tgt);
	tgt->buffer_free(tgt, &vi_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_kv_quant_parity(backend *b, int n_kv_heads, int head_dim, int n_ctx, int pos) {
	char label[128];
	snprintf(label, sizeof(label), "kv_quant q8_0 vs f16 h=%d d=%d pos=%d [%s]", n_kv_heads,
			 head_dim, pos, b->name);

	if (!b->kv_put || !b->attention || !b->kv_alloc) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP,
					  "backend has no native kv_put/attention");
		return;
	}
	if (!backend_has_cap(b, BCAP_KV_QUANT_Q8_0)) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP,
					  "backend does not advertise BCAP_KV_QUANT_Q8_0");
		return;
	}

	int	  n_kv	= n_kv_heads * head_dim;
	int	  n		= n_kv_heads * head_dim;
	float scale = 1.0f / sqrtf((float)head_dim);
	int	  n_t	= pos + 1;

	seed_test_rng(0x4B56ULL + 0x8000ULL + ((uint64_t)n_kv_heads * 7) + ((uint64_t)head_dim * 3) +
				  (uint64_t)pos);

	float *q = xmalloc((size_t)n * sizeof(float));
	fill_random_f32(q, n, 1.0f);
	float **k_all = xmalloc((size_t)n_t * sizeof(float *));
	float **v_all = xmalloc((size_t)n_t * sizeof(float *));
	for (int t = 0; t < n_t; t++) {
		k_all[t] = xmalloc((size_t)n_kv * sizeof(float));
		v_all[t] = xmalloc((size_t)n_kv * sizeof(float));
		fill_random_f32(k_all[t], n_kv, 1.0f);
		fill_random_f32(v_all[t], n_kv, 1.0f);
	}

	kv_desc kvd_ref = {.n_ctx		= n_ctx,
					   .n_kv_heads	= n_kv_heads,
					   .head_dim	= head_dim,
					   .n_layers	= 1,
					   .n_kv_layers = 1,
					   .kv_quant	= KV_QUANT_F16};

	buffer kc_ref  = {0};
	buffer vc_ref  = {0};
	buffer ki_ref  = {0};
	buffer vi_ref  = {0};
	buffer q_ref_b = {0};
	buffer out_ref = {0};
	b->kv_alloc(b, &kvd_ref, &kc_ref, &vc_ref);
	b->buffer_alloc_scratch(b, (size_t)n_kv * sizeof(float), &ki_ref);
	b->buffer_alloc_scratch(b, (size_t)n_kv * sizeof(float), &vi_ref);
	for (int t = 0; t < n_t; t++) {
		b->buffer_write_f32(b, &ki_ref, k_all[t], n_kv);
		b->buffer_write_f32(b, &vi_ref, v_all[t], n_kv);
		b->kv_put(b, &kc_ref, &vc_ref, 0, t, &ki_ref, &vi_ref, n_kv_heads, head_dim, n_ctx,
				  n_kv_heads);
	}
	b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &q_ref_b);
	b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &out_ref);
	b->buffer_write_f32(b, &q_ref_b, q, n);
	float *y_f16[2];
	for (int fl = 0; fl < 2; fl++) {
		b->attention(b, &q_ref_b, &kc_ref, &vc_ref, &out_ref, 0, pos, n_kv_heads, n_kv_heads,
					 head_dim, n_ctx, fl, scale, n_kv_heads);
		if (b->synchronize)
			b->synchronize(b);
		y_f16[fl] = xmalloc((size_t)n * sizeof(float));
		b->buffer_read_f32(b, &out_ref, y_f16[fl], n);
	}

	kv_desc kvd_q8 = {.n_ctx	   = n_ctx,
					  .n_kv_heads  = n_kv_heads,
					  .head_dim	   = head_dim,
					  .n_layers	   = 1,
					  .n_kv_layers = 1,
					  .kv_quant	   = KV_QUANT_Q8_0};

	buffer		kc_q8	= {0};
	buffer		vc_q8	= {0};
	buffer		ki_q8	= {0};
	buffer		vi_q8	= {0};
	buffer		q_q8_b	= {0};
	buffer		out_q8	= {0};
	status_code s_alloc = b->kv_alloc(b, &kvd_q8, &kc_q8, &vc_q8);
	if (s_alloc != OK) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP,
					  "kv_alloc(q8_0) failed despite cap flag");
		goto cleanup_ref;
	}
	b->buffer_alloc_scratch(b, (size_t)n_kv * sizeof(float), &ki_q8);
	b->buffer_alloc_scratch(b, (size_t)n_kv * sizeof(float), &vi_q8);
	for (int t = 0; t < n_t; t++) {
		b->buffer_write_f32(b, &ki_q8, k_all[t], n_kv);
		b->buffer_write_f32(b, &vi_q8, v_all[t], n_kv);
		b->kv_put(b, &kc_q8, &vc_q8, 0, t, &ki_q8, &vi_q8, n_kv_heads, head_dim, n_ctx, n_kv_heads);
		if (b->synchronize)
			b->synchronize(b);
	}
	b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &q_q8_b);
	b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &out_q8);
	b->buffer_write_f32(b, &q_q8_b, q, n);
	float *k_flat = xmalloc((size_t)n_t * (size_t)n_kv * sizeof(float));
	float *v_flat = xmalloc((size_t)n_t * (size_t)n_kv * sizeof(float));
	for (int t = 0; t < n_t; t++) {
		memcpy(k_flat + (size_t)t * n_kv, k_all[t], (size_t)n_kv * sizeof(float));
		memcpy(v_flat + (size_t)t * n_kv, v_all[t], (size_t)n_kv * sizeof(float));
	}
	float *y_ref =
		ref_softmax_attn(q, k_flat, v_flat, n_kv_heads, n_kv_heads, head_dim, 0, n_t, scale);
	free(k_flat);
	free(v_flat);

	for (int fl = 0; fl < 2; fl++) {
		status_code s_got = b->attention(b, &q_q8_b, &kc_q8, &vc_q8, &out_q8, 0, pos, n_kv_heads,
										 n_kv_heads, head_dim, n_ctx, fl, scale, n_kv_heads);
		if (b->synchronize)
			b->synchronize(b);
		float *y_q8 = xmalloc((size_t)n * sizeof(float));
		b->buffer_read_f32(b, &out_q8, y_q8, n);

		const char *qn[2]	= {"f16", "q8_0"};
		float	   *ys[2]	= {y_f16[fl], y_q8};
		status_code s_cs[2] = {OK, s_got};
		for (int ci = 0; ci < 2; ci++) {
			char flabel[160];
			snprintf(flabel, sizeof(flabel), "%s %s flash=%d", label, qn[ci], fl);
			char	detail[256];
			verdict v =
				classify_output("kv_quant", y_ref, ys[ci], n, s_cs[ci], detail, sizeof(detail));
			if (v != V_PASS && v != V_SKIP)
				compute_debug(y_ref, ys[ci], n);
			int dl = (int)strlen(detail);
			snprintf(detail + dl, sizeof(detail) - dl,
					 " | kv cache quant parity: %s attention output vs f32 reference", qn[ci]);
			record_result(OPFAM_KV_QUANT_PARITY, flabel, v, detail);
		}
		free(y_q8);
	}
	for (int fl = 0; fl < 2; fl++)
		free(y_f16[fl]);
	free(y_ref);
	b->buffer_free(b, &ki_q8);
	b->buffer_free(b, &vi_q8);
	b->buffer_free(b, &q_q8_b);
	b->buffer_free(b, &out_q8);
	if (b->kv_free) {
		b->kv_free(b, &kc_q8, &vc_q8);
	} else {
		if (kc_q8.owner)
			kc_q8.owner->buffer_free(kc_q8.owner, &kc_q8);
		if (vc_q8.owner)
			vc_q8.owner->buffer_free(vc_q8.owner, &vc_q8);
	}

cleanup_ref:
	for (int t = 0; t < n_t; t++) {
		free(k_all[t]);
		free(v_all[t]);
	}
	free(k_all);
	free(v_all);
	free(q);
	b->buffer_free(b, &ki_ref);
	b->buffer_free(b, &vi_ref);
	b->buffer_free(b, &q_ref_b);
	b->buffer_free(b, &out_ref);
	if (b->kv_free) {
		b->kv_free(b, &kc_ref, &vc_ref);
	} else {
		if (kc_ref.owner)
			kc_ref.owner->buffer_free(kc_ref.owner, &kc_ref);
		if (vc_ref.owner)
			vc_ref.owner->buffer_free(vc_ref.owner, &vc_ref);
	}
}

static float *ref_softmax_attn(const float *q, const float *k, const float *v, int n_heads, int kvh,
							   int hd, int t0, int n_pos, float scale) {
	float *ref = xmalloc((size_t)n_heads * hd * sizeof(float));
	memset(ref, 0, (size_t)n_heads * hd * sizeof(float));
	int			 n_groups = n_heads / kvh;
	static float sc[4096];
	for (int h = 0; h < n_heads; h++) {
		int			 hh	  = h / n_groups;
		const float *qh	  = q + (size_t)h * hd;
		float		 maxs = -INFINITY;
		for (int t = t0; t < n_pos; t++) {
			float		 s	= 0;
			const float *kt = k + (size_t)t * kvh * hd + (size_t)hh * hd;
			for (int d = 0; d < hd; d++)
				s += qh[d] * kt[d];
			sc[t] = scale * s;
			if (sc[t] > maxs)
				maxs = sc[t];
		}
		float sum = 0;
		for (int t = t0; t < n_pos; t++) {
			sc[t] = expf(sc[t] - maxs);
			sum += sc[t];
		}
		float *out = ref + (size_t)h * hd;
		for (int t = t0; t < n_pos; t++) {
			float		 w	= sc[t] / sum;
			const float *vt = v + (size_t)t * kvh * hd + (size_t)hh * hd;
			for (int d = 0; d < hd; d++)
				out[d] += w * vt[d];
		}
	}
	return ref;
}

static void test_model_kv_size_shared(void) {
	const int n_layers = 12, n_kv_layers = 10, n_ctx = 512, win = 64;
	const int hd_swa = 64, hd_global = 128, kvh_swa = 4, kvh_global = 2, period = 6;
	char	  label[160];
	snprintf(label, sizeof(label), "kv size accounting shared+varlayers L=%d kvL=%d ctx=%d win=%d",
			 n_layers, n_kv_layers, n_ctx, win);

	arch_info ai			   = {0};
	ai.has_variable_layer_dims = true;
	ai.sliding_window_period   = period;

	model m;
	memset(&m, 0, sizeof(m));
	m.arch_info						   = &ai;
	m.n_layers						   = n_layers;
	m.sliding_window				   = win;
	m.layer_dims.n_layer_kv_from_start = n_kv_layers;
	m.layers						   = xcalloc((size_t)n_layers, sizeof(*m.layers));
	m.layer_dims.is_global_layer	   = xcalloc((size_t)n_layers, sizeof(uint8_t));
	for (int li = 0; li < n_layers; li++) {
		int glob						 = (li % period) == (period - 1);
		m.layers[li].is_global_layer	 = (uint8_t)glob;
		m.layer_dims.is_global_layer[li] = (uint8_t)glob;
		m.layers[li].is_sliding			 = !glob;
		m.layers[li].head_dim			 = glob ? hd_global : hd_swa;
		m.layers[li].n_kv_heads			 = glob ? kvh_global : kvh_swa;
	}

	int	   expect_cap_swa = win + win / 4;
	int	   n_swa_blocks = 0, n_glob_blocks = 0, cap_bad = 0;
	size_t expect = 0;
	for (int kvl = 0; kvl < n_kv_layers; kvl++) {
		int pcap = model_kv_layer_pos_cap(&m, n_ctx, kvl);
		int want = m.layers[kvl].is_sliding ? expect_cap_swa : n_ctx;
		if (pcap != want)
			cap_bad++;
		if (m.layers[kvl].is_sliding)
			n_swa_blocks++;
		else
			n_glob_blocks++;
		expect += 2 * (size_t)m.layers[kvl].n_kv_heads * (size_t)m.layers[kvl].head_dim *
				  (size_t)pcap * sizeof(uint16_t);
	}

	size_t got	= model_kv_cache_bytes_quant(&m, n_ctx, KV_QUANT_F16, 1);
	size_t full = 0;
	for (int kvl = 0; kvl < n_kv_layers; kvl++)
		full += 2 * (size_t)m.layers[kvl].n_kv_heads * (size_t)m.layers[kvl].head_dim *
				(size_t)n_ctx * sizeof(uint16_t);

	char detail[256];
	snprintf(detail, sizeof(detail),
			 "swa_blocks=%d glob_blocks=%d | reported %.1f KB, n_ctx-equiv %.1f KB (%.0f%%) | "
			 "cap mismatches=%d",
			 n_swa_blocks, n_glob_blocks, got / 1024.0, full / 1024.0,
			 full ? 100.0 * (double)got / (double)full : 0.0, cap_bad);
	verdict v =
		(got == expect && cap_bad == 0 && n_swa_blocks > 0 && n_glob_blocks > 0 && got < full)
			? V_PASS
			: V_FAIL;
	record_result(OPFAM_KV_PUT, label, v, detail);

	free(m.layer_dims.is_global_layer);
	free(m.layers);
}

static void test_kv_size_matches_alloc(backend *b, int compacted) {
	const int		 n_ctx	  = 256;
	static const int LHD[]	  = {64, 128};
	static const int LKVH[]	  = {1, 2};
	int				 n_layers = 2, n_kv_layers = 2;
	const int		 cap = 100;
	char			 label[192];
	snprintf(label, sizeof(label), "kv estimate == allocated bytes cap=%s [%s]",
			 compacted ? "compacted" : "full", b->name);
	if (!b->kv_alloc || !b->kv_free) {
		record_result(OPFAM_KV_PUT, label, V_SKIP, "backend has no kv_alloc/kv_free");
		return;
	}
	if (compacted && !backend_has_cap(b, BCAP_KV_POS_CAP)) {
		record_result(OPFAM_KV_PUT, label, V_SKIP, "backend does not advertise BCAP_KV_POS_CAP");
		return;
	}
	int *hd	 = xcalloc((size_t)n_layers, sizeof(int));
	int *kvh = xcalloc((size_t)n_layers, sizeof(int));
	for (int i = 0; i < n_layers; i++) {
		hd[i]  = LHD[i];
		kvh[i] = LKVH[i];
	}
	int			pcaps[2] = {cap, cap};
	kv_desc		d		 = {.n_layers		  = n_layers,
							.n_kv_layers	  = n_kv_layers,
							.n_kv_heads		  = LKVH[1],
							.head_dim		  = LHD[1],
							.n_ctx			  = n_ctx,
							.kv_quant		  = KV_QUANT_F16,
							.layer_head_dim	  = hd,
							.layer_n_kv_heads = kvh,
							.layer_pos_cap	  = compacted ? pcaps : NULL};
	buffer		k = {0}, v = {0};
	status_code sa	   = b->kv_alloc(b, &d, &k, &v);
	size_t		alloc  = (sa == OK) ? (k.size + v.size) : 0;
	int			rows   = compacted ? cap : n_ctx;
	size_t		expect = 0;
	for (int i = 0; i < n_kv_layers; i++)
		expect += 2 * (size_t)kvh[i] * (size_t)hd[i] * (size_t)rows * sizeof(uint16_t);
	char detail[192];
	snprintf(detail, sizeof(detail), "allocated %.2f MiB, expected %.2f MiB (%d rows/layer)",
			 alloc / 1048576.0, expect / 1048576.0, rows);
	record_result(OPFAM_KV_PUT, label, (sa == OK && alloc == expect) ? V_PASS : V_FAIL, detail);
	if (sa == OK)
		b->kv_free(b, &k, &v);
	free(hd);
	free(kvh);
}

static void test_op_kv_packed_layers(backend *b, kv_quant_type kq) {
	char label[128];
	snprintf(label, sizeof(label), "kv packed per-layer variable dims %s [%s]",
			 kq == KV_QUANT_Q8_0 ? "q8_0" : "f16", b->name);
	if (kq == KV_QUANT_Q8_0 && !backend_has_cap(b, BCAP_KV_QUANT_Q8_0)) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP,
					  "backend does not advertise BCAP_KV_QUANT_Q8_0");
		return;
	}
	static const int LHD[]	  = {32, 48, 64};
	static const int LKVH[]	  = {1, 2, 4};
	int				 n_layers = 3;
	int				 n_ctx	  = 16;
	int				 n_heads  = 8;
	int				 n_pos	  = 13;
	int				 hd_max	  = 64;
	if (!b->kv_put || !b->attention || !b->kv_alloc) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP,
					  "backend has no native kv_put/attention");
		return;
	}

	kv_desc desc = {.n_layers		  = n_layers,
					.n_kv_layers	  = n_layers,
					.n_kv_heads		  = LKVH[n_layers - 1],
					.head_dim		  = hd_max,
					.n_ctx			  = n_ctx,
					.kv_quant		  = kq,
					.layer_head_dim	  = LHD,
					.layer_n_kv_heads = LKVH};
	buffer	kc = {0}, vc = {0};
	if (b->kv_alloc(b, &desc, &kc, &vc) != OK) {
		record_result(OPFAM_KV_QUANT_PARITY, label, V_SKIP, "kv_alloc failed");
		return;
	}

	float *k_in = xmalloc((size_t)hd_max * 4 * sizeof(float));
	float *v_in = xmalloc((size_t)hd_max * 4 * sizeof(float));
	buffer kb = {0}, vb = {0};
	b->buffer_alloc_scratch(b, (size_t)hd_max * 4 * sizeof(float), &kb);
	b->buffer_alloc_scratch(b, (size_t)hd_max * 4 * sizeof(float), &vb);
	float *q  = xmalloc((size_t)n_heads * hd_max * sizeof(float));
	buffer qb = {0}, ob = {0};
	b->buffer_alloc_scratch(b, (size_t)n_heads * hd_max * sizeof(float), &qb);
	b->buffer_alloc_scratch(b, (size_t)n_heads * hd_max * sizeof(float), &ob);
	float *k_store = xmalloc((size_t)n_pos * LKVH[2] * LHD[2] * sizeof(float));
	float *v_store = xmalloc((size_t)n_pos * LKVH[2] * LHD[2] * sizeof(float));
	float *out	   = xmalloc((size_t)n_heads * hd_max * sizeof(float));

	int	  worst = 0;
	char  worst_d[256];
	float worst_reld2 = 0;
	seed_test_rng(0x9A51ULL);

	for (int l = 0; l < n_layers; l++) {
		int hd	= LHD[l];
		int kvh = LKVH[l];
		int kv	= kvh * hd;
		for (int t = 0; t < n_pos; t++) {
			fill_random_f32(k_in, kv, 1.0f);
			fill_random_f32(v_in, kv, 1.0f);
			memcpy(k_store + (size_t)t * kv, k_in, (size_t)kv * sizeof(float));
			memcpy(v_store + (size_t)t * kv, v_in, (size_t)kv * sizeof(float));
			b->buffer_write_f32(b, &kb, k_in, kv);
			b->buffer_write_f32(b, &vb, v_in, kv);
			b->kv_put(b, &kc, &vc, l, t, &kb, &vb, LKVH[n_layers - 1], hd, n_ctx, kvh);
			if (b->synchronize)
				b->synchronize(b);
		}
		fill_random_f32(q, n_heads * hd, 1.0f);
		b->buffer_write_f32(b, &qb, q, n_heads * hd);
		float		scale = 1.0f / sqrtf((float)hd);
		status_code s_got = b->attention(b, &qb, &kc, &vc, &ob, l, n_pos - 1, n_heads,
										 LKVH[n_layers - 1], hd, n_ctx, 0, scale, kvh);
		if (b->synchronize)
			b->synchronize(b);
		b->buffer_read_f32(b, &ob, out, n_heads * hd);

		float *ref	 = ref_softmax_attn(q, k_store, v_store, n_heads, kvh, hd, 0, n_pos, scale);
		float  reld2 = 0, refn2 = 0;
		for (int h = 0; h < n_heads; h++) {
			for (int d = 0; d < hd; d++) {
				float r = ref[(size_t)h * hd + d];
				float g = out[(size_t)h * hd + d];
				reld2 += (r - g) * (r - g);
				refn2 += r * r;
			}
		}
		float reld = reld2 > 0 ? sqrtf(reld2 / (refn2 > 0 ? refn2 : 1.0f)) : 0;
		if (reld > worst_reld2) {
			worst_reld2 = reld;
			worst		= l;
		}
		free(ref);
		if (s_got != OK) {
			snprintf(worst_d, sizeof(worst_d), "attention layer %d -> %d", l, (int)s_got);
			break;
		}
	}
	snprintf(worst_d, sizeof(worst_d), "worst layer=%d rel=%.3e", worst, worst_reld2);
	verdict v = worst_reld2 < 3e-2f ? V_PASS : V_FAIL;
	record_result(OPFAM_KV_QUANT_PARITY, label, v, worst_d);
	if (v != V_PASS)
		compute_debug(out, out, 1);

	free(out);
	free(k_store);
	free(v_store);
	free(q);
	free(k_in);
	free(v_in);
	b->buffer_free(b, &kb);
	b->buffer_free(b, &vb);
	b->buffer_free(b, &qb);
	b->buffer_free(b, &ob);
	if (b->kv_free)
		b->kv_free(b, &kc, &vc);
	else {
		if (kc.owner)
			kc.owner->buffer_free(kc.owner, &kc);
		if (vc.owner)
			vc.owner->buffer_free(vc.owner, &vc);
	}
}

static void run_kv_quant_parity_tests(backend *ref, backend *tgt) {
	test_op_kv_packed_layers(ref, KV_QUANT_F16);
	test_op_kv_batch_put_layers(ref, 0);
	test_op_kv_batch_put_layers(ref, 1);
	test_op_kv_packed_layers(ref, KV_QUANT_Q8_0);
	test_op_kv_quant_parity(ref, 4, 64, 1024, 0);
	test_op_kv_quant_parity(ref, 4, 64, 1024, 127);
	test_op_kv_quant_parity(ref, 8, 128, 2048, 511);
	test_op_kv_quant_parity(ref, 1, 256, 2048, 0);
	test_op_kv_quant_parity(ref, 32, 128, 2048, 1023);
	test_op_kv_quant_parity(ref, 4, 80, 512, 0);
	test_op_kv_quant_parity(ref, 4, 80, 512, 63);
	test_op_kv_quant_parity(ref, 2, 96, 512, 33);
	if (tgt && tgt != ref) {
		test_op_kv_packed_layers(tgt, KV_QUANT_F16);
		test_op_kv_packed_layers(tgt, KV_QUANT_Q8_0);
		test_op_kv_quant_parity(tgt, 4, 64, 1024, 0);
		test_op_kv_quant_parity(tgt, 8, 128, 2048, 511);
	}
	flush_family(OPFAM_KV_QUANT_PARITY);
}

static void test_op_argmax(backend *ref, backend *tgt, int n) {
	if (!tgt->argmax) {
		char label[96];
		snprintf(label, sizeof(label), "argmax n=%d (%s)", n, tgt->name);
		record_result(OPFAM_ARGMAX, label, V_SKIP, "backend has no native argmax");
		return;
	}
	float *logits = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0xA6A6ULL + (uint64_t)n);
	fill_random_f32(logits, n, 10.0f);
	int winner = (int)(next_u32() % (uint32_t)n);
	logits[winner] += 1000.0f;

	buffer l_ref = {0};
	buffer l_tgt = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &l_ref);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &l_tgt);
	ref->buffer_write_f32(ref, &l_ref, logits, n);
	tgt->buffer_write_f32(tgt, &l_tgt, logits, n);

	int32_t idx_ref = -1;
	int32_t idx_tgt = -1;
	ref->argmax(ref, &l_ref, n, &idx_ref);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	status_code s_tgt;
	{
		s_tgt = tgt->argmax(tgt, &l_tgt, n, &idx_tgt);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "argmax n=%d", n);
	if (s_tgt != OK) {
		snprintf(detail, sizeof(detail), "status=%d", s_tgt);
		record_result(OPFAM_ARGMAX, label, V_FAIL, detail);
	} else if (idx_ref == idx_tgt) {
		snprintf(detail, sizeof(detail), "idx=%d (match)", idx_ref);
		record_result(OPFAM_ARGMAX, label, V_PASS, detail);
	} else {
		snprintf(detail, sizeof(detail), "idx_ref=%d idx_tgt=%d (argmax diverged)", idx_ref,
				 idx_tgt);
		record_result(OPFAM_ARGMAX, label, V_FAIL, detail);
	}

	free(logits);
	ref->buffer_free(ref, &l_ref);
	tgt->buffer_free(tgt, &l_tgt);
}

static void test_op_matmul_residual(backend *ref, backend *tgt, const qtype_info *qt, int n,
									int k) {
	char label[128];
	if (!ref->matmul_residual || !tgt->matmul_residual) {
		snprintf(label, sizeof(label), "%s matmul_residual N=%d K=%d", qt->name, n, k);
		record_result(OPFAM_MATMUL_RESIDUAL, label, V_SKIP,
					  !ref->matmul_residual ? "reference has no matmul_residual"
											: "backend has no native matmul_residual");
		return;
	}
	if (k % qt->block != 0)
		return;
	if (tgt->matmul_type_native && !tgt->matmul_type_native(tgt, qt->type)) {
		snprintf(label, sizeof(label), "%s matmul_residual N=%d K=%d (%s)", qt->name, n, k,
				 tgt->name);
		record_result(OPFAM_MATMUL_RESIDUAL, label, V_SKIP, "missing native implementation");
		return;
	}

	seed_test_rng((0xA5A5ULL * (qt->type + 1) * 1000003ULL) + ((uint64_t)n * 31) + (uint64_t)k +
				  0x1234);
	void *blocks = test_make_weight(ref, qt, n, k, NULL);
	if (!blocks) {
		record_result(OPFAM_MATMUL_RESIDUAL, label, V_SKIP, "reference cannot repack weight");
		return;
	}

	float *x		= xmalloc((size_t)k * sizeof(float));
	float *residual = xmalloc((size_t)n * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	fill_random_f32(residual, n, 1.0f);

	tensor_desc wd = {
		.host_data = blocks,
		.type	   = qt->type,
		.n_dims	   = 2,
		.dims	   = {k, n},
	};

	buffer w_ref   = {0};
	buffer x_ref   = {0};
	buffer r_ref   = {0};
	buffer y_ref_b = {0};
	ref->buffer_alloc_weight(ref, &wd, &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)k * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &r_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &y_ref_b);
	ref->buffer_write_f32(ref, &x_ref, x, k);
	ref->buffer_write_f32(ref, &r_ref, residual, n);
	ref->matmul_residual(ref, &w_ref, qt->type, &x_ref, &r_ref, &y_ref_b, n, k);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref_b, y_ref, n);

	buffer w_tgt = {0};
	buffer x_tgt = {0};
	buffer r_tgt = {0};
	buffer y_tgt = {0};
	tgt->buffer_alloc_weight(tgt, &wd, &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)k * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &r_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, k);
	tgt->buffer_write_f32(tgt, &r_tgt, residual, n);
	status_code s_tgt;
	{
		s_tgt = tgt->matmul_residual(tgt, &w_tgt, qt->type, &x_tgt, &r_tgt, &y_tgt, n, k);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, y_got, n);

	char detail[256];
	snprintf(label, sizeof(label), "%s matmul_residual N=%d K=%d", qt->name, n, k);
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(OPFAM_MATMUL_RESIDUAL, label, v, detail);

	free(y_ref);
	free(y_got);
	free(x);
	free(residual);
	free(blocks);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &r_ref);
	ref->buffer_free(ref, &y_ref_b);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &r_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rope_qk(backend *ref, backend *tgt, int n_heads, int n_kv_heads, int head_dim,
							int pos) {
	if (!ref->rope_qk || !tgt->rope_qk) {
		char label[128];
		snprintf(label, sizeof(label), "rope_qk h=%d/%d d=%d pos=%d", n_heads, n_kv_heads, head_dim,
				 pos);
		record_result(OPFAM_ROPE_QK, label, V_SKIP,
					  !ref->rope_qk ? "reference has no rope_qk" : "backend has no native rope_qk");
		return;
	}
	int nq	  = n_heads * head_dim;
	int nk	  = n_kv_heads * head_dim;
	int half  = head_dim / 2;
	int n_ctx = pos + 1;

	float *q	 = xmalloc((size_t)nq * sizeof(float));
	float *k	 = xmalloc((size_t)nk * sizeof(float));
	float *cos_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	float *sin_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	seed_test_rng(0x5EEDULL + ((uint64_t)n_heads * 131) + ((uint64_t)n_kv_heads * 17) +
				  ((uint64_t)head_dim * 7) + (uint64_t)pos);
	fill_random_f32(q, nq, 1.0f);
	fill_random_f32(k, nk, 1.0f);
	for (int j = 0; j < half; j++) {
		float c = cosf(((float)j * 0.0731f) + 0.1f);
		float s = sinf(((float)j * 0.0731f) + 0.1f);
		for (int p = 0; p < n_ctx; p++) {
			cos_v[(p * half) + j] = c;
			sin_v[(p * half) + j] = s;
		}
	}

	kv_desc kvd = {.n_ctx		= n_ctx,
				   .n_kv_heads	= n_kv_heads,
				   .head_dim	= head_dim,
				   .n_layers	= 1,
				   .n_kv_layers = 1};

	ref->rope_neox = 0;
	tgt->rope_neox = 0;

	buffer kc_ref  = {0};
	buffer vc_ref  = {0};
	buffer q_ref_b = {0};
	buffer k_ref_b = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)nq * sizeof(float), &q_ref_b);
	ref->buffer_alloc_scratch(ref, (size_t)nk * sizeof(float), &k_ref_b);
	ref->buffer_write_f32(ref, &q_ref_b, q, nq);
	ref->buffer_write_f32(ref, &k_ref_b, k, nk);
	ref->rope_qk(ref, &q_ref_b, &k_ref_b, n_heads, n_kv_heads, head_dim, pos, cos_v, sin_v);
	if (ref && ref->synchronize)
		ref->synchronize(ref);
	float *q_ref = xmalloc((size_t)nq * sizeof(float));
	float *k_ref = xmalloc((size_t)nk * sizeof(float));
	ref->buffer_read_f32(ref, &q_ref_b, q_ref, nq);
	ref->buffer_read_f32(ref, &k_ref_b, k_ref, nk);

	buffer kc_tgt = {0};
	buffer vc_tgt = {0};
	buffer q_tgt  = {0};
	buffer k_tgt  = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)nq * sizeof(float), &q_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)nk * sizeof(float), &k_tgt);
	tgt->buffer_write_f32(tgt, &q_tgt, q, nq);
	tgt->buffer_write_f32(tgt, &k_tgt, k, nk);
	status_code s_tgt;
	{
		s_tgt = tgt->rope_qk(tgt, &q_tgt, &k_tgt, n_heads, n_kv_heads, head_dim, pos, cos_v, sin_v);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
	}
	float *q_got = xmalloc((size_t)nq * sizeof(float));
	float *k_got = xmalloc((size_t)nk * sizeof(float));
	tgt->buffer_read_f32(tgt, &q_tgt, q_got, nq);
	tgt->buffer_read_f32(tgt, &k_tgt, k_got, nk);

	char label[128];
	char detail[256];
	snprintf(label, sizeof(label), "rope_qk Q h=%d/%d d=%d pos=%d", n_heads, n_kv_heads, head_dim,
			 pos);
	verdict vq = classify_output("loose", q_ref, q_got, nq, s_tgt, detail, sizeof(detail));
	if (vq != V_PASS && vq != V_SKIP)
		compute_debug(q_ref, q_got, nq);
	record_result(OPFAM_ROPE_QK, label, vq, detail);

	snprintf(label, sizeof(label), "rope_qk K h=%d/%d d=%d pos=%d", n_heads, n_kv_heads, head_dim,
			 pos);
	verdict vk = classify_output("loose", k_ref, k_got, nk, s_tgt, detail, sizeof(detail));
	if (vk != V_PASS && vk != V_SKIP)
		compute_debug(k_ref, k_got, nk);
	record_result(OPFAM_ROPE_QK, label, vk, detail);

	free(q);
	free(k);
	free(cos_v);
	free(sin_v);
	free(q_ref);
	free(k_ref);
	free(q_got);
	free(k_got);
	ref->buffer_free(ref, &q_ref_b);
	ref->buffer_free(ref, &k_ref_b);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &k_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_batch_matmul_parity(backend *ref, backend *tgt, const qtype_info *qt, int n, int k,
									 int m) {
	char label[160];
	if (!ref->matmul || !ref->matmul_batch) {
		snprintf(label, sizeof(label), "%s batch N=%d K=%d M=%d", qt->name, n, k, m);
		record_result(OPFAM_BATCH_PARITY, label, V_SKIP, "reference has no matmul_batch");
		return;
	}
	if (k % qt->block != 0)
		return;

	void *blocks = test_make_weight(ref, qt, n, k, NULL);
	if (!blocks) {
		snprintf(label, sizeof(label), "%s batch N=%d K=%d M=%d", qt->name, n, k, m);
		record_result(OPFAM_BATCH_PARITY, label, V_SKIP, "reference cannot repack weight");
		return;
	}
	float *x = xmalloc((size_t)k * (size_t)m * sizeof(float));
	seed_test_rng(0xBA7CULL + ((uint64_t)qt->type * 131) + ((uint64_t)n * 17) + (uint64_t)k +
				  (uint64_t)m);
	fill_random_f32(x, k * m, 1.0f);

	tensor_desc wd	  = {.host_data = blocks, .type = qt->type, .n_dims = 2, .dims = {k, n}};
	buffer		w_ref = {0}, xb_ref = {0}, yb_ref = {0};
	ref->buffer_alloc_weight(ref, &wd, &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)k * (size_t)m * sizeof(float), &xb_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * (size_t)m * sizeof(float), &yb_ref);
	ref->buffer_write_f32(ref, &xb_ref, x, k * m);

	float *y_batch	= xmalloc((size_t)n * (size_t)m * sizeof(float));
	float *y_single = xmalloc((size_t)n * (size_t)m * sizeof(float));
	ref->matmul_batch(ref, &w_ref, qt->type, &xb_ref, &yb_ref, n, k, m);
	if (ref->synchronize)
		ref->synchronize(ref);
	ref->buffer_read_f32(ref, &yb_ref, y_batch, n * m);

	for (int t = 0; t < m; t++) {
		buffer xt = {0}, yt = {0};
		ref->buffer_alloc_scratch(ref, (size_t)k * sizeof(float), &xt);
		ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &yt);
		ref->buffer_write_f32(ref, &xt, x + (size_t)t * k, k);
		ref->matmul(ref, &w_ref, qt->type, &xt, &yt, n, k);
		if (ref->synchronize)
			ref->synchronize(ref);
		ref->buffer_read_f32(ref, &yt, y_single + (size_t)t * n, n);
		ref->buffer_free(ref, &xt);
		ref->buffer_free(ref, &yt);
	}
	snprintf(label, sizeof(label), "%s batch-vs-single N=%d K=%d M=%d", qt->name, n, k, m);
	test_parity_compare(OPFAM_BATCH_PARITY, label, y_single, y_batch, n * m, "loose");

	if (tgt->matmul_batch) {
		buffer w_tgt = {0}, xb_tgt = {0}, yb_tgt = {0};
		tgt->buffer_alloc_weight(tgt, &wd, &w_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)k * (size_t)m * sizeof(float), &xb_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n * (size_t)m * sizeof(float), &yb_tgt);
		tgt->buffer_write_f32(tgt, &xb_tgt, x, k * m);
		status_code s_tgt = tgt->matmul_batch(tgt, &w_tgt, qt->type, &xb_tgt, &yb_tgt, n, k, m);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
		float *y_tgt = xmalloc((size_t)n * (size_t)m * sizeof(float));
		if (s_tgt == OK)
			tgt->buffer_read_f32(tgt, &yb_tgt, y_tgt, n * m);
		snprintf(label, sizeof(label), "%s batch cross N=%d K=%d M=%d", qt->name, n, k, m);
		test_parity_compare_status(OPFAM_BATCH_PARITY, label, y_batch, y_tgt, n * m, s_tgt);
		free(y_tgt);
		tgt->buffer_free(tgt, &w_tgt);
		tgt->buffer_free(tgt, &xb_tgt);
		tgt->buffer_free(tgt, &yb_tgt);
	}

	free(blocks);
	free(x);
	free(y_batch);
	free(y_single);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &xb_ref);
	ref->buffer_free(ref, &yb_ref);
}

static void test_batch_attention_parity(backend *ref, backend *tgt, int n_heads, int n_kv_heads,
										int head_dim, int n_ctx, int pos_start, int m, int flash) {
	char label[160];
	if (!ref->attention || !ref->attention_batch || !ref->kv_put) {
		snprintf(label, sizeof(label), "attention batch h=%d/%d d=%d pos=%d m=%d", n_heads,
				 n_kv_heads, head_dim, pos_start, m);
		record_result(OPFAM_BATCH_PARITY, label, V_SKIP, "reference has no attention_batch");
		return;
	}
	int	  n		= n_heads * head_dim;
	int	  n_kv	= n_kv_heads * head_dim;
	float scale = 1.0f / sqrtf((float)head_dim);
	int	  n_t	= pos_start + m;

	seed_test_rng(0xBA77ULL + ((uint64_t)n_heads * 31) + ((uint64_t)head_dim * 7) +
				  (uint64_t)pos_start + (uint64_t)m);
	float *q = xmalloc((size_t)n * (size_t)m * sizeof(float));
	fill_random_f32(q, n * m, 1.0f);
	float *k_all = xmalloc((size_t)n_kv * (size_t)n_t * sizeof(float));
	float *v_all = xmalloc((size_t)n_kv * (size_t)n_t * sizeof(float));
	fill_random_f32(k_all, n_kv * n_t, 1.0f);
	fill_random_f32(v_all, n_kv * n_t, 1.0f);

	kv_desc kvd	   = {.n_ctx	   = n_ctx,
					  .n_kv_heads  = n_kv_heads,
					  .head_dim	   = head_dim,
					  .n_layers	   = 1,
					  .n_kv_layers = 1};
	buffer	kc_ref = {0}, vc_ref = {0}, ki_ref = {0}, vi_ref = {0};
	buffer	qb_ref = {0}, yb_ref = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &ki_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n_kv * sizeof(float), &vi_ref);
	for (int t = 0; t < n_t; t++) {
		ref->buffer_write_f32(ref, &ki_ref, k_all + (size_t)t * n_kv, n_kv);
		ref->buffer_write_f32(ref, &vi_ref, v_all + (size_t)t * n_kv, n_kv);
		ref->kv_put(ref, &kc_ref, &vc_ref, 0, t, &ki_ref, &vi_ref, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
	}
	ref->buffer_alloc_scratch(ref, (size_t)n * (size_t)m * sizeof(float), &qb_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * (size_t)m * sizeof(float), &yb_ref);
	ref->buffer_write_f32(ref, &qb_ref, q, n * m);
	ref->attention_batch(ref, &qb_ref, &kc_ref, &vc_ref, &yb_ref, 0, pos_start, n_heads, n_kv_heads,
						 head_dim, n_ctx, 0, scale, n_kv_heads, m);
	if (ref->synchronize)
		ref->synchronize(ref);
	float *y_batch = xmalloc((size_t)n * (size_t)m * sizeof(float));
	ref->buffer_read_f32(ref, &yb_ref, y_batch, n * m);

	float *y_single = xmalloc((size_t)n * (size_t)m * sizeof(float));
	for (int t = 0; t < m; t++) {
		buffer qt = {0}, yt = {0};
		ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &qt);
		ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &yt);
		ref->buffer_write_f32(ref, &qt, q + (size_t)t * n, n);
		ref->attention(ref, &qt, &kc_ref, &vc_ref, &yt, 0, pos_start + t, n_heads, n_kv_heads,
					   head_dim, n_ctx, 0, scale, n_kv_heads);
		if (ref->synchronize)
			ref->synchronize(ref);
		ref->buffer_read_f32(ref, &yt, y_single + (size_t)t * n, n);
		ref->buffer_free(ref, &qt);
		ref->buffer_free(ref, &yt);
	}
	snprintf(label, sizeof(label), "attention batch-vs-single h=%d/%d d=%d pos=%d m=%d", n_heads,
			 n_kv_heads, head_dim, pos_start, m);
	test_parity_compare(OPFAM_BATCH_PARITY, label, y_single, y_batch, n * m, "loose");

	if (tgt->attention_batch && tgt->kv_put) {
		buffer kc_tgt = {0}, vc_tgt = {0}, ki_tgt = {0}, vi_tgt = {0};
		buffer qb_tgt = {0}, yb_tgt = {0};
		tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &ki_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &vi_tgt);
		for (int t = 0; t < n_t; t++) {
			tgt->buffer_write_f32(tgt, &ki_tgt, k_all + (size_t)t * n_kv, n_kv);
			tgt->buffer_write_f32(tgt, &vi_tgt, v_all + (size_t)t * n_kv, n_kv);
			tgt->kv_put(tgt, &kc_tgt, &vc_tgt, 0, t, &ki_tgt, &vi_tgt, n_kv_heads, head_dim, n_ctx,
						n_kv_heads);
		}
		tgt->buffer_alloc_scratch(tgt, (size_t)n * (size_t)m * sizeof(float), &qb_tgt);
		tgt->buffer_alloc_scratch(tgt, (size_t)n * (size_t)m * sizeof(float), &yb_tgt);
		tgt->buffer_write_f32(tgt, &qb_tgt, q, n * m);
		status_code s_tgt =
			tgt->attention_batch(tgt, &qb_tgt, &kc_tgt, &vc_tgt, &yb_tgt, 0, pos_start, n_heads,
								 n_kv_heads, head_dim, n_ctx, flash, scale, n_kv_heads, m);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
		float *y_tgt = xmalloc((size_t)n * (size_t)m * sizeof(float));
		if (s_tgt == OK)
			tgt->buffer_read_f32(tgt, &yb_tgt, y_tgt, n * m);
		snprintf(label, sizeof(label), "attention batch cross h=%d/%d d=%d pos=%d m=%d fl=%d",
				 n_heads, n_kv_heads, head_dim, pos_start, m, flash);
		if (getenv("KQDBG") && flash && m == 16 && head_dim == 64) {
			for (int rr = 0; rr < 8; rr++)
				fprintf(stderr, "[W%d] wroteBy=%.0f  val=%7.3f | ref=%7.3f\n", rr,
						y_tgt[rr * n] - 500.0, y_tgt[rr * n + 1], y_batch[rr * n]);
		}
		test_parity_compare_status(OPFAM_BATCH_PARITY, label, y_batch, y_tgt, n * m, s_tgt);
		free(y_tgt);
		tgt->buffer_free(tgt, &kc_tgt);
		tgt->buffer_free(tgt, &vc_tgt);
		tgt->buffer_free(tgt, &ki_tgt);
		tgt->buffer_free(tgt, &vi_tgt);
		tgt->buffer_free(tgt, &qb_tgt);
		tgt->buffer_free(tgt, &yb_tgt);
	}

	free(q);
	free(k_all);
	free(v_all);
	free(y_batch);
	free(y_single);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	ref->buffer_free(ref, &ki_ref);
	ref->buffer_free(ref, &vi_ref);
	ref->buffer_free(ref, &qb_ref);
	ref->buffer_free(ref, &yb_ref);
}

static void test_batch_rope_parity(backend *ref, backend *tgt, int n_heads, int head_dim,
								   int pos_start, int m) {
	char label[160];
	if (!ref->rope || !ref->rope_batch) {
		snprintf(label, sizeof(label), "rope batch h=%d d=%d pos=%d m=%d", n_heads, head_dim,
				 pos_start, m);
		record_result(OPFAM_BATCH_PARITY, label, V_SKIP, "reference has no rope_batch");
		return;
	}
	int N	  = n_heads * head_dim;
	int half  = head_dim / 2;
	int n_ctx = pos_start + m;

	float *vec	 = xmalloc((size_t)N * (size_t)m * sizeof(float));
	float *cos_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	float *sin_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	seed_test_rng(0xBA11ULL + ((uint64_t)n_heads * 131) + ((uint64_t)head_dim * 17) +
				  (uint64_t)pos_start + (uint64_t)m);
	fill_random_f32(vec, N * m, 1.0f);
	for (int j = 0; j < half; j++) {
		float c = cosf(((float)j * 0.0731f) + 0.1f);
		float s = sinf(((float)j * 0.0731f) + 0.1f);
		for (int p = 0; p < n_ctx; p++) {
			cos_v[(p * half) + j] = c;
			sin_v[(p * half) + j] = s;
		}
	}

	ref->rope_neox = 0;
	buffer vb_ref  = {0};
	ref->buffer_alloc_scratch(ref, (size_t)N * (size_t)m * sizeof(float), &vb_ref);
	ref->buffer_write_f32(ref, &vb_ref, vec, N * m);
	ref->rope_batch(ref, &vb_ref, n_heads, head_dim, pos_start, cos_v, sin_v, m);
	if (ref->synchronize)
		ref->synchronize(ref);
	float *y_batch = xmalloc((size_t)N * (size_t)m * sizeof(float));
	ref->buffer_read_f32(ref, &vb_ref, y_batch, N * m);

	float *y_single = xmalloc((size_t)N * (size_t)m * sizeof(float));
	for (int t = 0; t < m; t++) {
		buffer vt = {0};
		ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &vt);
		ref->buffer_write_f32(ref, &vt, vec + (size_t)t * N, N);
		ref->rope(ref, &vt, n_heads, head_dim, pos_start + t, cos_v, sin_v);
		if (ref->synchronize)
			ref->synchronize(ref);
		ref->buffer_read_f32(ref, &vt, y_single + (size_t)t * N, N);
		ref->buffer_free(ref, &vt);
	}
	snprintf(label, sizeof(label), "rope batch-vs-single h=%d d=%d pos=%d m=%d", n_heads, head_dim,
			 pos_start, m);
	test_parity_compare(OPFAM_BATCH_PARITY, label, y_single, y_batch, N * m, "loose");

	if (tgt->rope_batch) {
		tgt->rope_neox = 0;
		buffer vb_tgt  = {0};
		tgt->buffer_alloc_scratch(tgt, (size_t)N * (size_t)m * sizeof(float), &vb_tgt);
		tgt->buffer_write_f32(tgt, &vb_tgt, vec, N * m);
		status_code s_tgt =
			tgt->rope_batch(tgt, &vb_tgt, n_heads, head_dim, pos_start, cos_v, sin_v, m);
		if (tgt->synchronize)
			tgt->synchronize(tgt);
		float *y_tgt = xmalloc((size_t)N * (size_t)m * sizeof(float));
		if (s_tgt == OK)
			tgt->buffer_read_f32(tgt, &vb_tgt, y_tgt, N * m);
		snprintf(label, sizeof(label), "rope batch cross h=%d d=%d pos=%d m=%d", n_heads, head_dim,
				 pos_start, m);
		test_parity_compare_status(OPFAM_BATCH_PARITY, label, y_batch, y_tgt, N * m, s_tgt);
		free(y_tgt);
		tgt->buffer_free(tgt, &vb_tgt);
	}

	free(vec);
	free(cos_v);
	free(sin_v);
	free(y_batch);
	free(y_single);
	ref->buffer_free(ref, &vb_ref);
}

static void test_edge_rmsnorm_zeros(backend *ref, backend *tgt) {
	if (!tgt->rmsnorm)
		return;
	int	   N = 256;
	float *x = xcalloc((size_t)N, sizeof(float));
	float *w = xmalloc((size_t)N * sizeof(float));
	for (int i = 0; i < N; i++)
		w[i] = 1.0f;

	buffer x_ref   = {0};
	buffer w_ref   = {0};
	buffer y_ref_b = {0};
	ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &x_ref);
	ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &w_ref);
	ref->buffer_alloc_scratch(ref, (size_t)N * sizeof(float), &y_ref_b);
	ref->buffer_write_f32(ref, &x_ref, x, N);
	ref->buffer_write_f32(ref, &w_ref, w, N);
	ref->rmsnorm(ref, &x_ref, &w_ref, &y_ref_b, N, 1e-5f);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	ref->buffer_read_f32(ref, &y_ref_b, y_ref, N);

	buffer x_tgt = {0};
	buffer w_tgt = {0};
	buffer y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, N);
	tgt->buffer_write_f32(tgt, &w_tgt, w, N);
	status_code s_tgt = tgt->rmsnorm(tgt, &x_tgt, &w_tgt, &y_tgt, N, 1e-5f);
	if (tgt->synchronize)
		tgt->synchronize(tgt);
	float *y_got = xmalloc((size_t)N * sizeof(float));
	tgt->buffer_read_f32(tgt, &y_tgt, y_got, N);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "rmsnorm all-zero input (div-by-zero)");
	int nf = count_nonfinite(y_got, N);
	if (nf > 0) {
		snprintf(detail, sizeof(detail), "%d/%d non-finite (should be 0 with eps)", nf, N);
		compute_debug(y_ref, y_got, N);
		record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
	} else {
		verdict v = classify_output("exact", y_ref, y_got, N, s_tgt, detail, sizeof(detail));
		if (v != V_PASS)
			compute_debug(y_ref, y_got, N);
		record_result(OPFAM_EDGE_CASE, label, v, detail);
	}

	free(x);
	free(w);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &x_ref);
	ref->buffer_free(ref, &w_ref);
	ref->buffer_free(ref, &y_ref_b);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_edge_determinism(backend *ref, backend *tgt) {
	if (!tgt->matmul)
		return;
	const qtype_info *qt	   = &QTYPES[0];
	int				  N		   = 128;
	int				  K		   = 256;
	int				  n_blocks = N * (K / qt->block);
	void			 *blocks   = xcalloc((size_t)n_blocks, qt->bytes);
	seed_test_rng(0xD31ULL);
	fill_random_blocks(blocks, n_blocks, qt->bytes, qt->type);

	float *x = xmalloc((size_t)K * sizeof(float));
	fill_random_f32(x, K, 1.0f);

	tensor_desc wd = {
		.host_data = blocks,
		.type	   = qt->type,
		.n_dims	   = 2,
		.dims	   = {K, N},
	};

	buffer w_tgt  = {0};
	buffer x_tgt  = {0};
	buffer y1_tgt = {0};
	buffer y2_tgt = {0};
	tgt->buffer_alloc_weight(tgt, &wd, &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)K * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &y1_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)N * sizeof(float), &y2_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, K);

	tgt->matmul(tgt, &w_tgt, qt->type, &x_tgt, &y1_tgt, N, K);
	if (tgt->synchronize)
		tgt->synchronize(tgt);
	tgt->matmul(tgt, &w_tgt, qt->type, &x_tgt, &y2_tgt, N, K);
	if (tgt->synchronize)
		tgt->synchronize(tgt);

	float *y1 = xmalloc((size_t)N * sizeof(float));
	float *y2 = xmalloc((size_t)N * sizeof(float));
	tgt->buffer_read_f32(tgt, &y1_tgt, y1, N);
	tgt->buffer_read_f32(tgt, &y2_tgt, y2, N);

	int mismatch = 0;
	for (int i = 0; i < N; i++) {
		if (y1[i] != y2[i])
			mismatch++;
	}

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "matmul determinism (same input twice)");
	if (mismatch == 0) {
		snprintf(detail, sizeof(detail), "%d elements, bit-exact", N);
		record_result(OPFAM_EDGE_CASE, label, V_PASS, detail);
	} else {
		compute_debug(y1, y2, N);
		snprintf(detail, sizeof(detail), "%d/%d elements differ (non-deterministic)", mismatch, N);
		record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
	}

	free(y1);
	free(y2);
	free(x);
	free(blocks);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &y1_tgt);
	tgt->buffer_free(tgt, &y2_tgt);
	(void)ref;
}

static void test_edge_argmax_all_equal(backend *ref, backend *tgt) {
	if (!tgt->argmax)
		return;
	int	   n	  = 512;
	float *logits = xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++)
		logits[i] = 3.0f;

	buffer l_ref = {0};
	buffer l_tgt = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &l_ref);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &l_tgt);
	ref->buffer_write_f32(ref, &l_ref, logits, n);
	tgt->buffer_write_f32(tgt, &l_tgt, logits, n);

	int32_t idx_ref = -1;
	int32_t idx_tgt = -1;
	ref->argmax(ref, &l_ref, n, &idx_ref);
	status_code s_tgt = tgt->argmax(tgt, &l_tgt, n, &idx_tgt);
	if (tgt->synchronize)
		tgt->synchronize(tgt);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "argmax all-equal logits (tie-break)");
	if (s_tgt != OK) {
		snprintf(detail, sizeof(detail), "status=%d", s_tgt);
		record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
	} else if (idx_ref == idx_tgt) {
		snprintf(detail, sizeof(detail), "idx=%d (match)", idx_ref);
		record_result(OPFAM_EDGE_CASE, label, V_PASS, detail);
	} else {
		snprintf(detail, sizeof(detail), "idx_ref=%d idx_tgt=%d (tie-break diverged)", idx_ref,
				 idx_tgt);
		record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
	}

	free(logits);
	ref->buffer_free(ref, &l_ref);
	tgt->buffer_free(tgt, &l_tgt);
}

static void test_edge_ffn_activate_extremes(backend *ref, backend *tgt) {
	if (!tgt->ffn_activate)
		return;
	int	   n = 256;
	float *g = xmalloc((size_t)n * sizeof(float));
	float *u = xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++) {
		switch (i % 4) {
		case 0:
			g[i] = 40.0f;
			u[i] = 40.0f;
			break;
		case 1:
			g[i] = -40.0f;
			u[i] = -40.0f;
			break;
		case 2:
			g[i] = 0.0f;
			u[i] = 0.0f;
			break;
		default:
			g[i] = -40.0f;
			u[i] = 40.0f;
			break;
		}
	}

	buffer g_ref = {0};
	buffer u_ref = {0};
	buffer o_ref = {0};
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &g_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &u_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &o_ref);
	ref->buffer_write_f32(ref, &g_ref, g, n);
	ref->buffer_write_f32(ref, &u_ref, u, n);
	ref->ffn_activate(ref, &g_ref, &u_ref, &o_ref, n);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &o_ref, y_ref, n);

	buffer g_tgt = {0};
	buffer u_tgt = {0};
	buffer o_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &g_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &u_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &o_tgt);
	tgt->buffer_write_f32(tgt, &g_tgt, g, n);
	tgt->buffer_write_f32(tgt, &u_tgt, u, n);
	status_code s_tgt = tgt->ffn_activate(tgt, &g_tgt, &u_tgt, &o_tgt, n);
	if (tgt->synchronize)
		tgt->synchronize(tgt);
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &o_tgt, y_got, n);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "ffn_activate saturated inputs (SwiGLU overflow)");
	verdict v = classify_output("loose", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
	if (v != V_PASS && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	record_result(OPFAM_EDGE_CASE, label, v, detail);

	free(g);
	free(u);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &g_ref);
	ref->buffer_free(ref, &u_ref);
	ref->buffer_free(ref, &o_ref);
	tgt->buffer_free(tgt, &g_tgt);
	tgt->buffer_free(tgt, &u_tgt);
	tgt->buffer_free(tgt, &o_tgt);
}

static void test_edge_rope_identity_table(backend *ref, backend *tgt) {
	if (!tgt->rope)
		return;
	int n_heads	 = 8;
	int head_dim = 64;
	int half	 = head_dim / 2;
	int n_ctx	 = 4;
	int pos		 = 2;
	int n		 = n_heads * head_dim;

	float *x	 = xmalloc((size_t)n * sizeof(float));
	float *cos_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	float *sin_v = xmalloc((size_t)n_ctx * half * sizeof(float));
	seed_test_rng(0xF05E0ULL);
	fill_random_f32(x, n, 2.0f);
	for (int i = 0; i < n_ctx * half; i++) {
		cos_v[i] = 1.0f;
		sin_v[i] = 0.0f;
	}

	kv_desc kvd = {.n_ctx		= n_ctx,
				   .n_kv_heads	= n_heads,
				   .head_dim	= head_dim,
				   .n_layers	= 1,
				   .n_kv_layers = 1};

	buffer kc_ref = {0};
	buffer vc_ref = {0};
	buffer x_ref  = {0};
	ref->kv_alloc(ref, &kvd, &kc_ref, &vc_ref);
	ref->buffer_alloc_scratch(ref, (size_t)n * sizeof(float), &x_ref);
	ref->buffer_write_f32(ref, &x_ref, x, n);
	ref->rope(ref, &x_ref, n_heads, head_dim, pos, cos_v, sin_v);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	ref->buffer_read_f32(ref, &x_ref, y_ref, n);

	buffer kc_tgt = {0};
	buffer vc_tgt = {0};
	buffer x_tgt  = {0};
	tgt->kv_alloc(tgt, &kvd, &kc_tgt, &vc_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &x_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, n);
	status_code s_tgt = tgt->rope(tgt, &x_tgt, n_heads, head_dim, pos, cos_v, sin_v);
	if (tgt->synchronize)
		tgt->synchronize(tgt);
	float *y_got = xmalloc((size_t)n * sizeof(float));
	tgt->buffer_read_f32(tgt, &x_tgt, y_got, n);

	char label[112];
	char detail[256];
	snprintf(label, sizeof(label), "rope identity table (cos=1,sin=0) is a no-op");
	int nf = count_nonfinite(y_got, n);
	if (nf > 0) {
		snprintf(detail, sizeof(detail), "%d/%d non-finite", nf, n);
		compute_debug(y_ref, y_got, n);
		record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
	} else {
		int	  at		= -1;
		float noop_diff = max_abs_diff_at(x, y_ref, n, &at);
		if (noop_diff > EPS_EXACT) {
			snprintf(detail, sizeof(detail), "ref rope(cos=1,sin=0) is not identity: diff=%.3e@%d",
					 noop_diff, at);
			compute_debug(x, y_ref, n);
			record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
		} else {
			verdict v = classify_output("exact", y_ref, y_got, n, s_tgt, detail, sizeof(detail));
			if (v != V_PASS)
				compute_debug(y_ref, y_got, n);
			record_result(OPFAM_EDGE_CASE, label, v, detail);
		}
	}

	free(x);
	free(cos_v);
	free(sin_v);
	free(y_ref);
	free(y_got);
	ref->buffer_free(ref, &kc_ref);
	ref->buffer_free(ref, &vc_ref);
	ref->buffer_free(ref, &x_ref);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
	tgt->buffer_free(tgt, &x_tgt);
}

struct backend_op_coverage_row {
	const char *name;
	int			has_ref;
	int			has_tgt;
};

static void print_op_coverage(backend *ref, backend *tgt) {
	struct backend_op_coverage_row rows[] = {
		{"matmul", ref->matmul != NULL, tgt->matmul != NULL},
		{"matmul_multi", ref->matmul_multi != NULL, tgt->matmul_multi != NULL},
		{"matmul_batch", ref->matmul_batch != NULL, tgt->matmul_batch != NULL},
		{"embd_lookup", ref->embd_lookup != NULL, tgt->embd_lookup != NULL},
		{"rmsnorm", ref->rmsnorm != NULL, tgt->rmsnorm != NULL},
		{"rmsnorm_per_head", ref->rmsnorm_per_head != NULL, tgt->rmsnorm_per_head != NULL},
		{"rmsnorm_noweight", ref->rmsnorm_noweight != NULL, tgt->rmsnorm_noweight != NULL},
		{"rope", ref->rope != NULL, tgt->rope != NULL},
		{"rope_ext", ref->rope_ext != NULL, tgt->rope_ext != NULL},
		{"rope_qk", ref->rope_qk != NULL, tgt->rope_qk != NULL},
		{"add_inplace", ref->add_inplace != NULL, tgt->add_inplace != NULL},
		{"ffn_activate", ref->ffn_activate != NULL, tgt->ffn_activate != NULL},
		{"ffn_activate_ex", ref->ffn_activate_ex != NULL, tgt->ffn_activate_ex != NULL},
		{"attention", ref->attention != NULL, tgt->attention != NULL},
		{"attention_swa", ref->attention_swa != NULL, tgt->attention_swa != NULL},
		{"kv_put", ref->kv_put != NULL, tgt->kv_put != NULL},
		{"argmax", ref->argmax != NULL, tgt->argmax != NULL},
		{"matmul_residual", ref->matmul_residual != NULL, tgt->matmul_residual != NULL},
	};
	int n_gaps = 0;
	for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
		if (rows[i].has_ref != rows[i].has_tgt)
			n_gaps++;
	}
	if (n_gaps == 0)
		return;

	printf("native op coverage (%s vs ref):\n", tgt->name);
	for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
		if (rows[i].has_ref == rows[i].has_tgt)
			continue;
		printf("  %-18s ref=%c  tgt=%c\n", rows[i].name, rows[i].has_ref ? '+' : '-',
			   rows[i].has_tgt ? '+' : '-');
	}
}
#define RS_FAM OPFAM_REAL_SHAPE
#define RS_DIM 1536
#define RS_PLE 256

static const uint32_t RS_MATMUL_TYPES[] = {GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
										   GGML_TYPE_F32, GGML_TYPE_BF16};
#define RS_MATMUL_TYPES_N ((int)(sizeof(RS_MATMUL_TYPES) / sizeof(RS_MATMUL_TYPES[0])))

static const int RS_MATMUL_SHAPES[][2] = {
	{256, 1536},   {512, 1536},	 {2048, 1536}, {4096, 1536}, {8960, 1536},	{6144, 1536},
	{12288, 1536}, {1536, 2048}, {1536, 4096}, {1536, 6144}, {1536, 12288},
};
#define RS_MATMUL_SHAPES_N ((int)(sizeof(RS_MATMUL_SHAPES) / sizeof(RS_MATMUL_SHAPES[0])))

static const int RS_ROW_SPLIT_ROWS[] = {65540, 262148};

static const qtype_info *rs_qtype(uint32_t type) {
	for (int i = 0; i < QTYPES_N; i++)
		if (QTYPES[i].type == type)
			return &QTYPES[i];
	return NULL;
}

static buffer rs_scratch(backend *b, const float *src, int n) {
	buffer buf = {0};
	b->buffer_alloc_scratch(b, (size_t)n * sizeof(float), &buf);
	if (src)
		b->buffer_write_f32(b, &buf, src, n);
	return buf;
}

static buffer rs_weight(backend *b, const qtype_info *qt, const void *blocks, int n, int k) {
	tensor_desc wd = {.host_data = blocks, .type = qt->type, .n_dims = 2, .dims = {k, n}};
	buffer		w  = {0};
	b->buffer_alloc_weight(b, &wd, &w);
	return w;
}

static void rs_read(backend *b, const buffer *buf, float *dst, int n) {
	if (b->synchronize)
		b->synchronize(b);
	b->buffer_read_f32(b, buf, dst, n);
}

static void rs_free(backend *b, buffer **bufs, int n) {
	for (int i = 0; i < n; i++)
		b->buffer_free(b, bufs[i]);
}

static void rs_skip(const char *label, const char *why) {
	record_result(RS_FAM, label, V_SKIP, why);
}

static void run_rs_matmul_tests(backend *ref, backend *tgt) {
	for (int t = 0; t < RS_MATMUL_TYPES_N; t++) {
		const qtype_info *qt = rs_qtype(RS_MATMUL_TYPES[t]);
		if (!qt)
			continue;
		for (int sh = 0; sh < RS_MATMUL_SHAPES_N; sh++)
			test_op_matmul(ref, tgt, RS_FAM, qt, RS_MATMUL_SHAPES[sh][0], RS_MATMUL_SHAPES[sh][1]);
		int k_split = qt->block >= 256 ? 256 : 32;
		for (int i = 0; i < (int)(sizeof(RS_ROW_SPLIT_ROWS) / sizeof(RS_ROW_SPLIT_ROWS[0])); i++)
			test_op_matmul(ref, tgt, RS_FAM, qt, RS_ROW_SPLIT_ROWS[i], k_split);
	}
}

static status_code rs_embd_fill_rows(backend *b, const qtype_info *qt, const void *blocks,
									 int vocab, int dim, const int *tokens, int n_rows, float scale,
									 const float *junk, float *out) {
	int			n  = n_rows * dim;
	buffer		w  = rs_weight(b, qt, blocks, vocab, dim);
	buffer		xs = rs_scratch(b, junk, n);
	status_code s  = b->scale_inplace(b, &xs, 2.0f, n);
	for (int row = 0; row < n_rows && s == OK; row++) {
		buffer xrow =
			buffer_slice(&xs, (size_t)row * dim * sizeof(float), (size_t)dim * sizeof(float));
		s = b->embd_lookup(b, &w, qt->type, tokens[row], dim, &xrow);
	}
	if (s == OK)
		s = b->scale_inplace(b, &xs, scale, n);
	rs_read(b, &xs, out, n);
	buffer *all[] = {&w, &xs};
	rs_free(b, all, 2);
	return s;
}

static void test_rs_embd_fill_rows(backend *ref, backend *tgt, uint32_t type, int dim, int n_rows,
								   float scale) {
	const qtype_info *qt = rs_qtype(type);
	char			  label[128];
	snprintf(label, sizeof(label), "embd rows into slices then scale %s dim=%d rows=%d", qt->name,
			 dim, n_rows);
	if (!tgt->embd_lookup || !tgt->scale_inplace || dim % qt->block != 0) {
		rs_skip(label, "backend lacks embd_lookup/scale_inplace");
		return;
	}
	const int vocab = 64;
	int		  n		= n_rows * dim;
	seed_test_rng(0xE3BDULL + (uint64_t)type * 131 + (uint64_t)dim);
	void  *blocks = test_make_weight(ref, qt, vocab, dim, NULL);
	float *junk	  = xmalloc((size_t)n * sizeof(float));
	float *y_ref  = xmalloc((size_t)n * sizeof(float));
	float *y_got  = xmalloc((size_t)n * sizeof(float));
	int	   tokens[16];
	fill_random_f32(junk, n, 1.0f);
	for (int i = 0; i < n_rows; i++)
		tokens[i] = (int)(next_u32() % (uint32_t)vocab);
	rs_embd_fill_rows(ref, qt, blocks, vocab, dim, tokens, n_rows, scale, junk, y_ref);
	status_code s =
		rs_embd_fill_rows(tgt, qt, blocks, vocab, dim, tokens, n_rows, scale, junk, y_got);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, n, s);
	free(blocks);
	free(junk);
	free(y_ref);
	free(y_got);
}

static void test_rs_ffn_inplace(backend *ref, backend *tgt, int n, int activation) {
	char label[128];
	snprintf(label, sizeof(label), "ffn_activate_ex in-place gate==out %s N=%d",
			 activation == 1 ? "GELU" : "SiLU", n);
	if (!tgt->ffn_activate_ex) {
		rs_skip(label, "backend has no native ffn_activate_ex");
		return;
	}
	float *g	 = xmalloc((size_t)n * sizeof(float));
	float *u	 = xmalloc((size_t)n * sizeof(float));
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));
	seed_test_rng(0x1A9ULL + (uint64_t)n * 7 + (uint64_t)activation);
	fill_random_f32(g, n, 4.0f);
	fill_random_f32(u, n, 4.0f);

	buffer g_ref = rs_scratch(ref, g, n);
	buffer u_ref = rs_scratch(ref, u, n);
	buffer o_ref = rs_scratch(ref, NULL, n);
	ref->ffn_activate_ex(ref, &g_ref, &u_ref, &o_ref, n, activation);
	rs_read(ref, &o_ref, y_ref, n);

	buffer		g_tgt = rs_scratch(tgt, g, n);
	buffer		u_tgt = rs_scratch(tgt, u, n);
	status_code s	  = tgt->ffn_activate_ex(tgt, &g_tgt, &u_tgt, &g_tgt, n, activation);
	rs_read(tgt, &g_tgt, y_got, n);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, n, s);

	buffer *rb[] = {&g_ref, &u_ref, &o_ref};
	buffer *tb[] = {&g_tgt, &u_tgt};
	rs_free(ref, rb, 3);
	rs_free(tgt, tb, 2);
	free(g);
	free(u);
	free(y_ref);
	free(y_got);
}

static status_code rs_ple_chain(backend *b, const qtype_info *qt, const void *gate_blocks,
								const void *proj_blocks, const float *norm_w, const float *x,
								const float *slice, float *out) {
	buffer gate_w	= rs_weight(b, qt, gate_blocks, RS_PLE, RS_DIM);
	buffer proj_w	= rs_weight(b, qt, proj_blocks, RS_DIM, RS_PLE);
	buffer xb		= rs_scratch(b, x, RS_DIM);
	buffer slice_b	= rs_scratch(b, slice, RS_PLE);
	buffer inp		= rs_scratch(b, NULL, RS_PLE);
	buffer proj_out = rs_scratch(b, NULL, RS_DIM);
	buffer norm_b	= rs_scratch(b, norm_w, RS_DIM);

	status_code s = b->matmul(b, &gate_w, qt->type, &xb, &inp, RS_PLE, RS_DIM);
	if (s == OK)
		s = b->ffn_activate_ex(b, &inp, &slice_b, &inp, RS_PLE, 1);
	if (s == OK)
		s = b->matmul(b, &proj_w, qt->type, &inp, &proj_out, RS_DIM, RS_PLE);
	if (s == OK)
		s = b->rmsnorm_add(b, &proj_out, &norm_b, &xb, &xb, RS_DIM, 1e-6f, 0.37f);
	rs_read(b, &xb, out, RS_DIM);
	buffer *all[] = {&gate_w, &proj_w, &xb, &slice_b, &inp, &proj_out, &norm_b};
	rs_free(b, all, 7);
	return s;
}

static void test_rs_ple_chain(backend *ref, backend *tgt) {
	const char label[] = "ple inject chain f32 gate/proj + in-place GELU + rmsnorm_add scale";
	if (!tgt->matmul || !tgt->ffn_activate_ex || !tgt->rmsnorm_add ||
		!backend_has_cap(tgt, BCAP_RMSNORM_ADD)) {
		rs_skip(label, "backend lacks matmul/ffn_activate_ex/rmsnorm_add");
		return;
	}
	const qtype_info *qt = rs_qtype(GGML_TYPE_F32);
	seed_test_rng(0x9133ULL);
	void  *gate_blocks = test_make_weight(ref, qt, RS_PLE, RS_DIM, NULL);
	void  *proj_blocks = test_make_weight(ref, qt, RS_DIM, RS_PLE, NULL);
	float *norm_w	   = xmalloc(RS_DIM * sizeof(float));
	float *x		   = xmalloc(RS_DIM * sizeof(float));
	float *slice	   = xmalloc(RS_PLE * sizeof(float));
	float *y_ref	   = xmalloc(RS_DIM * sizeof(float));
	float *y_got	   = xmalloc(RS_DIM * sizeof(float));
	fill_random_f32(norm_w, RS_DIM, 1.0f);
	fill_random_f32(x, RS_DIM, 1.0f);
	fill_random_f32(slice, RS_PLE, 1.0f);
	rs_ple_chain(ref, qt, gate_blocks, proj_blocks, norm_w, x, slice, y_ref);
	status_code s = rs_ple_chain(tgt, qt, gate_blocks, proj_blocks, norm_w, x, slice, y_got);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, RS_DIM, s);
	free(gate_blocks);
	free(proj_blocks);
	free(norm_w);
	free(x);
	free(slice);
	free(y_ref);
	free(y_got);
}

static status_code rs_ffn_chain(backend *b, const qtype_info *qt, const void *gate_blocks,
								const void *up_blocks, const void *down_blocks, int inter, int m,
								const float *x, float *out) {
	size_t inter_bytes = (size_t)inter * m * sizeof(float);
	buffer gate_w	   = rs_weight(b, qt, gate_blocks, inter, RS_DIM);
	buffer up_w		   = rs_weight(b, qt, up_blocks, inter, RS_DIM);
	buffer down_w	   = rs_weight(b, qt, down_blocks, RS_DIM, inter);
	buffer xb		   = rs_scratch(b, x, m * RS_DIM);
	buffer fused	   = rs_scratch(b, NULL, 2 * inter * m);
	buffer act		   = rs_scratch(b, NULL, inter * m);
	buffer y		   = rs_scratch(b, NULL, RS_DIM * m);
	buffer gate		   = buffer_slice(&fused, 0, inter_bytes);
	buffer up		   = buffer_slice(&fused, inter_bytes, inter_bytes);

	status_code s = OK;
	if (m > 1) {
		s = b->matmul_batch(b, &gate_w, qt->type, &xb, &gate, inter, RS_DIM, m);
		if (s == OK)
			s = b->matmul_batch(b, &up_w, qt->type, &xb, &up, inter, RS_DIM, m);
		if (s == OK)
			s = b->ffn_activate_batch(b, &gate, &up, &act, inter, 1, m);
		if (s == OK)
			s = b->matmul_batch(b, &down_w, qt->type, &act, &y, RS_DIM, inter, m);
	} else {
		if (b->matmul_multi && backend_has_cap(b, BCAP_MULTI_MATMUL)) {
			const buffer *ws[2]	   = {&gate_w, &up_w};
			uint32_t	  types[2] = {qt->type, qt->type};
			buffer		 *ys[2]	   = {&gate, &up};
			int			  ns[2]	   = {inter, inter};
			s					   = b->matmul_multi(b, ws, types, &xb, ys, ns, RS_DIM, 2);
		} else {
			s = b->matmul(b, &gate_w, qt->type, &xb, &gate, inter, RS_DIM);
			if (s == OK)
				s = b->matmul(b, &up_w, qt->type, &xb, &up, inter, RS_DIM);
		}
		if (s == OK)
			s = b->ffn_activate_ex(b, &gate, &up, &act, inter, 1);
		if (s == OK)
			s = b->matmul(b, &down_w, qt->type, &act, &y, RS_DIM, inter);
	}
	rs_read(b, &y, out, RS_DIM * m);
	buffer *all[] = {&gate_w, &up_w, &down_w, &xb, &fused, &act, &y};
	rs_free(b, all, 7);
	return s;
}

static void test_rs_ffn_chain(backend *ref, backend *tgt, int inter, int m) {
	char label[128];
	snprintf(label, sizeof(label),
			 "ffn chain iq4_nl gate/up fused slot + GELU + down inter=%d m=%d", inter, m);
	const qtype_info *qt = rs_qtype(GGML_TYPE_IQ4_NL);
	if (!tgt->matmul || !tgt->ffn_activate_ex ||
		(m > 1 && (!tgt->matmul_batch || !tgt->ffn_activate_batch)) ||
		(tgt->matmul_type_native && !tgt->matmul_type_native(tgt, qt->type))) {
		rs_skip(label, "backend lacks required native ops");
		return;
	}
	seed_test_rng(0xF1F1ULL + (uint64_t)inter * 13 + (uint64_t)m);
	void  *gate_blocks = test_make_weight(ref, qt, inter, RS_DIM, NULL);
	void  *up_blocks   = test_make_weight(ref, qt, inter, RS_DIM, NULL);
	void  *down_blocks = test_make_weight(ref, qt, RS_DIM, inter, NULL);
	float *x		   = xmalloc((size_t)m * RS_DIM * sizeof(float));
	float *y_ref	   = xmalloc((size_t)m * RS_DIM * sizeof(float));
	float *y_got	   = xmalloc((size_t)m * RS_DIM * sizeof(float));
	fill_random_f32(x, m * RS_DIM, 1.0f);
	rs_ffn_chain(ref, qt, gate_blocks, up_blocks, down_blocks, inter, m, x, y_ref);
	status_code s = rs_ffn_chain(tgt, qt, gate_blocks, up_blocks, down_blocks, inter, m, x, y_got);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, m * RS_DIM, s);
	free(gate_blocks);
	free(up_blocks);
	free(down_blocks);
	free(x);
	free(y_ref);
	free(y_got);
}

static status_code rs_qkv(backend *b, int fused, const qtype_info *const qt[3],
						  const void *const blocks[3], const int n[3], int k, const float *x,
						  float *out) {
	buffer		  w[3], y[3];
	const buffer *wp[3];
	buffer		 *yp[3];
	uint32_t	  types[3];
	for (int i = 0; i < 3; i++) {
		w[i]	 = rs_weight(b, qt[i], blocks[i], n[i], k);
		y[i]	 = rs_scratch(b, NULL, n[i]);
		wp[i]	 = &w[i];
		yp[i]	 = &y[i];
		types[i] = qt[i]->type;
	}
	buffer		xb = rs_scratch(b, x, k);
	status_code s  = OK;
	if (fused)
		s = b->matmul_multi(b, wp, types, &xb, yp, n, k, 3);
	for (int i = 0; i < 3 && !fused && s == OK; i++)
		s = b->matmul(b, &w[i], types[i], &xb, &y[i], n[i], k);
	int off = 0;
	for (int i = 0; i < 3; i++) {
		rs_read(b, &y[i], out + off, n[i]);
		off += n[i];
	}
	for (int i = 0; i < 3; i++) {
		buffer *pair[] = {&w[i], &y[i]};
		rs_free(b, pair, 2);
	}
	b->buffer_free(b, &xb);
	return s;
}

static void test_rs_qkv_multi(backend *ref, backend *tgt, int n_q, int n_kv, int k) {
	char label[160];
	snprintf(label, sizeof(label), "matmul_multi q=iq4_nl k=iq4_nl v=q5_K N=%d/%d/%d K=%d", n_q,
			 n_kv, n_kv, k);
	const qtype_info *iq = rs_qtype(GGML_TYPE_IQ4_NL);
	const qtype_info *q5 = rs_qtype(GGML_TYPE_Q5_K);
	if (!tgt->matmul_multi || !backend_has_cap(tgt, BCAP_MULTI_MATMUL) ||
		(tgt->matmul_type_native &&
		 (!tgt->matmul_type_native(tgt, iq->type) || !tgt->matmul_type_native(tgt, q5->type)))) {
		rs_skip(label, "backend lacks native matmul_multi for these types");
		return;
	}
	const qtype_info *qt[3] = {iq, iq, q5};
	const int		  n[3]	= {n_q, n_kv, n_kv};
	int				  total = n_q + 2 * n_kv;
	seed_test_rng(0x9C7ULL + (uint64_t)n_q * 3 + (uint64_t)k);
	void *blocks[3];
	for (int i = 0; i < 3; i++)
		blocks[i] = test_make_weight(ref, qt[i], n[i], k, NULL);
	float *x	 = xmalloc((size_t)k * sizeof(float));
	float *y_ref = xmalloc((size_t)total * sizeof(float));
	float *y_got = xmalloc((size_t)total * sizeof(float));
	fill_random_f32(x, k, 1.0f);
	rs_qkv(ref, 0, qt, (const void *const *)blocks, n, k, x, y_ref);
	status_code s = rs_qkv(tgt, 1, qt, (const void *const *)blocks, n, k, x, y_got);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, total, s);
	for (int i = 0; i < 3; i++)
		free(blocks[i]);
	free(x);
	free(y_ref);
	free(y_got);
}

static status_code rs_unary(backend *b, int softcap, int n, float param, const float *x,
							float *out) {
	buffer		buf = rs_scratch(b, x, n);
	status_code s	= softcap ? b->softcap(b, &buf, param, n) : b->scale_inplace(b, &buf, param, n);
	rs_read(b, &buf, out, n);
	b->buffer_free(b, &buf);
	return s;
}

static void test_rs_unary(backend *ref, backend *tgt, int softcap, int n, float param,
						  float magnitude) {
	char label[128];
	snprintf(label, sizeof(label), "%s param=%g N=%d", softcap ? "softcap" : "scale_inplace", param,
			 n);
	if ((softcap && !tgt->softcap) || (!softcap && !tgt->scale_inplace)) {
		rs_skip(label, "backend lacks native op");
		return;
	}
	seed_test_rng(0x50F7ULL + (uint64_t)n + (uint64_t)softcap);
	float *x	 = xmalloc((size_t)n * sizeof(float));
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	float *y_got = xmalloc((size_t)n * sizeof(float));
	fill_random_f32(x, n, magnitude);
	rs_unary(ref, softcap, n, param, x, y_ref);
	status_code s = rs_unary(tgt, softcap, n, param, x, y_got);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, n, s);
	free(x);
	free(y_ref);
	free(y_got);
}

static status_code rs_copy_slices(backend *b, const float *dst_init, const float *src_init,
								  int total, int dst_off, int src_off, int len, float *out) {
	buffer		dst = rs_scratch(b, dst_init, total);
	buffer		src = rs_scratch(b, src_init, total);
	status_code s	= b->scale_inplace(b, &dst, 2.0f, total);
	if (s == OK)
		s = b->scale_inplace(b, &src, 3.0f, total);
	buffer dv = buffer_slice(&dst, (size_t)dst_off * sizeof(float), (size_t)len * sizeof(float));
	buffer sv = buffer_slice(&src, (size_t)src_off * sizeof(float), (size_t)len * sizeof(float));
	if (s == OK)
		s = b->copy_buffer(b, &sv, &dv, len);
	if (s == OK)
		s = b->scale_inplace(b, &dst, 0.5f, total);
	rs_read(b, &dst, out, total);
	buffer *all[] = {&dst, &src};
	rs_free(b, all, 2);
	return s;
}

static void test_rs_copy_slices(backend *ref, backend *tgt, int total, int dst_off, int src_off,
								int len) {
	char label[128];
	snprintf(label, sizeof(label),
			 "copy_buffer slice into device-dirty buffer total=%d dst_off=%d src_off=%d len=%d",
			 total, dst_off, src_off, len);
	if (!tgt->copy_buffer || !tgt->scale_inplace) {
		rs_skip(label, "backend lacks copy_buffer/scale_inplace");
		return;
	}
	seed_test_rng(0xC0B7ULL + (uint64_t)total + (uint64_t)dst_off);
	float *d	 = xmalloc((size_t)total * sizeof(float));
	float *s	 = xmalloc((size_t)total * sizeof(float));
	float *y_ref = xmalloc((size_t)total * sizeof(float));
	float *y_got = xmalloc((size_t)total * sizeof(float));
	fill_random_f32(d, total, 1.0f);
	fill_random_f32(s, total, 1.0f);
	rs_copy_slices(ref, d, s, total, dst_off, src_off, len, y_ref);
	status_code st = rs_copy_slices(tgt, d, s, total, dst_off, src_off, len, y_got);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, total, st);
	free(d);
	free(s);
	free(y_ref);
	free(y_got);
}

enum { RS_KV_LAYERS = 2, RS_KV_HEADS_Q = 8, RS_KV_CTX = 64, RS_KV_HD_MAX = 512 };
static const int RS_KV_HD[RS_KV_LAYERS] = {256, 512};

static status_code rs_kv_mixed(backend *b, int n_kv, int n_steps, const float *kf, const float *vf,
							   const float *qf, float *out) {
	int			layer_hd[RS_KV_LAYERS]	= {RS_KV_HD[0], RS_KV_HD[1]};
	int			layer_kvh[RS_KV_LAYERS] = {n_kv, n_kv};
	int			host					= backend_has_cap(b, BCAP_IS_HOST);
	kv_desc		kvd						= {.n_layers		 = RS_KV_LAYERS,
										   .n_kv_layers		 = RS_KV_LAYERS,
										   .n_kv_heads		 = n_kv,
										   .head_dim		 = RS_KV_HD_MAX,
										   .n_ctx			 = RS_KV_CTX,
										   .kv_quant		 = KV_QUANT_F16,
										   .layer_head_dim	 = host ? layer_hd : NULL,
										   .layer_n_kv_heads = host ? layer_kvh : NULL};
	buffer		kc = {0}, vc = {0};
	status_code s	   = b->kv_alloc(b, &kvd, &kc, &vc);
	int			kv_row = n_kv * RS_KV_HD_MAX;
	int			q_row  = RS_KV_HEADS_Q * RS_KV_HD_MAX;
	buffer		ki	   = rs_scratch(b, NULL, kv_row);
	buffer		vi	   = rs_scratch(b, NULL, kv_row);
	buffer		qb	   = rs_scratch(b, NULL, q_row);
	buffer		ob	   = rs_scratch(b, NULL, q_row);
	for (int p = 0; p < n_steps && s == OK; p++) {
		for (int l = 0; l < RS_KV_LAYERS && s == OK; l++) {
			int	   hd  = RS_KV_HD[l];
			size_t row = (size_t)p * RS_KV_LAYERS + l;
			b->buffer_write_f32(b, &ki, kf + row * kv_row, n_kv * hd);
			b->buffer_write_f32(b, &vi, vf + row * kv_row, n_kv * hd);
			b->buffer_write_f32(b, &qb, qf + row * q_row, RS_KV_HEADS_Q * hd);
			s = b->kv_put(b, &kc, &vc, l, p, &ki, &vi, n_kv, hd, RS_KV_CTX, n_kv);
			if (s == OK && b->synchronize)
				b->synchronize(b);
			if (s == OK)
				s = b->attention(b, &qb, &kc, &vc, &ob, l, p, RS_KV_HEADS_Q, n_kv, hd, RS_KV_CTX, 1,
								 1.0f, n_kv);
			rs_read(b, &ob, out + row * q_row, RS_KV_HEADS_Q * hd);
		}
	}
	buffer *all[] = {&ki, &vi, &qb, &ob};
	rs_free(b, all, 4);
	b->kv_free(b, &kc, &vc);
	return s;
}

static void test_rs_kv_mixed(backend *ref, backend *tgt, int n_kv) {
	char label[128];
	snprintf(label, sizeof(label), "kv_put+attention mixed head_dim 256/512 layers n_kv=%d", n_kv);
	if (!tgt->kv_alloc || !tgt->kv_put || !tgt->attention) {
		rs_skip(label, "backend lacks kv_alloc/kv_put/attention");
		return;
	}
	const int n_steps = 6;
	size_t	  rows	  = (size_t)n_steps * RS_KV_LAYERS;
	size_t	  kv_n	  = rows * n_kv * RS_KV_HD_MAX;
	size_t	  q_n	  = rows * RS_KV_HEADS_Q * RS_KV_HD_MAX;
	seed_test_rng(0x4E55ULL + (uint64_t)n_kv);
	float *kf	 = xmalloc(kv_n * sizeof(float));
	float *vf	 = xmalloc(kv_n * sizeof(float));
	float *qf	 = xmalloc(q_n * sizeof(float));
	float *y_ref = xcalloc(q_n, sizeof(float));
	float *y_got = xcalloc(q_n, sizeof(float));
	fill_random_f32(kf, (int)kv_n, 1.0f);
	fill_random_f32(vf, (int)kv_n, 1.0f);
	fill_random_f32(qf, (int)q_n, 1.0f);
	rs_kv_mixed(ref, n_kv, n_steps, kf, vf, qf, y_ref);
	status_code s = rs_kv_mixed(tgt, n_kv, n_steps, kf, vf, qf, y_got);
	test_parity_compare_status(RS_FAM, label, y_ref, y_got, (int)q_n, s);
	free(kf);
	free(vf);
	free(qf);
	free(y_ref);
	free(y_got);
}

static void run_real_shape_tests(backend *ref, backend *tgt) {
	run_rs_matmul_tests(ref, tgt);

	test_rs_embd_fill_rows(ref, tgt, GGML_TYPE_Q4_K, 1536, 8, sqrtf(1536.0f));
	test_rs_embd_fill_rows(ref, tgt, GGML_TYPE_Q5_K, 8960, 4, 16.0f);
	test_rs_embd_fill_rows(ref, tgt, GGML_TYPE_IQ4_NL, 1536, 3, 1.0f);

	test_rs_ffn_inplace(ref, tgt, 256, 1);
	test_rs_ffn_inplace(ref, tgt, 12288, 1);
	test_rs_ple_chain(ref, tgt);

	test_rs_ffn_chain(ref, tgt, 6144, 1);
	test_rs_ffn_chain(ref, tgt, 12288, 1);
	test_rs_ffn_chain(ref, tgt, 6144, 8);
	test_rs_ffn_chain(ref, tgt, 12288, 8);

	test_rs_qkv_multi(ref, tgt, 2048, 256, RS_DIM);
	test_rs_qkv_multi(ref, tgt, 4096, 512, RS_DIM);

	test_rs_unary(ref, tgt, 1, 262144, 30.0f, 300.0f);
	test_rs_unary(ref, tgt, 1, 4096, 30.0f, 300.0f);
	test_rs_unary(ref, tgt, 0, 8 * RS_DIM, sqrtf(1536.0f), 1.0f);

	test_rs_copy_slices(ref, tgt, 4096, 1024, 2048, 1024);
	test_rs_copy_slices(ref, tgt, 8960, 256, 0, 256);

	test_rs_kv_mixed(ref, tgt, 1);
	test_rs_kv_mixed(ref, tgt, 2);
}

void run_per_op_tests(backend *ref, backend *tgt) {
	printf("\n========================================\n");
	printf("Per-op validation: %s  vs  %s (reference)\n", tgt->name, ref->name);
	printf("========================================\n");
	print_op_coverage(ref, tgt);

	int shapes[][2] = {{32, 32}, {128, 256}, {64, 512}, {17, 256}, {512, 512}};
	for (int sh = 0; sh < (int)(sizeof(shapes) / sizeof(shapes[0])); sh++) {
		int N = shapes[sh][0];
		int K = shapes[sh][1];
		for (int qi = 0; qi < QTYPES_N; qi++) {
			if (K % QTYPES[qi].block != 0)
				continue;
			test_op_matmul(ref, tgt, OPFAM_MATMUL, &QTYPES[qi], N, K);
		}
	}
	flush_family(OPFAM_MATMUL);

	for (int qi = 0; qi < QTYPES_N; qi++)
		test_op_embd_lookup(ref, tgt, &QTYPES[qi], QTYPES[qi].block * 4, 32);
	test_op_embd_lookup(ref, tgt, rs_qtype(GGML_TYPE_Q4_K), 1536, 64);
	test_op_embd_lookup(ref, tgt, rs_qtype(GGML_TYPE_Q5_K), 8960, 64);
	test_op_embd_lookup_f32(ref, tgt);
	flush_family(OPFAM_EMBD_LOOKUP);

	test_op_rmsnorm(ref, tgt, 256);
	test_op_rmsnorm(ref, tgt, 1536);
	test_op_rmsnorm(ref, tgt, 2048);
	test_op_rmsnorm(ref, tgt, 4096);
	flush_family(OPFAM_RMSNORM);

	test_op_rmsnorm_per_head(ref, tgt, 8, 64);
	test_op_rmsnorm_per_head(ref, tgt, 4, 64);
	test_op_rmsnorm_per_head(ref, tgt, 32, 128);
	test_op_rmsnorm_per_head(ref, tgt, 8, 256);
	test_op_rmsnorm_per_head(ref, tgt, 8, 512);
	test_op_rmsnorm_per_head(ref, tgt, 1, 256);
	test_op_rmsnorm_per_head(ref, tgt, 1, 512);
	flush_family(OPFAM_RMSNORM_PER_HEAD);

	test_op_rmsnorm_noweight(ref, tgt, 64);
	test_op_rmsnorm_noweight(ref, tgt, 128);
	test_op_rmsnorm_noweight(ref, tgt, 256);
	test_op_rmsnorm_noweight(ref, tgt, 512);
	flush_family(OPFAM_RMSNORM_NOWEIGHT);

	test_op_rope(ref, tgt, 16, 64, 0);
	test_op_rope(ref, tgt, 16, 64, 1);
	test_op_rope(ref, tgt, 16, 64, 127);
	test_op_rope(ref, tgt, 32, 128, 50);
	flush_family(OPFAM_ROPE);

	test_op_rope_ext(ref, tgt, 8, 64, 0, 0);
	test_op_rope_ext(ref, tgt, 8, 64, 1, 0);
	test_op_rope_ext(ref, tgt, 8, 64, 127, 0);
	test_op_rope_ext(ref, tgt, 8, 64, 0, 1);
	test_op_rope_ext(ref, tgt, 8, 64, 127, 1);
	test_op_rope_ext(ref, tgt, 32, 128, 511, 1);
	test_op_rope_ext(ref, tgt, 8, 512, 0, 1);
	test_op_rope_ext(ref, tgt, 8, 512, 777, 1);
	test_op_rope_ext(ref, tgt, 1, 512, 300, 1);
	test_op_rope_ext(ref, tgt, 8, 256, 300, 0);
	test_op_rope_ext(ref, tgt, 1, 256, 1000, 0);
	flush_family(OPFAM_ROPE_EXT);

	test_op_add_inplace(ref, tgt, 256);
	test_op_add_inplace(ref, tgt, 1536);
	test_op_add_inplace(ref, tgt, 4096);
	flush_family(OPFAM_ADD_INPLACE);

	test_op_ple_combine(ref, tgt, 256);
	test_op_ple_combine(ref, tgt, 4096);
	test_op_ple_combine(ref, tgt, 8960);
	flush_family(OPFAM_PLE_COMBINE);

	test_op_rmsnorm_add(ref, tgt, 256);
	test_op_rmsnorm_add(ref, tgt, 1536);
	test_op_rmsnorm_add(ref, tgt, 4096);
	flush_family(OPFAM_RMSNORM_ADD);

	test_op_matmul_ffn_down_batch(ref, tgt, 128, 64);
	flush_family(OPFAM_MATMUL_FFN_DOWN);

	test_op_ffn_activate(ref, tgt, 256);
	test_op_ffn_activate(ref, tgt, 4096);
	flush_family(OPFAM_FFN_ACTIVATE);

	test_op_ffn_activate_ex(ref, tgt, 256, 1);
	test_op_ffn_activate_ex(ref, tgt, 4096, 1);
	test_op_ffn_activate_ex(ref, tgt, 256, 0);
	test_op_ffn_activate_ex(ref, tgt, 4096, 0);
	test_op_ffn_activate_ex(ref, tgt, 6144, 1);
	test_op_ffn_activate_ex(ref, tgt, 12288, 1);
	flush_family(OPFAM_FFN_ACTIVATE_EX);

	test_op_attention(ref, tgt, 8, 4, 64, 1024, 0, 0);
	test_op_attention(ref, tgt, 8, 4, 64, 1024, 1, 1);
	test_op_attention(ref, tgt, 8, 4, 64, 1024, 127, 0);
	test_op_attention(ref, tgt, 8, 4, 64, 1024, 127, 1);
	test_op_attention(ref, tgt, 8, 8, 64, 1024, 63, 1);
	test_op_attention(ref, tgt, 32, 8, 128, 2048, 511, 0);
	test_op_attention(ref, tgt, 32, 8, 128, 2048, 511, 1);
	test_op_attention(ref, tgt, 8, 1, 256, 2048, 0, 0);
	test_op_attention(ref, tgt, 8, 1, 256, 2048, 0, 1);
	test_op_attention(ref, tgt, 8, 1, 256, 2048, 1023, 0);
	test_op_attention(ref, tgt, 8, 1, 256, 2048, 1023, 1);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 0, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 0, 1);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 1, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 2, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 3, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 4, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 8, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 16, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 63, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 511, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 1023, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 2048, 1023, 1);
	test_op_attention(ref, tgt, 8, 1, 512, 64, 1, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 64, 3, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 64, 16, 0);
	test_op_attention(ref, tgt, 8, 1, 512, 64, 63, 0);
	flush_family(OPFAM_ATTENTION);

	test_op_attention_swa(ref, tgt, 8, 4, 64, 1024, 63, 32, 0);
	test_op_attention_swa(ref, tgt, 8, 4, 64, 1024, 63, 32, 1);
	test_op_attention_swa(ref, tgt, 8, 4, 64, 1024, 511, 128, 0);
	test_op_attention_swa(ref, tgt, 8, 4, 64, 1024, 511, 128, 1);
	test_op_attention_swa(ref, tgt, 8, 8, 64, 1024, 20, 32, 0);
	test_op_attention_swa(ref, tgt, 32, 8, 128, 2048, 1000, 512, 1);
	test_op_attention_swa(ref, tgt, 8, 1, 256, 2048, 511, 512, 1);
	test_op_attention_swa(ref, tgt, 8, 1, 256, 2048, 512, 512, 1);
	test_op_attention_swa(ref, tgt, 8, 1, 256, 2048, 1000, 512, 1);
	test_op_attention_swa(ref, tgt, 8, 1, 256, 2048, 1000, 512, 0);
	test_op_attention_swa_slide(ref, 4, 2, 64, 192, 160, 1000, 0, 0, 0, KV_QUANT_F16, NULL, NULL,
								1);
	test_op_attention_swa_slide(ref, 4, 2, 64, 192, 160, 32, 0, 0, 0, KV_QUANT_F16, NULL, NULL, 1);
	test_op_attention_swa_slide(ref, 4, 2, 64, 192, 160, 32, 1, 0, 0, KV_QUANT_F16, NULL, NULL, 1);
	test_op_attention_swa_slide(ref, 4, 2, 64, 192, 160, 32, 0, 48, 0, KV_QUANT_F16, NULL, NULL, 1);
	test_op_attention_swa_slide(ref, 8, 2, 128, 256, 200, 64, 1, 0, 0, KV_QUANT_F16, NULL, NULL, 1);
	test_op_attention_swa_slide(ref, 8, 2, 128, 256, 200, 64, 1, 64, 0, KV_QUANT_F16, NULL, NULL,
								1);
	test_op_attention_swa_slide(ref, 8, 8, 64, 160, 150, 24, 1, 40, 0, KV_QUANT_F16, NULL, NULL, 1);
	test_op_attention_swa_compact_parity(ref, KV_QUANT_F16, 8, 2, 128, 320, 240, 64, 0, 64);
	test_op_attention_swa_compact_parity(ref, KV_QUANT_F16, 8, 2, 128, 384, 300, 128, 1, 64);
	test_op_attention_swa_compact_parity(ref, KV_QUANT_F16, 4, 2, 64, 256, 200, 32, 1, 0);
	test_op_attention_swa_compact_parity(ref, KV_QUANT_Q8_0, 8, 2, 128, 320, 240, 64, 0, 64);
	test_op_attention_swa_compact_parity(tgt, KV_QUANT_F16, 8, 2, 128, 320, 240, 64, 0, 64);
	test_op_attention_swa_slide(tgt, 8, 2, 128, 256, 200, 64, 1, 64, 1, KV_QUANT_F16, NULL, NULL,
								1);
	flush_family(OPFAM_ATTENTION_SWA);

	test_op_attention_mla(ref, tgt, 2, 16, 8, 8, 16, 16, 64, 31);
	test_op_attention_mla(ref, tgt, 4, 40, 16, 24, 16, 32, 128, 47);
	test_op_attention_mla(ref, tgt, 8, 192, 64, 128, 64, 512, 256, 63);
	test_op_attention_mla(ref, tgt, 8, 64, 32, 32, 64, 128, 35, 18);
	flush_family(OPFAM_ATTENTION_MLA);

	test_op_kv_put(ref, tgt, 4, 64, 1024, 0);
	test_op_kv_put(ref, tgt, 4, 64, 1024, 127);
	test_op_kv_put(ref, tgt, 8, 128, 2048, 511);
	test_op_kv_put_batch(ref, tgt, 4, 64, 1024, 0, 7);
	test_op_kv_put_batch(ref, tgt, 8, 128, 2048, 511, 33);
	test_model_kv_size_shared();
	test_kv_size_matches_alloc(ref, 0);
	test_kv_size_matches_alloc(ref, 1);
	flush_family(OPFAM_KV_PUT);

	run_kv_quant_parity_tests(ref, tgt);

	test_op_argmax(ref, tgt, 256);
	test_op_argmax(ref, tgt, 4096);
	test_op_argmax(ref, tgt, 32000);
	test_op_argmax(ref, tgt, 262144);
	flush_family(OPFAM_ARGMAX);

	for (int qi = 0; qi < QTYPES_N; qi++) {
		test_quant_determinism(ref, &QTYPES[qi]);
		test_quant_finiteness(ref, &QTYPES[qi]);
	}
	test_quant_q8_0_roundtrip(ref);
	flush_family(OPFAM_QUANT);

	for (int qi = 0; qi < QTYPES_N; qi++)
		test_dequant_parity_cross(ref, tgt, &QTYPES[qi], QTYPES[qi].block * 4, 8);
	flush_family(OPFAM_DEQUANT_PARITY);

	run_repack_parity_tests(ref, tgt);
	flush_family(OPFAM_REPACK_PARITY);

	for (int qi = 0; qi < QTYPES_N; qi++) {
		if (256 % QTYPES[qi].block != 0)
			continue;
		test_op_matmul_residual(ref, tgt, &QTYPES[qi], 64, 256);
	}
	flush_family(OPFAM_MATMUL_RESIDUAL);

	test_op_rope_qk(ref, tgt, 8, 4, 64, 0);
	test_op_rope_qk(ref, tgt, 8, 4, 64, 1);
	test_op_rope_qk(ref, tgt, 8, 4, 64, 127);
	test_op_rope_qk(ref, tgt, 32, 8, 128, 511);
	flush_family(OPFAM_ROPE_QK);

	for (int qi = 0; qi < QTYPES_N; qi++) {
		if (64 % QTYPES[qi].block != 0)
			continue;
		test_batch_matmul_parity(ref, tgt, &QTYPES[qi], 64, 256, 4);
	}
	for (int qi = 0; qi < QTYPES_N; qi++) {
		if (512 % QTYPES[qi].block != 0)
			continue;
		test_batch_matmul_parity(ref, tgt, &QTYPES[qi], 512, 512, 2);
	}
	{
		int bat_sh[][6] = {
			{8, 4, 64, 256, 60, 2},	 {8, 4, 64, 256, 60, 10},	{8, 2, 128, 512, 100, 8},
			{8, 4, 64, 256, 0, 16},	 {8, 4, 64, 64, 0, 16},		{8, 2, 128, 256, 0, 16},
			{8, 1, 256, 256, 0, 16}, {32, 8, 128, 512, 100, 8}, {8, 4, 64, 256, 40, 16},
			{8, 1, 512, 256, 0, 16}, {8, 1, 512, 256, 100, 8},	{8, 1, 256, 600, 520, 8},
		};
		for (int i = 0; i < (int)(sizeof(bat_sh) / sizeof(bat_sh[0])); i++)
			for (int fl = 0; fl < 2; fl++)
				test_batch_attention_parity(ref, tgt, bat_sh[i][0], bat_sh[i][1], bat_sh[i][2],
											bat_sh[i][3], bat_sh[i][4], bat_sh[i][5], fl);
	}
	for (int t = 0; t < RS_MATMUL_TYPES_N; t++) {
		const qtype_info *qt = rs_qtype(RS_MATMUL_TYPES[t]);
		if (!qt)
			continue;
		test_batch_matmul_parity(ref, tgt, qt, 2048, 1536, 8);
		test_batch_matmul_parity(ref, tgt, qt, 1536, 6144, 8);
		test_batch_matmul_parity(ref, tgt, qt, 1536, 12288, 8);
	}
	test_batch_rope_parity(ref, tgt, 8, 64, 60, 2);
	test_batch_rope_parity(ref, tgt, 8, 64, 60, 8);
	test_batch_rope_parity(ref, tgt, 16, 128, 200, 5);
	flush_family(OPFAM_BATCH_PARITY);

	test_edge_rmsnorm_zeros(ref, tgt);
	test_edge_determinism(ref, tgt);
	test_edge_argmax_all_equal(ref, tgt);
	test_edge_ffn_activate_extremes(ref, tgt);
	test_edge_rope_identity_table(ref, tgt);
	flush_family(OPFAM_EDGE_CASE);

	run_real_shape_tests(ref, tgt);
	flush_family(OPFAM_REAL_SHAPE);
}
