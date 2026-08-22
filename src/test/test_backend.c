#include "test_core.h"

static void test_op_matmul(backend *cpu, backend *tgt, const qtype_info *qt, int n, int k) {
	char label[128];
	if (!tgt->matmul || !tgt->buffer_alloc_weight) {
		snprintf(label, sizeof(label), "%s matmul N=%d K=%d (%s)", qt->name, n, k, tgt->name);
		record_result(OPFAM_MATMUL, label, V_SKIP, "backend has no native matmul");
		return;
	}
	if (k % qt->block != 0)
		return;
	if (tgt->matmul_type_native && !tgt->matmul_type_native(tgt, qt->type)) {
		snprintf(label, sizeof(label), "%s matmul N=%d K=%d (%s)", qt->name, n, k, tgt->name);
		record_result(OPFAM_MATMUL, label, V_SKIP, "missing native implementation");
		return;
	}

	seed_test_rng((0xA5A5ULL * (qt->type + 1) * 1000003ULL) + ((uint64_t)n * 31) + (uint64_t)k);
	void *blocks = test_make_weight(qt, n, k, NULL);

	float *x = xmalloc((size_t)k * sizeof(float));
	fill_random_f32(x, k, 1.0f);

	tensor_desc wd = {
		.host_data = blocks,
		.type	   = qt->type,
		.n_dims	   = 2,
		.dims	   = {k, n},
	};

	buffer w_cpu = {0};
	buffer x_cpu = {0};
	buffer y_cpu = {0};
	cpu->buffer_alloc_weight(cpu, &wd, &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)k * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, k);
	cpu->matmul(cpu, &w_cpu, qt->type, &x_cpu, &y_cpu, n, k);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &y_cpu, y_ref, n);

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
	record_result(OPFAM_MATMUL, label, v, detail);

	free(y_ref);
	free(y_got);
	free(x);
	free(blocks);
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_embd_lookup(backend *cpu, backend *tgt, const qtype_info *qt, int dim,
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
	void *blocks = test_make_weight(qt, vocab, dim, NULL);
	int	  token	 = (int)(next_u32() % (uint32_t)vocab);

	tensor_desc wd = {
		.host_data = blocks,
		.type	   = qt->type,
		.n_dims	   = 2,
		.dims	   = {dim, vocab},
	};

	buffer w_cpu   = {0};
	buffer out_cpu = {0};
	cpu->buffer_alloc_weight(cpu, &wd, &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)dim * sizeof(float), &out_cpu);
	status_code s_cpu;
	{
		s_cpu = cpu->embd_lookup(cpu, &w_cpu, qt->type, token, dim, &out_cpu);
	}
	float *y_ref = xmalloc((size_t)dim * sizeof(float));
	cpu->buffer_read_f32(cpu, &out_cpu, y_ref, dim);

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
	if (s_cpu != OK) {
		snprintf(detail, sizeof(detail), "cpu reference status=%d", s_cpu);
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
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &out_cpu);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &out_tgt);
}

static void test_op_embd_lookup_f32(backend *cpu, backend *tgt) {
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

	buffer w_cpu   = {0};
	buffer out_cpu = {0};
	cpu->buffer_alloc_weight(cpu, &wd, &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)dim * sizeof(float), &out_cpu);
	cpu->embd_lookup(cpu, &w_cpu, GGML_TYPE_F32, token, dim, &out_cpu);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)dim * sizeof(float));
	cpu->buffer_read_f32(cpu, &out_cpu, y_ref, dim);

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
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &out_cpu);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &out_tgt);
}

static void test_op_rmsnorm(backend *cpu, backend *tgt, int n) {
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

	buffer x_cpu = {0};
	buffer w_cpu = {0};
	buffer y_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, n);
	cpu->buffer_write_f32(cpu, &w_cpu, w, n);
	cpu->rmsnorm(cpu, &x_cpu, &w_cpu, &y_cpu, n, 1e-5f);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &y_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rmsnorm_per_head(backend *cpu, backend *tgt, int n_heads, int head_dim) {
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

	buffer x_cpu = {0};
	buffer w_cpu = {0};
	buffer y_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)N * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)head_dim * sizeof(float), &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)N * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, N);
	cpu->buffer_write_f32(cpu, &w_cpu, w, head_dim);
	cpu->rmsnorm_per_head(cpu, &x_cpu, &w_cpu, &y_cpu, n_heads, head_dim, 1e-5f);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	cpu->buffer_read_f32(cpu, &y_cpu, y_ref, N);

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
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rmsnorm_noweight(backend *cpu, backend *tgt, int n) {
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

	buffer x_cpu = {0};
	buffer y_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, n);
	cpu->rmsnorm_noweight(cpu, &x_cpu, &y_cpu, n, 1e-5f);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &y_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rope(backend *cpu, backend *tgt, int n_heads, int head_dim, int pos) {
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

	cpu->rope_neox = 0;
	tgt->rope_neox = 0;

	buffer kc_cpu = {0};
	buffer vc_cpu = {0};
	buffer v_cpu  = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)N * sizeof(float), &v_cpu);
	cpu->buffer_write_f32(cpu, &v_cpu, vec, N);
	cpu->rope(cpu, &v_cpu, n_heads, head_dim, pos, cos_v, sin_v);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	cpu->buffer_read_f32(cpu, &v_cpu, y_ref, N);

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
	cpu->buffer_free(cpu, &v_cpu);
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
	tgt->buffer_free(tgt, &v_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_rope_ext(backend *cpu, backend *tgt, int n_heads, int head_dim, int pos,
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

	cpu->rope_theta = 10000.0f;
	tgt->rope_theta = 10000.0f;
	cpu->rope_neox	= 0;
	tgt->rope_neox	= 0;

	kv_desc kvd = {.n_ctx		= n_ctx,
				   .n_kv_heads	= n_heads,
				   .head_dim	= head_dim,
				   .n_layers	= 1,
				   .n_kv_layers = 1};

	buffer kc_cpu = {0};
	buffer vc_cpu = {0};
	buffer v_cpu  = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)N * sizeof(float), &v_cpu);
	cpu->buffer_write_f32(cpu, &v_cpu, vec, N);
	cpu->rope_ext(cpu, &v_cpu, n_heads, head_dim, pos, cos_v, sin_v, ff);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	cpu->buffer_read_f32(cpu, &v_cpu, y_ref, N);

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
	cpu->buffer_free(cpu, &v_cpu);
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
	tgt->buffer_free(tgt, &v_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_add_inplace(backend *cpu, backend *tgt, int n) {
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

	buffer x_cpu = {0};
	buffer y_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, n);
	cpu->buffer_write_f32(cpu, &y_cpu, y, n);
	cpu->add_inplace(cpu, &x_cpu, &y_cpu, n);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *r_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &x_cpu, r_ref, n);

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
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_ple_combine(backend *cpu, backend *tgt, int n) {
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

	buffer ple_cpu	= {0};
	buffer proj_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &ple_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &proj_cpu);
	cpu->buffer_write_f32(cpu, &ple_cpu, ple, n);
	cpu->buffer_write_f32(cpu, &proj_cpu, proj, n);
	cpu->ple_combine(cpu, &ple_cpu, &proj_cpu, n, combine_scale);
	if (cpu->synchronize)
		cpu->synchronize(cpu);
	float *r_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &ple_cpu, r_ref, n);

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
	cpu->buffer_free(cpu, &ple_cpu);
	cpu->buffer_free(cpu, &proj_cpu);
	tgt->buffer_free(tgt, &ple_tgt);
	tgt->buffer_free(tgt, &proj_tgt);
}

static void test_op_rmsnorm_add(backend *cpu, backend *tgt, int n) {
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

	buffer x_cpu = {0}, w_cpu = {0}, r_cpu = {0}, y_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &r_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, n);
	cpu->buffer_write_f32(cpu, &w_cpu, w, n);
	cpu->buffer_write_f32(cpu, &r_cpu, residual, n);
	cpu->rmsnorm_add(cpu, &x_cpu, &w_cpu, &r_cpu, &y_cpu, n, eps);
	if (cpu->synchronize)
		cpu->synchronize(cpu);
	float *r_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &y_cpu, r_ref, n);

	buffer x_tgt = {0}, w_tgt = {0}, r_tgt = {0}, y_tgt = {0};
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &x_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &w_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &r_tgt);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &y_tgt);
	tgt->buffer_write_f32(tgt, &x_tgt, x, n);
	tgt->buffer_write_f32(tgt, &w_tgt, w, n);
	tgt->buffer_write_f32(tgt, &r_tgt, residual, n);
	status_code s_tgt = tgt->rmsnorm_add(tgt, &x_tgt, &w_tgt, &r_tgt, &y_tgt, n, eps);
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
	free(w);
	free(residual);
	free(r_ref);
	free(r_got);
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &r_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &r_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_ffn_activate(backend *cpu, backend *tgt, int n) {
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

	buffer g_cpu = {0};
	buffer u_cpu = {0};
	buffer o_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &g_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &u_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &o_cpu);
	cpu->buffer_write_f32(cpu, &g_cpu, g, n);
	cpu->buffer_write_f32(cpu, &u_cpu, u, n);
	cpu->ffn_activate(cpu, &g_cpu, &u_cpu, &o_cpu, n);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &o_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &g_cpu);
	cpu->buffer_free(cpu, &u_cpu);
	cpu->buffer_free(cpu, &o_cpu);
	tgt->buffer_free(tgt, &g_tgt);
	tgt->buffer_free(tgt, &u_tgt);
	tgt->buffer_free(tgt, &o_tgt);
}

static void test_op_ffn_activate_ex(backend *cpu, backend *tgt, int n, int activation) {
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

	buffer g_cpu = {0};
	buffer u_cpu = {0};
	buffer o_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &g_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &u_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &o_cpu);
	cpu->buffer_write_f32(cpu, &g_cpu, g, n);
	cpu->buffer_write_f32(cpu, &u_cpu, u, n);
	cpu->ffn_activate_ex(cpu, &g_cpu, &u_cpu, &o_cpu, n, activation);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &o_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &g_cpu);
	cpu->buffer_free(cpu, &u_cpu);
	cpu->buffer_free(cpu, &o_cpu);
	tgt->buffer_free(tgt, &g_tgt);
	tgt->buffer_free(tgt, &u_tgt);
	tgt->buffer_free(tgt, &o_tgt);
}

static void test_op_attention(backend *cpu, backend *tgt, int n_heads, int n_kv_heads, int head_dim,
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

	buffer kc_cpu  = {0};
	buffer vc_cpu  = {0};
	buffer ki_cpu  = {0};
	buffer vi_cpu  = {0};
	buffer q_cpu   = {0};
	buffer out_cpu = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &ki_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &vi_cpu);
	for (int t = 0; t < n_t; t++) {
		cpu->buffer_write_f32(cpu, &ki_cpu, k_all[t], n_kv);
		cpu->buffer_write_f32(cpu, &vi_cpu, v_all[t], n_kv);
		cpu->kv_put(cpu, &kc_cpu, &vc_cpu, 0, t, &ki_cpu, &vi_cpu, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
	}
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &q_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &out_cpu);
	cpu->buffer_write_f32(cpu, &q_cpu, q, n);
	cpu->attention(cpu, &q_cpu, &kc_cpu, &vc_cpu, &out_cpu, 0, pos, n_heads, n_kv_heads, head_dim,
				   n_ctx, 0, scale, n_kv_heads);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &out_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &ki_cpu);
	cpu->buffer_free(cpu, &vi_cpu);
	cpu->buffer_free(cpu, &q_cpu);
	cpu->buffer_free(cpu, &out_cpu);
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
	tgt->buffer_free(tgt, &ki_tgt);
	tgt->buffer_free(tgt, &vi_tgt);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &out_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_attention_swa(backend *cpu, backend *tgt, int n_heads, int n_kv_heads,
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

	buffer kc_cpu  = {0};
	buffer vc_cpu  = {0};
	buffer q_cpu   = {0};
	buffer out_cpu = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &q_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &out_cpu);
	cpu->buffer_write_f32(cpu, &q_cpu, q, n);

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
		buffer k_in_cpu = {0};
		buffer v_in_cpu = {0};
		cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &k_in_cpu);
		cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &v_in_cpu);
		cpu->buffer_write_f32(cpu, &k_in_cpu, kbuf, n_kv);
		cpu->buffer_write_f32(cpu, &v_in_cpu, vbuf, n_kv);
		cpu->kv_put(cpu, &kc_cpu, &vc_cpu, 0, p, &k_in_cpu, &v_in_cpu, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
		cpu->buffer_free(cpu, &k_in_cpu);
		cpu->buffer_free(cpu, &v_in_cpu);

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
	cpu->attention_swa(cpu, &q_cpu, &kc_cpu, &vc_cpu, &out_cpu, 0, pos, n_heads, n_kv_heads,
					   head_dim, n_ctx, 0, scale, sliding_window, n_kv_heads);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &out_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &q_cpu);
	cpu->buffer_free(cpu, &out_cpu);
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &out_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_kv_put(backend *cpu, backend *tgt, int n_kv_heads, int head_dim, int n_ctx,
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

		buffer kc_cpu = {0};
		buffer vc_cpu = {0};
		buffer ki_cpu = {0};
		buffer vi_cpu = {0};
		cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
		cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &ki_cpu);
		cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &vi_cpu);
		cpu->buffer_write_f32(cpu, &ki_cpu, k, n_kv);
		cpu->buffer_write_f32(cpu, &vi_cpu, v, n_kv);
		cpu->kv_put(cpu, &kc_cpu, &vc_cpu, 0, pos, &ki_cpu, &vi_cpu, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
		if (cpu && cpu->synchronize)
			cpu->synchronize(cpu);

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

		uint16_t *kc_cpu_p = kc_cpu.handle;
		uint16_t *vc_cpu_p = vc_cpu.handle;
		uint16_t *kc_tgt_p = kc_tgt.handle;
		uint16_t *vc_tgt_p = vc_tgt.handle;

		size_t kvh_stride = (size_t)n_ctx * head_dim;
		size_t pos_off	  = (size_t)pos * head_dim;
		for (int h = 0; h < n_kv_heads; h++) {
			size_t base = (h * kvh_stride) + pos_off;
			memcpy(k_ref + (h * head_dim), kc_cpu_p + base, (size_t)head_dim * sizeof(uint16_t));
			memcpy(v_ref + (h * head_dim), vc_cpu_p + base, (size_t)head_dim * sizeof(uint16_t));
			memcpy(k_got + (h * head_dim), kc_tgt_p + base, (size_t)head_dim * sizeof(uint16_t));
			memcpy(v_got + (h * head_dim), vc_tgt_p + base, (size_t)head_dim * sizeof(uint16_t));
		}

		float *k_ref_f = xmalloc((size_t)n_kv * sizeof(float));
		float *k_got_f = xmalloc((size_t)n_kv * sizeof(float));
		float *v_ref_f = xmalloc((size_t)n_kv * sizeof(float));
		float *v_got_f = xmalloc((size_t)n_kv * sizeof(float));
		for (int i = 0; i < n_kv; i++) {
			k_ref_f[i] = f16_to_f32(k_ref[i]);
			k_got_f[i] = f16_to_f32(k_got[i]);
			v_ref_f[i] = f16_to_f32(v_ref[i]);
			v_got_f[i] = f16_to_f32(v_got[i]);
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
		cpu->buffer_free(cpu, &ki_cpu);
		cpu->buffer_free(cpu, &vi_cpu);
		cpu->buffer_free(cpu, &kc_cpu);
		cpu->buffer_free(cpu, &vc_cpu);
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

	buffer kc_cpu  = {0};
	buffer vc_cpu  = {0};
	buffer ki_cpu  = {0};
	buffer vi_cpu  = {0};
	buffer q_cpu   = {0};
	buffer out_cpu = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &ki_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &vi_cpu);
	for (int t = 0; t < n_t; t++) {
		cpu->buffer_write_f32(cpu, &ki_cpu, k_all[t], n_kv);
		cpu->buffer_write_f32(cpu, &vi_cpu, v_all[t], n_kv);
		cpu->kv_put(cpu, &kc_cpu, &vc_cpu, 0, t, &ki_cpu, &vi_cpu, n_kv_heads, head_dim, n_ctx,
					n_kv_heads);
	}
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &q_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &out_cpu);
	cpu->buffer_write_f32(cpu, &q_cpu, q, n);
	cpu->attention(cpu, &q_cpu, &kc_cpu, &vc_cpu, &out_cpu, 0, pos, n_kv_heads, n_kv_heads,
				   head_dim, n_ctx, 0, scale, n_kv_heads);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &out_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &ki_cpu);
	cpu->buffer_free(cpu, &vi_cpu);
	cpu->buffer_free(cpu, &q_cpu);
	cpu->buffer_free(cpu, &out_cpu);
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
	tgt->buffer_free(tgt, &ki_tgt);
	tgt->buffer_free(tgt, &vi_tgt);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &out_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_op_kv_put_batch(backend *cpu, backend *tgt, int n_kv_heads, int head_dim,
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

	buffer kc_cpu = {0}, vc_cpu = {0}, ki_cpu = {0}, vi_cpu = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &ki_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n_kv * sizeof(float), &vi_cpu);
	for (int r = 0; r < m; r++) {
		cpu->buffer_write_f32(cpu, &ki_cpu, k_all + (size_t)r * n_kv, n_kv);
		cpu->buffer_write_f32(cpu, &vi_cpu, v_all + (size_t)r * n_kv, n_kv);
		cpu->kv_put(cpu, &kc_cpu, &vc_cpu, 0, pos_start + r, &ki_cpu, &vi_cpu, n_kv_heads, head_dim,
					n_ctx, n_kv_heads);
	}
	if (cpu->synchronize)
		cpu->synchronize(cpu);

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
		uint16_t *kc_cpu_p	 = kc_cpu.handle;
		uint16_t *vc_cpu_p	 = vc_cpu.handle;
		uint16_t *kc_tgt_p	 = kc_tgt.handle;
		uint16_t *vc_tgt_p	 = vc_tgt.handle;
		size_t	  kvh_stride = (size_t)n_ctx * head_dim;
		int		  fail		 = 0;
		for (int r = 0; r < m && !fail; r++) {
			for (int h = 0; h < n_kv_heads && !fail; h++) {
				size_t base = ((size_t)h * kvh_stride) + (size_t)(pos_start + r) * head_dim;
				for (int d = 0; d < head_dim; d++) {
					float kr = f16_to_f32(kc_cpu_p[base + d]);
					float kt = f16_to_f32(kc_tgt_p[base + d]);
					float vr = f16_to_f32(vc_cpu_p[base + d]);
					float vt = f16_to_f32(vc_tgt_p[base + d]);
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
		for (int r = 0; r < m; r++) {
			buffer row_k, row_v;
			tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &row_k);
			tgt->buffer_alloc_scratch(tgt, (size_t)n_kv * sizeof(float), &row_v);
			tgt->buffer_write_f32(tgt, &row_k, k_all + (size_t)r * n_kv, n_kv);
			tgt->buffer_write_f32(tgt, &row_v, v_all + (size_t)r * n_kv, n_kv);
			status_code prs = tgt->kv_put(tgt, &kc_ref, &vc_ref, 0, pos_start + r, &row_k, &row_v,
										  n_kv_heads, head_dim, n_ctx, n_kv_heads);
			if (prs != OK)
				s_tgt = prs;
			tgt->buffer_free(tgt, &row_k);
			tgt->buffer_free(tgt, &row_v);
		}
		if (tgt->synchronize)
			tgt->synchronize(tgt);

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
	cpu->buffer_free(cpu, &ki_cpu);
	cpu->buffer_free(cpu, &vi_cpu);
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
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
	b->attention(b, &q_ref_b, &kc_ref, &vc_ref, &out_ref, 0, pos, n_kv_heads, n_kv_heads, head_dim,
				 n_ctx, 0, scale, n_kv_heads);
	if (b && b->synchronize)
		b->synchronize(b);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	b->buffer_read_f32(b, &out_ref, y_ref, n);

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
	status_code s_got;
	{
		s_got = b->attention(b, &q_q8_b, &kc_q8, &vc_q8, &out_q8, 0, pos, n_kv_heads, n_kv_heads,
							 head_dim, n_ctx, 0, scale, n_kv_heads);
		if (b->synchronize)
			b->synchronize(b);
	}
	float *y_got = xmalloc((size_t)n * sizeof(float));
	b->buffer_read_f32(b, &out_q8, y_got, n);

	char	detail[256];
	verdict v = classify_output("kv_quant", y_ref, y_got, n, s_got, detail, sizeof(detail));
	if (v != V_PASS && v != V_LOSSY && v != V_SKIP)
		compute_debug(y_ref, y_got, n);
	int dl = (int)strlen(detail);
	snprintf(detail + dl, sizeof(detail) - dl,
			 " | kv cache quant parity: q8_0 attention output vs f16 reference");
	record_result(OPFAM_KV_QUANT_PARITY, label, v, detail);

	free(y_got);
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
	free(y_ref);
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
							   int hd, int n_pos, float scale) {
	float *ref = xmalloc((size_t)n_heads * hd * sizeof(float));
	memset(ref, 0, (size_t)n_heads * hd * sizeof(float));
	int			 n_groups = n_heads / kvh;
	static float sc[4096];
	for (int h = 0; h < n_heads; h++) {
		int			 hh	  = h / n_groups;
		const float *qh	  = q + (size_t)h * hd;
		float		 maxs = -INFINITY;
		for (int t = 0; t < n_pos; t++) {
			float		 s	= 0;
			const float *kt = k + (size_t)t * kvh * hd + (size_t)hh * hd;
			for (int d = 0; d < hd; d++)
				s += qh[d] * kt[d];
			sc[t] = scale * s;
			if (sc[t] > maxs)
				maxs = sc[t];
		}
		float sum = 0;
		for (int t = 0; t < n_pos; t++) {
			sc[t] = expf(sc[t] - maxs);
			sum += sc[t];
		}
		float *out = ref + (size_t)h * hd;
		for (int t = 0; t < n_pos; t++) {
			float		 w	= sc[t] / sum;
			const float *vt = v + (size_t)t * kvh * hd + (size_t)hh * hd;
			for (int d = 0; d < hd; d++)
				out[d] += w * vt[d];
		}
	}
	return ref;
}

static void test_op_kv_packed_layers(backend *b) {
	char label[128];
	snprintf(label, sizeof(label), "kv packed per-layer variable dims [%s]", b->name);
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
					.kv_quant		  = KV_QUANT_F16,
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

		float *ref	 = ref_softmax_attn(q, k_store, v_store, n_heads, kvh, hd, n_pos, scale);
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
	verdict v = worst_reld2 < 6e-3f ? V_PASS : (worst_reld2 < 3e-2f ? V_LOSSY : V_FAIL);
	record_result(OPFAM_KV_QUANT_PARITY, label, v, worst_d);
	if (v != V_PASS && v != V_LOSSY)
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

static void run_kv_quant_parity_tests(backend *cpu, backend *tgt) {
	test_op_kv_packed_layers(cpu);
	test_op_kv_quant_parity(cpu, 4, 64, 1024, 0);
	test_op_kv_quant_parity(cpu, 4, 64, 1024, 127);
	test_op_kv_quant_parity(cpu, 8, 128, 2048, 511);
	test_op_kv_quant_parity(cpu, 1, 256, 2048, 0);
	test_op_kv_quant_parity(cpu, 32, 128, 2048, 1023);
	test_op_kv_quant_parity(cpu, 4, 80, 512, 0);
	test_op_kv_quant_parity(cpu, 4, 80, 512, 63);
	test_op_kv_quant_parity(cpu, 2, 96, 512, 33);
	if (tgt && tgt != cpu) {
		test_op_kv_packed_layers(tgt);
		test_op_kv_quant_parity(tgt, 4, 64, 1024, 0);
		test_op_kv_quant_parity(tgt, 8, 128, 2048, 511);
	}
	flush_family(OPFAM_KV_QUANT_PARITY);
}

static void test_op_argmax(backend *cpu, backend *tgt, int n) {
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

	buffer l_cpu = {0};
	buffer l_tgt = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &l_cpu);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &l_tgt);
	cpu->buffer_write_f32(cpu, &l_cpu, logits, n);
	tgt->buffer_write_f32(tgt, &l_tgt, logits, n);

	int32_t idx_cpu = -1;
	int32_t idx_tgt = -1;
	cpu->argmax(cpu, &l_cpu, n, &idx_cpu);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
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
	} else if (idx_cpu == idx_tgt) {
		snprintf(detail, sizeof(detail), "idx=%d (match)", idx_cpu);
		record_result(OPFAM_ARGMAX, label, V_PASS, detail);
	} else {
		snprintf(detail, sizeof(detail), "idx_cpu=%d idx_tgt=%d (argmax diverged)", idx_cpu,
				 idx_tgt);
		record_result(OPFAM_ARGMAX, label, V_FAIL, detail);
	}

	free(logits);
	cpu->buffer_free(cpu, &l_cpu);
	tgt->buffer_free(tgt, &l_tgt);
}

static void test_op_matmul_residual(backend *cpu, backend *tgt, const qtype_info *qt, int n,
									int k) {
	if (!cpu->matmul_residual || !tgt->matmul_residual) {
		char label[128];
		snprintf(label, sizeof(label), "%s matmul_residual N=%d K=%d", qt->name, n, k);
		record_result(OPFAM_MATMUL_RESIDUAL, label, V_SKIP,
					  !cpu->matmul_residual ? "CPU reference has no matmul_residual"
											: "backend has no native matmul_residual");
		return;
	}
	if (k % qt->block != 0)
		return;
	if (tgt->matmul_type_native && !tgt->matmul_type_native(tgt, qt->type)) {
		char label[128];
		snprintf(label, sizeof(label), "%s matmul_residual N=%d K=%d (%s)", qt->name, n, k,
				 tgt->name);
		record_result(OPFAM_MATMUL_RESIDUAL, label, V_SKIP, "missing native implementation");
		return;
	}

	seed_test_rng((0xA5A5ULL * (qt->type + 1) * 1000003ULL) + ((uint64_t)n * 31) + (uint64_t)k +
				  0x1234);
	void *blocks = test_make_weight(qt, n, k, NULL);

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

	buffer w_cpu = {0};
	buffer x_cpu = {0};
	buffer r_cpu = {0};
	buffer y_cpu = {0};
	cpu->buffer_alloc_weight(cpu, &wd, &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)k * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &r_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, k);
	cpu->buffer_write_f32(cpu, &r_cpu, residual, n);
	cpu->matmul_residual(cpu, &w_cpu, qt->type, &x_cpu, &r_cpu, &y_cpu, n, k);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &y_cpu, y_ref, n);

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

	char label[128];
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
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &r_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &r_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_op_rope_qk(backend *cpu, backend *tgt, int n_heads, int n_kv_heads, int head_dim,
							int pos) {
	if (!cpu->rope_qk || !tgt->rope_qk) {
		char label[128];
		snprintf(label, sizeof(label), "rope_qk h=%d/%d d=%d pos=%d", n_heads, n_kv_heads, head_dim,
				 pos);
		record_result(OPFAM_ROPE_QK, label, V_SKIP,
					  !cpu->rope_qk ? "CPU reference has no rope_qk"
									: "backend has no native rope_qk");
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

	cpu->rope_neox = 0;
	tgt->rope_neox = 0;

	buffer kc_cpu = {0};
	buffer vc_cpu = {0};
	buffer q_cpu  = {0};
	buffer k_cpu  = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)nq * sizeof(float), &q_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)nk * sizeof(float), &k_cpu);
	cpu->buffer_write_f32(cpu, &q_cpu, q, nq);
	cpu->buffer_write_f32(cpu, &k_cpu, k, nk);
	cpu->rope_qk(cpu, &q_cpu, &k_cpu, n_heads, n_kv_heads, head_dim, pos, cos_v, sin_v);
	if (cpu && cpu->synchronize)
		cpu->synchronize(cpu);
	float *q_ref = xmalloc((size_t)nq * sizeof(float));
	float *k_ref = xmalloc((size_t)nk * sizeof(float));
	cpu->buffer_read_f32(cpu, &q_cpu, q_ref, nq);
	cpu->buffer_read_f32(cpu, &k_cpu, k_ref, nk);

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
	cpu->buffer_free(cpu, &q_cpu);
	cpu->buffer_free(cpu, &k_cpu);
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
	tgt->buffer_free(tgt, &q_tgt);
	tgt->buffer_free(tgt, &k_tgt);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
}

static void test_edge_rmsnorm_zeros(backend *cpu, backend *tgt) {
	if (!tgt->rmsnorm)
		return;
	int	   N = 256;
	float *x = xcalloc((size_t)N, sizeof(float));
	float *w = xmalloc((size_t)N * sizeof(float));
	for (int i = 0; i < N; i++)
		w[i] = 1.0f;

	buffer x_cpu = {0};
	buffer w_cpu = {0};
	buffer y_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)N * sizeof(float), &x_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)N * sizeof(float), &w_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)N * sizeof(float), &y_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, N);
	cpu->buffer_write_f32(cpu, &w_cpu, w, N);
	cpu->rmsnorm(cpu, &x_cpu, &w_cpu, &y_cpu, N, 1e-5f);
	float *y_ref = xmalloc((size_t)N * sizeof(float));
	cpu->buffer_read_f32(cpu, &y_cpu, y_ref, N);

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
	cpu->buffer_free(cpu, &x_cpu);
	cpu->buffer_free(cpu, &w_cpu);
	cpu->buffer_free(cpu, &y_cpu);
	tgt->buffer_free(tgt, &x_tgt);
	tgt->buffer_free(tgt, &w_tgt);
	tgt->buffer_free(tgt, &y_tgt);
}

static void test_edge_determinism(backend *cpu, backend *tgt) {
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
	(void)cpu;
}

static void test_edge_argmax_all_equal(backend *cpu, backend *tgt) {
	if (!tgt->argmax)
		return;
	int	   n	  = 512;
	float *logits = xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++)
		logits[i] = 3.0f;

	buffer l_cpu = {0};
	buffer l_tgt = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &l_cpu);
	tgt->buffer_alloc_scratch(tgt, (size_t)n * sizeof(float), &l_tgt);
	cpu->buffer_write_f32(cpu, &l_cpu, logits, n);
	tgt->buffer_write_f32(tgt, &l_tgt, logits, n);

	int32_t idx_cpu = -1;
	int32_t idx_tgt = -1;
	cpu->argmax(cpu, &l_cpu, n, &idx_cpu);
	status_code s_tgt = tgt->argmax(tgt, &l_tgt, n, &idx_tgt);
	if (tgt->synchronize)
		tgt->synchronize(tgt);

	char label[96];
	char detail[256];
	snprintf(label, sizeof(label), "argmax all-equal logits (tie-break)");
	if (s_tgt != OK) {
		snprintf(detail, sizeof(detail), "status=%d", s_tgt);
		record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
	} else if (idx_cpu == idx_tgt) {
		snprintf(detail, sizeof(detail), "idx=%d (match)", idx_cpu);
		record_result(OPFAM_EDGE_CASE, label, V_PASS, detail);
	} else {
		snprintf(detail, sizeof(detail), "idx_cpu=%d idx_tgt=%d (tie-break diverged)", idx_cpu,
				 idx_tgt);
		record_result(OPFAM_EDGE_CASE, label, V_FAIL, detail);
	}

	free(logits);
	cpu->buffer_free(cpu, &l_cpu);
	tgt->buffer_free(tgt, &l_tgt);
}

static void test_edge_ffn_activate_extremes(backend *cpu, backend *tgt) {
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

	buffer g_cpu = {0};
	buffer u_cpu = {0};
	buffer o_cpu = {0};
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &g_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &u_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &o_cpu);
	cpu->buffer_write_f32(cpu, &g_cpu, g, n);
	cpu->buffer_write_f32(cpu, &u_cpu, u, n);
	cpu->ffn_activate(cpu, &g_cpu, &u_cpu, &o_cpu, n);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &o_cpu, y_ref, n);

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
	cpu->buffer_free(cpu, &g_cpu);
	cpu->buffer_free(cpu, &u_cpu);
	cpu->buffer_free(cpu, &o_cpu);
	tgt->buffer_free(tgt, &g_tgt);
	tgt->buffer_free(tgt, &u_tgt);
	tgt->buffer_free(tgt, &o_tgt);
}

static void test_edge_rope_identity_table(backend *cpu, backend *tgt) {
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

	buffer kc_cpu = {0};
	buffer vc_cpu = {0};
	buffer x_cpu  = {0};
	cpu->kv_alloc(cpu, &kvd, &kc_cpu, &vc_cpu);
	cpu->buffer_alloc_scratch(cpu, (size_t)n * sizeof(float), &x_cpu);
	cpu->buffer_write_f32(cpu, &x_cpu, x, n);
	cpu->rope(cpu, &x_cpu, n_heads, head_dim, pos, cos_v, sin_v);
	float *y_ref = xmalloc((size_t)n * sizeof(float));
	cpu->buffer_read_f32(cpu, &x_cpu, y_ref, n);

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
			snprintf(detail, sizeof(detail), "cpu rope(cos=1,sin=0) is not identity: diff=%.3e@%d",
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
	cpu->buffer_free(cpu, &kc_cpu);
	cpu->buffer_free(cpu, &vc_cpu);
	cpu->buffer_free(cpu, &x_cpu);
	tgt->buffer_free(tgt, &kc_tgt);
	tgt->buffer_free(tgt, &vc_tgt);
	tgt->buffer_free(tgt, &x_tgt);
}

struct backend_op_coverage_row {
	const char *name;
	int			has_cpu;
	int			has_tgt;
};

static void print_op_coverage(backend *cpu, backend *tgt) {
	struct backend_op_coverage_row rows[] = {
		{"matmul", cpu->matmul != NULL, tgt->matmul != NULL},
		{"matmul_multi", cpu->matmul_multi != NULL, tgt->matmul_multi != NULL},
		{"matmul_batch", cpu->matmul_batch != NULL, tgt->matmul_batch != NULL},
		{"embd_lookup", cpu->embd_lookup != NULL, tgt->embd_lookup != NULL},
		{"rmsnorm", cpu->rmsnorm != NULL, tgt->rmsnorm != NULL},
		{"rmsnorm_per_head", cpu->rmsnorm_per_head != NULL, tgt->rmsnorm_per_head != NULL},
		{"rmsnorm_noweight", cpu->rmsnorm_noweight != NULL, tgt->rmsnorm_noweight != NULL},
		{"rope", cpu->rope != NULL, tgt->rope != NULL},
		{"rope_ext", cpu->rope_ext != NULL, tgt->rope_ext != NULL},
		{"rope_qk", cpu->rope_qk != NULL, tgt->rope_qk != NULL},
		{"add_inplace", cpu->add_inplace != NULL, tgt->add_inplace != NULL},
		{"ffn_activate", cpu->ffn_activate != NULL, tgt->ffn_activate != NULL},
		{"ffn_activate_ex", cpu->ffn_activate_ex != NULL, tgt->ffn_activate_ex != NULL},
		{"attention", cpu->attention != NULL, tgt->attention != NULL},
		{"attention_swa", cpu->attention_swa != NULL, tgt->attention_swa != NULL},
		{"kv_put", cpu->kv_put != NULL, tgt->kv_put != NULL},
		{"argmax", cpu->argmax != NULL, tgt->argmax != NULL},
		{"matmul_residual", cpu->matmul_residual != NULL, tgt->matmul_residual != NULL},
	};
	int n_gaps = 0;
	for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
		if (rows[i].has_cpu != rows[i].has_tgt)
			n_gaps++;
	}
	if (n_gaps == 0)
		return;

	printf("native op coverage (%s vs cpu):\n", tgt->name);
	for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
		if (rows[i].has_cpu == rows[i].has_tgt)
			continue;
		printf("  %-18s cpu=%c  tgt=%c\n", rows[i].name, rows[i].has_cpu ? '+' : '-',
			   rows[i].has_tgt ? '+' : '-');
	}
}
void run_per_op_tests(backend *cpu, backend *tgt) {
	printf("\n========================================\n");
	printf("Per-op validation: %s  vs  cpu (reference)\n", tgt->name);
	printf("========================================\n");
	print_op_coverage(cpu, tgt);

	int shapes[][2] = {{32, 32}, {128, 256}, {64, 512}, {17, 256}};
	for (int sh = 0; sh < (int)(sizeof(shapes) / sizeof(shapes[0])); sh++) {
		int N = shapes[sh][0];
		int K = shapes[sh][1];
		for (int qi = 0; qi < QTYPES_N; qi++) {
			if (K % QTYPES[qi].block != 0)
				continue;
			test_op_matmul(cpu, tgt, &QTYPES[qi], N, K);
		}
	}
	flush_family(OPFAM_MATMUL);

	for (int qi = 0; qi < QTYPES_N; qi++)
		test_op_embd_lookup(cpu, tgt, &QTYPES[qi], QTYPES[qi].block * 4, 32);
	test_op_embd_lookup_f32(cpu, tgt);
	flush_family(OPFAM_EMBD_LOOKUP);

	test_op_rmsnorm(cpu, tgt, 256);
	test_op_rmsnorm(cpu, tgt, 2048);
	test_op_rmsnorm(cpu, tgt, 4096);
	flush_family(OPFAM_RMSNORM);

	test_op_rmsnorm_per_head(cpu, tgt, 8, 64);
	test_op_rmsnorm_per_head(cpu, tgt, 4, 64);
	test_op_rmsnorm_per_head(cpu, tgt, 32, 128);
	flush_family(OPFAM_RMSNORM_PER_HEAD);

	test_op_rmsnorm_noweight(cpu, tgt, 64);
	test_op_rmsnorm_noweight(cpu, tgt, 128);
	flush_family(OPFAM_RMSNORM_NOWEIGHT);

	test_op_rope(cpu, tgt, 16, 64, 0);
	test_op_rope(cpu, tgt, 16, 64, 1);
	test_op_rope(cpu, tgt, 16, 64, 127);
	test_op_rope(cpu, tgt, 32, 128, 50);
	flush_family(OPFAM_ROPE);

	test_op_rope_ext(cpu, tgt, 8, 64, 0, 0);
	test_op_rope_ext(cpu, tgt, 8, 64, 1, 0);
	test_op_rope_ext(cpu, tgt, 8, 64, 127, 0);
	test_op_rope_ext(cpu, tgt, 8, 64, 0, 1);
	test_op_rope_ext(cpu, tgt, 8, 64, 127, 1);
	test_op_rope_ext(cpu, tgt, 32, 128, 511, 1);
	flush_family(OPFAM_ROPE_EXT);

	test_op_add_inplace(cpu, tgt, 256);
	test_op_add_inplace(cpu, tgt, 4096);
	flush_family(OPFAM_ADD_INPLACE);

	test_op_ple_combine(cpu, tgt, 256);
	test_op_ple_combine(cpu, tgt, 4096);
	flush_family(OPFAM_PLE_COMBINE);

	test_op_rmsnorm_add(cpu, tgt, 256);
	test_op_rmsnorm_add(cpu, tgt, 4096);
	flush_family(OPFAM_RMSNORM_ADD);

	test_op_ffn_activate(cpu, tgt, 256);
	test_op_ffn_activate(cpu, tgt, 4096);
	flush_family(OPFAM_FFN_ACTIVATE);

	test_op_ffn_activate_ex(cpu, tgt, 256, 1);
	test_op_ffn_activate_ex(cpu, tgt, 4096, 1);
	test_op_ffn_activate_ex(cpu, tgt, 256, 0);
	test_op_ffn_activate_ex(cpu, tgt, 4096, 0);
	flush_family(OPFAM_FFN_ACTIVATE_EX);

	test_op_attention(cpu, tgt, 8, 4, 64, 1024, 0, 0);
	test_op_attention(cpu, tgt, 8, 4, 64, 1024, 1, 1);
	test_op_attention(cpu, tgt, 8, 4, 64, 1024, 127, 0);
	test_op_attention(cpu, tgt, 8, 4, 64, 1024, 127, 1);
	test_op_attention(cpu, tgt, 8, 8, 64, 1024, 63, 1);
	test_op_attention(cpu, tgt, 32, 8, 128, 2048, 511, 0);
	test_op_attention(cpu, tgt, 32, 8, 128, 2048, 511, 1);
	test_op_attention(cpu, tgt, 8, 1, 256, 2048, 0, 0);
	test_op_attention(cpu, tgt, 8, 1, 256, 2048, 0, 1);
	test_op_attention(cpu, tgt, 8, 1, 256, 2048, 1023, 0);
	test_op_attention(cpu, tgt, 8, 1, 256, 2048, 1023, 1);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 0, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 0, 1);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 1, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 2, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 3, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 4, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 8, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 16, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 63, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 511, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 1023, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 2048, 1023, 1);
	test_op_attention(cpu, tgt, 8, 1, 512, 64, 1, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 64, 3, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 64, 16, 0);
	test_op_attention(cpu, tgt, 8, 1, 512, 64, 63, 0);
	flush_family(OPFAM_ATTENTION);

	test_op_attention_swa(cpu, tgt, 8, 4, 64, 1024, 63, 32, 0);
	test_op_attention_swa(cpu, tgt, 8, 4, 64, 1024, 63, 32, 1);
	test_op_attention_swa(cpu, tgt, 8, 4, 64, 1024, 511, 128, 0);
	test_op_attention_swa(cpu, tgt, 8, 4, 64, 1024, 511, 128, 1);
	test_op_attention_swa(cpu, tgt, 8, 8, 64, 1024, 20, 32, 0);
	test_op_attention_swa(cpu, tgt, 32, 8, 128, 2048, 1000, 512, 1);
	flush_family(OPFAM_ATTENTION_SWA);

	test_op_kv_put(cpu, tgt, 4, 64, 1024, 0);
	test_op_kv_put(cpu, tgt, 4, 64, 1024, 127);
	test_op_kv_put(cpu, tgt, 8, 128, 2048, 511);
	test_op_kv_put_batch(cpu, tgt, 4, 64, 1024, 0, 7);
	test_op_kv_put_batch(cpu, tgt, 8, 128, 2048, 511, 33);
	flush_family(OPFAM_KV_PUT);

	run_kv_quant_parity_tests(cpu, tgt);

	test_op_argmax(cpu, tgt, 256);
	test_op_argmax(cpu, tgt, 4096);
	test_op_argmax(cpu, tgt, 32000);
	flush_family(OPFAM_ARGMAX);

	for (int qi = 0; qi < QTYPES_N; qi++) {
		test_quant_determinism(&QTYPES[qi]);
		test_quant_finiteness(&QTYPES[qi]);
	}
	test_quant_q8_0_roundtrip();
	flush_family(OPFAM_QUANT);

	for (int qi = 0; qi < QTYPES_N; qi++)
		test_dequant_parity_cross(cpu, tgt, &QTYPES[qi], QTYPES[qi].block * 4, 8);
	flush_family(OPFAM_DEQUANT_PARITY);

	run_repack_parity_tests(cpu);
	flush_family(OPFAM_REPACK_PARITY);

	for (int qi = 0; qi < QTYPES_N; qi++) {
		if (256 % QTYPES[qi].block != 0)
			continue;
		test_op_matmul_residual(cpu, tgt, &QTYPES[qi], 64, 256);
	}
	flush_family(OPFAM_MATMUL_RESIDUAL);

	test_op_rope_qk(cpu, tgt, 8, 4, 64, 0);
	test_op_rope_qk(cpu, tgt, 8, 4, 64, 1);
	test_op_rope_qk(cpu, tgt, 8, 4, 64, 127);
	test_op_rope_qk(cpu, tgt, 32, 8, 128, 511);
	flush_family(OPFAM_ROPE_QK);

	test_edge_rmsnorm_zeros(cpu, tgt);
	test_edge_determinism(cpu, tgt);
	test_edge_argmax_all_equal(cpu, tgt);
	test_edge_ffn_activate_extremes(cpu, tgt);
	test_edge_rope_identity_table(cpu, tgt);
	flush_family(OPFAM_EDGE_CASE);
}
