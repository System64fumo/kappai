#include "test_core.h"
#include "test_synth_gguf.h"

#include "gguf.h"
#include "recipe/internal.h"

#define CROSS_DEFAULT_PREFILL 24
#define CROSS_DEFAULT_DECODE 8
#define CROSS_LAYER_REL_TOL 1e-3
#define CROSS_MAX_ARCH_FILTERS 16

typedef struct {
	const char *name;
	model_arch	arch;
	const char *(*fixture_path)(void);
	void (*build)(const char *path, const tsg_arch_spec *spec);
	int n_layers;
} cross_arch;

static const char *fixture_llama(void) {
	return synth_chat_model_path;
}

static const char *fixture_lfm2(void) {
	return synth_lfm2_model_path;
}

static const char *fixture_glm_dsa(void) {
	return synth_dsa_model_path;
}

static const cross_arch CROSS_ARCHS[] = {
	{"llama", ARCH_LLAMA, fixture_llama, NULL, 0},
	{"lfm2", ARCH_LFM2, fixture_lfm2, NULL, 0},
	{"glm-dsa", ARCH_GLM_DSA, fixture_glm_dsa, NULL, 0},
	{"gemma4", ARCH_GEMMA4, NULL, tsg_build_gemma4, 6},
	{"gemma4-moe", ARCH_GEMMA4_MOE, NULL, tsg_build_gemma4_moe, 6},
	{"qwen35", ARCH_QWEN35, NULL, tsg_build_qwen35, 4},
	{"lfm3", ARCH_LFM3, NULL, tsg_build_lfm3, 6},
	{"glm5-next", ARCH_GLM5_NEXT, NULL, tsg_build_glm5_next, 4},
	{"qwen3vl", ARCH_QWEN3VL, NULL, tsg_build_qwen3vl, 3},
};

static int		   cross_wide_dims;
static const char *cross_quant_name = "f32";

static const char *cross_fixture(const cross_arch *e, char *buf, size_t sz) {
	if (e->fixture_path)
		return e->fixture_path();
	tsg_arch_spec spec = {.dim			= cross_wide_dims ? 256 : 64,
						  .n_heads		= 4,
						  .n_kv_heads	= 2,
						  .head_dim		= cross_wide_dims ? 64 : 16,
						  .n_layers		= e->n_layers,
						  .intermediate = cross_wide_dims ? 512 : 96,
						  .ctx			= 256,
						  .vocab		= 96,
						  .seed			= 0xA2C4ULL + (uint64_t)e->arch};
	snprintf(buf, sz, "%s/archs_%s_%s.gguf", synth_fixture_dir, e->name, cross_quant_name);
	e->build(buf, &spec);
	return buf;
}

typedef struct {
	double max_abs;
	size_t at;
	double ref_at;
	double got_at;
	double sum_d2;
	double sum_r2;
	double max_ref;
	size_t nonfinite;
} div_stats;

static void div_measure(const float *ref, const float *got, size_t n, div_stats *o) {
	memset(o, 0, sizeof(*o));
	for (size_t i = 0; i < n; i++) {
		if (!isfinite(ref[i]) || !isfinite(got[i])) {
			o->nonfinite++;
			continue;
		}
		double d = fabs((double)ref[i] - (double)got[i]);
		o->sum_d2 += d * d;
		o->sum_r2 += (double)ref[i] * (double)ref[i];
		if (fabs((double)ref[i]) > o->max_ref)
			o->max_ref = fabs((double)ref[i]);
		if (d > o->max_abs) {
			o->max_abs = d;
			o->at	   = i;
			o->ref_at  = ref[i];
			o->got_at  = got[i];
		}
	}
}

static double div_rel_rms(const div_stats *s) {
	return s->sum_r2 > 0.0 ? sqrt(s->sum_d2 / s->sum_r2) : (s->sum_d2 > 0.0 ? INFINITY : 0.0);
}

static double div_rel_peak(const div_stats *s) {
	return s->max_ref > 0.0 ? s->max_abs / s->max_ref : (s->max_abs > 0.0 ? INFINITY : 0.0);
}

static double top1_margin(const float *logits, int n, int *top) {
	int	  best	 = 0;
	float second = -INFINITY;
	for (int i = 1; i < n; i++) {
		if (logits[i] > logits[best]) {
			second = logits[best];
			best   = i;
		} else if (logits[i] > second) {
			second = logits[i];
		}
	}
	*top = best;
	return (double)logits[best] - (double)second;
}

typedef struct {
	model			*m;
	compute_scratch *s;
	float		   **hid;
	int				 total_tokens;
	int				 chunk_offset;
	int				 chunk_tokens;
} capture_ctx;

static status_code capture_after(void *ud, int layer) {
	capture_ctx *c	= ud;
	backend		*be = c->m->backend;
	if (!be || !be->buffer_read_f32 || !c->s->bs || layer < 0 || layer >= c->m->n_layers)
		return OK;
	if (be->synchronize)
		be->synchronize(be);
	size_t dim	 = (size_t)c->m->dim;
	size_t n	 = (size_t)c->chunk_tokens * dim;
	float *chunk = xmalloc(n * sizeof(float));
	if (be->buffer_read_f32(be, &c->s->bs->pair[RECIPE_SLOT_X].b, chunk, (int)n) != OK) {
		free(chunk);
		return OK;
	}
	if (!c->hid[layer])
		c->hid[layer] = xcalloc((size_t)c->total_tokens * dim, sizeof(float));
	memcpy(c->hid[layer] + ((size_t)c->chunk_offset * dim), chunk, n * sizeof(float));
	free(chunk);
	return OK;
}

typedef struct {
	model			m;
	kvcache			kv;
	compute_scratch s;
	backend		   *be;
	float		  **hid;
	int				n_layers;
	int				loaded;
} cross_side;

static void cross_side_free(cross_side *c) {
	if (c->hid) {
		for (int i = 0; i < c->n_layers; i++)
			free(c->hid[i]);
		free(c->hid);
	}
	if (c->loaded) {
		compute_scratch_free(&c->s);
		kvcache_free(&c->kv);
		model_free(&c->m);
	}
	memset(c, 0, sizeof(*c));
}

static status_code cross_side_init(cross_side *c, const char *path, backend *be, int n_ctx) {
	memset(c, 0, sizeof(*c));
	c->be = be;
	if (model_load_backend_ex_repack(&c->m, path, be, 1, NULL) != OK)
		return ERR_FORMAT;
	c->loaded  = 1;
	c->m.n_ctx = n_ctx;
	if (c->m.recipe)
		recipe_free(c->m.recipe);
	c->m.recipe = recipe_build(&c->m);
	compute_scratch_init(&c->s);
	if (kvcache_init(&c->kv, &c->m, n_ctx, KV_QUANT_F16) != OK)
		return ERR_OUT_OF_MEMORY;
	c->n_layers = c->m.n_layers;
	c->hid		= xcalloc((size_t)c->n_layers, sizeof(float *));
	return compute_scratch_ensure(&c->s, &c->m, n_ctx);
}

static int cross_chunk_limit(const model *m, int n_ctx) {
	if (m->sliding_window <= 0)
		return n_ctx;
	int limit = n_ctx;
	for (int li = 0; li < m->n_layers; li++) {
		int cap = model_kv_layer_pos_cap(m, n_ctx, li);
		if (cap < n_ctx && cap - m->sliding_window < limit)
			limit = cap - m->sliding_window;
	}
	return limit < 1 ? 1 : limit;
}

static status_code cross_prefill(cross_side *c, const int32_t *toks, int n, int chunk, int flash,
								 float *logits) {
	status_code st = OK;
	capture_ctx cc = {.m = &c->m, .s = &c->s, .hid = c->hid, .total_tokens = n};
	for (int off = 0; off < n && st == OK; off += chunk) {
		int rows = n - off < chunk ? n - off : chunk;
		compute_scratch_reserve_rows(&c->m, &c->s, rows);
		backend_bind_phase(TPOOL_PHASE_PREFILL);
		cc.chunk_offset		   = off;
		cc.chunk_tokens		   = rows;
		recipe_layer_gate gate = {.after = capture_after, .ud = &cc};
		st = compute_prefill_range(&c->m, &c->kv, &c->s, toks + off, rows, off, flash, &gate,
								   logits);
		if (st == ERR_UNSUPPORTED)
			st = compute_forward_batch(&c->m, &c->kv, &c->s, toks + off, rows, off, flash, logits);
		if (c->be->synchronize)
			c->be->synchronize(c->be);
	}
	return st;
}

static status_code cross_decode(cross_side *c, int32_t token, int pos, int flash, float *logits) {
	status_code st = compute_forward(&c->m, &c->kv, &c->s, token, pos, flash, logits);
	if (c->be->synchronize)
		c->be->synchronize(c->be);
	return st;
}

typedef struct {
	int	   n_steps;
	int	   argmax_match;
	double worst_rel_peak;
	int	   worst_step;
	int	   nonfinite;
	int	   classify_fail;
	int	   first_layer_over;
} cross_summary;

static void cross_report_layers(const cross_side *r, const cross_side *t, int n_tokens, int dim,
								cross_summary *sum) {
	sum->first_layer_over = -1;
	int have			  = 0;
	for (int l = 0; l < r->n_layers; l++)
		if (r->hid[l] && t->hid[l])
			have = 1;
	if (!have) {
		printf("  per-layer residual: not captured (backend has no readback or plan is split)\n");
		return;
	}
	printf("  per-layer residual divergence after each layer (%d tokens x %d dims):\n", n_tokens,
		   dim);
	printf("    %-6s %-12s %-12s %-12s %s\n", "layer", "max|d|", "rel_rms", "rel_peak",
		   "worst (token,dim)  ref -> got");
	for (int l = 0; l < r->n_layers; l++) {
		if (!r->hid[l] || !t->hid[l])
			continue;
		div_stats st;
		div_measure(r->hid[l], t->hid[l], (size_t)n_tokens * (size_t)dim, &st);
		double rel = div_rel_rms(&st);
		int	   hot = st.nonfinite > 0 || rel > CROSS_LAYER_REL_TOL;
		if (hot && sum->first_layer_over < 0)
			sum->first_layer_over = l;
		printf("    %-6d %-12.3e %-12.3e %-12.3e (%zu,%zu)  %.6g -> %.6g%s\n", l, st.max_abs, rel,
			   div_rel_peak(&st), st.at / (size_t)dim, st.at % (size_t)dim, st.ref_at, st.got_at,
			   st.nonfinite ? "  NON-FINITE" : (hot ? "  <--" : ""));
	}
	if (sum->first_layer_over >= 0)
		printf("  first layer over rel_rms %.0e: %d\n", CROSS_LAYER_REL_TOL, sum->first_layer_over);
	else
		printf("  no layer exceeds rel_rms %.0e\n", CROSS_LAYER_REL_TOL);
}

static void cross_report_step(const char *label, const float *lr, const float *lt, int vocab,
							  cross_summary *sum, int step) {
	div_stats st;
	div_measure(lr, lt, (size_t)vocab, &st);
	int	   top_r	= 0;
	int	   top_t	= 0;
	double margin_r = top1_margin(lr, vocab, &top_r);
	top1_margin(lt, vocab, &top_t);
	double	rel_peak = div_rel_peak(&st);
	char	detail[256];
	verdict v	 = classify_output("loose", lr, lt, vocab, OK, detail, sizeof(detail));
	int		same = top_r == top_t;

	sum->n_steps++;
	sum->argmax_match += same;
	sum->nonfinite += st.nonfinite > 0;
	sum->classify_fail += v == V_FAIL;
	if (rel_peak > sum->worst_rel_peak) {
		sum->worst_rel_peak = rel_peak;
		sum->worst_step		= step;
	}
	printf("    %-9s max|d|=%-10.3e rel_peak=%-10.3e rel_rms=%-10.3e worst_idx=%-6zu top1 %d/%d "
		   "margin=%.3e %s%s\n",
		   label, st.max_abs, rel_peak, div_rel_rms(&st), st.at, top_r, top_t, margin_r,
		   same ? "match" : "MISMATCH", st.nonfinite ? " NON-FINITE" : "");
}

static void cross_quant_summary(const char *path, char *out, size_t sz) {
	gguf_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	snprintf(out, sz, "unknown");
	if (gguf_load_metadata(&ctx, path) != OK)
		return;
	size_t weights	 = 0;
	size_t quantized = 0;
	for (size_t i = 0; i < ctx.n_tensors; i++) {
		if (ctx.tensors[i].n_dims < 2)
			continue;
		weights++;
		quantized += ctx.tensors[i].type != GGML_TYPE_F32;
	}
	snprintf(out, sz, "%zu of %zu matrices stored as %s", quantized, weights, cross_quant_name);
	gguf_free(&ctx);
}

static uint32_t cross_rng_state;

static uint32_t cross_rng(void) {
	cross_rng_state = cross_rng_state * 1664525u + 1013904223u;
	return cross_rng_state >> 8;
}

static void cross_run_arch(const cross_arch *e, backend *ref, const char *ref_spec, backend *tgt,
						   const char *tgt_spec, int n_prefill, int n_decode, int flash) {
	char label[160];
	snprintf(label, sizeof(label), "archs.%s[%s] %s vs %s", e->name, cross_quant_name, tgt_spec,
			 ref_spec);

	char		built_path[512];
	const char *path = cross_fixture(e, built_path, sizeof(built_path));

	printf("\n=== %s: %s vs %s ===\n", e->name, tgt_spec, ref_spec);
	char quant_info[96];
	cross_quant_summary(path, quant_info, sizeof(quant_info));
	printf("  weights: %s\n", quant_info);

	int		   n_ctx = n_prefill + n_decode + 16;
	cross_side r;
	cross_side t;
	if (cross_side_init(&r, path, ref, n_ctx) != OK) {
		record_result(OPFAM_ARCH_PIPELINE, label, V_FAIL, "failed to load fabricated GGUF on ref");
		cross_side_free(&r);
		return;
	}
	if (cross_side_init(&t, path, tgt, n_ctx) != OK) {
		record_result(OPFAM_ARCH_PIPELINE, label, V_FAIL,
					  "failed to load fabricated GGUF on target");
		cross_side_free(&r);
		cross_side_free(&t);
		return;
	}
	if (r.m.arch != e->arch) {
		record_result(OPFAM_ARCH_PIPELINE, label, V_FAIL, "fixture loaded as a different arch");
		cross_side_free(&r);
		cross_side_free(&t);
		return;
	}

	int vocab		= r.m.vocab_size;
	int dim			= r.m.dim;
	cross_rng_state = 0x9E3779B9u;
	int32_t *prompt = xmalloc((size_t)n_prefill * sizeof(int32_t));
	for (int i = 0; i < n_prefill; i++)
		prompt[i] = 1 + (int32_t)(cross_rng() % (uint32_t)(vocab - 1));

	float *lr = xmalloc((size_t)vocab * sizeof(float));
	float *lt = xmalloc((size_t)vocab * sizeof(float));

	cross_summary sum	  = {.first_layer_over = -1};
	int			  chunk	  = cross_chunk_limit(&r.m, n_ctx);
	int			  chunk_t = cross_chunk_limit(&t.m, n_ctx);
	if (chunk_t < chunk)
		chunk = chunk_t;
	status_code sr = cross_prefill(&r, prompt, n_prefill, chunk, flash, lr);
	status_code st = cross_prefill(&t, prompt, n_prefill, chunk, flash, lt);
	if (sr != OK || st != OK) {
		char detail[128];
		snprintf(detail, sizeof(detail), "prefill failed (ref status %d, target status %d)",
				 (int)sr, (int)st);
		record_result(OPFAM_ARCH_PIPELINE, label, V_FAIL, detail);
		goto done;
	}

	printf("  prefill %d tokens in chunks of %d, then %d teacher-forced decode steps (flash=%d)\n",
		   n_prefill, chunk, n_decode, flash);
	cross_report_layers(&r, &t, n_prefill, dim, &sum);
	printf("  logits per step:\n");
	cross_report_step("prefill", lr, lt, vocab, &sum, -1);

	int32_t next = sampler_argmax(lr, vocab);
	for (int i = 0; i < n_decode; i++) {
		int pos = n_prefill + i;
		sr		= cross_decode(&r, next, pos, flash, lr);
		st		= cross_decode(&t, next, pos, flash, lt);
		if (sr != OK || st != OK) {
			char detail[128];
			snprintf(detail, sizeof(detail), "decode step %d failed (ref %d, target %d)", i,
					 (int)sr, (int)st);
			record_result(OPFAM_ARCH_PIPELINE, label, V_FAIL, detail);
			goto done;
		}
		char step_label[24];
		snprintf(step_label, sizeof(step_label), "decode+%d", i);
		cross_report_step(step_label, lr, lt, vocab, &sum, i);
		next = sampler_argmax(lr, vocab);
	}

	{
		int	 ok = sum.argmax_match == sum.n_steps && sum.nonfinite == 0 && sum.classify_fail == 0;
		char where[32];
		if (sum.worst_step < 0)
			snprintf(where, sizeof(where), "prefill");
		else
			snprintf(where, sizeof(where), "decode+%d", sum.worst_step);
		char layer[48];
		if (sum.first_layer_over < 0)
			snprintf(layer, sizeof(layer), "none");
		else
			snprintf(layer, sizeof(layer), "layer %d", sum.first_layer_over);
		char detail[256];
		snprintf(detail, sizeof(detail),
				 "argmax %d/%d, worst logit rel_peak %.2e @ %s, first layer over tol: %s",
				 sum.argmax_match, sum.n_steps, sum.worst_rel_peak, where, layer);
		printf("  => %s\n", detail);
		record_result(OPFAM_ARCH_PIPELINE, label, ok ? V_PASS : V_FAIL, detail);
	}

done:
	free(prompt);
	free(lr);
	free(lt);
	cross_side_free(&r);
	cross_side_free(&t);
}

typedef struct {
	int			n_prefill;
	int			n_decode;
	int			flash;
	uint32_t	quant;
	const char *quant_name;
	const char *filters[CROSS_MAX_ARCH_FILTERS];
	int			n_filters;
} cross_options;

static int cross_parse_options(int argc, char **argv, cross_options *o) {
	o->n_prefill  = CROSS_DEFAULT_PREFILL;
	o->n_decode	  = CROSS_DEFAULT_DECODE;
	o->flash	  = 1;
	o->quant	  = tsg_quant_by_name("f32");
	o->quant_name = "f32";
	o->n_filters  = 0;
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--n-prefill") == 0 && ai + 1 < argc) {
			o->n_prefill = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--n-decode") == 0 && ai + 1 < argc) {
			o->n_decode = atoi(argv[++ai]);
		} else if (strcmp(argv[ai], "--no-flash") == 0) {
			o->flash = 0;
		} else if (strcmp(argv[ai], "--quant") == 0 && ai + 1 < argc) {
			o->quant_name = argv[++ai];
			o->quant	  = tsg_quant_by_name(o->quant_name);
			if (o->quant == UINT32_MAX) {
				fprintf(stderr, "error: unknown quant '%s' (choose from: %s)\n", o->quant_name,
						tsg_quant_names());
				return -1;
			}
		} else if (strcmp(argv[ai], "--arch") == 0 && ai + 1 < argc) {
			if (o->n_filters >= CROSS_MAX_ARCH_FILTERS) {
				fprintf(stderr, "error: too many --arch filters\n");
				return -1;
			}
			const char *name  = argv[++ai];
			int			found = 0;
			for (size_t i = 0; i < ARRAY_LEN(CROSS_ARCHS); i++)
				found |= strcmp(CROSS_ARCHS[i].name, name) == 0;
			if (!found) {
				fprintf(stderr, "error: unknown architecture '%s'\n", name);
				return -1;
			}
			o->filters[o->n_filters++] = name;
		}
	}
	if (o->n_prefill < 1 || o->n_decode < 0) {
		fprintf(stderr, "error: --n-prefill must be >= 1 and --n-decode >= 0\n");
		return -1;
	}
	return 0;
}

static int cross_arch_wanted(const cross_options *o, const char *name) {
	if (o->n_filters == 0)
		return 1;
	for (int i = 0; i < o->n_filters; i++)
		if (strcmp(o->filters[i], name) == 0)
			return 1;
	return 0;
}

int run_arch_cross_mode(int argc, char **argv, test_device *devs, int n_devs) {
	stats_reset();

	cross_options opt;
	if (cross_parse_options(argc, argv, &opt) != 0)
		return 1;

	device_selection sel;
	if (test_parse_device_selection(argc, argv, devs, n_devs, &sel) != 0) {
		usage(argv[0]);
		return 1;
	}

	backend *ref = NULL;
	if (test_device_create(test_find_device(devs, n_devs, sel.ref), &ref) != OK) {
		fprintf(stderr, "ERROR: reference device '%s' is unavailable\n", sel.ref);
		return 1;
	}
	printf("reference device: %s\n", sel.ref);

	tsg_set_quant(opt.quant);
	cross_quant_name = opt.quant_name;
	cross_wide_dims	 = tsg_quant_block_elems(opt.quant) >= 256;
	synth_suite_common_init();

	int run_all = wants_all(argc, argv);
	int any_run = 0;
	for (int di = 0; di < n_devs; di++) {
		int want = run_all;
		for (int t = 0; t < sel.n_targets && !want; t++)
			want = strcmp(sel.targets[t], devs[di].spec) == 0;
		if (!want || strcmp(devs[di].spec, sel.ref) == 0)
			continue;
		for (size_t ai = 0; ai < ARRAY_LEN(CROSS_ARCHS); ai++) {
			if (!cross_arch_wanted(&opt, CROSS_ARCHS[ai].name))
				continue;
			backend *tgt = NULL;
			if (test_device_create(&devs[di], &tgt) != OK) {
				any_run = 1;
				printf("\n=== %s: SKIPPED (failed to init) ===\n", devs[di].spec);
				continue;
			}
			any_run = 1;
			cross_run_arch(&CROSS_ARCHS[ai], ref, sel.ref, tgt, devs[di].spec, opt.n_prefill,
						   opt.n_decode, opt.flash);
			backend_destroy(tgt);
		}
	}
	if (!any_run)
		fprintf(stderr, "No matching/available devices were tested.\n");
	flush_family(OPFAM_ARCH_PIPELINE);
	print_final_results();
	backend_destroy(ref);
	return g_fail > 0 ? 1 : 0;
}
