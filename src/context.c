#include "context.h"
#include "compute.h"
#include "config.h"
#include "log.h"
#include "memconfig.h"
#include "moe/moe_stream.h"
#include "recipe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	progress		 *prog;
	int				  chunk_offset;
	int				  chunk_size;
	int				  total_tokens;
	int				  prev_layer_idx;
	int				  tokens_in_chunk_done;
	int				  slow_path;
	layer_progress_cb chained_cb;
	void			 *chained_ud;
} prefill_progress;

static void prefill_progress_start_chunk(prefill_progress *pp, int chunk_offset, int chunk_size) {
	pp->chunk_offset		 = chunk_offset;
	pp->chunk_size			 = chunk_size;
	pp->prev_layer_idx		 = 0;
	pp->tokens_in_chunk_done = 0;
	pp->slow_path			 = 0;
}

static void prefill_progress_on_layer(int layer_idx, int n_layers, int token_progress, int batched,
									  void *ud) {
	prefill_progress *pp = (prefill_progress *)ud;
	(void)token_progress;

	if (pp->prev_layer_idx > 0 && layer_idx <= pp->prev_layer_idx) {
		pp->slow_path = 1;
		pp->tokens_in_chunk_done++;
	}
	pp->prev_layer_idx = layer_idx;

	double chunk_progress;
	if (n_layers <= 0) {
		chunk_progress = 1.0;
	} else if (batched) {
		double token_frac = (double)layer_idx / (double)n_layers;
		chunk_progress	  = token_frac;
	} else if (pp->slow_path || pp->chunk_size <= 1) {
		double token_frac = (double)layer_idx / (double)n_layers;
		chunk_progress = ((double)pp->tokens_in_chunk_done + token_frac) / (double)pp->chunk_size;
	} else {
		double token_frac = (double)layer_idx / (double)n_layers;
		chunk_progress	  = token_frac / (double)pp->chunk_size;
	}
	if (chunk_progress > 1.0)
		chunk_progress = 1.0;

	uint64_t done =
		(uint64_t)pp->chunk_offset + (uint64_t)(chunk_progress * (double)pp->chunk_size);
	if (done > (uint64_t)pp->total_tokens)
		done = (uint64_t)pp->total_tokens;
	progress_update(pp->prog, done);

	if (pp->chained_cb)
		pp->chained_cb(layer_idx, n_layers, (int)done, batched, pp->chained_ud);
}

void context_set_show_template(context *c, bool enabled) {
	c->show_template = enabled;
}

status_code context_init(context *c, const config *cfg) {
	memset(c, 0, sizeof(*c));
	monitor_reset(&c->monitor);

	const char *mon_path =
		cfg->monitor ? (cfg->monitor[0] ? cfg->monitor : "/tmp/kappai.monitor") : NULL;
	if (mon_path) {
		monitor_init(&c->monitor, mon_path);
		g_monitor = &c->monitor;
		monitor_emit_load_phase_backend(&c->monitor, cfg->backend);
	}

	status_code s = (cfg->backend && strcmp(cfg->backend, "auto") != 0)
						? backend_create(cfg->backend, cfg->gpu_device, &c->backend)
						: backend_create_best(cfg->gpu_device, &c->backend);
	if (s != OK) {
		ERROR("failed to initialize accelerator '%s'", cfg->backend ? cfg->backend : "auto");
		return s;
	}
	backend_host_use(c->backend);
	recipe_init();

	monitor_emit_load_phase_model_start(&c->monitor, cfg->model);

	uint64_t load_t0 = time_us();
	s = model_load_backend_ex_repack(&c->m, cfg->model, c->backend, cfg->use_mmap, cfg->repack,
									 cfg->ctx_size);
	uint64_t load_us = time_us() - load_t0;
	if (s != OK) {
		ERROR("failed to load model '%s'", cfg->model);
		monitor_emit_load_phase_model_failed(&c->monitor, s);
		goto fail_backend;
	}
	monitor_emit_load_phase_model_done(&c->monitor, load_us / 1000, c->m.n_layers, c->m.dim,
									   c->m.vocab_size);

	c->backend->rope_neox = c->m.arch_info->uses_neox_rope;

	s = tokenizer_init(&c->tok, &c->m.gctx);
	if (s != OK) {
		ERROR("failed to init tokenizer");
		goto fail_model;
	}

	s = chat_template_init(&c->chat, &c->m.gctx, &c->tok);
	if (s != OK) {
		ERROR("failed to init chat template");
		goto fail_tokenizer;
	}

	int n_ctx = cfg->ctx_size;
	if (n_ctx <= 0)
		n_ctx = c->m.n_ctx;
	if (n_ctx > c->m.n_ctx)
		n_ctx = c->m.n_ctx;
	c->n_ctx = n_ctx;

	backend		 *kv_owner = c->backend->kv_alloc ? c->backend : backend_host();
	kv_quant_type kv_quant = (kv_quant_type)cfg->kv_quant;
	{
		size_t kv_bytes = model_kv_cache_bytes_quant(&c->m, n_ctx, kv_quant);
		size_t avail	= backend_mem_available(kv_owner);
		if (!cfg->disable_failsafes && avail > 0 && kv_bytes > avail) {
			ERROR("KV cache (ctx=%d) needs %.1f MB but only %.1f MB is available on "
				  "backend '%s' -- refusing to run (see --disable-failsafes).",
				  n_ctx, kv_bytes / (1024.0 * 1024.0), avail / (1024.0 * 1024.0), kv_owner->name);
			s = ERR_OUT_OF_MEMORY;
			goto fail_chat;
		}
		DEBUG("KV cache: %.1f MB for ctx=%d on backend '%s' (quant=%s)",
			  kv_bytes / (1024.0 * 1024.0), n_ctx, kv_owner->name,
			  kv_quant == KV_QUANT_Q8_0 ? "q8_0" : "f16");
	}

	c->flash_attn = cfg->flash_attn ? 1 : 0;

	s = kvcache_init(&c->kv, &c->m, n_ctx, kv_quant);
	if (s != OK) {
		ERROR("failed to init KV cache");
		goto fail_chat;
	}

	compute_scratch_init(&c->scratch);
	c->scratch.interrupt = &c->interrupt;
	sampler_init(&c->samp, cfg->seed);
	sampler_set_vocab(&c->samp, c->m.vocab_size);

	return OK;

fail_chat:
	chat_template_free(&c->chat);
fail_tokenizer:
	tokenizer_free(&c->tok);
fail_model:
	model_free(&c->m);
fail_backend:
	monitor_free(&c->monitor);
	g_monitor = NULL;
	backend_destroy(c->backend);
	c->backend = NULL;
	return s;
}

void context_free(context *c) {
	if (!c)
		return;
	monitor_free(&c->monitor);
	g_monitor = NULL;
	sampler_free(&c->samp);
	compute_scratch_free(&c->scratch);
	kvcache_free(&c->kv);
	chat_template_free(&c->chat);
	tokenizer_free(&c->tok);
	model_free(&c->m);
	backend_destroy(c->backend);
	memset(c, 0, sizeof(*c));
}

void context_reset(context *c) {
	kvcache_reset(&c->kv);
	c->samp.recent_count = 0;
	c->samp.recent_head	 = 0;
	chat_template_clear_messages(&c->chat);
}

static int context_feed_token_inner(context *c, int32_t token, float *logits_out) {
	if (c->kv.n_pos >= c->n_ctx)
		return -1;
	if (token < 0 || token >= c->m.vocab_size)
		return -1;
	int			pos = c->kv.n_pos;
	status_code st	= compute_scratch_ensure(&c->scratch, &c->m, c->n_ctx);
	if (st != OK)
		return CTX_COMPUTE_ERROR;
	status_code cst =
		compute_forward(&c->m, &c->kv, &c->scratch, token, pos, c->flash_attn, logits_out);
	if (cst == ERR_INTERRUPTED)
		return CTX_INTERRUPTED;
	if (cst != OK) {
		ERROR("compute_forward failed (status=%d) at pos=%d token=%d -- "
			  "aborting feed",
			  (int)cst, pos, (int)token);
		return CTX_COMPUTE_ERROR;
	}
	c->kv.n_pos++;
	return pos + 1;
}

int context_feed_tokens_batch(context *c, const int32_t *tokens, int n) {
	if (n <= 0)
		return 0;
	if (c->kv.n_pos + n > c->n_ctx)
		return -1;

	int			pos = c->kv.n_pos;
	status_code st	= compute_scratch_ensure(&c->scratch, &c->m, c->n_ctx);
	if (st != OK)
		return CTX_COMPUTE_ERROR;

	int n_threads = 1;
	if (c->backend && c->backend->get_pool) {
		tpool *pool = c->backend->get_pool(c->backend);
		if (pool)
			n_threads = tpool_n_threads(pool);
	}
	int chunk = n_threads > 1 ? (n_threads * 4) : 64;
	if (chunk < 32)
		chunk = 32;
	if (chunk > 256)
		chunk = 256;
	if (chunk > n)
		chunk = n;
	progress prog;
	int		 show_progress = n > 1;
	if (show_progress)
		progress_start(&prog, "prompt processing", (uint64_t)n);

	prefill_progress  pp		  = {0};
	layer_progress_cb saved_cb	  = c->scratch.layer_cb;
	void			 *saved_cb_ud = c->scratch.layer_cb_ud;
	if (show_progress) {
		pp.prog			= &prog;
		pp.total_tokens = n;
		pp.chained_cb	= saved_cb;
		pp.chained_ud	= saved_cb_ud;
		compute_set_layer_progress_cb(&c->scratch, prefill_progress_on_layer, &pp);
	}

	int status = CTX_COMPUTE_ERROR;
	int chunk_offset;
	for (chunk_offset = 0; chunk_offset < n; chunk_offset += chunk) {
		if (c->interrupt) {
			status = CTX_INTERRUPTED;
			goto fail;
		}

		int chunk_size = MIN(chunk, n - chunk_offset);
		prefill_progress_start_chunk(&pp, chunk_offset, chunk_size);

		status_code cfb_st = compute_forward_batch(
			&c->m, &c->kv, &c->scratch, tokens + chunk_offset, chunk_size, pos + chunk_offset,
			c->flash_attn, (chunk_offset + chunk_size == n) ? c->scratch.logits_host : NULL);
		if (cfb_st == ERR_INTERRUPTED) {
			status = CTX_INTERRUPTED;
			goto fail;
		}
		if (cfb_st != OK) {
			ERROR("compute_forward_batch failed (status=%d) at chunk_offset=%d chunk_size=%d -- "
				  "prompt processing aborted",
				  (int)cfb_st, chunk_offset, chunk_size);
			status = CTX_COMPUTE_ERROR;
			goto fail;
		}
	}

	if (show_progress) {
		compute_set_layer_progress_cb(&c->scratch, saved_cb, saved_cb_ud);
		progress_finish(&prog);
	}

	c->kv.n_pos += n;
	return pos + n;

fail:
	if (show_progress) {
		compute_set_layer_progress_cb(&c->scratch, saved_cb, saved_cb_ud);
		progress_finish(&prog);
	}
	c->kv.n_pos += chunk_offset;
	return status;
}

static void debug_print_logits(context *c) {
	buffer *logits = &c->scratch.slots[RECIPE_SLOT_LOGITS];
	if (c->backend && c->backend->argmax) {
		if (c->backend->synchronize)
			c->backend->synchronize(c->backend);
		backend *owner = logits->owner ? logits->owner : c->backend;
		if (owner->buffer_read_f32) {
			owner->buffer_read_f32(owner, logits, c->scratch.logits_host, c->m.vocab_size);
		}
	}
	int	   vocab = c->m.vocab_size;
	float *lg	 = c->scratch.logits_host;

	if (vocab <= 128) {
		fprintf(stderr, "[DEBUG] full logits (vocab=%d):", vocab);
		for (int i = 0; i < vocab; i++) {
			fprintf(stderr, " [%d]=%.6f", i, lg[i]);
		}
		fprintf(stderr, "\n");
	}

	sampler_top_k_entry top[8];
	int					n_top = sampler_top_k(lg, vocab, 8, top);
	fprintf(stderr, "[DEBUG] top logits:");
	for (int i = 0; i < n_top; i++) {
		fprintf(stderr, " [%d]=%.4f", top[i].i, top[i].v);
	}
	fprintf(stderr, "\n");
}

int32_t context_sample_next(context *c) {
	if (c->debug_forward)
		debug_print_logits(c);
	if (c->backend && c->backend->argmax && (c->samp.temperature <= 0.0f || c->samp.top_k == 1) &&
		c->samp.repeat_penalty == 1.0f) {
		int32_t		idx;
		status_code st = c->backend->argmax(c->backend, &c->scratch.slots[RECIPE_SLOT_LOGITS],
											c->m.vocab_size, &idx);
		if (st == OK) {
			sampler_observe(&c->samp, idx);
			return idx;
		}
	}
	int32_t tok = sampler_sample(&c->samp, c->scratch.logits_host, c->m.vocab_size);
	sampler_observe(&c->samp, tok);
	return tok;
}

typedef struct {
	context *c;
} sub_token_cb_ud;

static void sub_token_cb(int layer_idx, int n_layers, int token_progress, int batched, void *ud) {
	(void)batched;
	sub_token_cb_ud *cb = (sub_token_cb_ud *)ud;
	context			*c	= cb->c;
	if (!monitor_active(&c->monitor))
		return;
	moe_stats_summary ms;
	moe_stream_summarize(&c->m, &ms);
	monitor_record_layer_event(&c->monitor, &c->layer_tracker, layer_idx, n_layers, token_progress,
							   &ms);
}

static void monitor_setup_cb(context *c, sub_token_cb_ud *cb, const char *phase, int token_idx,
							 int phase_tokens) {
	if (!monitor_active(&c->monitor))
		return;
	monitor_begin_phase(&c->layer_tracker, phase, token_idx, phase_tokens);
	*cb = (sub_token_cb_ud){.c = c};
	compute_set_layer_progress_cb(&c->scratch, sub_token_cb, cb);
}

static int context_decode_loop(context *c, int max_tokens, float temperature, int top_k,
							   void (*on_token)(int32_t, const char *, int, void *), void *ud) {
	(void)temperature;
	(void)top_k;
	int				generated = 0;
	sub_token_cb_ud cb;

	for (int i = 0; max_tokens < 0 || i < max_tokens; i++) {
		if (c->interrupt)
			break;
		monitor_poll(&c->monitor);
		monitor_setup_cb(c, &cb, "decode", i, i + 1);
		int32_t tok = context_sample_next(c);
		if (tokenizer_is_eog(&c->tok, tok))
			break;
		if (on_token || monitor_active(&c->monitor)) {
			char piece[256];
			int	 pn = tokenizer_decode(&c->tok, &tok, 1, piece, sizeof(piece), &c->scratch.prof);
			if (pn < 0)
				pn = 0;
			piece[pn] = '\0';
			if (on_token)
				on_token(tok, piece, pn, ud);
			if (monitor_active(&c->monitor))
				monitor_emit_token(&c->monitor, i, tok, c->kv.n_pos - 1, piece, pn);
		}
		generated++;
		int rc = context_feed_token_inner(c, tok, c->scratch.logits_host);
		if (rc == CTX_COMPUTE_ERROR) {
			ERROR("decode aborted: GPU compute error at token %d (pos=%d)", i, c->kv.n_pos);
			break;
		}
		if (rc == CTX_INTERRUPTED)
			break;
		if (rc < 0) {
			c->context_limit_hit = true;
			break;
		}
	}

	compute_set_layer_progress_cb(&c->scratch, NULL, NULL);
	return generated;
}

static void send_generation_end(context *c, int generated, double pf_tps, double dec_tps) {
	moe_stats_summary ms;
	moe_stream_summarize(&c->m, &ms);
	monitor_emit_end(&c->monitor, generated, pf_tps, dec_tps, &ms);
}

void context_monitor_send_start(context *c) {
	monitor_emit_start(&c->monitor, c->m.arch_info ? c->m.arch_info->gguf_name : "?", c->m.n_layers,
					   c->m.dim, c->n_ctx, c->m.vocab_size,
					   c->m.arch_info ? c->m.arch_info->is_moe : 0, c->m.moe.n_experts,
					   c->m.moe.n_experts_used);
}

int context_prefill_preamble(context *c, const char *system) {
	if (!system || !*system)
		return 0;

	char  errbuf[512];
	char *preamble;
	if (chat_template_add_turn(&c->chat, "system", system, 0, &preamble, errbuf, sizeof(errbuf)) !=
		OK) {
		ERROR("chat template render failed: %s", errbuf);
		return -1;
	}

	if (c->show_template)
		fprintf(stderr, "=== chat template ===\n%s\n======================\n", preamble);

	int		 cap = c->n_ctx;
	int32_t *ids = xmalloc((size_t)(cap + 1) * sizeof(int32_t));
	int		 n	 = tokenizer_encode_with_specials(&c->tok, preamble, 0, ids, cap, &c->scratch.prof);
	free(preamble);
	if (n < 0) {
		free(ids);
		ERROR("preamble too long");
		return -1;
	}

	context_monitor_send_start(c);

	prefill_result pf = context_prefill_tokens(c, ids, n, "warmup");
	free(ids);
	if (pf.rc == CTX_COMPUTE_ERROR) {
		ERROR("GPU compute error during preamble prefill");
		return -1;
	}
	if (pf.rc < 0) {
		ERROR("context overflow during preamble");
		return -1;
	}
	DEBUG("preamble: %d tokens", n);
	return 0;
}

prefill_result context_prefill_tokens(context *c, const int32_t *tokens, int n_tokens,
									  const char *phase) {
	prefill_result	result = {0};
	sub_token_cb_ud cb;
	monitor_setup_cb(c, &cb, phase, 0, n_tokens);
	uint64_t t_start = time_us();
	result.rc		 = context_feed_tokens_batch(c, tokens, n_tokens);
	if (c->backend && c->backend->synchronize)
		c->backend->synchronize(c->backend);
	uint64_t t_end = time_us();
	result.us	   = t_end - t_start;
	result.tps	   = n_tokens > 0 ? n_tokens * 1.0e6 / (double)result.us : 0.0;
	for (int i = 0; i < n_tokens; i++)
		sampler_observe(&c->samp, tokens[i]);
	monitor_emit_prefill(&c->monitor, n_tokens, result.us / 1000, result.tps);
	return result;
}

static int run_generation(context *c, const int32_t *tokens, int n_tokens, int max_tokens,
						  const sampler_params *samp,
						  void (*on_token)(int32_t, const char *, int, void *), void *ud,
						  int show_pp_tg) {
	context_monitor_send_start(c);
	bool prof_was_on = c->scratch.prof.enabled;

	prefill_result pf = context_prefill_tokens(c, tokens, n_tokens, "prefill");
	if (pf.rc == CTX_INTERRUPTED)
		return 0;
	if (pf.rc == CTX_COMPUTE_ERROR) {
		ERROR("prompt processing failed (GPU compute error at pos=%d)", c->kv.n_pos);
		return -1;
	}
	if (pf.rc < 0) {
		ERROR("prompt too long (pos=%d, ctx=%d)", c->kv.n_pos, c->n_ctx);
		return -1;
	}
	if (prof_was_on)
		profile_print(&c->scratch.prof, "prefill", stderr);

	sampler_set_params(&c->samp, samp->temperature, samp->top_k, samp->top_p, samp->min_p, 1.0f,
					   64);

	uint64_t t_dec_start = time_us();
	int		 generated =
		context_decode_loop(c, max_tokens, samp->temperature, samp->top_k, on_token, ud);
	if (c->backend && c->backend->synchronize)
		c->backend->synchronize(c->backend);
	uint64_t t_dec_end = time_us();

	uint64_t decode_us	= t_dec_end - t_dec_start;
	double	 decode_tps = generated > 0 ? generated * 1.0e6 / (double)decode_us : 0.0;
	fputc('\n', stderr);
	if (show_pp_tg)
		fprintf(stderr, "\033[35m[ PP: %.2f t/s | TG: %.2f t/s ]\033[0m\n", pf.tps, decode_tps);
	if (prof_was_on)
		profile_print(&c->scratch.prof, "decode", stderr);

	send_generation_end(c, generated, pf.tps, decode_tps);
	return generated;
}

typedef struct {
	char  *buf;
	size_t len;
	size_t cap;
	void (*on_token)(int32_t, const char *, int, void *);
	void *ud;
} assistant_capture_ud;

static void assistant_capture_cb(int32_t id, const char *piece, int n, void *ud_) {
	assistant_capture_ud *ud = (assistant_capture_ud *)ud_;
	if (n > 0) {
		if (ud->len + (size_t)n + 1 > ud->cap) {
			ud->cap = (ud->len + (size_t)n + 1) * 2;
			ud->buf = xrealloc(ud->buf, ud->cap);
		}
		memcpy(ud->buf + ud->len, piece, (size_t)n);
		ud->len += (size_t)n;
		ud->buf[ud->len] = '\0';
	}
	if (ud->on_token)
		ud->on_token(id, piece, n, ud->ud);
}

int context_chat_turn(context *c, const char *role, const char *content, bool add_generation_prompt,
					  int max_tokens, const sampler_params						 *samp,
					  void (*on_token)(int32_t, const char *, int, void *), void *ud) {
	char  errbuf[512];
	char *turn_str;
	if (chat_template_add_turn(&c->chat, role, content, add_generation_prompt, &turn_str, errbuf,
							   sizeof(errbuf)) != OK) {
		ERROR("chat template render failed: %s", errbuf);
		return -1;
	}

	if (c->show_template)
		fprintf(stderr, "=== chat template ===\n%s\n======================\n", turn_str);

	int		 cap = c->n_ctx;
	int32_t *ids = xmalloc((size_t)(cap + 1) * sizeof(int32_t));

	profile_reset(&c->scratch.prof);
	int n = tokenizer_encode_with_specials(&c->tok, turn_str, 0, ids, cap, &c->scratch.prof);
	free(turn_str);
	if (n < 0) {
		free(ids);
		return -1;
	}

	assistant_capture_ud acap = {0};
	acap.on_token			  = on_token;
	acap.ud					  = ud;

	int generated =
		run_generation(c, ids, n, max_tokens, samp,
					   add_generation_prompt ? assistant_capture_cb : on_token, &acap, 1);
	free(ids);

	if (add_generation_prompt && generated > 0) {
		char *discard;
		if (chat_template_add_turn(&c->chat, "assistant", acap.buf ? acap.buf : "", 0, &discard,
								   errbuf, sizeof(errbuf)) == OK)
			free(discard);
	}
	free(acap.buf);
	return generated;
}