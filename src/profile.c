#include "profile.h"
#include <time.h>

uint64_t time_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

uint64_t time_us(void) {
	return time_ns() / 1000ull;
}

uint64_t time_ms(void) {
	return time_us() / 1000ull;
}

void profile_reset(profile *p) {
	if (!p)
		return;
	for (int i = 0; i < STAGE_COUNT; i++) {
		atomic_store_explicit(&p->total_us[i], 0, memory_order_relaxed);
		atomic_store_explicit(&p->calls[i], 0, memory_order_relaxed);
		atomic_store_explicit(&p->bytes[i], 0, memory_order_relaxed);
	}
	p->wall_us = 0;
}

static const char *stage_name(stage s) {
	static const char *names[] = {
		[STAGE_EMBD]			   = "embd_lookup",
		[STAGE_RMSNORM]			   = "rmsnorm",
		[STAGE_MATMUL]			   = "matmul",
		[STAGE_MATMUL_QKV]		   = "matmul_qkv",
		[STAGE_MATMUL_ATTN_OUT]	   = "matmul_attn_out",
		[STAGE_MATMUL_FFN_GATE_UP] = "matmul_ffn_gate_up",
		[STAGE_MATMUL_FFN_DOWN]	   = "matmul_ffn_down",
		[STAGE_PLE]				   = "ple",
		[STAGE_MOE_ROUTER]		   = "moe_router",
		[STAGE_MOE_IO_WAIT]		   = "moe_io_wait",
		[STAGE_MOE_SHARED]		   = "moe_shared",
		[STAGE_MOE_EXPERT]		   = "moe_expert",
		[STAGE_ROPE]			   = "rope",
		[STAGE_KVPUT]			   = "kv_put",
		[STAGE_ATTN]			   = "attention",
		[STAGE_FFN_ACT]			   = "ffn_activate",
		[STAGE_ADD]				   = "residual_add",
		[STAGE_LOGITS_NORM]		   = "logits_norm",
		[STAGE_LOGITS_MATMUL]	   = "logits_matmul",
		[STAGE_LOGITS_READBACK]	   = "logits_readback",
		[STAGE_TOKENIZE_ENCODE]	   = "tokenize_encode",
		[STAGE_TOKENIZE_DECODE]	   = "tokenize_decode",
	};
	if ((int)s < 0 || s >= ARRAY_LEN(names))
		return "?";
	return names[s];
}

void profile_print(const profile *p, const char *label, FILE *fp) {
	if (!p)
		return;
	uint64_t total = 0;
	uint64_t totals[STAGE_COUNT];
	uint64_t calls[STAGE_COUNT];
	for (int i = 0; i < STAGE_COUNT; i++) {
		totals[i] = atomic_load_explicit(&p->total_us[i], memory_order_relaxed);
		calls[i]  = atomic_load_explicit(&p->calls[i], memory_order_relaxed);
		total += totals[i];
	}

	uint64_t wall = p->wall_us;
	if (wall && total > wall) {
		fprintf(fp,
				"\n[profile] %s  WARNING: stage sum %llu us exceeds wall %llu us -- nested "
				"profile scopes are double-counting; shares below are unreliable\n",
				label ? label : "?", (unsigned long long)total, (unsigned long long)wall);
		total = wall;
	}

	fprintf(fp, "[profile] %s  accounted=%llu us", label ? label : "?", (unsigned long long)total);
	if (wall)
		fprintf(fp, "  wall=%llu us  (%.1f%% attributed)", (unsigned long long)wall,
				100.0 * (double)total / (double)wall);
	fputc('\n', fp);
	if (total == 0) {
		fprintf(fp, "  (no stages recorded)\n");
		return;
	}
	fprintf(fp, "  %-18s %12s %8s %10s %8s %9s %8s\n", "stage", "us", "calls", "us/call", "share",
			"wtMB", "MB/s");
	for (int i = 0; i < STAGE_COUNT; i++) {
		if (calls[i] == 0 && totals[i] == 0)
			continue;
		uint64_t by	   = atomic_load_explicit(&p->bytes[i], memory_order_relaxed);
		double	 share = total > 0 ? 100.0 * (double)totals[i] / (double)total : 0.0;
		double	 per   = calls[i] > 0 ? (double)totals[i] / (double)calls[i] : 0.0;
		double mbps = totals[i] ? (double)by / 1000.0 / (double)totals[i] : 0.0;
		fprintf(fp, "  %-18s %12llu %8llu %10.2f %7.2f%% %9.1f %8.2f\n", stage_name((stage)i),
				(unsigned long long)totals[i], (unsigned long long)calls[i], per, share,
				(double)by / 1e6, mbps);
	}
	if (wall > total)
		fprintf(fp, "  %-18s %12llu %12s %10s %9.2f%%\n", "  (unattributed)",
				(unsigned long long)(wall - total), "-", "-",
				100.0 * (double)(wall - total) / (double)wall);

	uint64_t mm_total = totals[STAGE_MATMUL] + totals[STAGE_MATMUL_QKV] +
						totals[STAGE_MATMUL_ATTN_OUT] + totals[STAGE_MATMUL_FFN_GATE_UP] +
						totals[STAGE_MATMUL_FFN_DOWN];
	uint64_t mm_calls = calls[STAGE_MATMUL] + calls[STAGE_MATMUL_QKV] +
						calls[STAGE_MATMUL_ATTN_OUT] + calls[STAGE_MATMUL_FFN_GATE_UP] +
						calls[STAGE_MATMUL_FFN_DOWN];
	if (mm_calls > 0) {
		double mm_share = total > 0 ? 100.0 * (double)mm_total / (double)total : 0.0;
		double mm_per	= (double)mm_total / (double)mm_calls;
		fprintf(fp, "  %-18s %12llu %12llu %10.2f %9.2f%%\n", "  matmul (subtotal)",
				(unsigned long long)mm_total, (unsigned long long)mm_calls, mm_per, mm_share);
	}
}
