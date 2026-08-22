#ifndef MONITOR_H
#define MONITOR_H

#include "common.h"
#include "moe/moe_stream.h"

#include <pthread.h>
#include <stdatomic.h>

#define MONITOR_MAX_CLIENTS 8

typedef struct monitor {
	int				fd;
	int				clients[MONITOR_MAX_CLIENTS];
	_Atomic int		n_clients;
	char			path[256];
	pthread_mutex_t mtx;
} monitor;

status_code monitor_init(monitor *mon, const char *path);

void monitor_reset(monitor *mon);

void monitor_send(monitor *mon, const char *json_fmt, ...) __attribute__((format(printf, 2, 3)));

void monitor_poll(monitor *mon);

void monitor_free(monitor *mon);

static inline int monitor_active(const monitor *mon) {
	return mon && mon->fd >= 0;
}

extern monitor *g_monitor;

int monitor_maybe_emit(void);

void monitor_emit_load_phase_backend(monitor *mon, const char *backend_name);
void monitor_emit_load_phase_model_start(monitor *mon, const char *model_path);
void monitor_emit_load_phase_model_done(monitor *mon, uint64_t ms, int n_layers, int dim,
										int vocab_size);
void monitor_emit_load_phase_model_failed(monitor *mon, status_code err);

void monitor_emit_start(monitor *mon, const char *arch_name, int n_layers, int dim, int n_ctx,
						int vocab_size, int is_moe, int n_experts, int n_experts_used);

void monitor_emit_prefill(monitor *mon, int n_tokens, uint64_t ms, double tps);

typedef struct {
	const char *phase;
	uint64_t	phase_cumulative_start_us;
	int			phase_cumulative_tokens;
	uint64_t	last_send_ms;
	uint64_t	phase_start_us;
	int			token_idx;
	int			phase_tokens;
} monitor_layer_tracker;

void monitor_layer_tracker_init(monitor_layer_tracker *t);
void monitor_begin_phase(monitor_layer_tracker *t, const char *phase, int token_idx,
						 int phase_tokens);
void monitor_record_layer_event(monitor *mon, monitor_layer_tracker *t, int layer_idx, int n_layers,
								int tokens_done, const moe_stats_summary *moe);

void monitor_emit_token(monitor *mon, int token_idx, int32_t token_id, int pos, const char *piece,
						int piece_len);

void monitor_emit_end(monitor *mon, int tokens_generated, double pp_tps, double tg_tps,
					  double ttft_ms, const moe_stats_summary *moe);

void monitor_emit_moe_experts(monitor *mon, int layer, int token_idx, const int32_t *expert_ids,
							  const float *weights, int K);

#endif