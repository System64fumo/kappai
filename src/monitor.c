#define _GNU_SOURCE
#include "monitor.h"
#include "log.h"
#include "profile.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define MONITOR_DEFAULT_PATH "/tmp/kappai.monitor"

monitor *g_monitor = NULL;

void monitor_reset(monitor *mon) {
	memset(mon, 0, sizeof(*mon));
	mon->fd = -1;
	for (int i = 0; i < MONITOR_MAX_CLIENTS; i++)
		mon->clients[i] = -1;
	mon->mtx = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
}

status_code monitor_init(monitor *mon, const char *path) {
	monitor_reset(mon);

	const char *sock_path = (path && path[0]) ? path : MONITOR_DEFAULT_PATH;
	snprintf(mon->path, sizeof(mon->path), "%s", sock_path);

	int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	if (fd < 0) {
		WARN("monitor: socket() failed: %s", strerror(errno));
		return ERR_IO;
	}

	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	size_t path_len = strlen(mon->path);
	if (path_len >= sizeof(addr.sun_path)) {
		WARN("monitor: socket path too long (%zu bytes, max %zu): %s", path_len,
			 sizeof(addr.sun_path) - 1, mon->path);
		close(fd);
		return ERR_IO;
	}
	memcpy(addr.sun_path, mon->path, path_len + 1);

	unlink(mon->path);

	if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		WARN("monitor: bind(%s) failed: %s", mon->path, strerror(errno));
		close(fd);
		return ERR_IO;
	}

	if (listen(fd, 4) < 0) {
		WARN("monitor: listen() failed: %s", strerror(errno));
		close(fd);
		unlink(mon->path);
		return ERR_IO;
	}

	mon->fd = fd;
	atomic_store_explicit(&mon->n_clients, 0, memory_order_relaxed);
	INFO("monitor: listening on %s", mon->path);
	return OK;
}

void monitor_poll(monitor *mon) {
	if (!mon || mon->fd < 0)
		return;

	pthread_mutex_lock(&mon->mtx);
	while (atomic_load_explicit(&mon->n_clients, memory_order_relaxed) < MONITOR_MAX_CLIENTS) {
		int cfd = accept4(mon->fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
		if (cfd < 0)
			break;

		int slot = -1;
		for (int i = 0; i < MONITOR_MAX_CLIENTS; i++) {
			if (mon->clients[i] < 0) {
				slot = i;
				break;
			}
		}
		if (slot < 0) {
			close(cfd);
			break;
		}
		mon->clients[slot] = cfd;
		atomic_fetch_add_explicit(&mon->n_clients, 1, memory_order_relaxed);
	}
	pthread_mutex_unlock(&mon->mtx);
}

void monitor_send(monitor *mon, const char *json_fmt, ...) {
	if (!mon || mon->fd < 0 || atomic_load_explicit(&mon->n_clients, memory_order_relaxed) <= 0)
		return;

	char	buf[4096];
	va_list ap;
	va_start(ap, json_fmt);
	int would_write = vsnprintf(buf, sizeof(buf), json_fmt, ap);
	va_end(ap);
	if (would_write <= 0)
		return;

	int n = would_write < (int)sizeof(buf) ? would_write : (int)sizeof(buf) - 1;

	if (n > 0 && buf[n - 1] != '\n' && n < (int)sizeof(buf) - 1) {
		buf[n++] = '\n';
		buf[n]	 = '\0';
	}

	pthread_mutex_lock(&mon->mtx);
	for (int i = 0; i < MONITOR_MAX_CLIENTS; i++) {
		int fd = mon->clients[i];
		if (fd < 0)
			continue;
		ssize_t w = send(fd, buf, (size_t)n, MSG_NOSIGNAL | MSG_DONTWAIT);
		if (w < 0 && (errno == EPIPE || errno == ECONNRESET || errno == EBADF || errno == EAGAIN ||
					  errno == EWOULDBLOCK)) {
			if (mon->clients[i] == fd) {
				close(mon->clients[i]);
				mon->clients[i] = -1;
				atomic_fetch_sub_explicit(&mon->n_clients, 1, memory_order_relaxed);
			}
		}
	}
	pthread_mutex_unlock(&mon->mtx);
}

void monitor_free(monitor *mon) {
	if (!mon)
		return;
	pthread_mutex_lock(&mon->mtx);
	for (int i = 0; i < MONITOR_MAX_CLIENTS; i++) {
		if (mon->clients[i] >= 0) {
			close(mon->clients[i]);
			mon->clients[i] = -1;
		}
	}
	if (mon->fd >= 0) {
		close(mon->fd);
		mon->fd = -1;
	}
	if (mon->path[0])
		unlink(mon->path);
	atomic_store_explicit(&mon->n_clients, 0, memory_order_relaxed);
	pthread_mutex_unlock(&mon->mtx);
	pthread_mutex_destroy(&mon->mtx);
}

static size_t json_escape(const char *in, size_t in_len, char *out, size_t out_cap) {
	size_t n = 0;
	for (size_t i = 0; i < in_len; i++) {
		unsigned char c	  = (unsigned char)in[i];
		const char	 *esc = NULL;
		char		  buf[8];
		switch (c) {
		case '"':
			esc = "\\\"";
			break;
		case '\\':
			esc = "\\\\";
			break;
		case '\n':
			esc = "\\n";
			break;
		case '\r':
			esc = "\\r";
			break;
		case '\t':
			esc = "\\t";
			break;
		default:
			if (c < 0x20) {
				snprintf(buf, sizeof(buf), "\\u%04x", c);
				esc = buf;
			}
			break;
		}
		size_t elen = esc ? strlen(esc) : 1;
		if (n + elen >= out_cap)
			break;
		if (esc)
			memcpy(out + n, esc, elen);
		else
			out[n] = (char)c;
		n += elen;
	}
	out[n] = '\0';
	return n;
}

static void append_moe_json(char *buf, size_t cap, const moe_stats_summary *moe) {
	buf[0] = '\0';
	if (!moe || !moe->has_moe)
		return;
	snprintf(buf, cap, ",\"moe_hit\":%.1f,\"moe_pin\":%.1f,\"moe_lru\":%.1f,\"moe_miss\":%llu",
			 moe->hit_rate, moe->pin_rate, moe->lru_rate, (unsigned long long)moe->cache_misses);
}

void monitor_emit_load_phase_backend(monitor *mon, const char *backend_name) {
	monitor_send(mon, "{\"type\":\"load\",\"phase\":\"backend_init\",\"backend\":\"%s\"}",
				 backend_name ? backend_name : "auto");
	monitor_poll(mon);
}

void monitor_emit_load_phase_model_start(monitor *mon, const char *model_path) {
	monitor_send(mon, "{\"type\":\"load\",\"phase\":\"model_load_start\",\"path\":\"%s\"}",
				 model_path);
	monitor_poll(mon);
}

void monitor_emit_load_phase_model_done(monitor *mon, uint64_t ms, int n_layers, int dim,
										int vocab_size) {
	monitor_send(mon,
				 "{\"type\":\"load\",\"phase\":\"model_load_done\",\"ms\":%llu,\"layers\":%d,"
				 "\"dim\":%d,\"vocab\":%d}",
				 (unsigned long long)ms, n_layers, dim, vocab_size);
	monitor_poll(mon);
}

void monitor_emit_load_phase_model_failed(monitor *mon, status_code err) {
	monitor_send(mon, "{\"type\":\"load\",\"phase\":\"model_load_failed\",\"error\":%d}", err);
}

void monitor_emit_start(monitor *mon, const char *arch_name, int n_layers, int dim, int n_ctx,
						int vocab_size, int is_moe, int n_experts, int n_experts_used) {
	monitor_poll(mon);
	monitor_send(mon,
				 "{\"type\":\"start\",\"arch\":\"%s\",\"n_layers\":%d,\"dim\":%d,\"n_ctx\":%d,"
				 "\"vocab\":%d,\"is_moe\":%d,\"n_experts\":%d,\"topk\":%d}",
				 arch_name ? arch_name : "?", n_layers, dim, n_ctx, vocab_size, is_moe, n_experts,
				 n_experts_used);
}

void monitor_emit_prefill(monitor *mon, int n_tokens, uint64_t ms, double tps) {
	monitor_send(mon, "{\"type\":\"prefill\",\"n_tokens\":%d,\"ms\":%llu,\"tps\":%.2f}", n_tokens,
				 (unsigned long long)ms, tps);
}

void monitor_layer_tracker_init(monitor_layer_tracker *t) {
	memset(t, 0, sizeof(*t));
}

void monitor_begin_phase(monitor_layer_tracker *t, const char *phase, int token_idx,
						 int phase_tokens) {
	int phase_changed = (t->phase == NULL) || (strcmp(t->phase, phase) != 0);
	t->phase		  = phase;
	if (phase_changed) {
		t->phase_cumulative_start_us = time_us();
		t->phase_cumulative_tokens	 = phase_tokens;
	}
	t->token_idx	  = token_idx;
	t->phase_tokens	  = phase_tokens;
	t->phase_start_us = time_us();
}

void monitor_record_layer_event(monitor *mon, monitor_layer_tracker *t, int layer_idx, int n_layers,
								int tokens_done, const moe_stats_summary *moe) {
	if (!monitor_active(mon))
		return;
	uint64_t now = time_ms();
	if (layer_idx < n_layers && now - t->last_send_ms < 50)
		return;
	t->last_send_ms = now;
	double pct		= n_layers > 0 ? (100.0 * (double)layer_idx / (double)n_layers) : 0.0;
	double live_tps = 0.0;
	if (t->phase_start_us > 0) {
		uint64_t elapsed_us = time_us() - t->phase_start_us;
		if (elapsed_us > 0)
			live_tps = 1.0e6 / (double)elapsed_us;
	}
	double cumulative_tps = 0.0;
	if (t->phase_cumulative_start_us > 0) {
		uint64_t cumulative_us = time_us() - t->phase_cumulative_start_us;
		if (cumulative_us > 0) {
			double effective_tokens = (double)t->phase_cumulative_tokens;
			if (strcmp(t->phase, "decode") == 0) {
				effective_tokens = (double)t->token_idx + ((double)layer_idx / (double)n_layers);
			}
			cumulative_tps = effective_tokens * 1.0e6 / (double)cumulative_us;
		}
	}
	char moe_json[160];
	append_moe_json(moe_json, sizeof(moe_json), moe);
	monitor_send(mon,
				 "{\"type\":\"layer\",\"phase\":\"%s\",\"token_idx\":%d,\"n_tokens\":%d,\""
				 "tokens_done\":%d,\"layer\":%d,\"n_layers\":%d,\"pct\":%.1f,\"tps\":%.2f,\""
				 "cumulative_tps\":%.2f%s}",
				 t->phase, t->token_idx, t->phase_tokens, tokens_done, layer_idx, n_layers, pct,
				 live_tps, cumulative_tps, moe_json);
	monitor_poll(mon);
}

void monitor_emit_token(monitor *mon, int token_idx, int32_t token_id, int pos, const char *piece,
						int piece_len) {
	char escaped[512];
	json_escape(piece, (size_t)piece_len, escaped, sizeof(escaped));
	monitor_send(mon,
				 "{\"type\":\"token\",\"token_idx\":%d,\"token_id\":%d,\"pos\":%d,"
				 "\"text\":\"%s\"}",
				 token_idx, (int)token_id, pos, escaped);
}

void monitor_emit_end(monitor *mon, int tokens_generated, double pp_tps, double tg_tps,
					  double ttft_ms, const moe_stats_summary *moe) {
	if (moe && moe->has_moe) {
		monitor_send(mon,
					 "{\"type\":\"end\",\"tokens_generated\":%d,\"pp_tps\":%.2f,\"tg_tps\":%.2f,"
					 "\"ttft_ms\":%.2f,"
					 "\"moe_hit\":%.1f,\"moe_pin\":%.1f,\"moe_lru\":%.1f,\"moe_miss\":%llu,"
					 "\"moe_direct_ok\":%llu,\"moe_direct_fallback\":%llu}",
					 tokens_generated, pp_tps, tg_tps, ttft_ms, moe->hit_rate, moe->pin_rate,
					 moe->lru_rate, (unsigned long long)moe->cache_misses,
					 (unsigned long long)moe->direct_io_ok,
					 (unsigned long long)moe->direct_io_fallback);
	} else {
		monitor_send(mon,
					 "{\"type\":\"end\",\"tokens_generated\":%d,\"pp_tps\":%.2f,\"tg_tps\":%.2f,"
					 "\"ttft_ms\":%.2f}",
					 tokens_generated, pp_tps, tg_tps, ttft_ms);
	}
}

void monitor_emit_moe_experts(monitor *mon, int layer, int token_idx, const int32_t *expert_ids,
							  const float *weights, int K) {
	if (!monitor_active(mon) || atomic_load_explicit(&mon->n_clients, memory_order_relaxed) <= 0)
		return;

	char buf[512];
	int	 n = 0;
	n += snprintf(buf + n, sizeof(buf) - n,
				  "{\"type\":\"moe_experts\",\"layer\":%d,\"token_idx\":%d,\"experts\":[", layer,
				  token_idx);
	for (int k = 0; k < K && n < (int)sizeof(buf) - 64; k++)
		n += snprintf(buf + n, sizeof(buf) - n, "%s%d", k ? "," : "", expert_ids[k]);
	n += snprintf(buf + n, sizeof(buf) - n, "],\"weights\":[");
	for (int k = 0; k < K && n < (int)sizeof(buf) - 64; k++)
		n += snprintf(buf + n, sizeof(buf) - n, "%s%.4f", k ? "," : "", weights[k]);
	snprintf(buf + n, sizeof(buf) - n, "]}\n");
	monitor_send(mon, "%s", buf);
}
