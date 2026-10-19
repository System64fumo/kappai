#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define CONFIG_MAX_LAYER_DEVICES 8
#define CONFIG_PRIORITY_UNSET (-1)

typedef struct config {
	const char *device;
	const char *model;
	const char *moe_pin_list;
	const char *monitor;
	const char *repack;
	const char *fuse;
	uint64_t	seed;

	int			n_threads;
	int			ctx_size;
	int			kv_quant;
	int			moe_cache_cap;
	int			moe_pin;
	int			ngl;
	int			n_layer_devices;
	const char *layer_devices[CONFIG_MAX_LAYER_DEVICES];
	int			layer_counts[CONFIG_MAX_LAYER_DEVICES];
	int			n_expert_devices;
	const char *expert_devices[CONFIG_MAX_LAYER_DEVICES];
	int			expert_counts[CONFIG_MAX_LAYER_DEVICES];
	int			expert_priorities[CONFIG_MAX_LAYER_DEVICES];
	const char *expert_pins[CONFIG_MAX_LAYER_DEVICES];
	int			host_priority;

	bool debug;
	bool use_mmap;
	bool flash_attn;
	bool debug_forward;
	bool moe_stream;
	bool moe_preload;
	bool profile_time;
	bool reasoning;
	bool disable_failsafes;
} config;

config config_defaults(void);

int parse_bool_token(const char *s);

void config_init(const config *cfg);

const config *config_get(void);

typedef struct {
	const char *prompt;
	const char *system;

	bool interactive;
	bool output_stream;

	int	  n_predict;
	float temperature;
	int	  top_k;
	float top_p;
	float min_p;

	float repeat_penalty;
	int	  repeat_last_n;

	bool		list_devices;
	bool		dump_metadata;
	bool		debug_prompt;
	const char *grep_vocab;
	char		metrics[32];
	bool		warmup;

	const char *server_host;
	int			server_port;
	const char *server_api_key;

	bool is_server;
} cli_args;

void config_usage(FILE *fp, bool is_server);

void cli_args_defaults(cli_args *a);

int parse_args(int argc, char **argv, config *cfg, cli_args *a);

#endif
