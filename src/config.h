#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct config {
	const char *backend;
	const char *model;
	const char *moe_pin_list;
	const char *monitor;
	const char *repack;
	const char *fuse;
	uint64_t	seed;

	int n_threads;
	int gpu_device;
	int ctx_size;
	int kv_quant;
	int moe_cache_cap;
	int moe_pin;
	int ngl;

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

	bool		list_accels;
	bool		dump_metadata;
	bool		show_template;
	const char *grep_vocab;
	char		metrics[32];
	bool		warmup;

	const char *server_host;
	int			server_port;
	const char *server_api_key;
} cli_args;

void usage(FILE *fp);

int parse_args(int argc, char **argv, config *cfg, cli_args *a);

#endif