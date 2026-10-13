#ifndef TUI_SCHEMA_H
#define TUI_SCHEMA_H

#include "config.h"

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define SC_NAME_MAX 128
#define SC_PATH_MAX PATH_MAX
#define SC_TEXT_MAX 64
#define SC_METRICS_MAX 32
#define SC_PINLIST_MAX 256
#define SC_SYSTEM_MAX 4096
#define SC_HOST_MAX 256

#define SC_MAX_CONFIGS 512
#define SC_ERR_MAX 512

typedef struct {
	char name[SC_NAME_MAX];
	char model[SC_PATH_MAX];
	char device[SC_TEXT_MAX];
	char repack[SC_TEXT_MAX];
	char fuse[SC_TEXT_MAX];
	char moe_pin_list[SC_PINLIST_MAX];
	char monitor[SC_PATH_MAX];
	char metrics[SC_METRICS_MAX];
	char system[SC_SYSTEM_MAX];
	char host[SC_HOST_MAX];
	char api_key[SC_HOST_MAX];

	uint64_t seed;

	int ctx_size;
	int n_threads;
	int ngl;
	int kv_quant;
	int moe_cache_cap;
	int moe_pin;
	int n_predict;
	int top_k;
	int repeat_last_n;
	int server_port;

	float temperature;
	float top_p;
	float min_p;
	float repeat_penalty;

	bool flash_attn;
	bool use_mmap;
	bool moe_stream;
	bool moe_preload;
	bool reasoning;
	bool warmup;
	bool output_stream;
	bool profile_time;
	bool debug;
	bool debug_forward;
	bool disable_failsafes;
} saved_config;

typedef enum {
	FLD_STR,
	FLD_INT,
	FLD_FLOAT,
	FLD_BOOL,
	FLD_ENUM,
	FLD_U64,
} fld_type;

typedef enum {
	UIK_TEXT = 0,
	UIK_PATH,
	UIK_SELECT,
	UIK_MULTI,
	UIK_DEVICE,
} fld_uikind;

typedef struct {
	const char		  *key;
	const char		  *label;
	const char		  *hint;
	fld_type		   type;
	size_t			   offset;
	size_t			   max_len;
	double			   lo;
	double			   hi;
	const char *const *choices;
	fld_uikind		   uik;
	bool			   empty_all;
} field_desc;

typedef struct {
	const char		 *name;
	const field_desc *fields;
	int				  n_fields;
} schema_tab;

extern const schema_tab SCHEMA_TABS[];
extern const int		SCHEMA_TAB_COUNT;

typedef struct {
	char name[SC_NAME_MAX];
	char path[SC_PATH_MAX];
} sc_entry;

void	   *sc_field_ptr(saved_config *sc, const field_desc *f);
const char *sc_format_value(const saved_config *sc, const field_desc *f, char *buf, size_t buflen);
int			sc_parse_value(saved_config *sc, const field_desc *f, const char *value, char *err,
						   size_t errlen);
const char *sc_field_hint(const field_desc *f);

void		sc_defaults(saved_config *sc);
void		sc_sanitize_name(const char *in, char *out, size_t outlen);
const char *sc_configs_dir(void);
int			sc_list(sc_entry *entries, int max, char *err, size_t errlen);
int			sc_load(const char *path, saved_config *sc, char *err, size_t errlen);
int			sc_save(const saved_config *sc, const char *orig_name, char *err, size_t errlen);
int			sc_delete(const sc_entry *entry, char *err, size_t errlen);

void sc_materialize(const saved_config *sc, config *cfg, cli_args *a);

void sc_multi_parse(const char *val, const char *const choices[], int n, bool empty_all, bool *sel);
void sc_multi_format(char *out, size_t cap, const char *const choices[], int n, bool empty_all,
					 const bool *sel);
int	 sc_enum_devices(char out[][SC_TEXT_MAX], int max);
void sc_default_name(char *out, size_t cap);

#endif
