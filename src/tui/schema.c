#include "schema.h"

#include "backend/backend.h"
#include "common.h"
#include "hcs.h"
#include "log.h"
#include "settings.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
	const char *p;
	int			line_no;
} ini_cursor;

static const char *const kv_quant_choices[] = {"f16", "q8_0", NULL};
static const char *const repack_choices[]	= {
	"iq3_s", "q8_0", "iq4_nl", "q4_0", "q4_k", "q5_k", "q6_k", NULL,
};
static const char *const fuse_choices[] = {"qkv", NULL};

#define S_STR(k, l, h, f, cap)                                                                     \
	{.key	  = k,                                                                                 \
	 .label	  = l,                                                                                 \
	 .hint	  = h,                                                                                 \
	 .type	  = FLD_STR,                                                                           \
	 .offset  = offsetof(saved_config, f),                                                         \
	 .max_len = cap,                                                                               \
	 .uik	  = UIK_TEXT}
#define S_PATH(k, l, h, f, cap)                                                                    \
	{.key	  = k,                                                                                 \
	 .label	  = l,                                                                                 \
	 .hint	  = h,                                                                                 \
	 .type	  = FLD_STR,                                                                           \
	 .offset  = offsetof(saved_config, f),                                                         \
	 .max_len = cap,                                                                               \
	 .uik	  = UIK_PATH}
#define S_MODEL_PATH(k, l, h, f, cap, ty)                                                          \
	{.key		 = k,                                                                              \
	 .label		 = l,                                                                              \
	 .hint		 = h,                                                                              \
	 .type		 = FLD_STR,                                                                        \
	 .offset	 = offsetof(saved_config, f),                                                      \
	 .max_len	 = cap,                                                                            \
	 .uik		 = UIK_MODEL_PATH,                                                                 \
	 .model_type = ty}
#define S_MULTI(k, l, h, f, cap, ch, ea)                                                           \
	{.key		= k,                                                                               \
	 .label		= l,                                                                               \
	 .hint		= h,                                                                               \
	 .type		= FLD_STR,                                                                         \
	 .offset	= offsetof(saved_config, f),                                                       \
	 .max_len	= cap,                                                                             \
	 .choices	= ch,                                                                              \
	 .uik		= UIK_MULTI,                                                                       \
	 .empty_all = ea}
#define S_INT(k, l, h, f, lo_, hi_)                                                                \
	{.key	 = k,                                                                                  \
	 .label	 = l,                                                                                  \
	 .hint	 = h,                                                                                  \
	 .type	 = FLD_INT,                                                                            \
	 .offset = offsetof(saved_config, f),                                                          \
	 .lo	 = (double)(lo_),                                                                      \
	 .hi	 = (double)(hi_),                                                                      \
	 .uik	 = UIK_TEXT}
#define S_INT_STEP(k, l, h, f, lo_, hi_, step)                                                     \
	{.key		  = k,                                                                             \
	 .label		  = l,                                                                             \
	 .hint		  = h,                                                                             \
	 .type		  = FLD_INT,                                                                       \
	 .offset	  = offsetof(saved_config, f),                                                     \
	 .lo		  = (double)(lo_),                                                                 \
	 .hi		  = (double)(hi_),                                                                 \
	 .uik		  = UIK_TEXT,                                                                      \
	 .multiple_of = step}
#define S_FLT(k, l, h, f, lo_, hi_)                                                                \
	{.key	 = k,                                                                                  \
	 .label	 = l,                                                                                  \
	 .hint	 = h,                                                                                  \
	 .type	 = FLD_FLOAT,                                                                          \
	 .offset = offsetof(saved_config, f),                                                          \
	 .lo	 = (lo_),                                                                              \
	 .hi	 = (hi_),                                                                              \
	 .uik	 = UIK_TEXT}
#define S_BOOL(k, l, h, f)                                                                         \
	{.key	 = k,                                                                                  \
	 .label	 = l,                                                                                  \
	 .hint	 = h,                                                                                  \
	 .type	 = FLD_BOOL,                                                                           \
	 .offset = offsetof(saved_config, f),                                                          \
	 .uik	 = UIK_TEXT}
#define S_ENUM(k, l, h, f, ch)                                                                     \
	{.key	  = k,                                                                                 \
	 .label	  = l,                                                                                 \
	 .hint	  = h,                                                                                 \
	 .type	  = FLD_ENUM,                                                                          \
	 .offset  = offsetof(saved_config, f),                                                         \
	 .choices = ch,                                                                                \
	 .uik	  = UIK_SELECT}
#define S_U64(k, l, h, f)                                                                          \
	{.key	 = k,                                                                                  \
	 .label	 = l,                                                                                  \
	 .hint	 = h,                                                                                  \
	 .type	 = FLD_U64,                                                                            \
	 .offset = offsetof(saved_config, f),                                                          \
	 .uik	 = UIK_TEXT}

static const field_desc MODEL_FIELDS[] = {
	S_MODEL_PATH("model", "Model path",
				 "GGUF model file, relative paths resolve against the models directory", model,
				 SC_PATH_MAX, "language"),
	S_INT("ctx_size", "Context size", "0 = use model default", ctx_size, 0, INT_MAX),
	S_INT("threads", "Threads", "0 = auto-detect", n_threads, 0, 4096),
};

static const field_desc DIFFUSION_FIELDS[] = {
	S_MODEL_PATH("vae", "VAE path", "VAE GGUF file, relative to the models directory", vae,
				 SC_PATH_MAX, "image"),
	S_MODEL_PATH("llm", "LLM path", "text encoder LLM GGUF file, relative to the models directory",
				 llm, SC_PATH_MAX, "language"),
	S_STR("negative_prompt", "Negative prompt", "what to steer away from, single line",
		  negative_prompt, SC_SYSTEM_MAX),
	S_INT_STEP("width", "Width", "pixels, multiple of 32", width, 32, 16384,
			   DIFFUSION_DIMENSION_STEP),
	S_INT_STEP("height", "Height", "pixels, multiple of 32", height, 32, 16384,
			   DIFFUSION_DIMENSION_STEP),
	S_INT("steps", "Steps", "denoising steps", steps, 1, 1000),
	S_FLT("cfg_scale", "CFG scale", "prompt guidance, 1.0 = no guidance", cfg_scale, 0.0, 1000.0),
	S_PATH("output_dir", "Output directory", "final images are saved here, created if missing",
		   output_dir, SC_PATH_MAX),
	S_PATH("preview_path", "Preview image", "PNG rewritten while generating, empty = off",
		   preview_path, SC_PATH_MAX),
	S_FLT("preview_interval", "Preview interval", "minimum seconds between preview updates",
		  preview_interval, 0.0, 3600.0),
};

static const field_desc ENGINE_FIELDS[] = {
	S_BOOL("flash_attn", "Flash attention", "flash attention kernels", flash_attn),
	S_ENUM("kv_quant", "KV cache quant", "KV cache precision", kv_quant, kv_quant_choices),
	S_BOOL("mmap", "mmap weights", "memory-map model weights", use_mmap),
	S_MULTI("repack", "Repack", "select weight types to repack", repack, SC_TEXT_MAX,
			repack_choices, true),
	S_MULTI("fuse", "Weight fusion", "select load-time fusions", fuse, SC_TEXT_MAX, fuse_choices,
			false),
	S_STR("monitor", "Monitor socket", "monitoring socket path, empty = off", monitor, SC_PATH_MAX),
	S_BOOL("warmup", "Warmup", "pre-process chat template prefix at startup", warmup),
	S_STR("metrics", "Metrics", "comma list: pp, tg, ttft", metrics, SC_METRICS_MAX),
	S_U64("seed", "Seed", "0 = random each run", seed),
	S_BOOL("profile_time", "Timing breakdown", "print per-stage timings", profile_time),
	S_BOOL("debug", "Debug logging", "verbose debug logs", debug),
	S_BOOL("debug_forward", "Forward debug", "per-layer activation stats", debug_forward),
	S_BOOL("disable_failsafes", "Failsafes off", "skip memory guardrails (unsafe)",
		   disable_failsafes),
};

static const field_desc MOE_FIELDS[] = {
	S_BOOL("stream", "Expert streaming", "streaming expert cache for MoE models", moe_stream),
	S_BOOL("preload", "Preload experts", "load all expert weights at startup", moe_preload),
	S_STR("pin_list", "Pin list", "comma-separated expert ids, e.g. 3,17,42", moe_pin_list,
		  SC_PINLIST_MAX),
};

static const field_desc SAMPLING_FIELDS[] = {
	S_STR("system", "System prompt", "default system prompt, single line", system, SC_SYSTEM_MAX),
	S_FLT("temperature", "Temperature", "0 = greedy", temperature, 0.0, 1000.0),
	S_INT("top_k", "Top-k", "0 = disabled", top_k, 0, INT_MAX),
	S_FLT("top_p", "Top-p", "1.0 = disabled", top_p, 0.0, 1.0),
	S_FLT("min_p", "Min-p", "0.0 = disabled", min_p, 0.0, 1.0),
	S_FLT("repeat_penalty", "Repeat penalty", "1.0 = disabled", repeat_penalty, 1e-6, 1000.0),
	S_INT("repeat_last_n", "Repeat window", "tokens considered by repeat penalty", repeat_last_n, 0,
		  INT_MAX),
	S_INT("n_predict", "Max tokens", "tokens per turn, -1 = unlimited", n_predict, -1, INT_MAX),
	S_BOOL("reasoning", "Reasoning", "allow model thinking", reasoning),
	S_BOOL("stream", "Stream output", "stream tokens while generating", output_stream),
};

static const field_desc SERVER_FIELDS[] = {
	S_STR("host", "Bind address", "address the server listens on", host, SC_HOST_MAX),
	S_INT("port", "Port", "listen port", server_port, 1, 65535),
	S_STR("api_key", "API key", "require Bearer auth, empty = off", api_key, SC_HOST_MAX),
	S_INT("slots", "Session slots", "max sessions; created on demand, 1 = off", server_slots, 1,
		  4096),
};

const schema_tab SCHEMA_TABS[] = {
	{"model", MODEL_FIELDS, ARRAY_LEN(MODEL_FIELDS)},
	{"diffusion", DIFFUSION_FIELDS, ARRAY_LEN(DIFFUSION_FIELDS)},
	{"engine", ENGINE_FIELDS, ARRAY_LEN(ENGINE_FIELDS)},
	{"moe", MOE_FIELDS, ARRAY_LEN(MOE_FIELDS)},
	{"sampling", SAMPLING_FIELDS, ARRAY_LEN(SAMPLING_FIELDS)},
	{"server", SERVER_FIELDS, ARRAY_LEN(SERVER_FIELDS)},
};
const int SCHEMA_TAB_COUNT = ARRAY_LEN(SCHEMA_TABS);

void *sc_field_ptr(saved_config *sc, const field_desc *f) {
	return (char *)sc + f->offset;
}

const char *sc_field_hint(const field_desc *f) {
	return f->hint ? f->hint : "";
}

const char *sc_format_value(const saved_config *sc, const field_desc *f, char *buf, size_t buflen) {
	switch (f->type) {
	case FLD_STR: {
		const char *val = (const char *)((const char *)sc + f->offset);
		if (f->uik == UIK_MULTI && !val[0] && f->empty_all)
			return "all";
		return val;
	}
	case FLD_INT:
		snprintf(buf, buflen, "%d", *(int *)((char *)sc + f->offset));
		return buf;
	case FLD_FLOAT:
		snprintf(buf, buflen, "%g", (double)*(float *)((char *)sc + f->offset));
		return buf;
	case FLD_BOOL:
		return *(bool *)((char *)sc + f->offset) ? "on" : "off";
	case FLD_ENUM: {
		int idx = *(int *)((char *)sc + f->offset);
		return f->choices[idx];
	}
	case FLD_U64:
		snprintf(buf, buflen, "%llu", (unsigned long long)*(uint64_t *)((char *)sc + f->offset));
		return buf;
	}
	return "";
}

int sc_parse_value(saved_config *sc, const field_desc *f, const char *value, char *err,
				   size_t errlen) {
	str_span	v	= span_trim(value, strlen(value));
	const char *s	= v.p;
	size_t		len = v.len;

	switch (f->type) {
	case FLD_STR: {
		if (len >= f->max_len) {
			snprintf(err, errlen, "value too long for '%s' (max %zu characters)", f->key,
					 f->max_len - 1);
			return -1;
		}
		char *dst = (char *)sc_field_ptr(sc, f);
		memcpy(dst, s, len);
		dst[len] = '\0';
		return 0;
	}
	case FLD_INT: {
		if (len == 0) {
			snprintf(err, errlen, "empty value for '%s'", f->key);
			return -1;
		}
		char *end	= NULL;
		errno		= 0;
		long long n = strtoll(s, &end, 10);
		if (end == s || (size_t)(end - s) != len) {
			snprintf(err, errlen, "'%s' expects an integer, got '%.*s'", f->key, (int)len, s);
			return -1;
		}
		if ((double)n < f->lo || (double)n > f->hi) {
			snprintf(err, errlen, "'%s' value %lld out of range [%lld, %lld]", f->key, n,
					 (long long)f->lo, (long long)f->hi);
			return -1;
		}
		if (f->multiple_of && n % f->multiple_of != 0) {
			snprintf(err, errlen, "'%s' must be a multiple of %d", f->key, f->multiple_of);
			return -1;
		}
		*(int *)sc_field_ptr(sc, f) = (int)n;
		return 0;
	}
	case FLD_FLOAT: {
		if (len == 0) {
			snprintf(err, errlen, "empty value for '%s'", f->key);
			return -1;
		}
		char *end = NULL;
		errno	  = 0;
		double d  = strtod(s, &end);
		if (end == s || (size_t)(end - s) != len || !isfinite(d)) {
			snprintf(err, errlen, "'%s' expects a number, got '%.*s'", f->key, (int)len, s);
			return -1;
		}
		if (d < f->lo || d > f->hi) {
			snprintf(err, errlen, "'%s' value %g out of range [%g, %g]", f->key, d, f->lo, f->hi);
			return -1;
		}
		*(float *)sc_field_ptr(sc, f) = (float)d;
		return 0;
	}
	case FLD_BOOL: {
		char tok[SC_TEXT_MAX];
		if (len == 0 || len >= sizeof(tok)) {
			snprintf(err, errlen, "empty value for '%s'", f->key);
			return -1;
		}
		memcpy(tok, s, len);
		tok[len] = '\0';
		int b	 = parse_bool_token(tok);
		if (b < 0) {
			snprintf(err, errlen, "'%s' expects on/off, got '%s'", f->key, tok);
			return -1;
		}
		*(bool *)sc_field_ptr(sc, f) = b ? true : false;
		return 0;
	}
	case FLD_ENUM: {
		for (int i = 0; f->choices[i]; i++) {
			if (strlen(f->choices[i]) == len && strncasecmp(f->choices[i], s, len) == 0) {
				*(int *)sc_field_ptr(sc, f) = i;
				return 0;
			}
		}
		snprintf(err, errlen, "'%s' expects one of:", f->key);
		for (int i = 0; f->choices[i]; i++) {
			strncat(err, " ", errlen - strlen(err) - 1);
			strncat(err, f->choices[i], errlen - strlen(err) - 1);
		}
		return -1;
	}
	case FLD_U64: {
		if (len == 0) {
			snprintf(err, errlen, "empty value for '%s'", f->key);
			return -1;
		}
		char *end			 = NULL;
		errno				 = 0;
		unsigned long long n = strtoull(s, &end, 10);
		if (end == s || (size_t)(end - s) != len) {
			snprintf(err, errlen, "'%s' expects an unsigned integer, got '%.*s'", f->key, (int)len,
					 s);
			return -1;
		}
		*(uint64_t *)sc_field_ptr(sc, f) = (uint64_t)n;
		return 0;
	}
	}
	return -1;
}

void sc_defaults(saved_config *sc) {
	memset(sc, 0, sizeof(*sc));

	config	 c = config_defaults();
	cli_args a;
	cli_args_defaults(&a);

	hcs_defaults(&sc->hcs);
	snprintf(sc->metrics, sizeof(sc->metrics), "%s", a.metrics);
	snprintf(sc->host, sizeof(sc->host), "%s", a.server_host);
	snprintf(sc->repack, sizeof(sc->repack), "all");
	snprintf(sc->fuse, sizeof(sc->fuse), "qkv");

	sc->use_mmap   = c.use_mmap;
	sc->flash_attn = c.flash_attn;
	sc->reasoning  = c.reasoning;
	sc->moe_stream = c.moe_stream;

	sc->n_predict	   = a.n_predict;
	sc->temperature	   = a.temperature;
	sc->top_k		   = a.top_k;
	sc->top_p		   = a.top_p;
	sc->min_p		   = a.min_p;
	sc->repeat_penalty = a.repeat_penalty;
	sc->repeat_last_n  = a.repeat_last_n;
	sc->output_stream  = a.output_stream;
	sc->warmup		   = a.warmup;
	sc->server_port	   = a.server_port;
	sc->server_slots   = a.server_slots;

	sc->kv_quant = KV_QUANT_F16;

	snprintf(sc->output_dir, sizeof(sc->output_dir), ".");
	diffusion_params defaults = diffusion_params_defaults();
	sc->width				  = defaults.width;
	sc->height				  = defaults.height;
	sc->steps				  = defaults.steps;
	sc->cfg_scale			  = defaults.cfg_scale;
	sc->preview_interval	  = defaults.preview_interval;
}

void sc_sanitize_name(const char *in, char *out, size_t outlen) {
	size_t n = 0;
	for (const char *p = in; *p && n + 1 < outlen; p++) {
		unsigned char c = (unsigned char)*p;
		if (isalnum(c) || c == '-' || c == '_' || c == '.')
			out[n++] = (char)c;
		else if (isspace(c))
			out[n++] = '_';
	}
	out[n] = '\0';
	if (out[0] == '.' || strcmp(out, ".") == 0 || strcmp(out, "..") == 0)
		out[0] = '_';
}

const char *sc_configs_dir(void) {
	static char dir[SC_PATH_MAX];
	if (dir[0])
		return dir;

	const char *base = getenv("XDG_CONFIG_HOME");
	if (!base || base[0] != '/' || !base[1]) {
		const char *home = getenv("HOME");
		if (!home || !home[0])
			return NULL;
		if (snprintf(dir, sizeof(dir), "%s/.config/sys64/kappai/configs", home) >=
			(int)sizeof(dir)) {
			dir[0] = '\0';
			return NULL;
		}
		return dir;
	}
	if (snprintf(dir, sizeof(dir), "%s/sys64/kappai/configs", base) >= (int)sizeof(dir)) {
		dir[0] = '\0';
		return NULL;
	}
	return dir;
}

static int entry_cmp(const void *pa, const void *pb) {
	const sc_entry *a = (const sc_entry *)pa;
	const sc_entry *b = (const sc_entry *)pb;
	return strcmp(a->name, b->name);
}

int sc_list(sc_entry *entries, int max, char *err, size_t errlen) {
	err[0]			= '\0';
	const char *dir = sc_configs_dir();
	if (!dir) {
		snprintf(err, errlen, "could not determine the config directory (HOME is not set)");
		return -1;
	}

	DIR *d = opendir(dir);
	if (!d) {
		if (errno == ENOENT)
			return 0;
		snprintf(err, errlen, "cannot open '%s': %s", dir, strerror(errno));
		return -1;
	}

	int			   n = 0;
	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		const char *dot = strrchr(de->d_name, '.');
		if (!dot || strcmp(dot, ".conf") != 0)
			continue;
		size_t stem_len = (size_t)(dot - de->d_name);
		if (stem_len == 0 || stem_len >= SC_NAME_MAX)
			continue;
		if (n >= max)
			break;
		memcpy(entries[n].name, de->d_name, stem_len);
		entries[n].name[stem_len] = '\0';
		if (snprintf(entries[n].path, sizeof(entries[n].path), "%s/%s", dir, de->d_name) >=
			(int)sizeof(entries[n].path))
			continue;
		n++;
	}
	closedir(d);

	qsort(entries, (size_t)n, sizeof(entries[0]), entry_cmp);
	return n;
}

static char *read_file(const char *path, size_t *out_len, char *err, size_t errlen) {
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		snprintf(err, errlen, "cannot open '%s': %s", path, strerror(errno));
		return NULL;
	}
	if (fseek(fp, 0, SEEK_END) != 0) {
		snprintf(err, errlen, "cannot read '%s': %s", path, strerror(errno));
		fclose(fp);
		return NULL;
	}
	long sz = ftell(fp);
	if (sz < 0 || sz > 4 * 1024 * 1024) {
		snprintf(err, errlen, "config file '%s' is too large", path);
		fclose(fp);
		return NULL;
	}
	rewind(fp);
	char *data = xmalloc((size_t)sz + 1);
	if (fread(data, 1, (size_t)sz, fp) != (size_t)sz) {
		snprintf(err, errlen, "cannot read '%s': %s", path, strerror(errno));
		free(data);
		fclose(fp);
		return NULL;
	}
	data[sz] = '\0';
	fclose(fp);
	*out_len = (size_t)sz;
	return data;
}

static int ini_next_line(ini_cursor *c, char *out, size_t outcap) {
	size_t used	  = 0;
	int	   joined = 0;
	out[0]		  = '\0';

	for (;;) {
		if (!*c->p)
			return 0;
		const char *nl	  = strchr(c->p, '\n');
		const char *start = c->p;
		size_t		n	  = nl ? (size_t)(nl - c->p) : strlen(c->p);
		c->p			  = nl ? nl + 1 : start + n;
		c->line_no++;

		size_t skip_ws = 0;
		if (joined) {
			while (skip_ws < n && (start[skip_ws] == ' ' || start[skip_ws] == '\t'))
				skip_ws++;
		}
		if (used + (n - skip_ws) + 2 > outcap)
			return -1;
		if (joined && used > 0)
			out[used++] = ' ';
		memcpy(out + used, start + skip_ws, n - skip_ws);
		used += n - skip_ws;
		out[used] = '\0';

		while (used > 0 && (out[used - 1] == '\r' || out[used - 1] == ' ' || out[used - 1] == '\t'))
			out[--used] = '\0';
		if (used > 0 && out[used - 1] == '\\') {
			out[--used] = '\0';
			joined		= 1;
			continue;
		}
		return 1;
	}
}

static const schema_tab *find_tab(const char *name, size_t len) {
	for (int i = 0; i < SCHEMA_TAB_COUNT; i++)
		if (strlen(SCHEMA_TABS[i].name) == len && strncasecmp(SCHEMA_TABS[i].name, name, len) == 0)
			return &SCHEMA_TABS[i];
	return NULL;
}

static const field_desc *find_field(const schema_tab *tab, const char *key, size_t len) {
	for (int i = 0; i < tab->n_fields; i++)
		if (strlen(tab->fields[i].key) == len && strncasecmp(tab->fields[i].key, key, len) == 0)
			return &tab->fields[i];
	return NULL;
}

#define HCS_SECTION_NONE (-2)
#define HCS_SECTION_ROOT (-1)

static bool section_is_hcs(str_span sec, str_span *device) {
	if (sec.len == 3 && strncasecmp(sec.p, "hcs", 3) == 0) {
		device->len = 0;
		return true;
	}
	if (sec.len <= 4 || strncasecmp(sec.p, "hcs.", 4) != 0)
		return false;
	*device = span_trim(sec.p + 4, sec.len - 4);
	return true;
}

static int take_legacy_key(hcs_legacy *legacy, const schema_tab *tab, const char *key,
						   const char *value, char *err, size_t errlen) {
	bool is_model = strcasecmp(tab->name, "model") == 0;
	bool is_moe	  = strcasecmp(tab->name, "moe") == 0;
	if (is_moe && strcasecmp(key, "pin") == 0) {
		char *end = NULL;
		(void)strtol(value, &end, 10);
		if (!*value || *end != '\0') {
			snprintf(err, errlen, "'pin' expects an integer, got '%.16s'", value);
			return -1;
		}
		return 1;
	}
	if (is_model && strcasecmp(key, "device") == 0) {
		if (strlen(value) >= sizeof(legacy->device)) {
			snprintf(err, errlen, "value too long for 'device'");
			return -1;
		}
		snprintf(legacy->device, sizeof(legacy->device), "%s", value);
		legacy->has_device = true;
		return 1;
	}
	if (!(is_model && strcasecmp(key, "ngl") == 0) && !(is_moe && strcasecmp(key, "cache") == 0))
		return 0;

	char *end = NULL;
	long  v	  = strtol(value, &end, 10);
	if (!*value || *end != '\0' || v < -1 || v > INT_MAX) {
		snprintf(err, errlen, "'%s' expects an integer, got '%s'", key, value);
		return -1;
	}
	if (is_model) {
		legacy->has_ngl = true;
		legacy->ngl		= (int)v;
	} else {
		legacy->has_cache = true;
		legacy->cache	  = (int)v;
	}
	return 1;
}

typedef struct {
	const schema_tab *tab;
	int				  hcs_section;
	bool			  hcs_used;
	bool			  skipping;
	bool			  lenient;
	hcs_loader		  loader;
	hcs_legacy		  legacy;
} ini_parser;

typedef struct {
	char  *buf;
	size_t cap;
	size_t len;
} ini_warnings;

static void ini_warn(ini_warnings *w, const char *msg) {
	size_t n = strlen(msg);
	if (!w->buf || w->len + n + 2 > w->cap)
		return;
	memcpy(w->buf + w->len, msg, n);
	w->len += n;
	w->buf[w->len++] = '\n';
	w->buf[w->len]	 = '\0';
}

static int ini_finish_hcs(saved_config *sc, ini_parser *p, char *err, size_t errlen) {
	const hcs_legacy *legacy = &p->legacy;
	if (!p->hcs_used)
		return hcs_migrate_legacy(&sc->hcs, legacy, err, errlen);
	if (legacy->has_device || legacy->has_ngl || legacy->has_cache) {
		snprintf(err, errlen,
				 "legacy keys ([model] device/ngl, [moe] cache) cannot be combined with [hcs]");
		return -1;
	}
	return hcs_loader_finish(&p->loader, &sc->hcs, err, errlen);
}

static int ini_section_line(ini_parser *p, const char *s, size_t len, int line_no, char *err,
							size_t errlen) {
	p->tab			  = NULL;
	p->hcs_section	  = HCS_SECTION_NONE;
	p->skipping		  = true;
	const char *close = memchr(s, ']', len);
	if (!close) {
		snprintf(err, errlen, "line %d: missing ']' after section name", line_no);
		return -1;
	}
	str_span sec = span_trim(s + 1, (size_t)(close - s - 1));
	if (sec.len == 0) {
		snprintf(err, errlen, "line %d: empty section name", line_no);
		return -1;
	}
	str_span device = {0};
	if (section_is_hcs(sec, &device)) {
		p->hcs_used	   = true;
		p->hcs_section = HCS_SECTION_ROOT;
		if (device.len > 0) {
			char name[HCS_NAME_MAX];
			char perr[SC_ERR_MAX];
			snprintf(name, sizeof(name), "%.*s", (int)device.len, device.p);
			p->hcs_section = hcs_loader_open_section(&p->loader, name, perr, sizeof(perr));
			if (p->hcs_section < 0) {
				p->hcs_section = HCS_SECTION_NONE;
				snprintf(err, errlen, "line %d: %s", line_no, perr);
				return -1;
			}
		}
		p->skipping = false;
		return 0;
	}
	p->tab = find_tab(sec.p, sec.len);
	if (!p->tab) {
		snprintf(err, errlen, "line %d: unknown section '%.*s'", line_no, (int)sec.len, sec.p);
		return -1;
	}
	p->skipping = false;
	return 0;
}

static int ini_key_line(ini_parser *p, saved_config *sc, const char *s, size_t len, int line_no,
						char *err, size_t errlen) {
	const char *eq = memchr(s, '=', len);
	if (!eq) {
		snprintf(err, errlen, "line %d: expected 'key = value'", line_no);
		return -1;
	}
	str_span key = span_trim(s, (size_t)(eq - s));
	str_span val = span_trim(eq + 1, len - (size_t)(eq - s) - 1);
	if (key.len == 0) {
		snprintf(err, errlen, "line %d: empty key", line_no);
		return -1;
	}
	if (p->skipping)
		return 0;
	if (!p->tab && p->hcs_section == HCS_SECTION_NONE) {
		snprintf(err, errlen, "line %d: '%.*s' appears before any section", line_no, (int)key.len,
				 key.p);
		return -1;
	}
	char perr[SC_ERR_MAX];
	char keybuf[SC_TEXT_MAX];
	char valbuf[SC_SYSTEM_MAX * 2];
	if (val.len >= sizeof(valbuf))
		val.len = sizeof(valbuf) - 1;
	snprintf(keybuf, sizeof(keybuf), "%.*s", (int)key.len, key.p);
	memcpy(valbuf, val.p, val.len);
	valbuf[val.len] = '\0';

	if (!p->tab) {
		if (hcs_loader_set(&p->loader, p->hcs_section, keybuf, valbuf, perr, sizeof(perr)) < 0) {
			snprintf(err, errlen, "line %d: %s", line_no, perr);
			return -1;
		}
		return 0;
	}
	int legacy_rc = take_legacy_key(&p->legacy, p->tab, keybuf, valbuf, perr, sizeof(perr));
	if (legacy_rc < 0) {
		snprintf(err, errlen, "line %d: %s", line_no, perr);
		return -1;
	}
	if (legacy_rc > 0)
		return 0;

	const field_desc *f = find_field(p->tab, key.p, key.len);
	if (!f) {
		snprintf(err, errlen, "line %d: unknown key '%.*s' in section [%s]", line_no, (int)key.len,
				 key.p, p->tab->name);
		return -1;
	}
	if (sc_parse_value(sc, f, valbuf, perr, sizeof(perr)) < 0) {
		snprintf(err, errlen, "line %d: %s", line_no, perr);
		return -1;
	}
	return 0;
}

static int ini_apply(saved_config *sc, const char *data, bool lenient, ini_warnings *warn,
					 char *err, size_t errlen) {
	ini_cursor cur = {data, 0};
	char	   line[SC_SYSTEM_MAX * 2];
	ini_parser p = {.hcs_section = HCS_SECTION_NONE, .lenient = lenient};
	hcs_loader_init(&p.loader);

	for (;;) {
		int r = ini_next_line(&cur, line, sizeof(line));
		if (r == 0)
			break;
		int		 rc = -1;
		str_span sp = span_trim(line, strlen(line));
		if (r < 0)
			snprintf(err, errlen, "line %d: line too long", cur.line_no);
		else if (sp.len == 0 || sp.p[0] == '#' || sp.p[0] == ';')
			continue;
		else if (sp.p[0] == '[')
			rc = ini_section_line(&p, sp.p, sp.len, cur.line_no, err, errlen);
		else
			rc = ini_key_line(&p, sc, sp.p, sp.len, cur.line_no, err, errlen);
		if (rc == 0)
			continue;
		if (!lenient)
			return -1;
		ini_warn(warn, err);
	}

	int rc = ini_finish_hcs(sc, &p, err, errlen);
	if (rc == 0 || !lenient)
		return rc;
	ini_warn(warn, err);
	if (p.hcs_used)
		hcs_loader_finish_lenient(&p.loader, &sc->hcs);
	return 0;
}

static int sc_load_impl(const char *path, saved_config *sc, bool lenient, ini_warnings *warn,
						char *err, size_t errlen) {
	sc_defaults(sc);

	size_t len	= 0;
	char  *data = read_file(path, &len, err, errlen);
	if (!data)
		return -1;

	const char *parse = data;
	if (len >= 3 && (unsigned char)data[0] == 0xEF && (unsigned char)data[1] == 0xBB &&
		(unsigned char)data[2] == 0xBF)
		parse = data + 3;

	int rc = ini_apply(sc, parse, lenient, warn, err, errlen);
	if (rc == 0) {
		if (!sc->hcs.enabled)
			hcs_set_simple(&sc->hcs, sc->hcs.devices[0].name);
		const char *base = path_basename(path);
		size_t		bl	 = strlen(base);
		if (bl > 5 && strcmp(base + bl - 5, ".conf") == 0)
			bl -= 5;
		if (bl >= SC_NAME_MAX)
			bl = SC_NAME_MAX - 1;
		memcpy(sc->name, base, bl);
		sc->name[bl] = '\0';
	}
	free(data);
	return rc;
}

int sc_load(const char *path, saved_config *sc, char *err, size_t errlen) {
	ini_warnings none = {0};
	return sc_load_impl(path, sc, false, &none, err, errlen);
}

int sc_load_lenient(const char *path, saved_config *sc, char *warn, size_t warnlen, char *err,
					size_t errlen) {
	if (warnlen)
		warn[0] = '\0';
	ini_warnings w = {warn, warnlen, 0};
	return sc_load_impl(path, sc, true, &w, err, errlen);
}

static void resolve_in_place(char *path, size_t cap) {
	char resolved[SC_PATH_MAX];
	tui_resolve_path(path, resolved, sizeof(resolved));
	snprintf(path, cap, "%s", resolved);
}

void sc_resolve_model_paths(saved_config *sc) {
	resolve_in_place(sc->model, sizeof(sc->model));
	resolve_in_place(sc->vae, sizeof(sc->vae));
	resolve_in_place(sc->llm, sizeof(sc->llm));
}

static void write_hcs(FILE *fp, const hcs_config *h) {
	char devices[HCS_MAX_DEVICES * (HCS_NAME_MAX + 2)];
	hcs_format_devices(h, devices, sizeof(devices));
	fprintf(fp, "[hcs]\nenabled = %s\ndevices = %s\n\n", h->enabled ? "on" : "off", devices);
	for (int i = 0; i < h->n_devices; i++) {
		const hcs_device *d = &h->devices[i];
		char			  tasks[SC_TEXT_MAX];
		char			  count[16];
		hcs_format_tasks(d->tasks, tasks, sizeof(tasks));
		fprintf(fp, "[hcs.%s]\n", d->name);
		if (d->tasks)
			fprintf(fp, "tasks = %s\n", tasks);
		if (d->layers != HCS_COUNT_UNSET)
			fprintf(fp, "layers = %s\n", hcs_format_count(d->layers, count, sizeof(count)));
		if (d->experts != HCS_COUNT_UNSET)
			fprintf(fp, "experts = %s\n", hcs_format_count(d->experts, count, sizeof(count)));
		if (d->pins[0])
			fprintf(fp, "pins = %s\n", d->pins);
		if (d->has_priority)
			fprintf(fp, "priority = %d\n", d->priority);
		fputc('\n', fp);
	}
}

int sc_save(const saved_config *sc, const char *orig_name, char *err, size_t errlen) {
	char stem[SC_NAME_MAX];
	sc_sanitize_name(sc->name, stem, sizeof(stem));
	if (!stem[0]) {
		snprintf(err, errlen, "config name is empty");
		return -1;
	}

	const char *dir = sc_configs_dir();
	if (!dir) {
		snprintf(err, errlen, "could not determine the config directory (HOME is not set)");
		return -1;
	}

	char path[SC_PATH_MAX];
	char tmp[SC_PATH_MAX];
	if (snprintf(path, sizeof(path), "%s/%s.conf", dir, stem) >= (int)sizeof(path) ||
		snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) {
		snprintf(err, errlen, "config path too long");
		return -1;
	}

	if (strcmp(stem, orig_name ? orig_name : "") != 0 && access(path, F_OK) == 0) {
		snprintf(err, errlen, "a config named '%s' already exists", stem);
		return -1;
	}
	if (mkdir_p(dir, err, errlen) < 0)
		return -1;

	FILE *fp = fopen(tmp, "w");
	if (!fp) {
		snprintf(err, errlen, "cannot write '%s': %s", tmp, strerror(errno));
		return -1;
	}

	for (int t = 0; t < SCHEMA_TAB_COUNT; t++) {
		const schema_tab *tab = &SCHEMA_TABS[t];
		fprintf(fp, "[%s]\n", tab->name);
		for (int i = 0; i < tab->n_fields; i++) {
			const field_desc *f = &tab->fields[i];
			char			  vbuf[SC_SYSTEM_MAX + 16];
			const char		 *val = sc_format_value(sc, f, vbuf, sizeof(vbuf));
			fprintf(fp, "%s = %s\n", f->key, val);
		}
		fputc('\n', fp);
	}

	write_hcs(fp, &sc->hcs);

	if (fclose(fp) != 0) {
		snprintf(err, errlen, "cannot write '%s': %s", tmp, strerror(errno));
		unlink(tmp);
		return -1;
	}
	if (rename(tmp, path) != 0) {
		snprintf(err, errlen, "cannot save '%s': %s", path, strerror(errno));
		unlink(tmp);
		return -1;
	}
	return 0;
}

int sc_delete(const sc_entry *entry, char *err, size_t errlen) {
	if (unlink(entry->path) != 0) {
		snprintf(err, errlen, "cannot delete '%s': %s", entry->path, strerror(errno));
		return -1;
	}
	return 0;
}

void sc_materialize(const saved_config *sc, config *cfg, cli_args *a) {
	*cfg = config_defaults();
	cli_args_defaults(a);

	cfg->model		  = sc->model[0] ? sc->model : NULL;
	cfg->repack		  = sc->repack[0] ? sc->repack : NULL;
	cfg->fuse		  = sc->fuse[0] ? sc->fuse : NULL;
	cfg->moe_pin_list = sc->moe_pin_list[0] ? sc->moe_pin_list : NULL;
	cfg->monitor	  = sc->monitor[0] ? sc->monitor : NULL;
	cfg->seed		  = sc->seed;
	cfg->n_threads	  = sc->n_threads;
	cfg->ctx_size	  = sc->ctx_size;
	cfg->kv_quant	  = sc->kv_quant == 1 ? KV_QUANT_Q8_0 : KV_QUANT_F16;
	hcs_materialize(&sc->hcs, sc->model, cfg);
	cfg->debug			   = sc->debug;
	cfg->use_mmap		   = sc->use_mmap;
	cfg->flash_attn		   = sc->flash_attn;
	cfg->debug_forward	   = sc->debug_forward;
	cfg->moe_stream		   = sc->moe_stream;
	cfg->moe_preload	   = sc->moe_preload;
	cfg->profile_time	   = sc->profile_time;
	cfg->reasoning		   = sc->reasoning;
	cfg->disable_failsafes = sc->disable_failsafes;

	a->system		  = sc->system;
	a->n_predict	  = sc->n_predict;
	a->temperature	  = sc->temperature;
	a->top_k		  = sc->top_k;
	a->top_p		  = sc->top_p;
	a->min_p		  = sc->min_p;
	a->repeat_penalty = sc->repeat_penalty;
	a->repeat_last_n  = sc->repeat_last_n;
	a->output_stream  = sc->output_stream;
	snprintf(a->metrics, sizeof(a->metrics), "%s", sc->metrics);
	a->warmup		  = sc->warmup;
	a->server_host	  = sc->host[0] ? sc->host : "127.0.0.1";
	a->server_port	  = sc->server_port;
	a->server_slots	  = sc->server_slots > 0 ? sc->server_slots : 4;
	a->server_api_key = sc->api_key[0] ? sc->api_key : NULL;
}

void sc_materialize_diffusion(const saved_config *sc, diffusion_files *files, diffusion_params *p,
							  diffusion_output *out) {
	*files = (diffusion_files){
		.diffusion_model = sc->model,
		.device			 = sc->hcs.devices[0].name,
		.vae			 = sc->vae,
		.llm			 = sc->llm,
	};
	*p = (diffusion_params){
		.negative_prompt  = sc->negative_prompt,
		.width			  = sc->width,
		.height			  = sc->height,
		.steps			  = sc->steps,
		.cfg_scale		  = sc->cfg_scale,
		.seed			  = sc->seed,
		.preview_path	  = sc->preview_path[0] ? sc->preview_path : NULL,
		.preview_interval = sc->preview_interval,
	};
	*out = (diffusion_output){.dir = sc->output_dir, .name = sc->name};
}

void sc_multi_parse(const char *val, const char *const choices[], int n, bool empty_all,
					bool *sel) {
	for (int i = 0; i < n; i++)
		sel[i] = false;

	str_span v = span_trim(val ? val : "", val ? strlen(val) : 0);
	if (v.len == 0) {
		if (empty_all)
			for (int i = 0; i < n; i++)
				sel[i] = true;
		return;
	}

	char   head[SC_TEXT_MAX];
	size_t hl = MIN(v.len, sizeof(head) - 1);
	memcpy(head, v.p, hl);
	head[hl] = '\0';
	if (strcasecmp(head, "all") == 0) {
		for (int i = 0; i < n; i++)
			sel[i] = true;
		return;
	}
	if (strcasecmp(head, "none") == 0)
		return;

	const char *p = head;
	while (*p) {
		const char *comma = strchr(p, ',');
		size_t		len	  = comma ? (size_t)(comma - p) : strlen(p);
		while (len > 0 && (*p == ' ' || *p == '\t')) {
			p++;
			len--;
		}
		while (len > 0 && (p[len - 1] == ' ' || p[len - 1] == '\t'))
			len--;
		for (int i = 0; i < n; i++)
			if (strlen(choices[i]) == len && strncasecmp(choices[i], p, len) == 0)
				sel[i] = true;
		if (!comma)
			break;
		p = comma + 1;
	}
}

void sc_multi_format(char *out, size_t cap, const char *const choices[], int n, bool empty_all,
					 const bool *sel) {
	int on = 0;
	for (int i = 0; i < n; i++)
		if (sel[i])
			on++;

	out[0] = '\0';
	if (on == 0) {
		if (empty_all)
			snprintf(out, cap, "none");
		return;
	}
	if (on == n && empty_all) {
		snprintf(out, cap, "all");
		return;
	}

	size_t used = 0;
	for (int i = 0; i < n && on > 0; i++) {
		if (!sel[i])
			continue;
		const char *tok	 = choices[i];
		size_t		tl	 = strlen(tok);
		size_t		need = used ? tl + 1 : tl;
		if (used + need + 1 > cap)
			break;
		if (used)
			out[used++] = ',';
		memcpy(out + used, tok, tl);
		used += tl;
		out[used] = '\0';
		on--;
	}
}

static bool name_taken(const char *dir, const char *name) {
	char path[SC_PATH_MAX];
	if (snprintf(path, sizeof(path), "%s/%s.conf", dir, name) >= (int)sizeof(path))
		return false;
	return access(path, F_OK) == 0;
}

void sc_default_name(char *out, size_t cap) {
	snprintf(out, cap, "new-config");
	const char *dir = sc_configs_dir();
	if (!dir || !name_taken(dir, out))
		return;
	for (int i = 2; i < 1000; i++) {
		char cand[SC_NAME_MAX];
		if (snprintf(cand, sizeof(cand), "new-config-%d", i) >= (int)sizeof(cand))
			return;
		if (!name_taken(dir, cand)) {
			snprintf(out, cap, "%s", cand);
			return;
		}
	}
}
