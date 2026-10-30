#include "settings.h"

#include "common.h"
#include "schema.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define SETTINGS_FILE "tui.conf"

static tui_settings current;

static void settings_defaults(tui_settings *t) {
	memset(t, 0, sizeof(*t));
	t->confirm_delete = true;
}

const tui_settings *tui_settings_get(void) {
	return &current;
}

const char *tui_models_dir(void) {
	return current.models_dir;
}

static int settings_path(char *out, size_t cap, char *err, size_t errlen) {
	const char *configs = sc_configs_dir();
	if (!configs) {
		snprintf(err, errlen, "could not determine the config directory (HOME is not set)");
		return -1;
	}
	const char *slash = strrchr(configs, '/');
	size_t		root  = slash ? (size_t)(slash - configs) : 0;
	if (snprintf(out, cap, "%.*s/%s", (int)root, configs, SETTINGS_FILE) >= (int)cap) {
		snprintf(err, errlen, "settings path too long");
		return -1;
	}
	return 0;
}

static bool key_is(str_span key, const char *name) {
	return key.len == strlen(name) && strncasecmp(key.p, name, key.len) == 0;
}

static void copy_value(char *dst, size_t cap, str_span val) {
	if (val.len < cap) {
		memcpy(dst, val.p, val.len);
		dst[val.len] = '\0';
	}
}

static bool parse_flag(str_span val, bool fallback) {
	char tok[16];
	if (val.len == 0 || val.len >= sizeof(tok))
		return fallback;
	memcpy(tok, val.p, val.len);
	tok[val.len] = '\0';
	int b		 = parse_bool_token(tok);
	return b < 0 ? fallback : b != 0;
}

static void apply_line(const char *line) {
	str_span sp = span_trim(line, strlen(line));
	if (sp.len == 0 || sp.p[0] == '#' || sp.p[0] == ';' || sp.p[0] == '[')
		return;
	const char *eq = memchr(sp.p, '=', sp.len);
	if (!eq)
		return;
	str_span key = span_trim(sp.p, (size_t)(eq - sp.p));
	str_span val = span_trim(eq + 1, sp.len - (size_t)(eq - sp.p) - 1);
	if (key_is(key, "models_dir"))
		copy_value(current.models_dir, sizeof(current.models_dir), val);
	else if (key_is(key, "default_mode"))
		current.default_server = val.len == 6 && strncasecmp(val.p, "server", 6) == 0;
	else if (key_is(key, "custom_model") && current.n_custom < TUI_CUSTOM_MAX)
		copy_value(current.custom_models[current.n_custom++], TUI_CUSTOM_PATH_MAX, val);
	else if (key_is(key, "setup_done"))
		current.setup_done = parse_flag(val, current.setup_done);
	else if (key_is(key, "confirm_delete"))
		current.confirm_delete = parse_flag(val, current.confirm_delete);
}

int tui_settings_load(char *err, size_t errlen) {
	settings_defaults(&current);
	char path[SC_PATH_MAX];
	if (settings_path(path, sizeof(path), err, errlen) < 0)
		return -1;

	FILE *fp = fopen(path, "r");
	if (!fp) {
		if (errno == ENOENT)
			return 0;
		snprintf(err, errlen, "cannot open '%s': %s", path, strerror(errno));
		return -1;
	}
	char line[SC_PATH_MAX + 64];
	while (fgets(line, sizeof(line), fp))
		apply_line(line);
	fclose(fp);
	return 0;
}

int tui_expand_path(const char *value, char *out, size_t cap, char *err, size_t errlen) {
	str_span	sp	 = span_trim(value, strlen(value));
	const char *home = getenv("HOME");
	int			n;
	if (sp.len >= 2 && sp.p[0] == '~' && sp.p[1] == '/' && home && home[0])
		n = snprintf(out, cap, "%s%.*s", home, (int)sp.len - 1, sp.p + 1);
	else
		n = snprintf(out, cap, "%.*s", (int)sp.len, sp.p);
	if (n >= (int)cap) {
		snprintf(err, errlen, "path too long");
		return -1;
	}
	while (n > 1 && out[n - 1] == '/')
		out[--n] = '\0';
	if (out[0] && out[0] != '/') {
		snprintf(err, errlen, "expected an absolute path");
		return -1;
	}
	return 0;
}

int tui_models_dir_check(const char *value, char *expanded, size_t cap, char *err, size_t errlen) {
	if (tui_expand_path(value, expanded, cap, err, errlen) < 0)
		return -1;
	if (!expanded[0])
		return 0;
	struct stat st;
	if (stat(expanded, &st) != 0 || !S_ISDIR(st.st_mode)) {
		snprintf(err, errlen, "'%s' is not a directory", expanded);
		return -1;
	}
	return 0;
}

int tui_custom_check(const char *value, char *expanded, size_t cap, char *err, size_t errlen) {
	if (tui_expand_path(value, expanded, cap, err, errlen) < 0)
		return -1;
	size_t len = strlen(expanded);
	if (!len) {
		snprintf(err, errlen, "enter the path of a .gguf file");
		return -1;
	}
	struct stat st;
	if (stat(expanded, &st) != 0 || !S_ISREG(st.st_mode)) {
		snprintf(err, errlen, "'%s' is not a file", expanded);
		return -1;
	}
	if (len <= 5 || strcasecmp(expanded + len - 5, ".gguf") != 0) {
		snprintf(err, errlen, "expected a .gguf file");
		return -1;
	}
	return 0;
}

int tui_settings_save(const tui_settings *next, char *err, size_t errlen) {
	char path[SC_PATH_MAX];
	char tmp[SC_PATH_MAX];
	if (settings_path(path, sizeof(path), err, errlen) < 0)
		return -1;
	if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) {
		snprintf(err, errlen, "settings path too long");
		return -1;
	}
	char root[SC_PATH_MAX];
	snprintf(root, sizeof(root), "%.*s", (int)(strrchr(path, '/') - path), path);
	if (mkdir_p(root, err, errlen) < 0)
		return -1;

	FILE *fp = fopen(tmp, "w");
	if (!fp) {
		snprintf(err, errlen, "cannot write '%s': %s", tmp, strerror(errno));
		return -1;
	}
	fchmod(fileno(fp), 0600);
	fprintf(fp,
			"[tui]\nmodels_dir = %s\ndefault_mode = %s\nconfirm_delete = "
			"%s\nsetup_done = %s\n",
			next->models_dir, next->default_server ? "server" : "chat",
			next->confirm_delete ? "on" : "off", next->setup_done ? "on" : "off");
	for (int i = 0; i < next->n_custom; i++)
		fprintf(fp, "custom_model = %s\n", next->custom_models[i]);
	if (fclose(fp) != 0 || rename(tmp, path) != 0) {
		snprintf(err, errlen, "cannot save '%s': %s", path, strerror(errno));
		unlink(tmp);
		return -1;
	}
	current = *next;
	return 0;
}

void tui_resolve_path(const char *path, char *out, size_t cap) {
	if (!path[0] || path[0] == '/' || !current.models_dir[0] ||
		snprintf(out, cap, "%s/%s", current.models_dir, path) >= (int)cap)
		snprintf(out, cap, "%s", path);
}

static int custom_index(const tui_settings *t, const char *path) {
	for (int i = 0; i < t->n_custom; i++)
		if (strcmp(t->custom_models[i], path) == 0)
			return i;
	return -1;
}

int tui_settings_add_custom(const char *path, char *err, size_t errlen) {
	tui_settings next = current;
	if (custom_index(&next, path) >= 0)
		return 0;
	if (next.n_custom >= TUI_CUSTOM_MAX) {
		snprintf(err, errlen, "at most %d custom models are supported", TUI_CUSTOM_MAX);
		return -1;
	}
	if (snprintf(next.custom_models[next.n_custom], TUI_CUSTOM_PATH_MAX, "%s", path) >=
		TUI_CUSTOM_PATH_MAX) {
		snprintf(err, errlen, "path too long");
		return -1;
	}
	next.n_custom++;
	return tui_settings_save(&next, err, errlen);
}

int tui_settings_remove_custom(const char *path, char *err, size_t errlen) {
	tui_settings next = current;
	int			 at	  = custom_index(&next, path);
	if (at < 0)
		return 0;
	for (int i = at; i + 1 < next.n_custom; i++)
		memcpy(next.custom_models[i], next.custom_models[i + 1], TUI_CUSTOM_PATH_MAX);
	next.n_custom--;
	return tui_settings_save(&next, err, errlen);
}
