#define _GNU_SOURCE
#include "model_library.h"

#include "common.h"
#include "schema.h"
#include "settings.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define LIB_MAX_DEPTH 4

static bool has_gguf_suffix(const char *name) {
	size_t n = strlen(name);
	return n > 5 && strcasecmp(name + n - 5, ".gguf") == 0;
}

static void classify(lib_model *m) {
	char		first[LIB_ARCH_MAX];
	const char *slash = strchr(m->path, '/');
	m->type			  = MODEL_TYPE_OTHER;
	m->arch[0]		  = '\0';
	if (!slash)
		return;
	snprintf(first, sizeof(first), "%.*s", (int)MIN((size_t)(slash - m->path), sizeof(first) - 1),
			 m->path);
	int type = model_type_index(first);
	if (type < 0)
		return;
	m->type			   = type;
	const char *rest   = slash + 1;
	const char *slash2 = strchr(rest, '/');
	if (slash2)
		snprintf(m->arch, sizeof(m->arch), "%.*s",
				 (int)MIN((size_t)(slash2 - rest), sizeof(m->arch) - 1), rest);
}

static void fill_model(lib_model *m, const char *path, size_t size, bool custom) {
	memset(m, 0, sizeof(*m));
	snprintf(m->path, sizeof(m->path), "%s", path);
	m->size	  = size;
	m->parts  = 1;
	m->custom = custom;
	m->type	  = MODEL_TYPE_OTHER;
	if (!custom)
		classify(m);
	model_detect_quant(path, m->quant, sizeof(m->quant));
}

static void scan_dir(model_lib *lib, const char *root, const char *rel, int depth) {
	char dir[SC_PATH_MAX];
	snprintf(dir, sizeof(dir), "%s%s%s", root, rel[0] ? "/" : "", rel);
	DIR *d = opendir(dir);
	if (!d)
		return;
	struct dirent *de;
	while ((de = readdir(d)) != NULL && lib->count < LIB_MAX_MODELS) {
		if (de->d_name[0] == '.')
			continue;
		char child[LIB_PATH_MAX], full[SC_PATH_MAX];
		snprintf(child, sizeof(child), "%s%s%s", rel, rel[0] ? "/" : "", de->d_name);
		snprintf(full, sizeof(full), "%s/%s", root, child);
		struct stat sb;
		if (stat(full, &sb) != 0)
			continue;
		if (S_ISDIR(sb.st_mode) && depth < LIB_MAX_DEPTH)
			scan_dir(lib, root, child, depth + 1);
		else if (S_ISREG(sb.st_mode) && has_gguf_suffix(de->d_name))
			fill_model(&lib->items[lib->count++], child, (size_t)sb.st_size, false);
	}
	closedir(d);
}

static int compare_models(const void *a, const void *b) {
	return strcasecmp(((const lib_model *)a)->path, ((const lib_model *)b)->path);
}

static void merge_shards(model_lib *lib) {
	char last_key[LIB_PATH_MAX] = "";
	int	 kept					= 0;
	for (int i = 0; i < lib->count; i++) {
		char key[LIB_PATH_MAX];
		bool sharded = model_shard_key(lib->items[i].path, key, sizeof(key), NULL);
		if (sharded && kept > 0 && strcmp(key, last_key) == 0) {
			lib->items[kept - 1].size += lib->items[i].size;
			lib->items[kept - 1].parts++;
			continue;
		}
		snprintf(last_key, sizeof(last_key), "%s", sharded ? key : "");
		lib->items[kept++] = lib->items[i];
	}
	lib->count = kept;
}

static void add_custom(model_lib *lib) {
	const tui_settings *cfg = tui_settings_get();
	for (int i = 0; i < cfg->n_custom && lib->count < LIB_MAX_MODELS; i++) {
		struct stat sb;
		bool		found = stat(cfg->custom_models[i], &sb) == 0 && S_ISREG(sb.st_mode);
		lib_model  *m	  = &lib->items[lib->count++];
		fill_model(m, cfg->custom_models[i], found ? (size_t)sb.st_size : 0, true);
		m->missing = !found;
	}
}

void model_lib_load(model_lib *lib) {
	lib->items = xcalloc(LIB_MAX_MODELS, sizeof(lib_model));
	lib->count = 0;
	if (tui_models_dir()[0]) {
		scan_dir(lib, tui_models_dir(), "", 0);
		qsort(lib->items, (size_t)lib->count, sizeof(lib_model), compare_models);
		merge_shards(lib);
	}
	add_custom(lib);
}

void model_lib_free(model_lib *lib) {
	free(lib->items);
	lib->items = NULL;
	lib->count = 0;
}

void model_lib_absolute(const lib_model *m, char *out, size_t cap) {
	if (m->custom)
		snprintf(out, cap, "%s", m->path);
	else
		snprintf(out, cap, "%s/%s", tui_models_dir(), m->path);
}

static int delete_parts(const char *abs, char *err, size_t errlen) {
	char key[SC_PATH_MAX];
	if (!model_shard_key(abs, key, sizeof(key), NULL)) {
		if (unlink(abs) != 0) {
			snprintf(err, errlen, "cannot delete '%s': %s", abs, strerror(errno));
			return -1;
		}
		return 0;
	}
	char dir[SC_PATH_MAX];
	snprintf(dir, sizeof(dir), "%s", abs);
	*strrchr(dir, '/') = '\0';
	DIR *d			   = opendir(dir);
	if (!d) {
		snprintf(err, errlen, "cannot open '%s': %s", dir, strerror(errno));
		return -1;
	}
	int			   rc = 0;
	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		char full[SC_PATH_MAX], other[SC_PATH_MAX];
		snprintf(full, sizeof(full), "%s/%s", dir, de->d_name);
		if (model_shard_key(full, other, sizeof(other), NULL) && strcmp(key, other) == 0 &&
			unlink(full) != 0) {
			snprintf(err, errlen, "cannot delete '%s': %s", full, strerror(errno));
			rc = -1;
		}
	}
	closedir(d);
	return rc;
}

static void prune_empty_parents(const char *abs) {
	size_t root_len = strlen(tui_models_dir());
	char   dir[SC_PATH_MAX];
	snprintf(dir, sizeof(dir), "%s", abs);
	for (char *slash = strrchr(dir, '/'); slash && (size_t)(slash - dir) > root_len;
		 slash		 = strrchr(dir, '/')) {
		*slash = '\0';
		if (rmdir(dir) != 0)
			return;
	}
}

int model_lib_delete(const lib_model *m, char *err, size_t errlen) {
	if (m->custom)
		return tui_settings_remove_custom(m->path, err, errlen);
	char abs[SC_PATH_MAX];
	model_lib_absolute(m, abs, sizeof(abs));
	if (delete_parts(abs, err, errlen) < 0)
		return -1;
	prune_empty_parents(abs);
	return 0;
}
