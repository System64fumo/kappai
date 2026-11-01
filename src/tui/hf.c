#define _GNU_SOURCE
#include "hf.h"

#include "common.h"
#include "dialog.h"
#include "hf_meta.h"
#include "json_helpers.h"
#include "model_names.h"
#include "schema.h"
#include "settings.h"
#include "ui.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define HF_HOST "https://huggingface.co"
#define HF_SEARCH_LIMIT 30
#define HF_MAX_FILES 200
#define HF_NAME_MAX 256
#define HF_LABEL_MAX 320
#define HF_BODY_MAX (8 * 1024 * 1024)
#define HF_PROC_ERR_MAX 512
#define HF_QUANT_MAX MODEL_QUANT_MAX
#define HF_HEADER_MAX (48 * 1024 * 1024)
#define HF_ENOUGH_FIRST (128 * 1024)
#define HF_ENOUGH_STEP (512 * 1024)
#define HF_JSON_TIMEOUT 30
#define HF_HEADER_TIMEOUT 120

typedef struct {
	pid_t pid;
	int	  out_fd;
	int	  err_fd;
} hf_proc;

typedef struct {
	char   name[HF_NAME_MAX];
	size_t size;
	char   quant[HF_QUANT_MAX];
	char   label[HF_LABEL_MAX];
} hf_entry;

static void url_encode(const char *in, char *out, size_t cap) {
	size_t n = 0;
	for (const unsigned char *p = (const unsigned char *)in; *p && n + 4 < cap; p++) {
		if (isalnum(*p) || *p == '-' || *p == '_' || *p == '.' || *p == '~')
			out[n++] = (char)*p;
		else
			n += (size_t)snprintf(out + n, cap - n, "%%%02X", *p);
	}
	out[n] = '\0';
}

static bool repo_is_valid(const char *repo) {
	const char *slash = strchr(repo, '/');
	if (!slash || slash == repo || !slash[1] || strchr(slash + 1, '/') || strstr(repo, ".."))
		return false;
	for (const char *p = repo; *p; p++)
		if (!isalnum((unsigned char)*p) && *p != '-' && *p != '_' && *p != '.' && *p != '/')
			return false;
	return true;
}

static bool path_is_safe(const char *path) {
	return path[0] && path[0] != '/' && !strstr(path, "..");
}

static int proc_start(char *const argv[], hf_proc *p, bool capture) {
	int out[2], err[2];
	if (pipe(err) != 0)
		return -1;
	if (capture && pipe(out) != 0) {
		close(err[0]);
		close(err[1]);
		return -1;
	}
	pid_t pid = fork();
	if (pid < 0) {
		close(err[0]);
		close(err[1]);
		if (capture) {
			close(out[0]);
			close(out[1]);
		}
		return -1;
	}
	if (pid == 0) {
		int null = open("/dev/null", O_RDWR);
		dup2(null, STDIN_FILENO);
		dup2(capture ? out[1] : null, STDOUT_FILENO);
		dup2(err[1], STDERR_FILENO);
		close(err[0]);
		close(err[1]);
		if (capture) {
			close(out[0]);
			close(out[1]);
		}
		execvp(argv[0], argv);
		_exit(127);
	}
	close(err[1]);
	p->pid	  = pid;
	p->err_fd = err[0];
	p->out_fd = -1;
	if (capture) {
		close(out[1]);
		p->out_fd = out[0];
	}
	return 0;
}

static void proc_describe(int status, const char *stderr_text, char *err, size_t errlen) {
	if (WIFEXITED(status) && WEXITSTATUS(status) == 127)
		snprintf(err, errlen, "curl is required for downloads but could not be run");
	else if (stderr_text[0])
		snprintf(err, errlen, "%s", stderr_text);
	else
		snprintf(err, errlen, "curl failed (status %d)",
				 WIFEXITED(status) ? WEXITSTATUS(status) : -1);
}

static void proc_kill(hf_proc *p) {
	kill(p->pid, SIGTERM);
	waitpid(p->pid, NULL, 0);
	close(p->err_fd);
	if (p->out_fd >= 0)
		close(p->out_fd);
}

static size_t drain(int fd, char *buf, size_t used, size_t cap) {
	ssize_t n = read(fd, buf + used, cap - used);
	return n > 0 ? used + (size_t)n : used;
}

static void trim_text(char *s) {
	size_t n = strlen(s);
	while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' '))
		s[--n] = '\0';
}

typedef struct {
	bool (*enough)(const unsigned char *buf, size_t len, void *ud);
	bool (*tick)(size_t used, void *ud);
	void *ud;
} fetch_hooks;

static void fetch_abort(hf_proc *p, char *buf, char *err, size_t errlen) {
	proc_kill(p);
	free(buf);
	ui_install_interrupt();
	snprintf(err, errlen, "cancelled");
}

static bool fetch_hooks_stop(const fetch_hooks *hooks, const char *buf, size_t used,
							 size_t *next_check, bool *cancelled) {
	if (!hooks)
		return false;
	if (hooks->tick && hooks->tick(used, hooks->ud)) {
		*cancelled = true;
		return true;
	}
	if (!hooks->enough || used < *next_check)
		return false;
	*next_check = used + HF_ENOUGH_STEP;
	return hooks->enough((const unsigned char *)buf, used, hooks->ud);
}

static int fetch_finish(hf_proc *p, const char *etext, char *err, size_t errlen) {
	int status;
	waitpid(p->pid, &status, 0);
	close(p->out_fd);
	close(p->err_fd);
	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return 0;
	proc_describe(status, etext, err, errlen);
	return -1;
}

static int fetch(const char *url, size_t cap, int max_time, const fetch_hooks *hooks, char **body,
				 size_t *body_len, char *err, size_t errlen) {
	char seconds[16];
	snprintf(seconds, sizeof(seconds), "%d", max_time);
	char *const argv[] = {"curl", "-fsSL", "--max-time", seconds, "-K", "-", (char *)url, NULL};
	hf_proc		p;
	if (proc_start(argv, &p, true) < 0) {
		snprintf(err, errlen, "cannot start curl: %s", strerror(errno));
		return -1;
	}
	char  *buf					  = xmalloc(cap + 1);
	char   etext[HF_PROC_ERR_MAX] = "";
	size_t used = 0, eused = 0, next_check = HF_ENOUGH_FIRST;
	bool   out_open = true, err_open = true, stopped = false, cancelled = false;
	while ((out_open || err_open) && used < cap && !stopped) {
		if (ui_interrupt) {
			fetch_abort(&p, buf, err, errlen);
			return -1;
		}
		struct pollfd fds[2] = {{out_open ? p.out_fd : -1, POLLIN, 0},
								{err_open ? p.err_fd : -1, POLLIN, 0}};
		if (poll(fds, 2, 200) > 0) {
			if (fds[0].revents) {
				size_t next = drain(p.out_fd, buf, used, cap);
				out_open	= next != used;
				used		= next;
			}
			if (fds[1].revents) {
				size_t next = drain(p.err_fd, etext, eused, sizeof(etext) - 1);
				err_open	= next != eused;
				eused		= next;
			}
		}
		stopped = fetch_hooks_stop(hooks, buf, used, &next_check, &cancelled);
	}
	if (cancelled) {
		fetch_abort(&p, buf, err, errlen);
		return -1;
	}
	if (stopped || used >= cap) {
		proc_kill(&p);
	} else {
		etext[eused] = '\0';
		trim_text(etext);
		if (fetch_finish(&p, etext, err, errlen) < 0) {
			free(buf);
			return -1;
		}
	}
	buf[used] = '\0';
	*body	  = buf;
	*body_len = used;
	return 0;
}

static json_object *fetch_json(const char *url, json_type type, char *err, size_t errlen) {
	char  *body = NULL;
	size_t len	= 0;
	if (fetch(url, HF_BODY_MAX, HF_JSON_TIMEOUT, NULL, &body, &len, err, errlen) < 0)
		return NULL;
	json_object *root = json_tokener_parse(body);
	free(body);
	if (!root || !json_object_is_type(root, type)) {
		snprintf(err, errlen, "unexpected response from Hugging Face");
		if (root)
			json_object_put(root);
		return NULL;
	}
	return root;
}

#define HF_PANEL_W 42
#define HF_PANEL_MIN_COLS 96
#define HF_TAGS_MAX 192
#define HF_PIPELINE_MAX 48
#define HF_DATE_LEN 10
#define HF_QUERY_MAX 160
#define HF_LIST_TOP 3
#define HF_DISK_TAG_W 12
#define HF_META_CACHE 8
#define HF_KEY_TIMEOUT_MS 500

typedef struct {
	char	  pipeline[HF_PIPELINE_MAX];
	char	  arch[HF_META_ARCH_MAX];
	long long params;
	long long ctx;
	int		  hidden;
} hf_repo_info;

typedef struct {
	char	  name[HF_NAME_MAX];
	long long downloads;
	long long likes;
	char	  pipeline[HF_PIPELINE_MAX];
	char	  created[HF_DATE_LEN + 1];
	char	  tags[HF_TAGS_MAX];
	char	  label[HF_LABEL_MAX];
} hf_model;

typedef struct {
	bool   active;
	char   file[HF_NAME_MAX];
	int	   index;
	int	   total;
	size_t done;
	size_t expected;
	double speed;
} hf_progress;

typedef struct {
	char	file[HF_NAME_MAX];
	hf_meta meta;
	bool	ok;
} hf_meta_entry;

enum { MODAL_NONE, MODAL_LOADING, MODAL_CONFIRM, MODAL_DOWNLOAD };

typedef struct {
	int			  tab;
	char		  query[HF_QUERY_MAX];
	hf_model	 *models;
	int			  n_models;
	int			  model_sel;
	int			  model_top;
	char		  repo[HF_NAME_MAX];
	hf_entry	 *files;
	int			  n_files;
	int			  file_sel;
	int			  file_top;
	hf_progress	  progress;
	hf_repo_info  info;
	int			  type_idx;
	hf_meta		  meta;
	bool		  meta_ok;
	char		  meta_err[SC_ERR_MAX];
	hf_meta_entry cache[HF_META_CACHE];
	int			  cache_n;
	int			  cache_next;
	int			  downloaded;
	int			  modal;
	int			  focus;
	size_t		  loading_bytes;
	int			  spinner;
	int			  group[HF_MAX_FILES];
	int			  group_count;
	char		  status[SC_ERR_MAX];
	int			  status_kind;
} hf_page;

enum { TAB_SEARCH, TAB_FILES, TAB_COUNT };

static bool tag_is_noise(const char *tag) {
	return strncmp(tag, "region:", 7) == 0 || strcmp(tag, "gguf") == 0 ||
		   strcmp(tag, "endpoints_compatible") == 0;
}

static void join_tags(json_object *tags, char *out, size_t cap) {
	out[0] = '\0';
	if (!tags || !json_object_is_type(tags, json_type_array))
		return;
	size_t used = 0;
	for (size_t i = 0; i < json_object_array_length(tags); i++) {
		const char *tag = json_object_get_string(json_object_array_get_idx(tags, i));
		if (!tag || tag_is_noise(tag))
			continue;
		int n = snprintf(out + used, cap - used, "%s%s", used ? ", " : "", tag);
		if (n < 0 || used + (size_t)n >= cap) {
			out[used] = '\0';
			return;
		}
		used += (size_t)n;
	}
}

static void fill_model(hf_model *m, const char *id, json_object *obj) {
	json_object *downloads = json_get(obj, "downloads");
	json_object *likes	   = json_get(obj, "likes");
	snprintf(m->name, sizeof(m->name), "%s", id);
	m->downloads = downloads ? json_object_get_int64(downloads) : 0;
	m->likes	 = likes ? json_object_get_int64(likes) : 0;
	snprintf(m->pipeline, sizeof(m->pipeline), "%s", json_get_str(obj, "pipeline_tag", ""));
	snprintf(m->created, sizeof(m->created), "%.*s", HF_DATE_LEN,
			 json_get_str(obj, "createdAt", ""));
	join_tags(json_get(obj, "tags"), m->tags, sizeof(m->tags));
	snprintf(m->label, sizeof(m->label), "%s  (%lld downloads)", id, m->downloads);
}

static int search_models(const char *query, hf_model *out, int max, char *err, size_t errlen) {
	char enc[256];
	char url[512];
	url_encode(query, enc, sizeof(enc));
	snprintf(url, sizeof(url),
			 HF_HOST "/api/models?search=%s&filter=gguf&sort=downloads&direction=-1&limit=%d", enc,
			 HF_SEARCH_LIMIT);
	json_object *root = fetch_json(url, json_type_array, err, errlen);
	if (!root)
		return -1;
	int n = 0;
	for (size_t i = 0; i < json_object_array_length(root) && n < max; i++) {
		json_object *obj = json_object_array_get_idx(root, i);
		const char	*id	 = json_get_str(obj, "id", NULL);
		if (!id || !repo_is_valid(id) || strlen(id) >= HF_NAME_MAX)
			continue;
		fill_model(&out[n++], id, obj);
	}
	json_object_put(root);
	if (n == 0)
		snprintf(err, errlen, "no GGUF models found for '%s'", query);
	return n > 0 ? n : -1;
}

static bool has_gguf_suffix(const char *path) {
	size_t n = strlen(path);
	return n > 5 && strcasecmp(path + n - 5, ".gguf") == 0;
}

static size_t entry_size(json_object *file) {
	json_object *lfs  = json_get(file, "lfs");
	json_object *size = json_get(lfs ? lfs : file, "size");
	return size ? (size_t)json_object_get_int64(size) : 0;
}

static void fill_repo_info(json_object *root, hf_repo_info *info) {
	json_object *gguf  = json_get(root, "gguf");
	json_object *total = json_get(gguf, "total");
	json_object *ctx   = json_get(gguf, "context_length");
	snprintf(info->pipeline, sizeof(info->pipeline), "%s", json_get_str(root, "pipeline_tag", ""));
	snprintf(info->arch, sizeof(info->arch), "%s", json_get_str(gguf, "architecture", ""));
	info->params = total ? json_object_get_int64(total) : 0;
	info->ctx	 = ctx ? json_object_get_int64(ctx) : 0;
}

static void fill_file(hf_entry *e, const char *path, size_t size) {
	char size_text[32];
	e->size = size;
	model_format_bytes((double)size, size_text, sizeof(size_text));
	snprintf(e->name, sizeof(e->name), "%s", path);
	model_detect_quant(path, e->quant, sizeof(e->quant));
	snprintf(e->label, sizeof(e->label), "%-8s %-10s %s", e->quant, size_text, path);
}

static bool quant_unsupported(const char *quant) {
	static const char *const prefixes[] = {"Q1",	 "Q2_K", "Q8_K", "IQ1_",
										   "IQ2_XS", "TQ",	 "MXFP", "NVFP"};
	for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++)
		if (strncmp(quant, prefixes[i], strlen(prefixes[i])) == 0)
			return true;
	return false;
}

static int fetch_repo(const char *repo, hf_entry *files, int max, hf_repo_info *info, char *err,
					  size_t errlen) {
	char url[512];
	snprintf(url, sizeof(url), HF_HOST "/api/models/%s?blobs=true", repo);
	json_object *root = fetch_json(url, json_type_object, err, errlen);
	if (!root)
		return -1;
	memset(info, 0, sizeof(*info));
	fill_repo_info(root, info);
	json_object *siblings = json_get(root, "siblings");
	int			 n		  = 0;
	size_t		 count	  = siblings ? json_object_array_length(siblings) : 0;
	for (size_t i = 0; i < count && n < max; i++) {
		json_object *f	  = json_object_array_get_idx(siblings, i);
		const char	*path = json_get_str(f, "rfilename", NULL);
		if (!path || !has_gguf_suffix(path) || !path_is_safe(path) || strlen(path) >= HF_NAME_MAX)
			continue;
		fill_file(&files[n], path, entry_size(f));
		if (quant_unsupported(files[n].quant))
			info->hidden++;
		else
			n++;
	}
	json_object_put(root);
	if (n == 0 && info->hidden)
		snprintf(err, errlen, "all %d GGUF files in '%s' use quantizations the engine cannot load",
				 info->hidden, repo);
	else if (n == 0)
		snprintf(err, errlen, "'%s' has no GGUF files", repo);
	return n > 0 ? n : -1;
}

static bool header_complete(const unsigned char *buf, size_t len, void *ud) {
	(void)ud;
	hf_meta probe;
	hf_meta_init(&probe);
	return hf_meta_scan(buf, len, &probe) != 1;
}

static int fetch_meta(const char *repo, const char *file, bool (*tick)(size_t, void *), void *ud,
					  hf_meta *meta, char *err, size_t errlen) {
	char url[1024];
	snprintf(url, sizeof(url), HF_HOST "/%s/resolve/main/%s", repo, file);
	fetch_hooks hooks = {header_complete, tick, ud};
	char	   *body  = NULL;
	size_t		len	  = 0;
	if (fetch(url, HF_HEADER_MAX, HF_HEADER_TIMEOUT, &hooks, &body, &len, err, errlen) < 0)
		return -1;
	hf_meta_init(meta);
	int rc = hf_meta_scan((const unsigned char *)body, len, meta);
	free(body);
	if (meta->kv_ok)
		return 0;
	snprintf(err, errlen,
			 rc < 0 ? "could not read GGUF metadata" : "GGUF metadata is larger than expected");
	return -1;
}

static int pick_group(const hf_entry *files, int n, int chosen, int *group) {
	char key[HF_NAME_MAX];
	if (!model_shard_key(files[chosen].name, key, sizeof(key), NULL)) {
		group[0] = chosen;
		return 1;
	}
	int count = 0;
	for (int i = 0; i < n; i++) {
		char other[HF_NAME_MAX];
		if (model_shard_key(files[i].name, other, sizeof(other), NULL) && strcmp(key, other) == 0)
			group[count++] = i;
	}
	return count;
}

typedef struct {
	const char *pipeline;
	int			type;
} type_rule;

static const type_rule TYPE_RULES[] = {
	{"text-generation", 0},
	{"text2text-generation", 0},
	{"image-text-to-text", 0},
	{"question-answering", 0},
	{"summarization", 0},
	{"translation", 0},
	{"text-to-image", 1},
	{"image-to-image", 1},
	{"text-to-video", 1},
	{"image-to-video", 1},
	{"unconditional-image-generation", 1},
	{"automatic-speech-recognition", 2},
	{"text-to-speech", 2},
	{"text-to-audio", 2},
	{"audio-to-audio", 2},
	{"feature-extraction", 3},
	{"sentence-similarity", 3},
	{"reinforcement-learning", 4},
	{"robotics", 4},
};

static int type_from_pipeline(const char *pipeline) {
	for (size_t i = 0; i < sizeof(TYPE_RULES) / sizeof(TYPE_RULES[0]); i++)
		if (strcmp(TYPE_RULES[i].pipeline, pipeline) == 0)
			return TYPE_RULES[i].type;
	return MODEL_TYPE_OTHER;
}

static double now_seconds(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static size_t file_size(const char *path) {
	struct stat st;
	return stat(path, &st) == 0 ? (size_t)st.st_size : 0;
}

static void sanitize_dir(const char *in, char *out, size_t cap) {
	size_t n = 0;
	for (; *in && n + 1 < cap; in++) {
		char c	 = (char)tolower((unsigned char)*in);
		out[n++] = isalnum((unsigned char)c) || c == '.' ? c : '-';
	}
	out[n] = '\0';
	if (!out[0])
		snprintf(out, cap, "unknown");
}

static void arch_dir(const hf_page *pg, char *out, size_t cap) {
	const char *arch = pg->info.arch[0] ? pg->info.arch : pg->meta_ok ? pg->meta.arch : "";
	sanitize_dir(arch, out, cap);
}

static int dest_rel(const hf_page *pg, const char *name, char *out, size_t cap) {
	char arch[HF_META_ARCH_MAX];
	arch_dir(pg, arch, sizeof(arch));
	return snprintf(out, cap, "%s/%s/%s", MODEL_TYPE_NAMES[pg->type_idx], arch, name) < (int)cap
			   ? 0
			   : -1;
}

static int dest_path(const hf_page *pg, const char *name, char *dest, size_t cap) {
	char rel[SC_PATH_MAX];
	if (dest_rel(pg, name, rel, sizeof(rel)) < 0)
		return -1;
	return snprintf(dest, cap, "%s/%s", tui_models_dir(), rel) < (int)cap ? 0 : -1;
}

static void set_status(hf_page *pg, int kind, const char *msg) {
	snprintf(pg->status, sizeof(pg->status), "%s", msg);
	pg->status_kind = kind;
}

static bool tab_visible(const hf_page *pg, int tab) {
	return tab == TAB_SEARCH || pg->repo[0];
}

static bool panel_visible(void) {
	return COLS >= HF_PANEL_MIN_COLS;
}

static int list_rows(void) {
	return MAX(LINES - HF_LIST_TOP - 2, 1);
}

static int list_width(void) {
	return panel_visible() ? COLS - HF_PANEL_W - 4 : COLS - 4;
}

static void keep_visible(int sel, int *top, int n, int rows) {
	if (sel < *top)
		*top = sel;
	if (sel >= *top + rows)
		*top = sel - rows + 1;
	*top = MAX(MIN(*top, n - rows), 0);
}

static int put_wrapped(int y, int x, int w, const char *text, chtype attr) {
	int lines = ui_wrap_count(text, w);
	if (y + lines <= LINES - 2)
		ui_draw_wrapped(y, x, w, text, attr);
	return y + lines;
}

static int put_field(int y, int x, int w, const char *label, const char *value) {
	char line[HF_LABEL_MAX];
	snprintf(line, sizeof(line), "%-11s %s", label, value);
	return put_wrapped(y, x, w, line, 0);
}

static void draw_tabs(const hf_page *pg) {
	static const char *const names[TAB_COUNT] = {"search", "files"};
	int						 tx				  = 2;
	for (int t = 0; t < TAB_COUNT; t++) {
		if (!tab_visible(pg, t))
			continue;
		char label[48];
		snprintf(label, sizeof(label), " %s ", names[t]);
		ui_putn(0, tx, label, (int)strlen(label),
				t == pg->tab ? (A_REVERSE | A_BOLD) : ui_attr(UI_DIM));
		tx += (int)strlen(label) + 2;
	}
	move(1, 1);
	attron(ui_attr(UI_DIM));
	for (int i = 1; i < COLS - 1; i++)
		addch(ACS_HLINE);
	attroff(ui_attr(UI_DIM));
}

static const char *model_label(const hf_page *pg, int i) {
	return pg->models[i].label;
}

static const char *file_label(const hf_page *pg, int i) {
	return pg->files[i].label;
}

typedef enum { DISK_NONE, DISK_PARTIAL, DISK_FULL } disk_state;

static disk_state file_disk_state(const hf_page *pg, const hf_entry *f, int *percent) {
	char dest[SC_PATH_MAX];
	char part[SC_PATH_MAX + 8];
	*percent = 0;
	if (dest_path(pg, f->name, dest, sizeof(dest)) < 0)
		return DISK_NONE;
	size_t whole = file_size(dest);
	if (whole && (!f->size || whole == f->size))
		return DISK_FULL;
	snprintf(part, sizeof(part), "%s.part", dest);
	size_t partial = file_size(part);
	if (partial == 0)
		return DISK_NONE;
	*percent = f->size ? (int)MIN(partial * 100 / f->size, 99) : 0;
	return DISK_PARTIAL;
}

static void hide_downloaded(hf_page *pg) {
	int percent;
	int kept = 0;
	for (int i = 0; i < pg->n_files; i++) {
		if (file_disk_state(pg, &pg->files[i], &percent) == DISK_FULL) {
			pg->downloaded++;
			continue;
		}
		if (kept != i)
			pg->files[kept] = pg->files[i];
		kept++;
	}
	pg->n_files	 = kept;
	pg->file_sel = MIN(pg->file_sel, MAX(kept - 1, 0));
}
static void draw_disk_tag(const hf_page *pg, int idx, int y, bool selected) {
	int	 percent;
	char tag[HF_DISK_TAG_W + 1];
	if (file_disk_state(pg, &pg->files[idx], &percent) != DISK_PARTIAL)
		return;
	snprintf(tag, sizeof(tag), "partial %d%%", percent);
	chtype attr = selected ? (A_REVERSE | A_BOLD) : A_BOLD | ui_attr(UI_WARN);
	ui_putn(y, 3 + list_width() - HF_DISK_TAG_W, tag, HF_DISK_TAG_W, attr);
}

static void draw_list(const hf_page *pg, int n, int sel, int top,
					  const char *(*label)(const hf_page *, int), bool show_disk) {
	int rows	= MIN(list_rows(), n - top);
	int label_w = show_disk ? list_width() - HF_DISK_TAG_W - 1 : list_width();
	for (int i = 0; i < rows; i++) {
		int	 idx	  = top + i;
		int	 y		  = HF_LIST_TOP + i;
		bool selected = idx == sel;
		if (selected) {
			ui_putn(y, 1, ">", 1, A_BOLD);
			ui_fill(y, 3, 1, list_width(), A_REVERSE);
		}
		ui_putn(y, 3, label(pg, idx), label_w, selected ? (A_REVERSE | A_BOLD) : 0);
		if (show_disk)
			draw_disk_tag(pg, idx, y, selected);
	}
}

static void format_params(long long params, char *out, size_t cap) {
	if (params <= 0)
		snprintf(out, cap, "-");
	else if (params >= 1000000000LL)
		snprintf(out, cap, "%.2fB", (double)params / 1e9);
	else
		snprintf(out, cap, "%.0fM", (double)params / 1e6);
}

static void draw_header(const hf_page *pg) {
	char line[HF_NAME_MAX + 128];
	if (pg->tab == TAB_SEARCH) {
		snprintf(line, sizeof(line), "Search: %s   (%d results)",
				 pg->query[0] ? pg->query : "(press / to search)", pg->n_models);
	} else {
		char params[16], hidden[96] = "";
		format_params(pg->info.params, params, sizeof(params));
		if (pg->info.hidden)
			snprintf(hidden, sizeof(hidden), ", %d hidden: unsupported quant", pg->info.hidden);
		if (pg->downloaded)
			snprintf(hidden + strlen(hidden), sizeof(hidden) - strlen(hidden),
					 ", %d already on disk", pg->downloaded);
		snprintf(line, sizeof(line), "Repository: %s   (%d GGUF files%s, %s params%s%s)", pg->repo,
				 pg->n_files, hidden, params, pg->info.arch[0] ? ", " : "", pg->info.arch);
	}
	ui_putn(2, 3, line, COLS - 6, A_BOLD);
}

static void draw_model_panel(const hf_model *m, int x, int w) {
	char num[32];
	int	 y = put_wrapped(2, x, w, m->name, A_BOLD) + 1;
	snprintf(num, sizeof(num), "%lld", m->downloads);
	y = put_field(y, x, w, "Downloads", num);
	snprintf(num, sizeof(num), "%lld", m->likes);
	y = put_field(y, x, w, "Likes", num);
	y = put_field(y, x, w, "Task", m->pipeline[0] ? m->pipeline : "-");
	y = put_field(y, x, w, "Created", m->created[0] ? m->created : "-") + 1;
	if (m->tags[0]) {
		ui_putn(y++, x, "Tags", w, ui_attr(UI_DIM));
		y = put_wrapped(y, x, w, m->tags, 0) + 1;
	}
	put_wrapped(y, x, w, "Enter: open files", ui_attr(UI_DIM));
}

static const char *local_state(const char *dest, size_t expected, char *buf, size_t cap) {
	char part[SC_PATH_MAX];
	snprintf(part, sizeof(part), "%s.part", dest);
	if (expected && file_size(dest) == expected)
		return "downloaded";
	size_t partial = file_size(part);
	if (partial == 0)
		return "not downloaded";
	snprintf(buf, cap, "partial (%d%%)", expected ? (int)(partial * 100 / expected) : 0);
	return buf;
}

static size_t group_bytes(const hf_page *pg, const int *group, int count) {
	size_t total = 0;
	for (int i = 0; i < count; i++)
		total += pg->files[group[i]].size;
	return total;
}

static const hf_meta_entry *cache_find(const hf_page *pg, const char *file) {
	for (int i = 0; i < pg->cache_n; i++)
		if (strcmp(pg->cache[i].file, file) == 0)
			return &pg->cache[i];
	return NULL;
}

static void engine_status(const hf_meta_entry *c, char *out, size_t cap) {
	char bad[64];
	if (!c || !c->ok)
		snprintf(out, cap, "not checked yet");
	else if (hf_meta_blocked(&c->meta)) {
		hf_meta_bad_list(&c->meta, bad, sizeof(bad));
		snprintf(out, cap, "cannot load (%s)", bad);
	} else if (!c->meta.tensors_ok)
		snprintf(out, cap, "partly verified");
	else
		snprintf(out, cap, "compatible");
}

static void draw_file_panel(const hf_page *pg, int x, int w) {
	const hf_entry *f = &pg->files[pg->file_sel];
	int				group[HF_MAX_FILES];
	int				count = pick_group(pg->files, pg->n_files, pg->file_sel, group);
	size_t			total = group_bytes(pg, group, count);

	char size[32], dest[SC_PATH_MAX], state[32], engine[96];
	model_format_bytes((double)f->size, size, sizeof(size));
	int y = put_wrapped(2, x, w, f->name, A_BOLD) + 1;
	y	  = put_field(y, x, w, "Quant", f->quant);
	y	  = put_field(y, x, w, "Size", size);
	if (count > 1) {
		char text[64], sum[32];
		model_format_bytes((double)total, sum, sizeof(sum));
		snprintf(text, sizeof(text), "%d files, %s", count, sum);
		y = put_field(y, x, w, "Shards", text);
	}
	const hf_meta_entry *cached = cache_find(pg, f->name);
	if (cached && cached->ok && cached->meta.tensors_ok) {
		char mix[96];
		hf_meta_mix(&cached->meta, mix, sizeof(mix));
		y = put_field(y, x, w, "Mix", mix);
	}
	engine_status(cached, engine, sizeof(engine));
	y = put_field(y, x, w, "Engine", engine);
	dest_path(pg, f->name, dest, sizeof(dest));
	y = put_field(y, x, w, "Status", local_state(dest, f->size, state, sizeof(state))) + 1;
	put_wrapped(y, x, w, "Enter: check compatibility and download", ui_attr(UI_DIM));
}

static double estimate_total(double weights, const hf_meta *m, bool have_meta, long long ctx,
							 double elem_bytes) {
	double kv = have_meta ? hf_meta_kv_per_token(m, elem_bytes) : 0.0;
	return weights * 1.05 + 256.0 * 1024 * 1024 + MAX(kv, 0.0) * (double)ctx;
}

static int estimate_contexts(const hf_page *pg, long long *out) {
	static const long long steps[] = {4096, 8192, 32768};
	long long max_ctx = pg->meta_ok && pg->meta.n_ctx > 0 ? pg->meta.n_ctx : pg->info.ctx;
	int		  n		  = 0;
	for (size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); i++)
		if (max_ctx <= 0 || steps[i] <= max_ctx)
			out[n++] = steps[i];
	if (max_ctx > steps[2])
		out[n++] = max_ctx;
	return MAX(n, 1);
}

static bool group_blocked(const hf_page *pg) {
	return pg->meta_ok && hf_meta_blocked(&pg->meta);
}

static int emit(bool draw, int y, int x, int w, const char *text, chtype attr) {
	if (draw)
		ui_draw_wrapped(y, x, w, text, attr);
	return y + ui_wrap_count(text, w);
}

static int emit_field(bool draw, int y, int x, int w, const char *label, const char *value) {
	char line[HF_LABEL_MAX];
	snprintf(line, sizeof(line), "%-11s %s", label, value);
	return emit(draw, y, x, w, line, 0);
}

static int emit_estimates(const hf_page *pg, double weights, bool draw, int x, int y, int w) {
	char a[24], b[24], line[HF_LABEL_MAX];
	if (!pg->meta_ok) {
		model_format_bytes(estimate_total(weights, &pg->meta, false, 0, 2.0), a, sizeof(a));
		snprintf(line, sizeof(line), "weights + overhead ~ %s (KV cache not estimated)", a);
		return emit(draw, y, x, w, line, ui_attr(UI_WARN));
	}
	if (draw)
		ui_putn(y, x, "ctx     f16 KV      q8_0 KV", w, ui_attr(UI_DIM));
	y++;
	long long ctxs[4];
	int		  n = estimate_contexts(pg, ctxs);
	for (int i = 0; i < n; i++, y++) {
		if (!draw)
			continue;
		char label[16];
		model_format_bytes(estimate_total(weights, &pg->meta, true, ctxs[i], 2.0), a, sizeof(a));
		model_format_bytes(estimate_total(weights, &pg->meta, true, ctxs[i], 34.0 / 32.0), b,
						   sizeof(b));
		snprintf(label, sizeof(label), "%lldK", ctxs[i] / 1024);
		snprintf(line, sizeof(line), "%-7s %-11s %s", label, a, b);
		ui_putn(y, x, line, w, 0);
	}
	return y;
}

static int emit_notices(const hf_page *pg, bool draw, int x, int y, int w) {
	char bad[64], text[SC_ERR_MAX + 96];
	if (!pg->meta_ok) {
		snprintf(text, sizeof(text),
				 "Could not read the model header (%s); quant support is not verified.",
				 pg->meta_err);
		return emit(draw, y, x, w, text, ui_attr(UI_WARN));
	}
	if (group_blocked(pg)) {
		hf_meta_bad_list(&pg->meta, bad, sizeof(bad));
		snprintf(text, sizeof(text),
				 "Blocked: %d tensor(s) use quantizations the engine cannot load (%s).",
				 pg->meta.n_bad_tensors, bad);
		y = emit(draw, y, x, w, text, A_BOLD | ui_attr(UI_ERR));
	} else if (!pg->meta.tensors_ok) {
		y = emit(draw, y, x, w,
				 "Tensor types could not be fully verified (header larger than expected).",
				 ui_attr(UI_WARN));
	} else if (pg->meta.n_bad_mtp_tensors > 0) {
		y = emit(draw, y, x, w,
				 "MTP tensors use quantizations the engine cannot load; they are skipped (MTP is "
				 "not supported yet).",
				 ui_attr(UI_WARN));
	} else {
		y = emit(draw, y, x, w, "All tensors use quantizations the engine can load.",
				 ui_attr(UI_OK));
	}
	if (pg->group_count > 1)
		y = emit(draw, y, x, w, "Multi-part model: only the first part was checked.",
				 ui_attr(UI_DIM));
	return y;
}

static int confirm_layout(const hf_page *pg, bool draw, int x, int y, int w) {
	const hf_entry *f	  = &pg->files[pg->group[0]];
	size_t			total = group_bytes(pg, pg->group, pg->group_count);
	char			text[96], sum[32], params[16], mix[96], rel[SC_PATH_MAX];

	y = emit(draw, y, x, w, f->name, A_BOLD) + 1;
	format_params(pg->info.params, params, sizeof(params));
	model_format_bytes((double)total, sum, sizeof(sum));
	if (pg->info.params > 0)
		snprintf(text, sizeof(text), "%s  (%.2f bpw)", f->quant,
				 (double)total * 8.0 / (double)pg->info.params);
	else
		snprintf(text, sizeof(text), "%s", f->quant);
	y = emit_field(draw, y, x, w, "Quant", text);
	if (pg->group_count > 1)
		snprintf(text, sizeof(text), "%d files, %s", pg->group_count, sum);
	else
		snprintf(text, sizeof(text), "%s", sum);
	y = emit_field(draw, y, x, w, "Size", text);
	if (pg->meta_ok && pg->meta.tensors_ok) {
		hf_meta_mix(&pg->meta, mix, sizeof(mix));
		y = emit_field(draw, y, x, w, "Mix", mix);
	}
	if (pg->meta_ok) {
		snprintf(text, sizeof(text), "%lld layers, ctx %lld", pg->meta.n_layer, pg->meta.n_ctx);
		y = emit_field(draw, y, x, w, "Layout", text);
		if (pg->meta.n_expert > 0) {
			snprintf(text, sizeof(text), "%lld experts (all resident unless streamed)",
					 pg->meta.n_expert);
			y = emit_field(draw, y, x, w, "Experts", text);
		}
	}
	y++;
	y = emit_notices(pg, draw, x, y, w) + 1;
	if (draw)
		ui_putn(y, x, "Memory estimate", w, ui_attr(UI_DIM));
	y = emit_estimates(pg, (double)total, draw, x, y + 1, w) + 1;
	dest_rel(pg, f->name, rel, sizeof(rel));
	if (draw)
		ui_putn(y, x, "Saves to", w, ui_attr(UI_DIM));
	y = emit(draw, y + 1, x, w, rel, 0);
	return emit_field(draw, y, x, w, "Type", MODEL_TYPE_NAMES[pg->type_idx]);
}

static int left_width(void) {
	return panel_visible() ? COLS - HF_PANEL_W : COLS;
}

static void modal_frame(int h, int w, const char *title, chtype attr, int *bx, int *by) {
	h	= MIN(h, LINES - 2);
	*bx = MAX((left_width() - w) / 2, 0);
	*by = MAX((LINES - h) / 2, 0);
	ui_box(*by, *bx, h, w, title, attr);
}

static void draw_buttons(const hf_page *pg, int y, int bx, int w) {
	static const char download[] = "[ Download ]";
	static const char cancel[]	 = "[ Cancel ]";
	bool			  blocked	 = group_blocked(pg);
	int				  gap		 = 4;
	int				  total		 = (int)strlen(download) + gap + (int)strlen(cancel);
	int				  x			 = bx + (w - total) / 2;
	chtype			  dl_attr	 = blocked			? ui_attr(UI_DIM)
								   : pg->focus == 0 ? (A_REVERSE | A_BOLD)
													: ui_attr(UI_OK);
	chtype			  cx_attr	 = pg->focus == 1 ? (A_REVERSE | A_BOLD) : 0;
	ui_putn(y, x, download, (int)strlen(download), dl_attr);
	ui_putn(y, x + (int)strlen(download) + gap, cancel, (int)strlen(cancel), cx_attr);
}

static void draw_confirm_modal(const hf_page *pg) {
	int	 w		 = MIN(left_width() - 4, 68);
	int	 inner	 = w - 4;
	int	 content = confirm_layout(pg, false, 0, 0, inner);
	int	 h		 = MIN(content + 6, LINES - 2);
	int	 bx, by;
	bool blocked = group_blocked(pg);
	modal_frame(h, w, blocked ? "Cannot download" : "Download this model?",
				blocked ? ui_attr(UI_ERR) : ui_attr(UI_TITLE), &bx, &by);
	confirm_layout(pg, true, bx + 2, by + 1, inner);
	draw_buttons(pg, by + h - 3, bx, w);
	ui_putn(by + h - 2, bx + 2, "Left/Right: choose   Enter: confirm   T: change type", inner,
			ui_attr(UI_DIM));
}

static void draw_loading_modal(const hf_page *pg) {
	static const char spin[] = "|/-\\";
	char			  a[24], line[HF_LABEL_MAX];
	int				  w = MIN(left_width() - 4, 60);
	int				  bx, by;
	modal_frame(7, w, "Checking compatibility", ui_attr(UI_TITLE), &bx, &by);
	ui_putn(by + 1, bx + 2, pg->files[pg->group[0]].name, w - 4, A_BOLD);
	model_format_bytes((double)pg->loading_bytes, a, sizeof(a));
	snprintf(line, sizeof(line), "%c  reading model header... %s", spin[pg->spinner % 4], a);
	ui_putn(by + 3, bx + 2, line, w - 4, 0);
	ui_putn(by + 5, bx + 2, "Esc: cancel", w - 4, ui_attr(UI_DIM));
}

static void draw_download_modal(const hf_page *pg) {
	const hf_progress *pr = &pg->progress;
	char			   a[24], b[24], s[24], line[HF_LABEL_MAX];
	int				   w	 = MIN(left_width() - 4, 64);
	int				   inner = w - 4;
	int				   bx, by;
	modal_frame(9, w, "Downloading", ui_attr(UI_TITLE), &bx, &by);
	snprintf(line, sizeof(line), "[%d/%d] %s", pr->index + 1, pr->total, pr->file);
	ui_putn(by + 1, bx + 2, line, inner, A_BOLD);
	char bar[128];
	int	 bar_w = MIN(inner, (int)sizeof(bar) - 1);
	int	 filled =
		pr->expected ? (int)(MIN((double)pr->done / (double)pr->expected, 1.0) * bar_w) : 0;
	for (int i = 0; i < bar_w; i++)
		bar[i] = i < filled ? '#' : '-';
	bar[bar_w] = '\0';
	ui_putn(by + 3, bx + 2, bar, inner, ui_attr(UI_OK));
	model_format_bytes((double)pr->done, a, sizeof(a));
	model_format_bytes((double)pr->expected, b, sizeof(b));
	model_format_bytes(pr->speed, s, sizeof(s));
	snprintf(line, sizeof(line), "%s / %s   %s/s", a, b, s);
	ui_putn(by + 4, bx + 2, line, inner, 0);
	ui_putn(by + 6, bx + 2, "Esc: cancel (the partial file is kept and resumes next time)", inner,
			ui_attr(UI_DIM));
}

static void draw_modal(const hf_page *pg) {
	if (pg->modal == MODAL_LOADING)
		draw_loading_modal(pg);
	else if (pg->modal == MODAL_CONFIRM)
		draw_confirm_modal(pg);
	else if (pg->modal == MODAL_DOWNLOAD)
		draw_download_modal(pg);
}

static void draw_panel(const hf_page *pg) {
	int px = COLS - HF_PANEL_W;
	attron(ui_attr(UI_DIM));
	for (int y = 2; y <= LINES - 3; y++)
		mvaddch(y, px - 2, ACS_VLINE);
	attroff(ui_attr(UI_DIM));
	int w = HF_PANEL_W - 2;
	if (pg->tab == TAB_SEARCH && pg->n_models > 0)
		draw_model_panel(&pg->models[pg->model_sel], px, w);
	else if (pg->tab == TAB_FILES && pg->n_files > 0)
		draw_file_panel(pg, px, w);
}

static const char *page_key_bar(const hf_page *pg) {
	if (pg->modal == MODAL_DOWNLOAD)
		return "Esc: cancel download";
	if (pg->modal == MODAL_LOADING)
		return "Esc: cancel";
	if (pg->modal == MODAL_CONFIRM)
		return "Left/Right: choose   Enter: confirm   T: change type   Esc: cancel";
	if (pg->tab == TAB_SEARCH)
		return "/: search   Up/Down: select   Enter: open files   Left/Right: switch tab   Esc: "
			   "back";
	return "Up/Down: select   Enter: check and download   /: new search   Left/Right: switch tab   "
		   "Esc: "
		   "back";
}

static void page_draw(const hf_page *pg) {
	erase();
	draw_tabs(pg);
	draw_header(pg);
	if (pg->tab == TAB_SEARCH) {
		draw_list(pg, pg->n_models, pg->model_sel, pg->model_top, model_label, false);
		if (pg->n_models == 0)
			ui_putn(HF_LIST_TOP, 3, "(no results)", COLS - 6, ui_attr(UI_DIM));
	} else {
		draw_list(pg, pg->n_files, pg->file_sel, pg->file_top, file_label, true);
		if (pg->n_files == 0)
			ui_putn(HF_LIST_TOP, 3, "(nothing left to download here)", COLS - 6, ui_attr(UI_DIM));
	}
	if (panel_visible())
		draw_panel(pg);
	if (pg->status[0] && pg->modal == MODAL_NONE)
		ui_putn(LINES - 2, 2, pg->status, COLS - 4, ui_attr(pg->status_kind));
	ui_putn(LINES - 1, 2, page_key_bar(pg), COLS - 4, ui_attr(UI_DIM));
	draw_modal(pg);
	refresh();
}

typedef enum { DL_OK, DL_CANCELLED, DL_FAILED } dl_result;

static dl_result download_one(hf_page *pg, const hf_entry *f, const char *dest,
							  size_t finished_before, char *err, size_t errlen) {
	char part[SC_PATH_MAX];
	char url[1024];
	snprintf(part, sizeof(part), "%s.part", dest);
	snprintf(url, sizeof(url), HF_HOST "/%s/resolve/main/%s", pg->repo, f->name);

	char *const argv[] = {"curl", "-fsSL", "-C", "-",  "--retry", "3",
						  "-K",	  "-",	   "-o", part, url,		  NULL};
	hf_proc		p;
	if (proc_start(argv, &p, false) < 0) {
		snprintf(err, errlen, "cannot start curl: %s", strerror(errno));
		return DL_FAILED;
	}
	fcntl(p.err_fd, F_SETFL, O_NONBLOCK);

	size_t base	   = file_size(part);
	double started = now_seconds();
	int	   status;
	snprintf(pg->progress.file, sizeof(pg->progress.file), "%s", f->name);
	while (waitpid(p.pid, &status, WNOHANG) == 0) {
		size_t cur		   = file_size(part);
		pg->progress.done  = finished_before + cur;
		pg->progress.speed = (double)(cur - MIN(cur, base)) / MAX(now_seconds() - started, 0.5);
		page_draw(pg);
		int ch = ui_poll();
		if (ch == 27 || ch == KBD_INTERRUPT) {
			proc_kill(&p);
			ui_install_interrupt();
			return DL_CANCELLED;
		}
	}
	char   etext[HF_PROC_ERR_MAX] = "";
	size_t used					  = drain(p.err_fd, etext, 0, sizeof(etext) - 1);
	etext[used]					  = '\0';
	trim_text(etext);
	close(p.err_fd);
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		proc_describe(status, etext, err, errlen);
		return DL_FAILED;
	}
	if (rename(part, dest) != 0) {
		snprintf(err, errlen, "cannot move '%s' into place: %s", part, strerror(errno));
		return DL_FAILED;
	}
	return DL_OK;
}

static int make_dest(const hf_page *pg, const char *name, char *dest, size_t cap, char *err,
					 size_t errlen) {
	if (dest_path(pg, name, dest, cap) < 0) {
		snprintf(err, errlen, "destination path too long");
		return -1;
	}
	char dir[SC_PATH_MAX];
	snprintf(dir, sizeof(dir), "%s", dest);
	*strrchr(dir, '/') = '\0';
	return mkdir_p(dir, err, errlen);
}

static void finish_download(hf_page *pg, dl_result r, const char *err) {
	pg->progress.active = false;
	if (r == DL_CANCELLED) {
		set_status(pg, UI_WARN, "download cancelled; the partial file will resume next time");
		return;
	}
	if (r == DL_FAILED) {
		set_status(pg, UI_ERR, err);
		return;
	}
	char rel[SC_PATH_MAX], text[SC_PATH_MAX + 32];
	dest_rel(pg, pg->files[pg->group[0]].name, rel, sizeof(rel));
	snprintf(text, sizeof(text), "download complete: %s", rel);
	hide_downloaded(pg);
	set_status(pg, UI_OK, text);
}

static void download_group(hf_page *pg) {
	pg->modal		   = MODAL_DOWNLOAD;
	pg->progress	   = (hf_progress){.active	 = true,
									   .total	 = pg->group_count,
									   .expected = group_bytes(pg, pg->group, pg->group_count)};
	size_t	  finished = 0;
	char	  err[SC_ERR_MAX] = "";
	dl_result r				  = DL_OK;
	for (int i = 0; i < pg->group_count && r == DL_OK; i++) {
		const hf_entry *f = &pg->files[pg->group[i]];
		char			dest[SC_PATH_MAX];
		pg->progress.index = i;
		if (make_dest(pg, f->name, dest, sizeof(dest), err, sizeof(err)) < 0) {
			r = DL_FAILED;
			break;
		}
		if (!f->size || file_size(dest) != f->size)
			r = download_one(pg, f, dest, finished, err, sizeof(err));
		finished += f->size;
	}
	pg->modal = MODAL_NONE;
	finish_download(pg, r, err);
}

static void open_repo(hf_page *pg, const char *repo) {
	char		 err[SC_ERR_MAX];
	hf_repo_info info;
	set_status(pg, UI_WARN, "loading repository...");
	page_draw(pg);
	int n = fetch_repo(repo, pg->files, HF_MAX_FILES, &info, err, sizeof(err));
	if (n < 0) {
		set_status(pg, UI_ERR, err);
		return;
	}
	pg->info	   = info;
	pg->type_idx   = type_from_pipeline(info.pipeline);
	pg->n_files	   = n;
	pg->file_sel   = 0;
	pg->file_top   = 0;
	pg->downloaded = 0;
	pg->meta_ok	   = false;
	pg->cache_n	   = 0;
	pg->cache_next = 0;
	pg->modal	   = MODAL_NONE;
	snprintf(pg->repo, sizeof(pg->repo), "%s", repo);
	hide_downloaded(pg);
	if (pg->n_files == 0) {
		pg->repo[0] = '\0';
		pg->tab		= TAB_SEARCH;
		set_status(pg, UI_OK, "everything in this repository is already on disk");
		return;
	}
	pg->tab		  = TAB_FILES;
	pg->status[0] = '\0';
}

static void run_search(hf_page *pg) {
	char buf[HF_QUERY_MAX];
	pg->tab = TAB_SEARCH;
	if (ui_input("Search Hugging Face, or enter owner/repo", pg->query, buf, sizeof(buf), NULL,
				 NULL, NULL) <= 0 ||
		!buf[0])
		return;
	snprintf(pg->query, sizeof(pg->query), "%s", buf);
	if (repo_is_valid(buf)) {
		open_repo(pg, buf);
		return;
	}
	char err[SC_ERR_MAX];
	set_status(pg, UI_WARN, "searching Hugging Face...");
	page_draw(pg);
	int n = search_models(buf, pg->models, HF_SEARCH_LIMIT, err, sizeof(err));
	if (n < 0) {
		set_status(pg, UI_ERR, err);
		return;
	}
	pg->n_models  = n;
	pg->model_sel = 0;
	pg->model_top = 0;
	pg->status[0] = '\0';
}

static bool loading_tick(size_t used, void *ud) {
	hf_page *pg		  = ud;
	pg->loading_bytes = used;
	pg->spinner++;
	page_draw(pg);
	timeout(0);
	int ch = getch();
	timeout(HF_KEY_TIMEOUT_MS);
	return ch == 27;
}

static void cache_store(hf_page *pg, const char *file, const hf_meta *meta) {
	hf_meta_entry *slot = &pg->cache[pg->cache_next];
	pg->cache_next		= (pg->cache_next + 1) % HF_META_CACHE;
	pg->cache_n			= MIN(pg->cache_n + 1, HF_META_CACHE);
	snprintf(slot->file, sizeof(slot->file), "%s", file);
	slot->meta = *meta;
	slot->ok   = true;
}

static bool load_meta(hf_page *pg) {
	const char			*file = pg->files[pg->group[0]].name;
	const hf_meta_entry *hit  = cache_find(pg, file);
	if (hit) {
		pg->meta	= hit->meta;
		pg->meta_ok = true;
		return true;
	}
	pg->modal		  = MODAL_LOADING;
	pg->loading_bytes = 0;
	hf_meta fetched;
	int		rc =
		fetch_meta(pg->repo, file, loading_tick, pg, &fetched, pg->meta_err, sizeof(pg->meta_err));
	pg->modal = MODAL_NONE;
	if (rc < 0 && strcmp(pg->meta_err, "cancelled") == 0) {
		set_status(pg, UI_WARN, "cancelled");
		return false;
	}
	pg->meta_ok = rc == 0;
	if (pg->meta_ok) {
		pg->meta = fetched;
		cache_store(pg, file, &fetched);
	}
	return true;
}

static void begin_confirm(hf_page *pg) {
	pg->group_count = pick_group(pg->files, pg->n_files, pg->file_sel, pg->group);
	if (!load_meta(pg))
		return;
	if (!pg->info.arch[0] && pg->meta_ok)
		snprintf(pg->info.arch, sizeof(pg->info.arch), "%s", pg->meta.arch);
	pg->modal	  = MODAL_CONFIRM;
	pg->focus	  = group_blocked(pg) ? 1 : 0;
	pg->status[0] = '\0';
}

static void confirm_accept(hf_page *pg) {
	if (pg->focus == 0 && !group_blocked(pg))
		download_group(pg);
	else
		pg->modal = MODAL_NONE;
}

static bool handle_confirm_key(hf_page *pg, int ch) {
	switch (ch) {
	case 27:
	case 'n':
	case 'N':
		pg->modal = MODAL_NONE;
		break;
	case 'y':
	case 'Y':
		pg->focus = 0;
		confirm_accept(pg);
		break;
	case KEY_LEFT:
	case KEY_RIGHT:
	case '\t':
	case KEY_BTAB:
		pg->focus = group_blocked(pg) ? 1 : 1 - pg->focus;
		break;
	case '\n':
	case KEY_ENTER:
		confirm_accept(pg);
		break;
	case 't':
	case 'T':
		pg->type_idx = (pg->type_idx + 1) % MODEL_TYPE_COUNT;
		break;
	default:
		break;
	}
	return true;
}

static void move_selection(hf_page *pg, int delta) {
	bool search	  = pg->tab == TAB_SEARCH;
	int	 n		  = search ? pg->n_models : pg->n_files;
	int *sel	  = search ? &pg->model_sel : &pg->file_sel;
	*sel		  = MAX(MIN(*sel + delta, n - 1), 0);
	pg->status[0] = '\0';
}

static void switch_tab(hf_page *pg, int dir) {
	int tab = pg->tab;
	do {
		tab = (tab + dir + TAB_COUNT) % TAB_COUNT;
	} while (!tab_visible(pg, tab));
	pg->tab		  = tab;
	pg->status[0] = '\0';
}

static void activate(hf_page *pg) {
	if (pg->tab == TAB_SEARCH && pg->n_models > 0) {
		open_repo(pg, pg->models[pg->model_sel].name);
	} else if (pg->tab == TAB_FILES && pg->n_files > 0) {
		begin_confirm(pg);
	}
}

static bool handle_key(hf_page *pg, int ch) {
	if (pg->modal == MODAL_CONFIRM)
		return handle_confirm_key(pg, ch);
	switch (ch) {
	case KEY_UP:
		move_selection(pg, -1);
		break;
	case KEY_DOWN:
		move_selection(pg, 1);
		break;
	case KEY_PPAGE:
		move_selection(pg, -list_rows());
		break;
	case KEY_NPAGE:
		move_selection(pg, list_rows());
		break;
	case KEY_HOME:
		move_selection(pg, -HF_MAX_FILES);
		break;
	case KEY_END:
		move_selection(pg, HF_MAX_FILES);
		break;
	case KEY_RIGHT:
		switch_tab(pg, 1);
		break;
	case KEY_LEFT:
		switch_tab(pg, -1);
		break;
	case '/':
		run_search(pg);
		break;
	case '\n':
	case KEY_ENTER:
		activate(pg);
		break;
	case 27:
		return false;
	default:
		break;
	}
	return true;
}

void hf_page_run(void) {
	if (!tui_models_dir()[0]) {
		ui_message("Models directory not set",
				   "set a models directory in settings first; downloads are stored there", true);
		return;
	}
	hf_page *pg = xcalloc(1, sizeof(*pg));
	pg->models	= xcalloc(HF_SEARCH_LIMIT, sizeof(hf_model));
	pg->files	= xcalloc(HF_MAX_FILES, sizeof(hf_entry));
	run_search(pg);

	for (;;) {
		keep_visible(pg->model_sel, &pg->model_top, pg->n_models, list_rows());
		keep_visible(pg->file_sel, &pg->file_top, pg->n_files, list_rows());
		page_draw(pg);
		int ch = ui_getch();
		if (ch == KBD_INTERRUPT || !handle_key(pg, ch))
			break;
	}
	free(pg->models);
	free(pg->files);
	free(pg);
}
