#include "hcs.h"

#include "backend/backend.h"
#include "common.h"
#include "log.h"
#include "model.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define HCS_DEVICE_LIST_MAX 32

typedef struct {
	char  *buf;
	size_t cap;
	size_t len;
} violations;

static bool next_token(const char **cursor, char *tok, size_t cap) {
	const char *p = *cursor;
	if (!p)
		return false;
	const char *comma = strchr(p, ',');
	size_t		n	  = comma ? (size_t)(comma - p) : strlen(p);
	str_span	trim  = span_trim(p, n);
	*cursor			  = comma ? comma + 1 : NULL;
	if (trim.len >= cap)
		trim.len = cap - 1;
	memcpy(tok, trim.p, trim.len);
	tok[trim.len] = '\0';
	return true;
}

static int parse_bounded_int(const char *text, int *out) {
	if (!*text)
		return -1;
	char *end = NULL;
	long  v	  = strtol(text, &end, 10);
	if (*end != '\0' || v < 0 || v > INT_MAX)
		return -1;
	*out = (int)v;
	return 0;
}

int hcs_parse_count(const char *text, int *out) {
	str_span trim = span_trim(text, strlen(text));
	char	 buf[32];
	if (trim.len == 0 || trim.len >= sizeof(buf))
		return -1;
	memcpy(buf, trim.p, trim.len);
	buf[trim.len] = '\0';
	if (strcmp(buf, "*") == 0) {
		*out = HCS_COUNT_ALL;
		return 0;
	}
	return parse_bounded_int(buf, out);
}

const char *hcs_format_count(int value, char *buf, size_t cap) {
	if (value == HCS_COUNT_UNSET)
		return "";
	if (value == HCS_COUNT_ALL)
		return "*";
	snprintf(buf, cap, "%d", value);
	return buf;
}

int hcs_parse_priority(const char *text, int *out) {
	str_span trim = span_trim(text, strlen(text));
	char	 buf[32];
	if (trim.len == 0 || trim.len >= sizeof(buf))
		return -1;
	memcpy(buf, trim.p, trim.len);
	buf[trim.len] = '\0';
	return parse_bounded_int(buf, out);
}

int hcs_parse_pinlist(const char *text, char *out, size_t cap) {
	out[0]			= '\0';
	size_t		len = 0;
	const char *p	= text;
	while (*p == ' ' || *p == '\t')
		p++;
	if (*p == '\0')
		return 0;
	for (;;) {
		while (*p == ' ' || *p == '\t')
			p++;
		char *tail = NULL;
		long  v	   = strtol(p, &tail, 10);
		if (tail == p)
			return -1;
		char piece[24];
		snprintf(piece, sizeof(piece), "%s%ld", len ? "," : "", v);
		size_t plen = strlen(piece);
		if (len + plen + 1 > cap)
			return -1;
		memcpy(out + len, piece, plen + 1);
		len += plen;
		p = tail;
		while (*p == ' ' || *p == '\t')
			p++;
		if (*p == ',') {
			p++;
			continue;
		}
		if (*p == '\0')
			return 0;
		return -1;
	}
}

int hcs_parse_tasks(const char *text, unsigned *out) {
	unsigned	mask = 0;
	char		tok[HCS_NAME_MAX];
	const char *cursor = text;
	while (next_token(&cursor, tok, sizeof(tok))) {
		if (strcasecmp(tok, "prefill") == 0)
			mask |= HCS_TASK_PREFILL;
		else if (strcasecmp(tok, "decode") == 0)
			mask |= HCS_TASK_DECODE;
		else
			return -1;
	}
	*out = mask;
	return 0;
}

void hcs_format_tasks(unsigned tasks, char *buf, size_t cap) {
	snprintf(buf, cap, "%s%s%s", (tasks & HCS_TASK_PREFILL) ? "prefill" : "",
			 tasks == HCS_TASKS_ALL ? ", " : "", (tasks & HCS_TASK_DECODE) ? "decode" : "");
}

void hcs_format_devices(const hcs_config *h, char *buf, size_t cap) {
	size_t used = 0;
	buf[0]		= '\0';
	for (int i = 0; i < h->n_devices; i++)
		used += snprintf(buf + used, used < cap ? cap - used : 0, "%s%s", i ? ", " : "",
						 h->devices[i].name);
}

void hcs_format_devices_subset(const hcs_config *h, const int *idx, int n, char *buf, size_t cap) {
	size_t used = 0;
	buf[0]		= '\0';
	for (int k = 0; k < n; k++)
		used += snprintf(buf + used, used < cap && idx[k] < h->n_devices ? cap - used : 0, "%s%s",
						 k ? ", " : "", h->devices[idx[k]].name);
}

static void device_defaults(hcs_device *d) {
	d->tasks   = 0;
	d->layers  = HCS_COUNT_UNSET;
	d->experts = HCS_COUNT_UNSET;
}

void hcs_set_simple(hcs_config *h, const char *device) {
	char name[HCS_NAME_MAX];
	snprintf(name, sizeof(name), "%s", device);
	memset(h->devices, 0, sizeof(h->devices));
	h->n_devices = 1;
	snprintf(h->devices[0].name, HCS_NAME_MAX, "%s", name);
	h->devices[0].tasks	  = HCS_TASKS_ALL;
	h->devices[0].layers  = HCS_COUNT_ALL;
	h->devices[0].experts = HCS_COUNT_UNSET;
}

void hcs_defaults(hcs_config *h) {
	char best[HCS_NAME_MAX];
	backend_best_device(best, sizeof(best));
	memset(h, 0, sizeof(*h));
	hcs_set_simple(h, best);
}

static hcs_config hcs_effective(const hcs_config *h) {
	if (h->enabled)
		return *h;
	hcs_config simple = {0};
	hcs_set_simple(&simple, h->devices[0].name);
	return simple;
}

void hcs_loader_init(hcs_loader *l) {
	memset(l, 0, sizeof(*l));
}

int hcs_loader_open_section(hcs_loader *l, const char *device, char *err, size_t errlen) {
	for (int i = 0; i < l->n_sections; i++) {
		if (strcmp(l->sections[i].name, device) == 0) {
			snprintf(err, errlen, "duplicate section [hcs.%s]", device);
			return -1;
		}
	}
	if (l->n_sections >= HCS_MAX_DEVICES || strlen(device) >= HCS_NAME_MAX) {
		snprintf(err, errlen, "too many or too long [hcs.<device>] sections (max %d)",
				 HCS_MAX_DEVICES);
		return -1;
	}
	hcs_section *s = &l->sections[l->n_sections];
	memset(s, 0, sizeof(*s));
	snprintf(s->name, sizeof(s->name), "%s", device);
	device_defaults(&s->dev);
	return l->n_sections++;
}

static int parse_flag(const char *text) {
	static const char *const on[]  = {"on", "yes", "true", "1"};
	static const char *const off[] = {"off", "no", "false", "0"};
	for (size_t i = 0; i < 4; i++) {
		if (strcasecmp(text, on[i]) == 0)
			return 1;
		if (strcasecmp(text, off[i]) == 0)
			return 0;
	}
	return -1;
}

static int set_root_key(hcs_loader *l, const char *key, const char *value, char *err,
						size_t errlen) {
	if (strcasecmp(key, "enabled") == 0) {
		int b = parse_flag(value);
		if (l->has_enabled || b < 0) {
			snprintf(err, errlen,
					 l->has_enabled ? "duplicate key 'enabled' in section [hcs]"
									: "'enabled' expects on/off, got '%s'",
					 value);
			return -1;
		}
		l->enabled	   = b != 0;
		l->has_enabled = true;
		return 0;
	}
	if (strcasecmp(key, "devices") != 0) {
		snprintf(err, errlen, "unknown key '%s' in section [hcs]", key);
		return -1;
	}
	if (l->has_devices) {
		snprintf(err, errlen, "duplicate key 'devices' in section [hcs]");
		return -1;
	}
	if (strlen(value) >= sizeof(l->devices)) {
		snprintf(err, errlen, "'devices' value too long");
		return -1;
	}
	snprintf(l->devices, sizeof(l->devices), "%s", value);
	l->has_devices = true;
	return 0;
}

static int set_section_key(hcs_section *s, const char *key, const char *value, char *err,
						   size_t errlen) {
	enum { SEEN_TASKS = 1, SEEN_LAYERS = 2, SEEN_EXPERTS = 4, SEEN_PRIORITY = 8, SEEN_PINS = 16 };
	unsigned bit = 0;
	int		 rc	 = 0;
	if (strcasecmp(key, "tasks") == 0) {
		bit = SEEN_TASKS;
		rc	= s->seen & bit ? 0 : hcs_parse_tasks(value, &s->dev.tasks);
	} else if (strcasecmp(key, "layers") == 0) {
		bit = SEEN_LAYERS;
		rc	= s->seen & bit ? 0 : hcs_parse_count(value, &s->dev.layers);
	} else if (strcasecmp(key, "experts") == 0) {
		bit = SEEN_EXPERTS;
		rc	= s->seen & bit ? 0 : hcs_parse_count(value, &s->dev.experts);
	} else if (strcasecmp(key, "priority") == 0) {
		bit = SEEN_PRIORITY;
		rc	= s->seen & bit ? 0 : hcs_parse_priority(value, &s->dev.priority);
	} else if (strcasecmp(key, "pins") == 0) {
		bit = SEEN_PINS;
		rc	= 0;
		if (s->seen & bit)
			rc = -1;
		else {
			str_span trim = span_trim(value, strlen(value));
			rc = trim.len == 0 ? 0 : hcs_parse_pinlist(value, s->dev.pins, sizeof(s->dev.pins));
		}
	} else {
		snprintf(err, errlen, "unknown key '%s' in section [hcs.%s]", key, s->name);
		return -1;
	}
	if (s->seen & bit) {
		snprintf(err, errlen, "duplicate key '%s' in section [hcs.%s]", key, s->name);
		return -1;
	}
	if (rc < 0) {
		if (bit == SEEN_TASKS)
			snprintf(err, errlen, "'tasks' expects a comma list of prefill, decode, got '%s'",
					 value);
		else if (bit == SEEN_PRIORITY)
			snprintf(err, errlen, "'priority' expects a non-negative integer, got '%s'", value);
		else if (bit == SEEN_PINS)
			snprintf(err, errlen, "'pins' expects a comma-separated expert id list, got '%s'",
					 value);
		else
			snprintf(err, errlen, "'%s' expects a non-negative integer or '*', got '%s'", key,
					 value);
		return -1;
	}
	s->seen |= bit;
	if (bit == SEEN_PRIORITY)
		s->dev.has_priority = true;
	return 0;
}

int hcs_loader_set(hcs_loader *l, int section, const char *key, const char *value, char *err,
				   size_t errlen) {
	if (section < 0)
		return set_root_key(l, key, value, err, errlen);
	return set_section_key(&l->sections[section], key, value, err, errlen);
}

int hcs_loader_finish(const hcs_loader *l, hcs_config *out, char *err, size_t errlen) {
	if (!l->has_devices) {
		snprintf(err, errlen, "section [hcs] requires 'devices'");
		return -1;
	}
	bool		used[HCS_MAX_DEVICES] = {false};
	char		tok[HCS_NAME_MAX];
	const char *cursor = l->devices;
	hcs_config	h	   = {0};
	while (next_token(&cursor, tok, sizeof(tok))) {
		if (!tok[0])
			continue;
		if (h.n_devices >= HCS_MAX_DEVICES) {
			snprintf(err, errlen, "'devices' lists more than %d devices", HCS_MAX_DEVICES);
			return -1;
		}
		int found = -1;
		for (int i = 0; i < l->n_sections; i++)
			if (strcmp(l->sections[i].name, tok) == 0)
				found = i;
		if (found < 0) {
			snprintf(err, errlen, "device '%s' has no [hcs.%s] section", tok, tok);
			return -1;
		}
		used[found]				 = true;
		h.devices[h.n_devices++] = l->sections[found].dev;
		snprintf(h.devices[h.n_devices - 1].name, HCS_NAME_MAX, "%s", tok);
	}
	for (int i = 0; i < l->n_sections; i++) {
		if (!used[i]) {
			snprintf(err, errlen, "section [hcs.%s] is not listed in [hcs] devices",
					 l->sections[i].name);
			return -1;
		}
	}
	h.enabled = l->has_enabled ? l->enabled : true;
	if (h.n_devices == 0 && !h.enabled)
		hcs_set_simple(&h, "");
	*out = h;
	return 0;
}

int hcs_migrate_legacy(hcs_config *h, const hcs_legacy *legacy, char *err, size_t errlen) {
	if (legacy->has_device && strcmp(legacy->device, "auto") != 0)
		snprintf(h->devices[0].name, HCS_NAME_MAX, "%s", legacy->device);
	if (legacy->has_ngl && legacy->ngl > 0) {
		snprintf(err, errlen,
				 "[model] ngl: partial offload is not supported by HCS v1; remove the key or "
				 "use kappai-cli --ngl");
		return -1;
	}
	if (legacy->has_ngl && legacy->ngl == 0)
		backend_format_device("cpu", 0, h->devices[0].name, HCS_NAME_MAX);
	if (legacy->has_cache && legacy->cache > 0) {
		h->enabled			  = true;
		h->devices[0].experts = legacy->cache;
	}
	return 0;
}

static void add_violation(violations *v, const char *device, const char *key, const char *fmt, ...)
	__attribute__((format(printf, 4, 5)));

static void add_violation(violations *v, const char *device, const char *key, const char *fmt,
						  ...) {
	char	msg[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(msg, sizeof(msg), fmt, ap);
	va_end(ap);

	char section[HCS_NAME_MAX + 8];
	if (device)
		snprintf(section, sizeof(section), "[hcs.%s]", device);
	else
		snprintf(section, sizeof(section), "[hcs]");
	if (v->len < v->cap)
		v->len += snprintf(v->buf + v->len, v->cap - v->len, "  %s %s: %s\n", section, key, msg);
}

static void check_device_name(violations *v, const hcs_config *h, int idx,
							  char (*available)[BACKEND_DEVICE_NAME_MAX], int n_available) {
	const char *name = h->devices[idx].name;
	char		backend_name[BACKEND_DEVICE_NAME_MAX];
	int			device_index = 0;
	if (backend_parse_device(name, backend_name, sizeof(backend_name), &device_index) != 0) {
		add_violation(v, name, "device", "'%s' is not a valid device name; expected <backend>_<n>",
					  name);
		return;
	}
	for (int j = 0; j < idx; j++)
		if (strcmp(h->devices[j].name, name) == 0) {
			add_violation(v, name, "device", "listed more than once in 'devices'");
			return;
		}
	if (strcmp(name, "cpu_0") == 0)
		return;
	for (int j = 0; j < n_available; j++)
		if (strcmp(available[j], name) == 0)
			return;

	char list[HCS_DEVICE_LIST_MAX * BACKEND_DEVICE_NAME_MAX] = "";
	for (int j = 0; j < n_available; j++) {
		strncat(list, j ? ", " : "", sizeof(list) - strlen(list) - 1);
		strncat(list, available[j], sizeof(list) - strlen(list) - 1);
	}
	add_violation(v, name, "device", "'%s' is not an available device; available: %s", name, list);
}

static void check_device_fields(violations *v, const hcs_config *h, int idx, int *star_owner) {
	const hcs_device *d = &h->devices[idx];
	if (d->tasks != HCS_TASKS_ALL)
		add_violation(
			v, d->name, "tasks",
			!d->tasks ? "no tasks listed; a device must run both prefill and decode"
					  : "splitting prefill/decode execution is not supported yet; "
						"a device must run both prefill and decode ('tasks = prefill, decode')");
	if (d->layers == HCS_COUNT_UNSET)
		add_violation(v, d->name, "layers", "not set; use a layer count or '*'");
	if (d->layers != HCS_COUNT_ALL)
		return;
	if (*star_owner >= 0)
		add_violation(v, d->name, "layers", "'*' already used by [hcs.%s]; at most one '*' per key",
					  h->devices[*star_owner].name);
	else
		*star_owner = idx;
}

static int check_experts_star(violations *v, const hcs_config *h) {
	int owner = -1;
	for (int i = 0; i < h->n_devices; i++) {
		if (h->devices[i].experts != HCS_COUNT_ALL)
			continue;
		if (owner >= 0)
			add_violation(v, h->devices[i].name, "experts",
						  "'*' already used by [hcs.%s]; at most one '*' per key",
						  h->devices[owner].name);
		else
			owner = i;
	}

	return owner;
}

static int device_count(const hcs_device *d, hcs_field field) {
	return field == HCS_FIELD_EXPERTS ? d->experts : d->layers;
}

int hcs_resolve_counts(const hcs_config *h, hcs_field field, int total, int *held) {
	long numeric				= 0;
	int	 star					= -1;
	int	 pin_n[HCS_MAX_DEVICES] = {0};
	for (int i = 0; i < h->n_devices; i++) {
		int count = device_count(&h->devices[i], field);
		held[i]	  = count >= 0 ? count : 0;
		numeric += held[i];
		if (count == HCS_COUNT_ALL && star < 0)
			star = i;
	}
	if (field == HCS_FIELD_EXPERTS) {
		int ids[128];
		for (int i = 0; i < h->n_devices; i++) {
			if (!h->devices[i].pins[0])
				continue;
			pin_n[i] = hcs_pin_list_ids(h->devices[i].pins, ids, 128);
			numeric += pin_n[i];
		}
	}
	if (star >= 0 && numeric < total) {
		held[star] = total - (int)numeric + pin_n[star];
		numeric	   = total;
	} else if (field == HCS_FIELD_EXPERTS) {
		for (int i = 0; i < h->n_devices; i++)
			held[i] += pin_n[i];
	}
	return (int)MIN(numeric, INT_MAX);
}

static bool check_coverage(violations *v, const hcs_config *h, hcs_field field, int total,
						   int star_owner, bool moe_stream, int *held) {
	bool		experts	 = field == HCS_FIELD_EXPERTS;
	const char *key		 = experts ? "experts" : "layers";
	const char *unit	 = experts ? "expert(s)" : "layer(s)";
	const char *scope	 = experts ? " per layer" : "";
	int			assigned = hcs_resolve_counts(h, field, total, held);
	int			report	 = star_owner >= 0 ? star_owner : h->n_devices - 1;
	const char *name	 = h->devices[report].name;

	if (assigned > total) {
		add_violation(v, name, key, "%d %s assigned, model has %d%s; remove %d", assigned, unit,
					  total, scope, assigned - total);
		return false;
	}
	if (star_owner >= 0 && held[star_owner] == 0) {
		add_violation(v, name, key, "'*' has no %s remaining; all %d are already assigned", unit,
					  total);
		return false;
	}
	if (assigned < total) {
		if (experts && moe_stream)
			return true;
		add_violation(v, name, key,
					  "%d %s assigned, model has %d%s; add '%s = *' to a device or assign %d more",
					  assigned, unit, total, scope, key, total - assigned);
		return false;
	}
	return true;
}

static int first_set_experts(const hcs_config *h) {
	for (int i = 0; i < h->n_devices; i++)
		if (h->devices[i].experts != HCS_COUNT_UNSET)
			return h->devices[i].experts;
	return HCS_COUNT_UNSET;
}

int hcs_pin_list_ids(const char *pins, int *out, int cap) {
	char canon[HCS_PINLIST_MAX];
	if (!pins[0] || hcs_parse_pinlist(pins, canon, sizeof(canon)) < 0)
		return 0;
	int	  n	   = 0;
	char *save = NULL;
	for (char *tok = strtok_r(canon, ",", &save); tok && n < cap; tok = strtok_r(NULL, ",", &save))
		out[n++] = atoi(tok);
	return n;
}

static void check_pin_rules(violations *v, const hcs_config *h, int n_experts) {
	int list[HCS_MAX_DEVICES][128];
	int len[HCS_MAX_DEVICES];
	for (int i = 0; i < h->n_devices; i++)
		len[i] = hcs_pin_list_ids(h->devices[i].pins, list[i], 128);
	for (int i = 0; i < h->n_devices; i++) {
		for (int k = 0; k < len[i]; k++) {
			for (int t = 0; t < k; t++)
				if (list[i][t] == list[i][k])
					add_violation(v, h->devices[i].name, "pins",
								  "expert %d is listed more than once", list[i][k]);
			if (n_experts > 0 && list[i][k] >= n_experts) {
				add_violation(v, h->devices[i].name, "pins",
							  "expert id %d out of range; the model has %d expert(s) per layer",
							  list[i][k], n_experts);
				continue;
			}
			for (int j = 0; j < i; j++)
				for (int t = 0; t < len[j]; t++)
					if (list[j][t] == list[i][k])
						add_violation(v, h->devices[i].name, "pins",
									  "expert %d is already pinned on '%s'; an expert can only "
									  "be pinned once",
									  list[i][k], h->devices[j].name);
		}
	}
}

static void check_model_rules(violations *v, const hcs_config *h, int n_layers, int n_experts,
							  int star_owner, int experts_star_owner, bool moe_stream) {
	int held[HCS_MAX_DEVICES];
	check_coverage(v, h, HCS_FIELD_LAYERS, n_layers, star_owner, false, held);
	if (n_experts > 0 && first_set_experts(h) != HCS_COUNT_UNSET) {
		int expert_held[HCS_MAX_DEVICES];
		check_coverage(v, h, HCS_FIELD_EXPERTS, n_experts, experts_star_owner, moe_stream,
					   expert_held);
	}

	if (n_experts > 0)
		check_pin_rules(v, h, n_experts);
	for (int i = 0; i < h->n_devices; i++) {
		const hcs_device *d = &h->devices[i];
		if (n_experts == 0 && d->experts != HCS_COUNT_UNSET)
			add_violation(v, d->name, "experts", "model is not a MoE model; remove this key");
		if (n_experts == 0 && d->has_priority)
			add_violation(v, d->name, "priority", "model is not a MoE model; remove this key");
		bool no_experts = (d->experts == 0 || d->experts == HCS_COUNT_UNSET) && !d->pins[0];
		if (held[i] == 0 && no_experts && d->layers != HCS_COUNT_UNSET)
			add_violation(v, d->name, "device", "holds nothing; remove it from 'devices'");
	}
}

int hcs_probe_model(const char *path, int *n_layers, int *n_experts, int *n_kv_layers) {
	if (!path || !*path)
		return -1;
	log_level saved = log_get_level();
	log_set_level((log_level)(LOG_ERROR + 1));
	status_code s = model_probe_shape(path, n_layers, n_experts, n_kv_layers);
	log_set_level(saved);
	return s == OK ? 0 : -1;
}

int hcs_validate(const hcs_config *config, const char *model_path, bool moe_stream, char *err,
				 size_t errlen) {
	hcs_config		  eff = hcs_effective(config);
	const hcs_config *h	  = &eff;
	violations		  v	  = {err, errlen, 0};
	err[0]				  = '\0';

	char available[HCS_DEVICE_LIST_MAX][BACKEND_DEVICE_NAME_MAX];
	int	 n_available = backend_enum_devices(available, HCS_DEVICE_LIST_MAX);

	if (h->n_devices <= 0) {
		char best[BACKEND_DEVICE_NAME_MAX];
		backend_best_device(best, sizeof(best));
		add_violation(&v, NULL, "devices", "no devices listed; e.g. 'devices = %s'", best);
	}

	int star_owner = -1;
	for (int i = 0; i < h->n_devices; i++) {
		check_device_name(&v, h, i, available, n_available);
		check_device_fields(&v, h, i, &star_owner);
	}
	int experts_star_owner = check_experts_star(&v, h);

	int n_layers = 0, n_experts = 0, n_kv_layers = 0;
	if (h->n_devices > 0) {
		if (!config->enabled)
			return err[0] ? -1 : 0;
		if (hcs_probe_model(model_path, &n_layers, &n_experts, &n_kv_layers) == 0)
			check_model_rules(&v, h, n_layers, n_experts, star_owner, experts_star_owner,
							  moe_stream);
		else
			v.len += snprintf(err + v.len, v.len < errlen ? errlen - v.len : 0,
							  "  cannot validate HCS: model '%s' unreadable\n",
							  model_path ? model_path : "");
	}
	return err[0] ? -1 : 0;
}

int hcs_layout_order(const hcs_config *h, int *order) {
	int n = 0;
	for (int pass = 0; pass < 2; pass++)
		for (int d = 0; d < h->n_devices; d++)
			if (backend_device_is_host(h->devices[d].name) == (pass == 1))
				order[n++] = d;
	return n;
}

static void materialize_layer_map(const hcs_config *h, const char *model_path, config *cfg) {
	int n_layers = 0, n_experts = 0, n_kv_layers = 0;
	if (hcs_probe_model(model_path, &n_layers, &n_experts, &n_kv_layers) != 0)
		return;

	int held[HCS_MAX_DEVICES];
	int order[HCS_MAX_DEVICES];
	hcs_resolve_counts(h, HCS_FIELD_LAYERS, n_layers, held);
	int n = hcs_layout_order(h, order);
	for (int i = 0; i < n && cfg->n_layer_devices < CONFIG_MAX_LAYER_DEVICES; i++) {
		int d = order[i];
		if (held[d] == 0)
			continue;
		cfg->layer_devices[cfg->n_layer_devices] = h->devices[d].name;
		cfg->layer_counts[cfg->n_layer_devices]	 = held[d];
		cfg->n_layer_devices++;
	}
	if (cfg->n_layer_devices > 0)
		cfg->device = cfg->layer_devices[0];
}

static void materialize_expert_parts(const hcs_config *h, int n_experts, config *cfg) {
	bool any_pins = false;
	for (int d = 0; d < h->n_devices; d++)
		any_pins |= h->devices[d].pins[0] != '\0';
	if (!h->enabled || n_experts <= 0 || (first_set_experts(h) == HCS_COUNT_UNSET && !any_pins))
		return;
	int expert_held[HCS_MAX_DEVICES];
	hcs_resolve_counts(h, HCS_FIELD_EXPERTS, n_experts, expert_held);
	for (int d = 0; d < h->n_devices && cfg->n_expert_devices < CONFIG_MAX_LAYER_DEVICES; d++) {
		int	 held	  = expert_held[d];
		bool has_pins = h->devices[d].pins[0] != '\0';
		if (held <= 0 && !has_pins)
			continue;
		cfg->expert_devices[cfg->n_expert_devices] = h->devices[d].name;
		cfg->expert_counts[cfg->n_expert_devices]  = held;
		cfg->expert_priorities[cfg->n_expert_devices] =
			h->devices[d].has_priority ? h->devices[d].priority : CONFIG_PRIORITY_UNSET;
		cfg->expert_pins[cfg->n_expert_devices] = has_pins ? h->devices[d].pins : NULL;
		cfg->n_expert_devices++;
	}
	for (int d = 0; d < h->n_devices; d++) {
		if (backend_device_is_host(h->devices[d].name) && h->devices[d].has_priority) {
			cfg->host_priority = h->devices[d].priority;
			break;
		}
	}
}

void hcs_materialize(const hcs_config *h, const char *model_path, config *cfg) {
	hcs_config eff = hcs_effective(h);
	cfg->ngl	   = -1;
	cfg->device	   = h->devices[0].name;
	if (eff.n_devices >= 2)
		materialize_layer_map(h, model_path, cfg);

	int n_layers = 0, n_experts = 0, n_kv_layers = 0;
	hcs_probe_model(model_path, &n_layers, &n_experts, &n_kv_layers);
	materialize_expert_parts(h, n_experts, cfg);

	int expert_held[HCS_MAX_DEVICES];
	hcs_resolve_counts(&eff, HCS_FIELD_EXPERTS, n_experts > 0 ? n_experts : 0, expert_held);
	int cached = 0;
	for (int d = 0; d < eff.n_devices; d++)
		cached += expert_held[d];
	int experts = first_set_experts(&eff);
	if (experts == HCS_COUNT_ALL)
		cfg->moe_cache_cap = INT_MAX;
	else if (experts != HCS_COUNT_UNSET)
		cfg->moe_cache_cap = cached > 0 ? cached : experts;
}
