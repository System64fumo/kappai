#ifndef JSON_HELPERS_H
#define JSON_HELPERS_H

#include <json-c/json.h>

#include <stdint.h>
#include <stdio.h>

#include "common.h"

static inline void json_escape_append(str_builder *sb, const char *s, size_t n) {
	for (size_t i = 0; i < n; i++) {
		unsigned char c = (unsigned char)s[i];
		switch (c) {
		case '"':
			sb_putb(sb, "\\\"", 2);
			break;
		case '\\':
			sb_putb(sb, "\\\\", 2);
			break;
		case '\n':
			sb_putb(sb, "\\n", 2);
			break;
		case '\r':
			sb_putb(sb, "\\r", 2);
			break;
		case '\t':
			sb_putb(sb, "\\t", 2);
			break;
		default:
			if (c < 0x20) {
				char buf[8];
				snprintf(buf, sizeof(buf), "\\u%04x", c);
				sb_puts(sb, buf);
			} else {
				sb_putb(sb, (const char *)&c, 1);
			}
			break;
		}
	}
}

static inline size_t json_escape_buf(const char *in, size_t in_len, char *out, size_t out_cap) {
	str_builder sb;
	sb_init(&sb);
	json_escape_append(&sb, in, in_len);
	size_t n = MIN(sb.len, out_cap > 0 ? out_cap - 1 : 0);
	if (out_cap > 0) {
		memcpy(out, sb.p, n);
		out[n] = '\0';
	}
	sb_free(&sb);
	return n;
}

static inline struct json_object *json_get(struct json_object *o, const char *key) {
	struct json_object *v = NULL;
	if (o && key && json_object_object_get_ex(o, key, &v))
		return v;
	return NULL;
}

static inline const char *json_get_str(struct json_object *o, const char *key,
									   const char *fallback) {
	struct json_object *v = json_get(o, key);
	if (!v || !json_object_is_type(v, json_type_string))
		return fallback;
	return json_object_get_string(v);
}

static inline double json_get_num(struct json_object *o, const char *key, double fallback) {
	struct json_object *v = json_get(o, key);
	if (!v)
		return fallback;
	return json_object_get_double(v);
}

static inline int64_t json_get_int(struct json_object *o, const char *key, int64_t fallback) {
	struct json_object *v = json_get(o, key);
	if (!v)
		return fallback;
	return json_object_get_int64(v);
}

static inline int json_get_bool(struct json_object *o, const char *key, int fallback) {
	struct json_object *v = json_get(o, key);
	if (!v || !json_object_is_type(v, json_type_boolean))
		return fallback;
	return json_object_get_boolean(v);
}

static inline struct json_object *json_get_obj(struct json_object *o, const char *key) {
	struct json_object *v = json_get(o, key);
	if (!v || !json_object_is_type(v, json_type_object))
		return NULL;
	return v;
}

static inline struct json_object *json_get_arr(struct json_object *o, const char *key) {
	struct json_object *v = json_get(o, key);
	if (!v || !json_object_is_type(v, json_type_array))
		return NULL;
	return v;
}

static inline void json_set_str(struct json_object *o, const char *key, const char *val) {
	json_object_object_add(o, key, json_object_new_string(val ? val : ""));
}

static inline void json_set_int(struct json_object *o, const char *key, int32_t val) {
	json_object_object_add(o, key, json_object_new_int(val));
}

static inline void json_set_int64(struct json_object *o, const char *key, int64_t val) {
	json_object_object_add(o, key, json_object_new_int64(val));
}

static inline void json_set_bool(struct json_object *o, const char *key, int val) {
	json_object_object_add(o, key, json_object_new_boolean(val ? 1 : 0));
}

static inline struct json_object *json_parse_len_ex_code(const char *s, size_t len, size_t *used,
														 enum json_tokener_error *code) {
	if (len > (size_t)INT_MAX) {
		if (code)
			*code = json_tokener_error_size;
		return NULL;
	}
	struct json_tokener	   *tok		 = json_tokener_new();
	struct json_object	   *obj		 = json_tokener_parse_ex(tok, s, (int)len);
	enum json_tokener_error err_code = json_tokener_get_error(tok);
	if (code)
		*code = err_code;
	if (used)
		*used = (size_t)json_tokener_get_parse_end(tok);
	json_tokener_free(tok);
	if (err_code != json_tokener_success && obj) {
		json_object_put(obj);
		obj = NULL;
	}
	return obj;
}

static inline struct json_object *json_parse_len_ex(const char *s, size_t len, size_t *used,
													const char **err) {
	if (len > (size_t)INT_MAX) {
		if (err)
			*err = "input too large";
		return NULL;
	}
	enum json_tokener_error err_code = json_tokener_success;
	struct json_object	   *obj		 = json_parse_len_ex_code(s, len, used, &err_code);
	if (err)
		*err = err_code != json_tokener_success ? json_tokener_error_desc(err_code) : NULL;
	return obj;
}

static inline struct json_object *json_parse_len(const char *s, size_t len, const char **err) {
	return json_parse_len_ex(s, len, NULL, err);
}

static inline struct json_object *json_parse_first(const char *s, const char **err) {
	return json_parse_len(s, strlen(s), err);
}

typedef struct {
	struct json_object *arr;
	size_t				idx;
	size_t				len;
} json_arr_iter;

static inline json_arr_iter json_arr_begin(struct json_object *arr) {
	json_arr_iter it = {arr, 0, arr ? json_object_array_length(arr) : 0};
	return it;
}

static inline int json_arr_next(json_arr_iter *it, struct json_object **out) {
	if (it->idx >= it->len)
		return 0;
	*out = json_object_array_get_idx(it->arr, it->idx);
	it->idx++;
	return 1;
}

#endif
