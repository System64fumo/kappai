#ifndef JSON_HELPERS_H
#define JSON_HELPERS_H

#include <json-c/json.h>

#include <stdint.h>

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

static inline void json_set_bool(struct json_object *o, const char *key, int val) {
	json_object_object_add(o, key, json_object_new_boolean(val ? 1 : 0));
}

#endif
