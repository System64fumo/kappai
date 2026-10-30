#include "model_names.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#define QUANT_PATH_MAX 512
#define SHARD_TAIL 20

const char *const MODEL_TYPE_NAMES[MODEL_TYPE_COUNT] = {"language",	 "image",	 "audio",
														"embedding", "decision", "other"};

int model_type_index(const char *name) {
	for (int i = 0; i < MODEL_TYPE_COUNT; i++)
		if (strcmp(MODEL_TYPE_NAMES[i], name) == 0)
			return i;
	return -1;
}

static bool is_quant_token(const char *token) {
	char   up[MODEL_QUANT_MAX];
	size_t i = 0;
	for (; token[i] && i + 1 < sizeof(up); i++)
		up[i] = (char)toupper((unsigned char)token[i]);
	up[i] = '\0';
	return strcmp(up, "BF16") == 0 || strcmp(up, "F16") == 0 || strcmp(up, "F32") == 0 ||
		   (up[0] == 'Q' && isdigit((unsigned char)up[1])) ||
		   (strncmp(up, "IQ", 2) == 0 && isdigit((unsigned char)up[2])) ||
		   strncmp(up, "MXFP", 4) == 0;
}

void model_detect_quant(const char *path, char *out, size_t cap) {
	char stem[QUANT_PATH_MAX];
	snprintf(stem, sizeof(stem), "%s", path);
	size_t len = strlen(stem);
	if (len > 5 && strcasecmp(stem + len - 5, ".gguf") == 0)
		stem[len - 5] = '\0';
	snprintf(out, cap, "?");
	char *save = NULL;
	for (char *tok = strtok_r(stem, "-./", &save); tok; tok = strtok_r(NULL, "-./", &save)) {
		if (!is_quant_token(tok))
			continue;
		size_t i = 0;
		for (; tok[i] && i + 1 < cap; i++)
			out[i] = (char)toupper((unsigned char)tok[i]);
		out[i] = '\0';
	}
}

void model_format_bytes(double bytes, char *out, size_t cap) {
	static const char *const units[] = {"B", "KB", "MB", "GB", "TB"};
	int						 unit	 = 0;
	while (bytes >= 1024.0 && unit < 4) {
		bytes /= 1024.0;
		unit++;
	}
	snprintf(out, cap, unit ? "%.2f %s" : "%.0f %s", bytes, units[unit]);
}

bool model_shard_key(const char *path, char *key, size_t cap, int *index) {
	size_t len = strlen(path);
	if (len < SHARD_TAIL || len >= cap)
		return false;
	const char *tail = path + len - SHARD_TAIL;
	if (tail[0] != '-' || strncmp(tail + 6, "-of-", 4) != 0 || strcmp(tail + 15, ".gguf") != 0)
		return false;
	int value = 0;
	for (int i = 1; i <= 5; i++) {
		if (!isdigit((unsigned char)tail[i]) || !isdigit((unsigned char)tail[10 + i - 1]))
			return false;
		value = value * 10 + (tail[i] - '0');
	}
	snprintf(key, cap, "%s", path);
	memcpy(key + len - SHARD_TAIL + 1, "*****", 5);
	if (index)
		*index = value;
	return true;
}
