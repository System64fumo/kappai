#ifndef JSON_ESCAPE_H
#define JSON_ESCAPE_H

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

#endif
