#ifndef JINJA_H
#define JINJA_H

#include "common.h"

typedef enum {
	JV_NONE = 0,
	JV_BOOL,
	JV_STRING,
	JV_DICT,
	JV_LIST,
	JV_MACRO,
} jinja_value_type;

typedef struct jinja_value		jinja_value;
typedef struct jinja_dict_entry jinja_dict_entry;
typedef struct stmt_node		stmt_node;

struct jinja_dict_entry {
	char			 *key;
	jinja_value		 *val;
	jinja_dict_entry *next;
};

struct jinja_value {
	jinja_value_type type;
	union {
		int				  b;
		char			 *s;
		jinja_dict_entry *dict;
		struct {
			jinja_value **items;
			size_t		  n;
		} list;
		stmt_node *macro;
	} as;
};

jinja_value *jinja_none(void);
jinja_value *jinja_bool(int b);
jinja_value *jinja_string(const char *s);
jinja_value *jinja_string_n(const char *s, size_t n);
jinja_value *jinja_dict(void);
jinja_value *jinja_list(void);
void		 jinja_dict_set(jinja_value *d, const char *key, jinja_value *val);
void		 jinja_list_append(jinja_value *l, jinja_value *val);
void		 jinja_value_free(jinja_value *v);

typedef struct stmt_node jinja_program;

jinja_program *jinja_compile(const char *template_src, char *errbuf, size_t errbuf_len);
void		   jinja_program_free(jinja_program *prog);

status_code jinja_render(jinja_program *prog, jinja_value *globals, char **out, char *errbuf,
						 size_t errbuf_len);

#endif