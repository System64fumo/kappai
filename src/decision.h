#ifndef DECISION_H
#define DECISION_H

#include "common.h"
#include "context.h"
#include "gguf.h"
#include "tokenizer.h"

#include <json-c/json.h>

typedef enum {
	DECISION_CHOICE,
	DECISION_SCORE,
	DECISION_NOUL,
} decision_qtype;

typedef struct {
	char		*key;
	json_object *description;
} decision_option;

typedef struct {
	const char		*id;
	decision_qtype	 type;
	json_object		*instructions;
	decision_option *options;
	int				 n_options;
} decision_question;

enum { DECISION_LABEL_FORMS = 3 };

typedef struct {
	int plain;
	int question;
	int option;
	int total;
} decision_limits;

typedef struct {
	char	code[8];
	int32_t tok[DECISION_LABEL_FORMS];
	int		n_tok;
} decision_label;

typedef struct decision_request decision_request;
typedef struct decision_model	decision_model;

typedef struct {
	const char *type;
	int			max_options;
	bool		true_first;
	status_code (*labels)(tokenizer *tok, const decision_question *q, decision_label *out);
	const model_variant *variant;
	bool				 escape;
	const char			*sep;
	const char			*state_mark;
	const char			*question_mark;
	const char			*option_mark;
	const char			*score_mark;
	void (*limits)(const decision_question *q, decision_limits *out);
	status_code (*forward)(const decision_model *dm, context *c, const int32_t *ids, int n,
						   const decision_question *q, const decision_label *labels, float *scores,
						   int *n_tokens);
} decision_ops;

struct decision_request {
	decision_question *questions;
	int				   n_questions;
	json_object		  *state;
};

typedef enum {
	DECISION_PARSE_OK,
	DECISION_PARSE_INVALID,
	DECISION_PARSE_UNSUPPORTED,
} decision_parse_status;

void decision_register(const decision_ops *ops);

#define DECISION_REGISTER(id, ops_ptr)                                                             \
	static void __attribute__((constructor)) decision_autoreg_##id(void) {                         \
		decision_register(ops_ptr);                                                                \
	}

const decision_ops *decision_ops_of(const gguf_ctx *g);
const char		   *decision_type_of(const gguf_ctx *g);

decision_model *decision_open(const gguf_ctx *g, char *err, size_t err_len);
void			decision_close(decision_model *dm);

decision_parse_status decision_parse(const decision_model *dm, json_object *body,
									 decision_request *req, char *err, size_t err_len);
void				  decision_request_free(decision_request *req);

status_code decision_run(const decision_model *dm, context *c, const decision_request *req,
						 json_object **answers, int *n_tokens);

jinja_value *decision_prompt_values(const decision_request *req, const decision_question *q,
									const decision_label *labels);

int32_t decision_single_token(tokenizer *tok, const char *text);

status_code decision_forward_prefill(const decision_model *dm, context *c, const int32_t *ids,
									 int n, const decision_question *q,
									 const decision_label *labels, float *scores, int *n_tokens);

status_code decision_forward_masked(const decision_model *dm, context *c, const int32_t *ids, int n,
									const decision_question *q, const decision_label *labels,
									float *scores, int *n_tokens);

#endif
