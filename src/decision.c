#include "decision.h"
#include "chat_template.h"
#include "json_helpers.h"
#include "log.h"

#include <math.h>
#include <stdio.h>

#define DECISION_MAX_TYPES 16
#define DECISION_SCORE_MIN 2
#define DECISION_SCORE_MAX 10

struct decision_model {
	const decision_ops *ops;
	jinja_program	   *tmpl;
	float				temperatures[3][4];
};

static const decision_ops *g_decision_registry[DECISION_MAX_TYPES];
static int				   g_decision_count;

void decision_register(const decision_ops *ops) {
	if (g_decision_count < DECISION_MAX_TYPES)
		g_decision_registry[g_decision_count++] = ops;
}

static const char *qtype_name(decision_qtype t) {
	return t == DECISION_CHOICE ? "choice" : t == DECISION_SCORE ? "score" : "noul";
}

const char *decision_type_of(const gguf_ctx *g) {
	const char *arch = NULL;
	char		key[128];
	const char *type = NULL;
	if (gguf_get_str(g, "general.architecture", &arch) != OK)
		return NULL;
	snprintf(key, sizeof(key), "%s.decision.type", arch);
	return gguf_get_str(g, key, &type) == OK ? type : NULL;
}

const decision_ops *decision_ops_of(const gguf_ctx *g) {
	const char *type = decision_type_of(g);
	for (int i = 0; type && i < g_decision_count; i++)
		if (!strcmp(g_decision_registry[i]->type, type))
			return g_decision_registry[i];
	return NULL;
}

decision_model *decision_open(const gguf_ctx *g, char *err, size_t err_len) {
	const char *type = decision_type_of(g);
	if (!type)
		return NULL;

	const decision_ops *ops = decision_ops_of(g);
	if (!ops) {
		snprintf(err, err_len, "unsupported decision model type '%s'", type);
		return NULL;
	}

	const char *tmpl_src = NULL;
	if (gguf_get_str(g, "tokenizer.chat_template.systemone", &tmpl_src) != OK) {
		snprintf(err, err_len, "decision model has no 'systemone' chat template");
		return NULL;
	}

	decision_model *dm				   = xcalloc(1, sizeof(*dm));
	dm->ops							   = ops;
	static const char *const buckets[] = {"2", "3_5", "6_10", "11"};
	const char				*arch	   = NULL;
	gguf_get_str(g, "general.architecture", &arch);
	for (int t = 0; t < 3; t++) {
		char  key[160];
		float fallback = 1.0f;
		snprintf(key, sizeof(key), "%s.decision.temperature.%s", arch, qtype_name(t));
		gguf_get_f32(g, key, &fallback);
		if (!isfinite(fallback) || fallback <= 0.0f)
			fallback = 1.0f;
		for (size_t i = 0; i < ARRAY_LEN(buckets); i++) {
			float value = fallback;
			snprintf(key, sizeof(key), "%s.decision.temperature.%s.%s", arch, qtype_name(t),
					 buckets[i]);
			gguf_get_f32(g, key, &value);
			dm->temperatures[t][i] = isfinite(value) && value > 0.0f ? value : fallback;
		}
	}
	dm->tmpl = jinja_compile(tmpl_src, err, err_len);
	if (!dm->tmpl) {
		free(dm);
		return NULL;
	}
	return dm;
}

void decision_close(decision_model *dm) {
	if (!dm)
		return;
	jinja_program_free(dm->tmpl);
	free(dm);
}

int32_t decision_single_token(tokenizer *tok, const char *text) {
	int32_t ids[4];
	int		n = tokenizer_encode_with_specials(tok, text, 0, ids, 4, NULL);
	return n == 1 ? ids[0] : -1;
}

static bool is_payload(json_object *v) {
	return v &&
		   (json_object_is_type(v, json_type_string) || json_object_is_type(v, json_type_object) ||
			json_object_is_type(v, json_type_array));
}

static void question_free(decision_question *q) {
	for (int i = 0; i < q->n_options; i++)
		free(q->options[i].key);
	free(q->options);
}

void decision_request_free(decision_request *req) {
	for (int i = 0; i < req->n_questions; i++)
		question_free(&req->questions[i]);
	free(req->questions);
	memset(req, 0, sizeof(*req));
}

static void set_option(decision_question *q, int i, const char *key, json_object *description) {
	q->options[i].key		  = xstrdup(key);
	q->options[i].description = description;
}

static bool parse_choice(decision_question *q, json_object *criteria, char *err, size_t err_len) {
	if (!criteria || !json_object_is_type(criteria, json_type_object) ||
		json_object_object_length(criteria) == 0) {
		snprintf(err, err_len, "\"criteria\" must be a non-empty object");
		return false;
	}
	q->n_options = (int)json_object_object_length(criteria);
	q->options	 = xcalloc((size_t)q->n_options, sizeof(*q->options));
	int i		 = 0;
	json_object_object_foreach(criteria, key, val) set_option(q, i++, key, val);
	return true;
}

static bool parse_score(decision_question *q, json_object *criteria, char *err, size_t err_len) {
	size_t n = criteria && json_object_is_type(criteria, json_type_array)
				   ? json_object_array_length(criteria)
				   : 0;
	if (n < DECISION_SCORE_MIN || n > DECISION_SCORE_MAX) {
		snprintf(err, err_len, "\"criteria\" must be an array of %d to %d levels",
				 DECISION_SCORE_MIN, DECISION_SCORE_MAX);
		return false;
	}
	q->n_options = (int)n;
	q->options	 = xcalloc(n, sizeof(*q->options));
	for (size_t i = 0; i < n; i++) {
		char key[16];
		snprintf(key, sizeof(key), "%zu", i);
		set_option(q, (int)i, key, json_object_array_get_idx(criteria, i));
	}
	return true;
}

static bool parse_noul(const decision_ops *ops, decision_question *q, json_object *criteria,
					   char *err, size_t err_len) {
	if (criteria && !json_object_is_type(criteria, json_type_null) &&
		!json_object_is_type(criteria, json_type_object)) {
		snprintf(err, err_len, "\"criteria\" must be an object");
		return false;
	}
	static const char *const keys[2] = {"false", "true"};
	q->n_options					 = 2;
	q->options						 = xcalloc(2, sizeof(*q->options));
	for (int i = 0; i < 2; i++) {
		int			 slot = ops->true_first ? 1 - i : i;
		json_object *desc = NULL;
		if (criteria && !json_object_object_get_ex(criteria, keys[i], &desc))
			desc = json_get(criteria, i ? "yes" : "no");
		set_option(q, slot, keys[i], desc);
	}
	return true;
}

static bool parse_question(const decision_ops *ops, const char *id, json_object *jq,
						   decision_question *q, char *err, size_t err_len) {
	char msg[192];
	if (!json_object_is_type(jq, json_type_object)) {
		snprintf(err, err_len, "questions.%s: must be an object", id);
		return false;
	}
	q->id			= id;
	q->instructions = json_get(jq, "instructions");
	if (!q->instructions || json_object_is_type(q->instructions, json_type_null)) {
		snprintf(err, err_len, "questions.%s: \"instructions\" must be provided", id);
		return false;
	}

	const char	*type	  = json_get_str(jq, "type", "");
	json_object *criteria = json_get(jq, "criteria");
	bool		 ok;
	msg[0] = '\0';
	if (!strcmp(type, "choice")) {
		q->type = DECISION_CHOICE;
		ok		= parse_choice(q, criteria, msg, sizeof(msg));
	} else if (!strcmp(type, "score")) {
		q->type = DECISION_SCORE;
		ok		= parse_score(q, criteria, msg, sizeof(msg));
	} else if (!strcmp(type, "noul")) {
		q->type = DECISION_NOUL;
		ok		= parse_noul(ops, q, criteria, msg, sizeof(msg));
	} else {
		snprintf(msg, sizeof(msg), "\"type\" must be one of: choice, score, noul");
		ok = false;
	}
	if (ok && q->n_options > ops->max_options) {
		snprintf(msg, sizeof(msg), "too many options (%d), this model supports at most %d",
				 q->n_options, ops->max_options);
		ok = false;
	}
	if (!ok)
		snprintf(err, err_len, "questions.%s: %s", id, msg);
	return ok;
}

static bool has_media(json_object *body) {
	static const char *const keys[] = {"files", "images"};
	for (size_t i = 0; i < ARRAY_LEN(keys); i++) {
		json_object *v = json_get(body, keys[i]);
		if (v && json_object_is_type(v, json_type_array) && json_object_array_length(v) > 0)
			return true;
	}
	return false;
}

decision_parse_status decision_parse(const decision_model *dm, json_object *body,
									 decision_request *req, char *err, size_t err_len) {
	memset(req, 0, sizeof(*req));
	json_object *state = NULL;
	if (!json_object_object_get_ex(body, "state", &state)) {
		snprintf(err, err_len, "\"state\" must be provided");
		return DECISION_PARSE_INVALID;
	}
	if (json_object_is_type(state, json_type_null))
		state = NULL;
	else if (!is_payload(state)) {
		snprintf(err, err_len, "\"state\" must be text, an object or an array");
		return DECISION_PARSE_INVALID;
	}
	if (has_media(body)) {
		snprintf(err, err_len, "this model does not support image input");
		return DECISION_PARSE_UNSUPPORTED;
	}

	json_object *questions = json_get_obj(body, "questions");
	if (!questions || json_object_object_length(questions) == 0) {
		snprintf(err, err_len, "\"questions\" must be a non-empty object");
		return DECISION_PARSE_INVALID;
	}

	req->state		 = state;
	req->n_questions = (int)json_object_object_length(questions);
	req->questions	 = xcalloc((size_t)req->n_questions, sizeof(*req->questions));
	int i			 = 0;
	json_object_object_foreach(questions, id, jq) {
		if (!parse_question(dm->ops, id, jq, &req->questions[i], err, err_len)) {
			req->n_questions = i + 1;
			decision_request_free(req);
			return DECISION_PARSE_INVALID;
		}
		i++;
	}
	return DECISION_PARSE_OK;
}

static jinja_value *options_to_jinja(const decision_question *q, const decision_label *labels) {
	jinja_value *list = jinja_list();
	for (int i = 0; i < q->n_options; i++) {
		jinja_value *o = jinja_dict();
		jinja_dict_set(o, "key", jinja_string(q->options[i].key));
		jinja_dict_set(o, "description", json_to_jinja(q->options[i].description));
		jinja_dict_set(o, "label", jinja_string(labels ? labels[i].code : ""));
		jinja_list_append(list, o);
	}
	return list;
}

jinja_value *decision_prompt_values(const decision_request *req, const decision_question *q,
									const decision_label *labels) {
	jinja_value *g = jinja_dict();
	jinja_dict_set(g, "id", jinja_string(q->id));
	jinja_dict_set(g, "type", jinja_string(qtype_name(q->type)));
	jinja_dict_set(g, "instructions", json_to_jinja(q->instructions));
	jinja_dict_set(g, "state", json_to_jinja(req->state));
	jinja_dict_set(g, "options", options_to_jinja(q, labels));
	jinja_dict_set(g, "images", jinja_list());

	return g;
}

static status_code render_prompt(const decision_model *dm, const decision_request *req,
								 const decision_question *q, const decision_label *labels,
								 char **out) {
	jinja_value *g = decision_prompt_values(req, q, labels);
	char		 errbuf[256];
	status_code	 rc = jinja_render(dm->tmpl, g, out, errbuf, sizeof(errbuf));
	jinja_value_free(g);
	if (rc != OK)
		ERROR("decision: systemone template render failed: %s", errbuf);
	return rc;
}

static void softmax_scores(const float *scores, int n, float temperature, double *probs) {
	float  max = scores[0];
	double sum = 0.0;
	for (int i = 1; i < n; i++)
		max = MAX(max, scores[i]);
	for (int i = 0; i < n; i++) {
		probs[i] = exp((double)(scores[i] - max) / temperature);
		sum += probs[i];
	}
	for (int i = 0; i < n; i++)
		probs[i] /= sum;
}

static double confidence_choice(const double *probs, int n) {
	if (n < 2)
		return 1.0;
	double uniform = 1.0 / n;
	double p_max   = 0.0;
	for (int i = 0; i < n; i++)
		p_max = MAX(p_max, probs[i]);
	return MAX(0.0, (p_max - uniform) / (1.0 - uniform));
}

static double confidence_score(const double *probs, int n) {
	if (n < 2)
		return 1.0;
	int mode = 0;
	for (int i = 1; i < n; i++)
		if (probs[i] > probs[mode])
			mode = i;
	double dist = 0.0, dist_uniform = 0.0;
	for (int i = 0; i < n; i++) {
		dist += probs[i] * fabs((double)(i - mode));
		dist_uniform += fabs(i - (n - 1) / 2.0) / n;
	}
	return MAX(0.0, 1.0 - dist / dist_uniform);
}

static json_object *probabilities_json(const decision_question *q, const double *probs) {
	json_object *obj = json_object_new_object();
	for (int i = 0; i < q->n_options; i++)
		json_object_object_add(obj, q->options[i].key, json_object_new_double(probs[i]));
	return obj;
}

static json_object *answer_json(const decision_question *q, const double *probs) {
	json_object *ans = json_object_new_object();
	int			 n	 = q->n_options;
	json_set_str(ans, "type", qtype_name(q->type));

	if (q->type == DECISION_NOUL) {
		for (int i = 0; i < n; i++)
			if (!strcmp(q->options[i].key, "true"))
				json_object_object_add(ans, "noul", json_object_new_double(probs[i]));
		return ans;
	}

	if (q->type == DECISION_CHOICE) {
		int best = 0;
		for (int i = 1; i < n; i++)
			if (probs[i] > probs[best])
				best = i;
		json_set_str(ans, "choice", q->options[best].key);
		json_object_object_add(ans, "probabilities", probabilities_json(q, probs));
		json_object_object_add(ans, "confidence",
							   json_object_new_double(confidence_choice(probs, n)));
		return ans;
	}

	double		 expected = 0.0;
	json_object *legend	  = json_object_new_object();
	for (int i = 0; i < n; i++) {
		expected += i * probs[i];
		json_object_object_add(legend, q->options[i].key,
							   json_object_get(q->options[i].description));
	}
	json_object_object_add(ans, "score", json_object_new_double(expected));
	json_object_object_add(ans, "legend", legend);
	json_object_object_add(ans, "probabilities", probabilities_json(q, probs));
	json_object_object_add(ans, "confidence", json_object_new_double(confidence_score(probs, n)));
	return ans;
}

static status_code question_scores(const decision_model *dm, context *c,
								   const decision_request *req, const decision_question *q,
								   float *scores, int *n_tokens) {
	if (dm->ops->scores)
		return dm->ops->scores(c, dm->tmpl, req, q, scores, n_tokens);
	decision_label *labels = xcalloc((size_t)q->n_options, sizeof(*labels));
	char		   *prompt = NULL;
	int32_t		   *ids	   = NULL;
	status_code		rc	   = dm->ops->labels(&c->tok, q, labels);
	if (rc == OK)
		rc = render_prompt(dm, req, q, labels, &prompt);
	if (rc != OK)
		goto out;

	int cap = (int)strlen(prompt) + 1;
	ids		= xmalloc((size_t)cap * sizeof(*ids));
	int n	= tokenizer_encode_with_specials(&c->tok, prompt, 0, ids, cap, &c->scratch.prof);
	if (n <= 0 || n >= c->n_ctx) {
		ERROR("decision: prompt of %d tokens does not fit the context of %d", n, c->n_ctx);
		rc = ERR_INVALID_ARG;
		goto out;
	}

	context_reset(c);
	c->interrupt = 0;
	int fed		 = context_prefill_tokens(c, ids, n, "decision", true).rc;
	if (fed < 0) {
		rc = (status_code)fed;
		if (rc == ERR_COMPUTE_FAIL)
			c->session_poisoned = true;
		goto out;
	}

	for (int i = 0; i < q->n_options; i++) {
		scores[i] = c->scratch.logits_host[labels[i].tok[0]];
		for (int t = 1; t < labels[i].n_tok; t++)
			scores[i] = MAX(scores[i], c->scratch.logits_host[labels[i].tok[t]]);
	}
	*n_tokens += n;

out:
	free(labels);
	free(prompt);
	free(ids);
	return rc;
}

status_code decision_run(const decision_model *dm, context *c, const decision_request *req,
						 json_object **answers, int *n_tokens) {
	json_object *out = json_object_new_object();
	*n_tokens		 = 0;
	for (int i = 0; i < req->n_questions; i++) {
		const decision_question *q		= &req->questions[i];
		float					*scores = xmalloc((size_t)q->n_options * sizeof(*scores));
		double					*probs	= xmalloc((size_t)q->n_options * sizeof(*probs));
		status_code				 rc		= question_scores(dm, c, req, q, scores, n_tokens);
		if (rc == OK) {
			int n	   = q->n_options;
			int bucket = n <= 2 ? 0 : n <= 5 ? 1 : n <= 10 ? 2 : 3;
			softmax_scores(scores, n, dm->temperatures[q->type][bucket], probs);
			json_object_object_add(out, q->id, answer_json(q, probs));
		}
		free(scores);
		free(probs);
		if (rc != OK) {
			json_object_put(out);
			return rc;
		}
	}
	*answers = out;
	return OK;
}
