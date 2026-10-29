#include "decision.h"
#include "log.h"

#include <ctype.h>
#include <stdio.h>

#define D1_MAX_OPTIONS 255
#define D1_MAX_LETTER_OPTIONS 26

static const char *const D1_YES_FORMS[DECISION_LABEL_FORMS] = {"yes", "Yes", "YES"};
static const char *const D1_NO_FORMS[DECISION_LABEL_FORMS]	= {"no", "No", "NO"};

static void label_add_token(decision_label *l, int32_t tok) {
	if (tok < 0)
		return;
	for (int i = 0; i < l->n_tok; i++)
		if (l->tok[i] == tok)
			return;
	l->tok[l->n_tok++] = tok;
}

static status_code label_from_forms(tokenizer *tok, const char *code, const char *const *forms,
									int n_forms, decision_label *out) {
	snprintf(out->code, sizeof(out->code), "%s", code);
	for (int i = 0; i < n_forms; i++)
		label_add_token(out, decision_single_token(tok, forms[i]));
	return out->n_tok > 0 ? OK : ERR_NOT_FOUND;
}

static void fallback_code(int i, char *buf, size_t len) {
	if (i < 26)
		snprintf(buf, len, "%c", 'A' + i);
	else if (i < 126)
		snprintf(buf, len, "%02d", i - 26);
	else if (i < 152)
		snprintf(buf, len, "%c", 'a' + i - 126);
	else if (i < 352)
		snprintf(buf, len, "#%d", i - 152);
	else
		snprintf(buf, len, "%c%c", 'A' + (i - 352) / 26, 'A' + (i - 352) % 26);
}

static bool code_taken(tokenizer *tok, const char *code, const decision_label *done, int n_done,
					   decision_label *out) {
	int32_t id = decision_single_token(tok, code);
	if (id < 0)
		return false;
	for (int i = 0; i < n_done; i++)
		if (done[i].tok[0] == id)
			return false;

	char spaced[16];
	snprintf(spaced, sizeof(spaced), " %s", code);
	snprintf(out->code, sizeof(out->code), "%s", code);
	label_add_token(out, id);
	label_add_token(out, decision_single_token(tok, spaced));
	return true;
}

static bool all_single_letters(const decision_question *q) {
	for (int i = 0; i < q->n_options; i++)
		if (strlen(q->options[i].key) != 1 || !isalpha((unsigned char)q->options[i].key[0]))
			return false;
	return true;
}

static status_code choice_labels(tokenizer *tok, const decision_question *q, decision_label *out) {
	bool letters = all_single_letters(q);
	for (int i = 0; i < q->n_options; i++) {
		char wanted[8];
		if (letters)
			snprintf(wanted, sizeof(wanted), "%s", q->options[i].key);
		else
			fallback_code(q->n_options <= D1_MAX_LETTER_OPTIONS ? i : 26 + i, wanted,
						  sizeof(wanted));

		bool ok = code_taken(tok, wanted, out, i, &out[i]);
		for (int p = 0; !ok && p < 352 + 26 * 26; p++) {
			char pool[8];
			fallback_code(p, pool, sizeof(pool));
			ok = code_taken(tok, pool, out, i, &out[i]);
		}
		if (!ok) {
			ERROR("decision: no single-token label left for %d options", q->n_options);
			return ERR_NOT_FOUND;
		}
	}
	return OK;
}

static status_code d1_labels(tokenizer *tok, const decision_question *q, decision_label *out) {
	if (q->type == DECISION_CHOICE)
		return choice_labels(tok, q, out);

	for (int i = 0; i < q->n_options; i++) {
		const char		  *key	   = q->options[i].key;
		const char *const *forms   = q->type == DECISION_SCORE ? &key
									 : !strcmp(key, "true")	   ? D1_YES_FORMS
															   : D1_NO_FORMS;
		int				   n_forms = q->type == DECISION_SCORE ? 1 : DECISION_LABEL_FORMS;
		if (label_from_forms(tok, key, forms, n_forms, &out[i]) != OK) {
			ERROR("decision: label '%s' is not a single token", key);
			return ERR_NOT_FOUND;
		}
	}
	return OK;
}

static const decision_ops lfm2_d1_ops = {
	.type		 = "lfm2-d1",
	.max_options = D1_MAX_OPTIONS,
	.true_first	 = true,
	.labels		 = d1_labels,
	.forward	 = decision_forward_prefill,
};

DECISION_REGISTER(lfm2_d1, &lfm2_d1_ops)
