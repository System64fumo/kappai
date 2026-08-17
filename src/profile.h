#ifndef PROFILE_H
#define PROFILE_H

#include "common.h"

typedef enum {
	STAGE_EMBD = 0,
	STAGE_RMSNORM,
	STAGE_MATMUL,
	STAGE_MATMUL_QKV,
	STAGE_MATMUL_ATTN_OUT,
	STAGE_MATMUL_FFN,
	STAGE_MOE_ROUTER,
	STAGE_MOE_IO_WAIT,
	STAGE_MOE_SHARED,
	STAGE_ROPE,
	STAGE_KVPUT,
	STAGE_ATTN,
	STAGE_FFN_ACT,
	STAGE_ADD,
	STAGE_LOGITS_NORM,
	STAGE_LOGITS_MATMUL,
	STAGE_LOGITS_READBACK,
	STAGE_TOKENIZE_ENCODE,
	STAGE_TOKENIZE_DECODE,
	STAGE_COUNT
} stage;

typedef struct {
	_Atomic uint64_t total_us[STAGE_COUNT];
	_Atomic uint64_t calls[STAGE_COUNT];
	bool			 enabled;
} profile;

typedef struct {
	uint64_t t0;
	stage	 s;
} profile_scope;

uint64_t time_ns(void);
uint64_t time_us(void);
uint64_t time_ms(void);
void	 profile_reset(profile *p);

static inline void profile_add(profile *p, stage s, uint64_t us) {
	if (!p || !p->enabled || s >= STAGE_COUNT)
		return;
	atomic_fetch_add_explicit(&p->total_us[s], us, memory_order_relaxed);
	atomic_fetch_add_explicit(&p->calls[s], 1, memory_order_relaxed);
}

static inline profile_scope profile_begin(profile *p, stage s) {
	profile_scope ps;
	ps.s  = s;
	ps.t0 = (p && p->enabled) ? time_us() : 0;
	return ps;
}

static inline void profile_end(profile *p, profile_scope *ps) {
	if (!p || !p->enabled || ps->t0 == 0)
		return;
	profile_add(p, ps->s, time_us() - ps->t0);
}

void profile_print(const profile *p, const char *label, FILE *fp);

const char *stage_name(stage s);

#endif