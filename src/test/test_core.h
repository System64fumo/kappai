#ifndef TEST_CORE_H
#define TEST_CORE_H

#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include "compute.h"
#include "kvcache.h"
#include "log.h"
#include "model.h"
#include "moe/moe_stream.h"
#include "profile.h"
#include "recipe.h"
#include "sampler.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS_EXACT 1e-5f

#define RTOL_EXACT 1e-3f
#define RTOL_LOOSE 4e-3f
#define ATOL_EXACT 1e-4f
#define ATOL_LOOSE 4e-4f

#define ATOL_KV_QUANT 6e-3f
#define RTOL_KV_QUANT 3e-2f

#define NULL_MAGNITUDE 1e-30f
#define REF_MAGNITUDE_MIN 1e-6f
#define ARCH_GENERATE_N_DECODE 3

#define MAX_RESULTS_PER_FAMILY 128
#define V_COUNT 3

typedef enum {
	V_PASS = 0,
	V_FAIL = 1,
	V_SKIP = 2,
} verdict;

typedef enum {
	OPFAM_MATMUL = 0,
	OPFAM_EMBD_LOOKUP,
	OPFAM_RMSNORM,
	OPFAM_RMSNORM_PER_HEAD,
	OPFAM_RMSNORM_NOWEIGHT,
	OPFAM_ROPE,
	OPFAM_ROPE_EXT,
	OPFAM_ADD_INPLACE,
	OPFAM_FFN_ACTIVATE,
	OPFAM_FFN_ACTIVATE_EX,
	OPFAM_ATTENTION,
	OPFAM_ATTENTION_SWA,
	OPFAM_KV_PUT,
	OPFAM_ARGMAX,
	OPFAM_ARCH_LAYER,
	OPFAM_ARCH_PIPELINE,
	OPFAM_QUANT,
	OPFAM_DEQUANT_PARITY,
	OPFAM_ARCH_DECODE,
	OPFAM_ARCH_COMPOUND,
	OPFAM_ARCH_GENERATE,
	OPFAM_MATMUL_RESIDUAL,
	OPFAM_ROPE_QK,
	OPFAM_EDGE_CASE,
	OPFAM_REPACK_PARITY,
	OPFAM_KV_QUANT_PARITY,
	OPFAM_RMSNORM_ADD,
	OPFAM_PLE_COMBINE,
	OPFAM_SAMPLER,
	OPFAM_TOKENIZER,
	OPFAM_HYBRID_STATE,
	OPFAM_ORCHESTRATION,
	OPFAM_MOE_STREAM,
	OPFAM_COUNT
} op_family;

typedef struct {
	int calls;
	int cnt[V_COUNT];

} op_stat;

typedef struct {
	char	label[160];
	verdict v;
	char	detail[256];

	char debug[512];
} result_entry;

typedef struct {
	const char *name;
	uint32_t	type;
	int			block;
	size_t		bytes;
} qtype_info;

extern int g_pass, g_fail, g_skip;

extern const qtype_info QTYPES[];
extern const int		QTYPES_N;

void		color_init(void);
const char *verdict_str(verdict v);
const char *verdict_color(verdict v);
const char *color_reset(void);
const char *color_dim(void);
const char *color_bold(void);
const char *op_family_name(op_family f);
void		stats_reset(void);
void		compute_debug(const float *y_ref, const float *y_got, int n);
void		record_result(op_family fam, const char *label, verdict v, const char *detail);
void		flush_family(op_family fam);
float		max_abs_diff_at(const float *a, const float *b, int n, int *at);
float		max_abs_val(const float *a, int n);
int			count_nonfinite(const float *a, int n);
float max_combined_ratio_at(const float *a, const float *b, int n, float atol, float rtol, int *at);
verdict	 classify_output(const char *tol_kind, const float *y_ref, const float *y_got, int n,
						 status_code tgt_status, char *detail, size_t detail_sz);
uint32_t next_u32(void);
void	 seed_test_rng(uint64_t s);
void	 fill_random_blocks(void *blocks, int n_blocks, size_t block_bytes, uint32_t type);
void	 fill_random_f32(float *x, int n, float scale);
int		 test_type_per_row(uint32_t type);
void	*test_make_weight(const qtype_info *qt, int n_rows, int k, size_t *out_bytes);

void print_summary_table(void);
void print_final_results(void);
int	 matches_name(int argc, char **argv, const char *name);
int	 wants_all(int argc, char **argv);
void usage(const char *prog);

int	 run_per_op_mode(int argc, char **argv, backend_info *infos, int n_backends);
void run_per_op_tests(backend *cpu, backend *tgt);
void run_arch_tests(backend *cpu, backend *tgt);

void		synth_suite_common_init(void);
extern char synth_fixture_dir[256];
extern char synth_chat_model_path[512];
extern char synth_lfm2_model_path[512];
extern char synth_dsa_model_path[512];
void		run_sampler_tests(void);
void		run_tokenizer_tests(void);
void		run_jinja_tests(void);
void		run_hybrid_state_tests(backend *cpu);
void		run_orchestration_tests(void);
void		run_moe_stream_tests(void);
void		test_quant_determinism(const qtype_info *qt);
void		test_quant_finiteness(const qtype_info *qt);
void		test_quant_q8_0_roundtrip(void);
void		run_repack_parity_tests(backend *cpu);
void test_dequant_parity_cross(backend *cpu, backend *tgt, const qtype_info *qt, int dim, int n);
int	 run_matmul_bench_mode(int argc, char **argv, backend_info *infos, int n_backends);
int	 run_model_mode(int argc, char **argv, backend_info *infos, int n_backends);

#endif