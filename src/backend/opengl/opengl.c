#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include "gl_context.h"
#include "log.h"
#include "memconfig.h"
#include "recipe.h"
#include "shaders_embedded.h"

#include <GLES3/gl31.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define GL_MAX_BINDINGS 8
#define GL_SCRATCH_POOL_CAP 32
#define GL_MAX_DISPATCH_PER_DIM 65535u
#define GL_PIPELINE_MAX_UNIFORMS 16
#define GL_UNIFORM_NAME_CAP 32
#define GL_PIPELINE_NAME_PTR_CAP 32
#define GL_MAX_PENDING_ACCESSES 256
#define GL_FLUSH_INTERVAL_DISPATCHES 64
#define GL_QK_NORM_ROPE_MAX_HEAD_DIM 1024
#define GL_ROPE_CACHE_SLOTS 4
#define GL_ROPE_CACHE_MIN_POSITIONS 1024
#define GL_WHOLE_BUFFER ((size_t)1 << 62)
#define GL_HC_MAX 8
#define GL_KDA_MAX_HIST 8
#define GL_MLA_MAX_LORA 1024
#define GL_KDA_L2_EPS 1e-6f
#define GL_COMPUTE_BARRIER_BITS (GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)

typedef struct gl_priv gl_priv;

typedef enum {
	GL_PIPE_ARGMAX_STAGE1,
	GL_PIPE_ARGMAX_STAGE2,
	GL_PIPE_RMSNORM,
	GL_PIPE_RMSNORM_PER_HEAD,
	GL_PIPE_ELEMENTWISE,
	GL_PIPE_EMBD_LOOKUP_F16,
	GL_PIPE_EMBD_LOOKUP_F32,
	GL_PIPE_EMBD_LOOKUP_BF16,
	GL_PIPE_EMBD_LOOKUP_QUANT,
	GL_PIPE_MATMUL_F32,
	GL_PIPE_MATMUL_IQ4_NL,
	GL_PIPE_MATMUL_Q4_0,
	GL_PIPE_MATMUL_Q4_1,
	GL_PIPE_MATMUL_Q5_0,
	GL_PIPE_MATMUL_Q5_1,
	GL_PIPE_MATMUL_Q8_0,
	GL_PIPE_MATMUL_Q4_K,
	GL_PIPE_MATMUL_F16,
	GL_PIPE_MATMUL_BF16,
	GL_PIPE_MATMUL_Q5_K,
	GL_PIPE_MATMUL_Q6_K,
	GL_PIPE_MATMUL_IQ3_S,
	GL_PIPE_MATMUL_IQ3_S_REPACKED,
	GL_PIPE_MATMUL_IQ2_S,
	GL_PIPE_MATMUL_IQ2_XXS,
	GL_PIPE_MATMUL_IQ3_XXS,
	GL_PIPE_MATMUL_IQ4_XS,
	GL_PIPE_QUANTIZE_Q8_K,
	GL_PIPE_ROPE,
	GL_PIPE_ROPE_BATCH,
	GL_PIPE_FFN_ACTIVATE,
	GL_PIPE_FFN_ACTIVATE_FUSED_BATCH,
	GL_PIPE_ATTENTION,
	GL_PIPE_ATTENTION_MERGE,
	GL_PIPE_ATTENTION_BATCH,
	GL_PIPE_PARTIAL_ROPE_QK,
	GL_PIPE_KV_PUT,
	GL_PIPE_KV_SHIFT_GATHER,
	GL_PIPE_MLA_KV_PUT,
	GL_PIPE_MLA_ATTENTION,
	GL_PIPE_MOE_GATHER,
	GL_PIPE_MOE_COMBINE,
	GL_PIPE_QUANTIZE_X,
	GL_PIPE_ROPE_QK,
	GL_PIPE_QUANTIZE_X_ACT,
	GL_PIPE_QUANTIZE_Q8_K_ACT,
	GL_PIPE_MATMUL_Q4_0_TILED,
	GL_PIPE_MATMUL_Q4_1_TILED,
	GL_PIPE_MATMUL_Q5_0_TILED,
	GL_PIPE_MATMUL_Q5_1_TILED,
	GL_PIPE_MATMUL_Q4_0_REPACKED,
	GL_PIPE_MATMUL_Q8_0_REPACKED,
	GL_PIPE_MATMUL_IQ4_NL_REPACKED,
	GL_PIPE_MATMUL_Q4_K_REPACKED,
	GL_PIPE_MATMUL_Q5_K_REPACKED,
	GL_PIPE_MATMUL_Q6_K_REPACKED,
	GL_PIPE_MATMUL_Q4_0_REPACKED_TILED,
	GL_PIPE_MATMUL_Q8_0_REPACKED_TILED,
	GL_PIPE_MATMUL_IQ4_NL_REPACKED_TILED,
	GL_PIPE_MATMUL_F32_TILED,
	GL_PIPE_MATMUL_F16_TILED,
	GL_PIPE_MATMUL_BF16_TILED,
	GL_PIPE_HC_INIT,
	GL_PIPE_HC_FINAL,
	GL_PIPE_HC_PRE_COEF,
	GL_PIPE_HC_PRE_MIX,
	GL_PIPE_HC_POST,
	GL_PIPE_KDA_CONV,
	GL_PIPE_KDA_L2NORM,
	GL_PIPE_KDA_GATE,
	GL_PIPE_KDA_DELTA,
	GL_PIPE_KDA_OUT,
	GL_PIPE_MLA_ABSORB_Q,
	GL_PIPE_MLA_ABSORB_ATTN,
	GL_PIPE_DEQUANT_Q8_0,
	GL_PIPE_QK_NORM_ROPE,
	GL_PIPE_QUANTIZE_X_NORM,
	GL_PIPE_MATMUL_F32_ACT,
	GL_PIPE_MATMUL_MM_IQ4_NL,
	GL_PIPE_MATMUL_MM_Q4_0,
	GL_PIPE_MATMUL_MM_Q8_0,
	GL_PIPE_COUNT
} gl_pipe_id;

typedef struct {
	const char *name;
	const char *src;
	uint32_t	flags;
	uint32_t	w_type;
	int			rows_per_wg;
	int			block_size;
} gl_pipeline_def;

enum {
	GL_PIPE_F_FUSED_ACT		= 1u << 0,
	GL_PIPE_F_TILED			= 1u << 1,
	GL_PIPE_F_REPACKED_LANE = 1u << 2,
	GL_PIPE_F_REPACKED		= 1u << 3,
	GL_PIPE_F_MATMUL_WG		= 1u << 4,
	GL_PIPE_F_USES_XQ		= 1u << 5,
	GL_PIPE_F_USES_XQK		= 1u << 6,
	GL_PIPE_F_IQ4NL_LUT		= 1u << 7,
	GL_PIPE_F_BF16			= 1u << 8,
	GL_PIPE_F_IQX_GRID		= 1u << 9,
	GL_PIPE_F_MM			= 1u << 10,
};

typedef struct gl_buf {
	gl_priv *owner;
	void	*host_mirror;
	GLuint	 name;
	size_t	 size_bytes;
	size_t	 store_bytes;
	GLenum	 usage;
	int		 device_dirty;
	int		 host_dirty;
	int		 scratch;
} gl_buf;

typedef struct {
	int		 n_layers;
	gl_buf **layers;
} gl_kv_store;

#define GL_KV_STORE_USAGE ((GLenum)0xFFFF0001u)

typedef struct {
	gl_buf		hdr;
	gl_kv_store st;
} gl_kv_wrap;

typedef struct {
	GLuint		program;
	const char *name;
	const char *src;
	uint32_t	flags;
	uint32_t	w_type;
	int			rows_per_wg;
	int			block_size;
	int			n_uni;
	uint32_t	uvalid;
	uint32_t	uval[GL_PIPELINE_MAX_UNIFORMS];
	GLint		uloc[GL_PIPELINE_MAX_UNIFORMS];
	char		uname[GL_PIPELINE_MAX_UNIFORMS][GL_UNIFORM_NAME_CAP];
	int			n_name_ptrs;
	const char *name_ptr[GL_PIPELINE_NAME_PTR_CAP];
	int			name_ptr_index[GL_PIPELINE_NAME_PTR_CAP];
} gl_pipeline;

typedef struct {
	const float *cos_base;
	const float *sin_base;
	int			 half;
	float		 theta;
	int			 capacity;
	gl_buf		*cos_buf;
	gl_buf		*sin_buf;
	uint8_t		*valid;
} gl_rope_cache;

typedef struct {
	GLuint name;
	size_t offset;
	size_t size;
} gl_binding;

typedef struct {
	GLuint name;
	size_t offset;
	size_t length;
	int	   write;
} gl_access;

typedef struct {
	int		 valid;
	gl_buf	*src;
	GLuint	 src_name;
	size_t	 src_offset;
	int		 k, m;
	int		 act;
	gl_buf	*up;
	int		 stride;
	float	 gs, us;
	gl_buf	*stage;
	uint64_t gen;
} gl_xq_cache;

#define GL_PROF_RING 4096
#define GL_TIME_ELAPSED_EXT 0x88BF
#define GL_QUERY_RESULT_EXT 0x8866

typedef struct {
	int enabled;
	int n;
	void (*gen)(GLsizei n, GLuint *ids);
	void (*del)(GLsizei n, const GLuint *ids);
	void (*begin)(GLenum target, GLuint id);
	void (*end)(GLenum target);
	void (*result)(GLuint id, GLenum pname, uint64_t *out);
	GLuint	 q[GL_PROF_RING];
	int		 pipe_of[GL_PROF_RING];
	uint64_t ns[GL_PIPE_COUNT];
	uint64_t calls[GL_PIPE_COUNT];
} gl_gpuprof;

typedef struct {
	uint64_t dispatches;
	uint64_t barriers;
	uint64_t flushes;
	uint64_t finishes;
	uint64_t readbacks;
	uint64_t launch_us;
	uint64_t wait_us;
} gl_stats;

static uint64_t gl_now_us(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

struct gl_priv {
	gl_context ctx;
	gl_stats   stats;
	gl_gpuprof gpuprof;

	gl_pipeline pipes[GL_PIPE_COUNT];

	GLuint		  current_program;
	gl_binding	  bound[GL_MAX_BINDINGS];
	gl_access	  access[GL_MAX_BINDINGS];
	unsigned	  access_mask;
	gl_access	  pending[GL_MAX_PENDING_ACCESSES];
	int			  n_pending;
	int			  dispatches_since_flush;
	size_t		  ssbo_align;
	gl_rope_cache rope_cache[GL_ROPE_CACHE_SLOTS];
	int			  rope_cache_next;

	gl_buf *qx_stage[2];
	gl_buf *qx;
	size_t	qx_row_words;

	gl_buf *qxk_stage[2];
	gl_buf *qxk;
	size_t	qxk_row_words;

	gl_xq_cache xq_cache[4];
	uint64_t	xq_gen;

	gl_buf *iq3s_grid;
	gl_buf *iq4nl_pairs;
	gl_buf *iqx_grid;

	gl_buf *argmax_partial;
	size_t	argmax_partial_bytes;
	gl_buf *tok_reg;
	gl_buf *tok_rb;
	GLsync	tok_fence[BACKEND_TOKEN_REGS];

	gl_buf *readback;
	size_t	readback_bytes;

	size_t kv_layer_stride;
	size_t kv_kvh_stride;
	int	   kv_head_dim;
	int	   kv_n_kv_heads;
	int	   kv_n_ctx;
	int	   kv_n_layers;
	int	   kv_quant;
	size_t kv_row_bytes;

	gl_buf *attn_part;
	size_t	attn_part_cap;

	gl_buf *dummy;

	gl_buf *scratch_pool[GL_SCRATCH_POOL_CAP];
	size_t	scratch_pool_sizes[GL_SCRATCH_POOL_CAP];
	int		scratch_pool_count;

	size_t device_local_allocated;
	size_t device_local_total_estimate;

	int device_lost;
	int matmul_wg;

	char *attention_src;
	int	  attention_batch_tile_t;
	char *attention_batch_src;

	float *fb_scores;
	size_t fb_scores_cap;
	float *fb_rowbuf;
	size_t fb_rowbuf_cap;

	gl_buf **all_bufs;
	int		 all_bufs_count;
	int		 all_bufs_cap;
};

static void gl_xq_cache_invalidate(gl_priv *p, const gl_buf *b) {
	for (int i = 0; i < (int)(sizeof(p->xq_cache) / sizeof(p->xq_cache[0])); i++) {
		gl_xq_cache *c = &p->xq_cache[i];
		if (!c->valid)
			continue;
		if (c->src == b || c->up == b)
			c->valid = 0;

		if (c->stage == b) {
			c->valid = 0;
			c->stage = NULL;
		}
	}
	if (p->qx == b)
		p->qx = NULL;
	if (p->qxk == b)
		p->qxk = NULL;
}

static void gl_xq_cache_scan_writes(gl_priv *p, unsigned mask) {
	int any_valid = 0;
	for (int i = 0; i < (int)(sizeof(p->xq_cache) / sizeof(p->xq_cache[0])); i++)
		any_valid |= p->xq_cache[i].valid;
	if (!any_valid)
		return;
	for (unsigned m = mask; m; m &= m - 1) {
		const gl_access *a = &p->access[__builtin_ctz(m)];
		if (!a->write)
			continue;
		for (int i = 0; i < (int)(sizeof(p->xq_cache) / sizeof(p->xq_cache[0])); i++) {
			gl_xq_cache *c = &p->xq_cache[i];
			if (!c->valid)
				continue;
			size_t len = (size_t)c->k * (size_t)c->m * sizeof(float);
			if (c->src_name == a->name && a->offset < c->src_offset + len &&
				c->src_offset < a->offset + a->length) {
				c->valid = 0;
				continue;
			}
			if (c->up && c->up->name == a->name)
				c->valid = 0;
		}
	}
}

static inline uint32_t gl_f32_to_bits(float f) {
	union {
		float	 v;
		uint32_t b;
	} u = {.v = f};
	return u.b;
}

static inline float gl_f32_from_bits(uint32_t b) {
	union {
		uint32_t b;
		float	 v;
	} u = {.b = b};
	return u.v;
}

static float gl_f16_to_f32(uint16_t h) {
	uint32_t w			  = (uint32_t)h << 16;
	uint32_t sign		  = w & 0x80000000u;
	uint32_t two_w		  = w + w;
	uint32_t exp_offset	  = 0xE0u << 23;
	float	 exp_scale	  = 0x1.0p-112f;
	float	 normalized	  = gl_f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
	uint32_t magic_mask	  = 126u << 23;
	float	 denormalized = gl_f32_from_bits((two_w >> 17) | magic_mask) - 0.5f;
	uint32_t cutoff		  = 1u << 27;
	uint32_t bits = two_w < cutoff ? gl_f32_to_bits(denormalized) : gl_f32_to_bits(normalized);
	return gl_f32_from_bits(sign | bits);
}

static void gl_softmax_masked(float *scores, int n_valid) {
	float m = scores[0];
	for (int i = 1; i < n_valid; i++)
		if (scores[i] > m)
			m = scores[i];
	float sum = 0.0f;
	for (int i = 0; i < n_valid; i++) {
		scores[i] = expf(scores[i] - m);
		sum += scores[i];
	}
	float inv = 1.0f / sum;
	for (int i = 0; i < n_valid; i++)
		scores[i] *= inv;
}

static _Thread_local const gl_priv *gl_thread_current;

static status_code gl_ensure_context(gl_priv *p) {
	if (!p || !p->ctx.dpy || p->device_lost)
		return ERR_INTERNAL;
	if (gl_thread_current == p)
		return OK;
	if (eglGetCurrentContext() == p->ctx.ctx) {
		gl_thread_current = p;
		return OK;
	}
	if (!eglMakeCurrent(p->ctx.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, p->ctx.ctx)) {
		ERROR("gl: eglMakeCurrent failed on the calling thread (eglErr=0x%04x); "
			  "EGL display/context unusable or device lost",
			  (unsigned)eglGetError());
		return ERR_INTERNAL;
	}
	gl_thread_current = p;
	return OK;
}

#define GL_ENTER(p_)                                                                               \
	do {                                                                                           \
		status_code gl_enter_st_ = gl_ensure_context(p_);                                          \
		if (gl_enter_st_ != OK)                                                                    \
			return gl_enter_st_;                                                                   \
	} while (0)

#define GL_ENTER_VOID(p_)                                                                          \
	do {                                                                                           \
		if (gl_ensure_context(p_) != OK)                                                           \
			return;                                                                                \
	} while (0)

static status_code gl_compile_shader(const char *src, GLenum type, GLuint *out_shader) {
	GLuint shader = glCreateShader(type);
	if (!shader) {
		ERROR("gl: glCreateShader failed (glErr=0x%04x)", (unsigned)glGetError());
		return ERR_INTERNAL;
	}
	glShaderSource(shader, 1, &src, NULL);
	glCompileShader(shader);

	GLint ok = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[4096] = {0};
		glGetShaderInfoLog(shader, sizeof(log), NULL, log);
		ERROR("gl: shader compile failed:\n%s", log);
		glDeleteShader(shader);
		return ERR_FORMAT;
	}
	*out_shader = shader;
	return OK;
}

static status_code gl_link_program_from_shader(GLuint shader, GLuint *out_program) {
	GLuint program = glCreateProgram();
	if (!program) {
		ERROR("gl: glCreateProgram failed (glErr=0x%04x)", (unsigned)glGetError());
		glDeleteShader(shader);
		return ERR_INTERNAL;
	}
	glAttachShader(program, shader);
	glLinkProgram(program);
	glDeleteShader(shader);

	GLint ok = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[4096] = {0};
		glGetProgramInfoLog(program, sizeof(log), NULL, log);
		ERROR("gl: program link failed:\n%s", log);
		glDeleteProgram(program);
		return ERR_FORMAT;
	}
	*out_program = program;
	return OK;
}

#define GL_ATTENTION_MAX_HEAD_DIM 512

#define GL_MLA_MAX_QK_HEAD 512
#define GL_MLA_MAX_KV_LORA 512
#define GL_ATTENTION_TILE_T_MAX 16
#define GL_ATTENTION_Q_CHUNK 4
#define GL_ATTENTION_WG 64
#define GL_ATTENTION_MAX_SPLITS 64
#define GL_ATTENTION_MIN_CHUNK 16
#define GL_ATTENTION_TARGET_WORKGROUPS 256
#define GL_MATMUL_COLS 4
#define GL_MM_TILE 64
#define GL_MM_THREADS 256
#define GL_MM_SHARED_BYTES ((2 * 32 * GL_MM_TILE + 2 * GL_MM_TILE) * 4)
#define GL_MM_MIN_M 16
#define GL_ATTENTION_KV_ROW_BYTES (GL_ATTENTION_MAX_HEAD_DIM * 4)
#define GL_ATTENTION_FIXED_SHARED_BYTES (GL_ATTENTION_Q_CHUNK * GL_ATTENTION_MAX_HEAD_DIM * 4)

static int gl_attention_batch_tile_t(const gl_context *ctx) {
	int shared_budget = ctx->max_shared_bytes > 0 ? ctx->max_shared_bytes : 16384;
	int per_t		  = GL_ATTENTION_KV_ROW_BYTES + GL_ATTENTION_Q_CHUNK * 2 * sizeof(float);
	int tile_t		  = (shared_budget - GL_ATTENTION_FIXED_SHARED_BYTES) / per_t;
	if (tile_t > GL_ATTENTION_TILE_T_MAX)
		tile_t = GL_ATTENTION_TILE_T_MAX;
	if (tile_t < 1)
		tile_t = 1;
	return tile_t;
}

static int gl_matmul_wg(const gl_context *ctx) {
	const char *override = getenv("KAPPAI_GL_MATMUL_WG");
	if (override) {
		int wg = atoi(override);
		if (wg >= 4 && wg <= ctx->max_wg_size && (wg & (wg - 1)) == 0)
			return wg;
		WARN("gl: ignoring KAPPAI_GL_MATMUL_WG=%s (need a power of two in [4, %d])", override,
			 ctx->max_wg_size);
	}
	if (strstr(ctx->renderer, "Mali") || strstr(ctx->renderer, "Panfrost"))
		return 4;
	return 32;
}

static char *gl_shader_with_define(const char *src, const char *name, int value) {
	if (!src || strncmp(src, "#version", 8) != 0) {
		ERROR("gl: shader source must start with #version on the first line");
		return NULL;
	}
	const char *after_version = strchr(src, '\n');
	if (!after_version) {
		ERROR("gl: shader source has no newline after #version");
		return NULL;
	}
	size_t head_len = (size_t)(after_version - src) + 1;
	char   define[64];
	int	   define_len = snprintf(define, sizeof(define), "#define %s %d\n", name, value);
	if (define_len < 0 || (size_t)define_len >= sizeof(define)) {
		ERROR("gl: %s define did not fit (value=%d)", name, value);
		return NULL;
	}
	size_t src_len = strlen(src);
	char  *out	   = xmalloc(src_len + (size_t)define_len + 1);
	memcpy(out, src, head_len);
	memcpy(out + head_len, define, (size_t)define_len);
	memcpy(out + head_len + define_len, src + head_len, src_len - head_len);
	out[src_len + (size_t)define_len] = '\0';
	return out;
}

static void gl_pipeline_cache_uniforms(gl_pipeline *p) {
	p->n_uni	   = 0;
	p->n_name_ptrs = 0;
	GLint count	   = 0;
	glGetProgramiv(p->program, GL_ACTIVE_UNIFORMS, &count);
	for (GLint i = 0; i < count && p->n_uni < GL_PIPELINE_MAX_UNIFORMS; i++) {
		GLsizei namelen					= 0;
		GLint	size					= 0;
		GLenum	type					= 0;
		char	nm[GL_UNIFORM_NAME_CAP] = {0};
		glGetActiveUniform(p->program, (GLuint)i, (GLsizei)sizeof(nm) - 1, &namelen, &size, &type,
						   nm);
		if (!nm[0])
			continue;
		GLint loc = glGetUniformLocation(p->program, nm);
		if (loc < 0)
			continue;
		memcpy(p->uname[p->n_uni], nm, strlen(nm) + 1);
		p->uloc[p->n_uni] = loc;
		p->n_uni++;
	}
}

static void gl_pipeline_warn_missing(const gl_pipeline *p, const char *name) {
	static const gl_pipeline *warned[32];
	static char				  names[32][GL_UNIFORM_NAME_CAP];
	static int				  n_warned = 0;
	for (int i = 0; i < n_warned; i++)
		if (warned[i] == p && strcmp(names[i], name) == 0)
			return;
	WARN("gl: pipeline '%s' has no active uniform '%s' (skipped)", p->name ? p->name : "?", name);
	if (n_warned < 32) {
		warned[n_warned] = p;
		snprintf(names[n_warned], GL_UNIFORM_NAME_CAP, "%s", name);
		n_warned++;
	}
}

static int gl_pipeline_index(gl_pipeline *p, const char *name) {
	for (int i = 0; i < p->n_name_ptrs; i++)
		if (p->name_ptr[i] == name)
			return p->name_ptr_index[i];
	int index = -1;
	for (int i = 0; i < p->n_uni && index < 0; i++)
		if (strcmp(p->uname[i], name) == 0)
			index = i;
	if (index < 0)
		gl_pipeline_warn_missing(p, name);
	if (p->n_name_ptrs < GL_PIPELINE_NAME_PTR_CAP) {
		p->name_ptr[p->n_name_ptrs]		  = name;
		p->name_ptr_index[p->n_name_ptrs] = index;
		p->n_name_ptrs++;
	}
	return index;
}

static GLint gl_pipeline_stage_uniform(gl_pipeline *p, const char *name, uint32_t bits) {
	int i = gl_pipeline_index(p, name);
	if (i < 0)
		return -1;
	uint32_t bit = 1u << i;
	if ((p->uvalid & bit) && p->uval[i] == bits)
		return -1;
	p->uval[i] = bits;
	p->uvalid |= bit;
	return p->uloc[i];
}

static void gl_pu(gl_pipeline *p, const char *name, int v) {
	GLint loc = gl_pipeline_stage_uniform(p, name, (uint32_t)v);
	if (loc >= 0)
		glUniform1i(loc, v);
}

static void gl_puf(gl_pipeline *p, const char *name, float v) {
	GLint loc = gl_pipeline_stage_uniform(p, name, gl_f32_to_bits(v));
	if (loc >= 0)
		glUniform1f(loc, (GLfloat)v);
}

static void gl_pui(gl_pipeline *p, const char *const *names, const int *vals, int count) {
	for (int i = 0; i < count; i++)
		gl_pu(p, names[i], vals[i]);
}

static status_code gl_build_pipeline(const char *src, const char *name, gl_pipeline *out) {
	GLuint		shader;
	status_code s = gl_compile_shader(src, GL_COMPUTE_SHADER, &shader);
	if (s != OK) {
		ERROR("gl: failed to compile shader for pipeline '%s'", name);
		return s;
	}
	s = gl_link_program_from_shader(shader, &out->program);
	if (s != OK) {
		ERROR("gl: failed to link program for pipeline '%s'", name);
		return s;
	}
	out->name	= name;
	out->uvalid = 0;
	gl_pipeline_cache_uniforms(out);
	return OK;
}

static inline gl_buf *as_glbuf(const buffer *b) {
	return (gl_buf *)((char *)b->handle - sizeof(gl_buf));
}

static inline float gl_u32_as_float(uint32_t u) {
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static size_t gl_ssbo_align(gl_priv *p) {
	if (!p->ssbo_align) {
		GLint align = 0;
		glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &align);
		p->ssbo_align = align > 0 ? (size_t)align : 1;
	}
	return p->ssbo_align;
}

static void gl_log_stale_errors(const char *where) {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR)
		ERROR("gl: stale glErr=0x%04x before %s", (unsigned)err, where);
}

static void gl_use_pipeline(gl_priv *p, const gl_pipeline *pipe) {
	if (p->current_program == pipe->program)
		return;
	p->current_program = pipe->program;
	glUseProgram(pipe->program);
}

static void gl_bind_range(gl_priv *p, int binding, GLuint name, size_t offset, size_t size,
						  size_t access_length, int write) {
	p->access[binding] =
		(gl_access){name, offset, access_length ? access_length : GL_WHOLE_BUFFER, write};
	p->access_mask |= 1u << binding;
	gl_binding *current = &p->bound[binding];
	if (current->name == name && current->offset == offset && current->size == size)
		return;
	*current = (gl_binding){name, offset, size};
	if (offset % gl_ssbo_align(p))
		ERROR("gl: binding %d offset %zu violates SSBO alignment %zu (size %zu)", binding, offset,
			  gl_ssbo_align(p), size);
	if (offset == 0 && size == 0)
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)binding, name);
	else
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, (GLuint)binding, name, (GLintptr)offset,
						  (GLsizeiptr)size);
}

static void gl_bind_raw(gl_priv *p, int binding, GLuint name, size_t offset, size_t size,
						int write) {
	gl_bind_range(p, binding, name, offset, size, size, write);
}

static void gl_bind_ssbo_access(int binding, const buffer *b, int write) {
	gl_buf *gb = as_glbuf(b);
	gl_bind_range(gb->owner, binding, gb->name, b->offset, b->offset ? b->size : 0, b->size, write);
}

static void gl_bind_ssbo(int binding, const buffer *b) {
	gl_bind_ssbo_access(binding, b, 0);
}

static void gl_bind_ssbo_out(int binding, const buffer *b) {
	gl_bind_ssbo_access(binding, b, 1);
}

static int gl_pipe_index(const gl_priv *p) {
	for (int i = 0; i < GL_PIPE_COUNT; i++)
		if (p->pipes[i].program == p->current_program)
			return i;
	return 0;
}

static void gl_prof_drain(gl_priv *p) {
	gl_gpuprof *g = &p->gpuprof;
	for (int i = 0; i < g->n; i++) {
		uint64_t ns = 0;
		g->result(g->q[i], GL_QUERY_RESULT_EXT, &ns);
		g->ns[g->pipe_of[i]] += ns;
		g->calls[g->pipe_of[i]]++;
	}
	g->n = 0;
}

static void gl_prof_begin(gl_priv *p) {
	gl_gpuprof *g = &p->gpuprof;
	if (!g->enabled)
		return;
	if (g->n == GL_PROF_RING) {
		glFinish();
		gl_prof_drain(p);
	}
	g->begin(GL_TIME_ELAPSED_EXT, g->q[g->n]);
}

static void gl_prof_end(gl_priv *p) {
	gl_gpuprof *g = &p->gpuprof;
	if (!g->enabled)
		return;
	g->end(GL_TIME_ELAPSED_EXT);
	g->pipe_of[g->n++] = gl_pipe_index(p);
}

static void gl_barrier(gl_priv *p) {
	uint64_t t0 = gl_now_us();
	p->stats.barriers++;
	glMemoryBarrier(GL_COMPUTE_BARRIER_BITS);
	p->stats.launch_us += gl_now_us() - t0;
	p->n_pending = 0;
}

static void gl_finish(gl_priv *p) {
	uint64_t t0 = gl_now_us();
	p->stats.finishes++;
	glFinish();
	p->stats.wait_us += gl_now_us() - t0;
	if (p->gpuprof.enabled)
		gl_prof_drain(p);
	p->dispatches_since_flush = 0;
}

static void gl_flush_if_due(gl_priv *p) {
	if (++p->dispatches_since_flush < GL_FLUSH_INTERVAL_DISPATCHES)
		return;
	p->stats.flushes++;
	glFlush();
	p->dispatches_since_flush = 0;
}

static void gl_dispatch_barrier(gl_priv *p) {
	uint64_t t0 = gl_now_us();
	p->stats.barriers++;
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	p->stats.launch_us += gl_now_us() - t0;
	p->n_pending = 0;
}

static void gl_barrier_if_pending(gl_priv *p) {
	if (p->n_pending)
		gl_barrier(p);
}

static int gl_access_overlaps(const gl_access *a, const gl_access *b) {
	return a->name == b->name && a->offset < b->offset + b->length &&
		   b->offset < a->offset + a->length;
}

static int gl_access_conflicts_with_pending(const gl_priv *p, const gl_access *a) {
	for (int i = 0; i < p->n_pending; i++)
		if ((a->write || p->pending[i].write) && gl_access_overlaps(a, &p->pending[i]))
			return 1;
	return 0;
}

static void gl_record_pending(gl_priv *p, const gl_access *a) {
	for (int i = 0; i < p->n_pending; i++) {
		gl_access *known = &p->pending[i];
		if (known->name == a->name && known->offset == a->offset && known->length == a->length) {
			known->write |= a->write;
			return;
		}
	}
	p->pending[p->n_pending++] = *a;
}

static void gl_launch(gl_priv *p, GLuint groups_x, GLuint groups_y, GLuint groups_z) {
	unsigned mask = p->access_mask;
	if (p->n_pending) {
		for (unsigned m = mask; m; m &= m - 1) {
			if (!gl_access_conflicts_with_pending(p, &p->access[__builtin_ctz(m)]))
				continue;
			gl_dispatch_barrier(p);
			break;
		}
	}
	uint64_t t0 = gl_now_us();
	p->stats.dispatches++;
	gl_prof_begin(p);
	glDispatchCompute(groups_x, groups_y, groups_z);
	gl_prof_end(p);
	p->stats.launch_us += gl_now_us() - t0;
	gl_xq_cache_scan_writes(p, mask);
	gl_flush_if_due(p);
	p->access_mask = 0;
	if (p->n_pending + GL_MAX_BINDINGS > GL_MAX_PENDING_ACCESSES) {
		gl_dispatch_barrier(p);
		return;
	}
	for (unsigned m = mask; m; m &= m - 1)
		gl_record_pending(p, &p->access[__builtin_ctz(m)]);
}

static void gl_forget_buffer_name(gl_priv *p, GLuint name) {
	for (int i = 0; i < GL_MAX_BINDINGS; i++)
		if (p->bound[i].name == name)
			p->bound[i] = (gl_binding){0};
	int kept = 0;
	for (int i = 0; i < p->n_pending; i++)
		if (p->pending[i].name != name)
			p->pending[kept++] = p->pending[i];
	p->n_pending = kept;
}

static void gl_buf_register(gl_priv *p, gl_buf *b) {
	if (p->all_bufs_count == p->all_bufs_cap) {
		int new_cap		= p->all_bufs_cap ? p->all_bufs_cap * 2 : 64;
		p->all_bufs		= xrealloc(p->all_bufs, (size_t)new_cap * sizeof(gl_buf *));
		p->all_bufs_cap = new_cap;
	}
	p->all_bufs[p->all_bufs_count++] = b;
}

static void gl_buf_unregister(gl_priv *p, gl_buf *b) {
	for (int i = 0; i < p->all_bufs_count; i++) {
		if (p->all_bufs[i] == b) {
			p->all_bufs[i] = p->all_bufs[p->all_bufs_count - 1];
			p->all_bufs_count--;
			return;
		}
	}
}

static bool gl_buf_fill_from_desc(gl_buf *b, const tensor_desc *desc, size_t size) {
	char *staging = desc->read_fn ? xmalloc_aligned(WEIGHT_COPY_CHUNK_BYTES, 4096) : NULL;
	if (!desc->read_fn)
		weight_source_prefetch(desc->host_data, MIN(size, WEIGHT_COPY_CHUNK_BYTES));
	for (size_t off = 0; off < size; off += WEIGHT_COPY_CHUNK_BYTES) {
		size_t		len	  = MIN(size - off, WEIGHT_COPY_CHUNK_BYTES);
		const char *chunk = staging;
		if (staging) {
			if (tensor_desc_read(desc, off, staging, len) != OK) {
				free(staging);
				return false;
			}
		} else {
			chunk = (const char *)desc->host_data + off;
			if (off + len < size)
				weight_source_prefetch(chunk + len, MIN(size - off - len, WEIGHT_COPY_CHUNK_BYTES));
		}
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)off, (GLsizeiptr)len, chunk);
		if (b->host_mirror)
			memcpy(b->host_mirror + off, chunk, len);
		if (!staging)
			weight_source_release(chunk, len);
	}
	free(staging);
	return true;
}

static gl_buf *gl_buf_alloc_ex(gl_priv *p, size_t size, GLenum usage, const void *initial_data,
							   bool keep_host_mirror, const tensor_desc *src_desc) {
	if (size == 0)
		size = 1;

	if (gl_ensure_context(p) != OK) {
		ERROR("gl: buf_alloc(%zu bytes) without a usable GL context", size);
		return NULL;
	}

	gl_buf *b		= xcalloc(1, sizeof(gl_buf) + (keep_host_mirror ? size : 0));
	b->owner		= p;
	b->size_bytes	= size;
	b->usage		= usage;
	b->host_mirror	= keep_host_mirror ? (char *)b + sizeof(gl_buf) : NULL;
	b->device_dirty = 0;
	b->host_dirty	= 0;

	gl_log_stale_errors("buffer allocation");
	glGenBuffers(1, &b->name);
	if (!b->name) {
		ERROR("gl: glGenBuffers returned 0 (glErr=0x%04x) -- no current EGL context on "
			  "this thread or the context was lost",
			  (unsigned)glGetError());
		free(b);
		return NULL;
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	size_t alloc_size = (size + 3u) & ~(size_t)3u;
	bool chunked = initial_data && src_desc && (src_desc->release_host_source || src_desc->read_fn);
	if (chunked) {
		glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)alloc_size, NULL, usage);
		if (!gl_buf_fill_from_desc(b, src_desc, size)) {
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			glDeleteBuffers(1, &b->name);
			free(b);
			return NULL;
		}
	} else if (initial_data && alloc_size != size) {
		glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)alloc_size, NULL, usage);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)size, initial_data);
	} else {
		glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)alloc_size, initial_data, usage);
	}
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: glBufferData(%zu bytes) failed (glErr=0x%04x)", size, (unsigned)err);
		glDeleteBuffers(1, &b->name);
		free(b);
		return NULL;
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	b->store_bytes = alloc_size;

	if (initial_data && keep_host_mirror && !chunked)
		memcpy(b->host_mirror, initial_data, size);
	b->host_dirty = (keep_host_mirror && !initial_data) ? 1 : 0;

	gl_buf_register(p, b);
	p->device_local_allocated += size;
	return b;
}

static gl_buf *gl_buf_alloc(gl_priv *p, size_t size, GLenum usage, const void *initial_data) {
	return gl_buf_alloc_ex(p, size, usage, initial_data, true, NULL);
}

static inline void *gl_buf_handle(gl_buf *b) {
	return (char *)b + sizeof(gl_buf);
}

static void gl_buf_sync_to_device(gl_buf *b) {
	if (!b)
		return;

	if (b->usage == GL_STATIC_DRAW && !b->host_dirty)
		return;

	if (b->size_bytes == 0 || b->size_bytes > (size_t)(1ull << 40)) {
		ERROR("gl: sync_to_device: invalid gl_buf (size=%zu, name=%u, "
			  "usage=0x%04x) — buffer handle is not a valid gl_buf pointer",
			  b->size_bytes, b->name, b->usage);
		return;
	}

	if (b->device_dirty) {
		b->host_dirty = 0;
		return;
	}

	if (!b->host_dirty)
		return;

	if (gl_ensure_context(b->owner) != OK)
		return;

	gl_barrier_if_pending(b->owner);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	size_t alloc_size = (b->size_bytes + 3u) & ~(size_t)3u;
	if (alloc_size != b->store_bytes) {
		glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)alloc_size, NULL, b->usage);
		b->store_bytes = alloc_size;
	}
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)b->size_bytes, b->host_mirror);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
		ERROR("gl: sync_to_device upload(%zu bytes) failed (glErr=0x%04x)", b->size_bytes,
			  (unsigned)err);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	b->host_dirty = 0;
	gl_xq_cache_invalidate(b->owner, b);
}

static void gl_buf_sync_to_host(gl_buf *b) {
	if (!b || !b->device_dirty)
		return;
	if (!b->host_mirror)
		return;
	if (gl_ensure_context(b->owner) != OK)
		return;
	gl_finish(b->owner);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	gl_barrier(b->owner);
	void *ptr =
		glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)b->size_bytes, GL_MAP_READ_BIT);
	if (!ptr) {
		ERROR("gl: glMapBufferRange(%zu bytes) failed (glErr=0x%04x), retrying", b->size_bytes,
			  (unsigned)glGetError());
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		gl_finish(b->owner);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
		ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)b->size_bytes,
							   GL_MAP_READ_BIT);
	}
	if (ptr) {
		memcpy(b->host_mirror, ptr, b->size_bytes);
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		b->device_dirty = 0;
	} else {
		ERROR("gl: glMapBufferRange(%zu bytes) failed again (glErr=0x%04x); keeping "
			  "device_dirty flag",
			  b->size_bytes, (unsigned)glGetError());
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

static void gl_buffer_mark_host_written(const buffer *b) {
	if (!b || !b->owner || !b->handle)
		return;
	gl_buf *gb		 = as_glbuf(b);
	gb->host_dirty	 = 1;
	gb->device_dirty = 0;
}

static void gl_buffer_sync_to_host(const buffer *b) {
	if (b && b->owner && b->handle)
		gl_buf_sync_to_host(as_glbuf(b));
}

static void gl_buffer_sync_to_device(const buffer *b) {
	if (b && b->owner && b->handle && strcmp(b->owner->name, "opengl") == 0)
		gl_buf_sync_to_device(as_glbuf(b));
}

static void gl_buf_free(gl_priv *p, gl_buf *b) {
	if (!b)
		return;
	gl_xq_cache_invalidate(p, b);
	gl_buf_unregister(p, b);
	if (b->name) {
		gl_forget_buffer_name(p, b->name);
		glDeleteBuffers(1, &b->name);
		b->name = 0;
	}
	p->device_local_allocated -= MIN(p->device_local_allocated, b->size_bytes);
	free(b);
}

static gl_buf *gl_scratch_pool_take(gl_priv *p, size_t size) {
	int	   best		 = -1;
	size_t best_size = (size_t)-1;
	for (int i = 0; i < p->scratch_pool_count; i++) {
		size_t s = p->scratch_pool_sizes[i];
		if (s >= size && s < best_size) {
			best	  = i;
			best_size = s;
		}
	}
	if (best < 0)
		return NULL;
	gl_buf *b					= p->scratch_pool[best];
	p->scratch_pool[best]		= p->scratch_pool[p->scratch_pool_count - 1];
	p->scratch_pool_sizes[best] = p->scratch_pool_sizes[p->scratch_pool_count - 1];
	p->scratch_pool_count--;
	return b;
}

static void gl_scratch_pool_release(gl_priv *p, gl_buf *b) {
	for (int i = 0; i < p->scratch_pool_count; i++)
		if (p->scratch_pool[i] == b)
			return;
	if (p->scratch_pool_count < GL_SCRATCH_POOL_CAP) {
		p->scratch_pool[p->scratch_pool_count]		 = b;
		p->scratch_pool_sizes[p->scratch_pool_count] = b->size_bytes;
		p->scratch_pool_count++;
		return;
	}
	gl_buf_free(p, b);
}

static status_code gl_dispatch(gl_priv *p, gl_pipeline *pipe, gl_buf **bufs, int n_bufs,
							   const char *const *uniform_names, const int *uniforms,
							   int n_uniforms, GLuint groups_x, GLuint groups_y, GLuint groups_z,
							   int n_outputs) {
	if (!pipe->program) {
		ERROR("gl: dispatch on '%s' with no compiled program", pipe->name);
		return ERR_INTERNAL;
	}
	if (p->device_lost)
		return ERR_INTERNAL;
	GL_ENTER(p);

	for (int i = 0; i < n_bufs && i < GL_MAX_BINDINGS; i++)
		gl_buf_sync_to_device(bufs[i]);

	gl_use_pipeline(p, pipe);
	for (int i = 0; i < n_bufs && i < GL_MAX_BINDINGS; i++) {
		if (!bufs[i])
			continue;
		gl_bind_raw(p, i, bufs[i]->name, 0, 0, i >= n_bufs - n_outputs);
	}
	gl_pui(pipe, uniform_names, uniforms, n_uniforms);

	gl_launch(p, groups_x, groups_y, groups_z);
	return OK;
}

static status_code gl_dispatch_all_dirty(gl_priv *p, gl_pipeline *pipe, gl_buf **bufs, int n_bufs,
										 const char *const *uniform_names, const int *uniforms,
										 int n_uniforms, GLuint gx, GLuint gy, GLuint gz,
										 int n_outputs) {
	status_code s = gl_dispatch(p, pipe, bufs, n_bufs, uniform_names, uniforms, n_uniforms, gx, gy,
								gz, n_outputs);
	if (s != OK)
		return s;
	if (n_outputs > n_bufs)
		n_outputs = n_bufs;
	for (int i = n_bufs - n_outputs; i < n_bufs; i++)
		if (bufs[i])
			bufs[i]->device_dirty = 1;
	return OK;
}

static status_code gl_probe(void) {
	return gl_context_probe();
}

static int gl_device_count(void) {
	return gl_context_probe() == OK ? 1 : 0;
}

static status_code gl_device_identity(int device_index, char *out, size_t cap) {
	if (device_index != 0)
		return ERR_INVALID_ARG;
	return gl_render_node_identity(out, cap);
}

static status_code gl_init(backend *self, int device_index) {
	(void)device_index;
	gl_priv *p = xcalloc(1, sizeof(gl_priv));
	self->priv = p;

	status_code s = gl_context_init(&p->ctx);
	if (s != OK) {
		free(p);
		self->priv = NULL;
		return s;
	}
	if (!p->ctx.has_compute) {
		ERROR("gl: GLES context up but compute shaders unavailable -- cannot use this backend");
		gl_context_free(&p->ctx);
		free(p);
		self->priv = NULL;
		return ERR_UNSUPPORTED;
	}

	p->matmul_wg	 = gl_matmul_wg(&p->ctx);
	p->attention_src = gl_shader_with_define(gl_shader_attention_src, "ATTN_WG", GL_ATTENTION_WG);
	p->attention_batch_tile_t = gl_attention_batch_tile_t(&p->ctx);
	p->attention_batch_src =
		gl_shader_with_define(gl_shader_attention_batch_src, "TILE_T", p->attention_batch_tile_t);
	if (!p->attention_src || !p->attention_batch_src) {
		free(p->attention_src);
		free(p->attention_batch_src);
		gl_context_free(&p->ctx);
		free(p);
		self->priv = NULL;
		return ERR_INTERNAL;
	}

	gl_pipeline_def defs[GL_PIPE_COUNT] = {
		[GL_PIPE_ARGMAX_STAGE1] = {"argmax_stage1", gl_shader_argmax_stage1_src, 0, UINT32_MAX, 1,
								   0},
		[GL_PIPE_ARGMAX_STAGE2] = {"argmax_stage2", gl_shader_argmax_stage2_src, 0, UINT32_MAX, 1,
								   0},
		[GL_PIPE_RMSNORM]		= {"rmsnorm", gl_shader_rmsnorm_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_RMSNORM_PER_HEAD] = {"rmsnorm_per_head", gl_shader_rmsnorm_per_head_src, 0,
									  UINT32_MAX, 1, 0},
		[GL_PIPE_ELEMENTWISE]	  = {"elementwise", gl_shader_elementwise_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_EMBD_LOOKUP_F16] = {"embd_lookup_f16", gl_shader_embd_lookup_f16_src, 0,
									 UINT32_MAX, 1, 0},
		[GL_PIPE_EMBD_LOOKUP_F32] = {"embd_lookup_f32", gl_shader_embd_lookup_f32_src, 0,
									 UINT32_MAX, 1, 0},
		[GL_PIPE_EMBD_LOOKUP_BF16]	= {"embd_lookup_bf16", gl_shader_embd_lookup_bf16_src, 0,
									   UINT32_MAX, 1, 0},
		[GL_PIPE_EMBD_LOOKUP_QUANT] = {"embd_lookup_quant", gl_shader_embd_lookup_quant_src, 0,
									   UINT32_MAX, 1, 0},
		[GL_PIPE_MATMUL_F32]	= {"matmul_f32", gl_shader_matmul_f32_src, 0, GGML_TYPE_F32, 4, 0},
		[GL_PIPE_MATMUL_IQ4_NL] = {"matmul_iq4_nl", gl_shader_matmul_iq4_nl_src,
								   GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQ | GL_PIPE_F_IQ4NL_LUT,
								   GGML_TYPE_IQ4_NL, 4, 32},
		[GL_PIPE_MATMUL_Q4_0]	= {"matmul_q4_0", gl_shader_matmul_q4_0_src,
								   GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQ, GGML_TYPE_Q4_0, 4, 32},
		[GL_PIPE_MATMUL_Q4_1]	= {"matmul_q4_1", gl_shader_matmul_q4_1_src,
								   GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQ, GGML_TYPE_Q4_1, 4, 32},
		[GL_PIPE_MATMUL_Q5_0]	= {"matmul_q5_0", gl_shader_matmul_q5_0_src,
								   GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQ, GGML_TYPE_Q5_0, 4, 32},
		[GL_PIPE_MATMUL_Q5_1]	= {"matmul_q5_1", gl_shader_matmul_q5_1_src,
								   GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQ, GGML_TYPE_Q5_1, 4, 32},
		[GL_PIPE_MATMUL_Q8_0]	= {"matmul_q8_0", gl_shader_matmul_q8_0_src,
								   GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQ, GGML_TYPE_Q8_0, 4, 32},
		[GL_PIPE_MATMUL_Q4_K] = {"matmul_q4_k", gl_shader_matmul_q4_k_src,
								 GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQK, GGML_TYPE_Q4_K, 4, 256},
		[GL_PIPE_MATMUL_F16]  = {"matmul_f16", gl_shader_matmul_f16_src, 0, GGML_TYPE_F16, 4, 0},
		[GL_PIPE_MATMUL_BF16] = {"matmul_bf16", gl_shader_matmul_f16_src, GL_PIPE_F_BF16,
								 GGML_TYPE_BF16, 4, 0},
		[GL_PIPE_MATMUL_F32_TILED] = {"matmul_f32_tiled", gl_shader_matmul_f32_src, GL_PIPE_F_TILED,
									  UINT32_MAX, 4, 0},
		[GL_PIPE_MATMUL_F16_TILED] = {"matmul_f16_tiled", gl_shader_matmul_f16_src, GL_PIPE_F_TILED,
									  UINT32_MAX, 4, 0},
		[GL_PIPE_MATMUL_BF16_TILED] = {"matmul_bf16_tiled", gl_shader_matmul_f16_src,
									   GL_PIPE_F_TILED | GL_PIPE_F_BF16, UINT32_MAX, 4, 0},
		[GL_PIPE_MATMUL_Q5_K]  = {"matmul_q5_k", gl_shader_matmul_q4_k_src,
								  GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQK, GGML_TYPE_Q5_K, 4, 256},
		[GL_PIPE_MATMUL_Q6_K]  = {"matmul_q6_k", gl_shader_matmul_q6_k_src,
								  GL_PIPE_F_MATMUL_WG | GL_PIPE_F_USES_XQK, GGML_TYPE_Q6_K, 4, 256},
		[GL_PIPE_MATMUL_IQ3_S] = {"matmul_iq3_s", gl_shader_matmul_iq3_s_src, GL_PIPE_F_USES_XQK,
								  GGML_TYPE_IQ3_S, 32, 256},
		[GL_PIPE_MATMUL_IQ3_S_REPACKED] = {"matmul_iq3_s_repacked", gl_shader_matmul_iq3_s_src,
										   GL_PIPE_F_REPACKED | GL_PIPE_F_USES_XQK,
										   GGML_TYPE_IQ3_S_REPACKED, 32, 256},
		[GL_PIPE_MATMUL_IQ2_S]	 = {"matmul_iq2_s", gl_shader_matmul_iq2_s_src,
									GL_PIPE_F_USES_XQK | GL_PIPE_F_IQX_GRID, GGML_TYPE_IQ2_S, 32,
									256},
		[GL_PIPE_MATMUL_IQ2_XXS] = {"matmul_iq2_xxs", gl_shader_matmul_iq2_xxs_src,
									GL_PIPE_F_USES_XQK | GL_PIPE_F_IQX_GRID, GGML_TYPE_IQ2_XXS, 32,
									256},
		[GL_PIPE_MATMUL_IQ3_XXS] = {"matmul_iq3_xxs", gl_shader_matmul_iq3_xxs_src,
									GL_PIPE_F_USES_XQK | GL_PIPE_F_IQX_GRID, GGML_TYPE_IQ3_XXS, 32,
									256},
		[GL_PIPE_MATMUL_IQ4_XS]	 = {"matmul_iq4_xs", gl_shader_matmul_iq4_xs_src,
									GL_PIPE_F_USES_XQK | GL_PIPE_F_IQX_GRID, GGML_TYPE_IQ4_XS, 32,
									256},
		[GL_PIPE_MATMUL_Q4_0_TILED]	   = {"matmul_q4_0_tiled", gl_shader_matmul_q4_0_src,
										  GL_PIPE_F_MATMUL_WG | GL_PIPE_F_TILED | GL_PIPE_F_USES_XQ,
										  UINT32_MAX, 4, 32},
		[GL_PIPE_MATMUL_Q4_1_TILED]	   = {"matmul_q4_1_tiled", gl_shader_matmul_q4_1_src,
										  GL_PIPE_F_MATMUL_WG | GL_PIPE_F_TILED | GL_PIPE_F_USES_XQ,
										  UINT32_MAX, 4, 32},
		[GL_PIPE_MATMUL_Q5_0_TILED]	   = {"matmul_q5_0_tiled", gl_shader_matmul_q5_0_src,
										  GL_PIPE_F_MATMUL_WG | GL_PIPE_F_TILED | GL_PIPE_F_USES_XQ,
										  UINT32_MAX, 4, 32},
		[GL_PIPE_MATMUL_Q5_1_TILED]	   = {"matmul_q5_1_tiled", gl_shader_matmul_q5_1_src,
										  GL_PIPE_F_MATMUL_WG | GL_PIPE_F_TILED | GL_PIPE_F_USES_XQ,
										  UINT32_MAX, 4, 32},
		[GL_PIPE_MATMUL_Q4_0_REPACKED] = {"matmul_q4_0_repacked", gl_shader_matmul_q4_0_src,
										  GL_PIPE_F_REPACKED_LANE | GL_PIPE_F_USES_XQ,
										  GGML_TYPE_Q4_0_REPACKED, 8, 32},
		[GL_PIPE_MATMUL_Q8_0_REPACKED] = {"matmul_q8_0_repacked", gl_shader_matmul_q8_0_src,
										  GL_PIPE_F_REPACKED_LANE | GL_PIPE_F_USES_XQ,
										  GGML_TYPE_Q8_0_REPACKED, 8, 32},
		[GL_PIPE_MATMUL_IQ4_NL_REPACKED] = {"matmul_iq4_nl_repacked", gl_shader_matmul_iq4_nl_src,
											GL_PIPE_F_REPACKED_LANE | GL_PIPE_F_USES_XQ |
												GL_PIPE_F_IQ4NL_LUT,
											GGML_TYPE_IQ4_NL_REPACKED, 8, 32},
		[GL_PIPE_MATMUL_Q4_K_REPACKED]	 = {"matmul_q4_k_repacked", gl_shader_matmul_q4_k_src,
											GL_PIPE_F_MATMUL_WG | GL_PIPE_F_REPACKED |
												GL_PIPE_F_USES_XQK,
											GGML_TYPE_Q4_K_REPACKED, 4, 256},
		[GL_PIPE_MATMUL_Q5_K_REPACKED]	 = {"matmul_q5_k_repacked", gl_shader_matmul_q4_k_src,
											GL_PIPE_F_MATMUL_WG | GL_PIPE_F_REPACKED |
												GL_PIPE_F_USES_XQK,
											GGML_TYPE_Q5_K_REPACKED, 4, 256},
		[GL_PIPE_MATMUL_Q6_K_REPACKED]	 = {"matmul_q6_k_repacked", gl_shader_matmul_q6_k_src,
											GL_PIPE_F_MATMUL_WG | GL_PIPE_F_REPACKED |
												GL_PIPE_F_USES_XQK,
											GGML_TYPE_Q6_K_REPACKED, 4, 256},
		[GL_PIPE_MATMUL_Q4_0_REPACKED_TILED]   = {"matmul_q4_0_repacked_tiled",
												  gl_shader_matmul_q4_0_src,
												  GL_PIPE_F_MATMUL_WG |
													  GL_PIPE_F_TILED | GL_PIPE_F_REPACKED_LANE |
													  GL_PIPE_F_USES_XQ,
												  UINT32_MAX, 8, 32},
		[GL_PIPE_MATMUL_Q8_0_REPACKED_TILED]   = {"matmul_q8_0_repacked_tiled",
												  gl_shader_matmul_q8_0_src,
												  GL_PIPE_F_MATMUL_WG |
													  GL_PIPE_F_TILED | GL_PIPE_F_REPACKED_LANE |
													  GL_PIPE_F_USES_XQ,
												  UINT32_MAX, 8, 32},
		[GL_PIPE_MATMUL_IQ4_NL_REPACKED_TILED] = {"matmul_iq4_nl_repacked_tiled",
												  gl_shader_matmul_iq4_nl_src,
												  GL_PIPE_F_MATMUL_WG | GL_PIPE_F_TILED |
													  GL_PIPE_F_REPACKED_LANE | GL_PIPE_F_USES_XQ |
													  GL_PIPE_F_IQ4NL_LUT,
												  UINT32_MAX, 8, 32},
		[GL_PIPE_QUANTIZE_Q8_K]	 = {"quantize_q8_k", gl_shader_quantize_q8_k_src, 0, UINT32_MAX, 1,
									0},
		[GL_PIPE_QUANTIZE_X]	 = {"quantize_x", gl_shader_quantize_x_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_ROPE_QK]		 = {"rope_qk", gl_shader_rope_qk_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_QUANTIZE_X_ACT] = {"quantize_x_act", gl_shader_quantize_x_src, GL_PIPE_F_FUSED_ACT,
									UINT32_MAX, 1, 0},
		[GL_PIPE_QUANTIZE_Q8_K_ACT] = {"quantize_q8_k_act", gl_shader_quantize_q8_k_src,
									   GL_PIPE_F_FUSED_ACT, UINT32_MAX, 1, 0},
		[GL_PIPE_ROPE]				= {"rope", gl_shader_rope_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_ROPE_BATCH]		= {"rope_batch", gl_shader_rope_batch_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_FFN_ACTIVATE] = {"ffn_activate", gl_shader_ffn_activate_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_FFN_ACTIVATE_FUSED_BATCH] = {"ffn_activate_fused_batch",
											  gl_shader_ffn_activate_fused_batch_src, 0, UINT32_MAX,
											  1, 0},
		[GL_PIPE_ATTENTION]				   = {"attention", p->attention_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_ATTENTION_MERGE]		   = {"attention_merge", gl_shader_attention_merge_src, 0,
											  UINT32_MAX, 1, 0},
		[GL_PIPE_ATTENTION_BATCH] = {"attention_batch", p->attention_batch_src, 0, UINT32_MAX, 1,
									 0},
		[GL_PIPE_PARTIAL_ROPE_QK] = {"partial_rope_qk", gl_shader_partial_rope_qk_src, 0,
									 UINT32_MAX, 1, 0},
		[GL_PIPE_KV_PUT]		  = {"kv_put", gl_shader_kv_put_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_KV_SHIFT_GATHER] = {"kv_shift_gather", gl_shader_kv_shift_gather_src, 0,
									 UINT32_MAX, 1, 0},
		[GL_PIPE_MLA_KV_PUT]	  = {"mla_kv_put", gl_shader_mla_kv_put_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_MLA_ATTENTION]	  = {"mla_attention", gl_shader_mla_attention_src, 0, UINT32_MAX, 1,
									 0},
		[GL_PIPE_MOE_GATHER]	  = {"moe_gather", gl_shader_moe_gather_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_MOE_COMBINE]	  = {"moe_combine", gl_shader_moe_combine_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_HC_INIT]		  = {"hc_init", gl_shader_hc_init_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_HC_FINAL]		  = {"hc_final", gl_shader_hc_final_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_HC_PRE_COEF]	  = {"hc_pre_coef", gl_shader_hc_pre_coef_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_HC_PRE_MIX]	  = {"hc_pre_mix", gl_shader_hc_pre_mix_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_HC_POST]		  = {"hc_post", gl_shader_hc_post_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_KDA_CONV]		  = {"kda_conv", gl_shader_kda_conv_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_KDA_L2NORM]	  = {"kda_l2norm", gl_shader_kda_l2norm_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_KDA_GATE]		  = {"kda_gate", gl_shader_kda_gate_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_KDA_DELTA]		  = {"kda_delta", gl_shader_kda_delta_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_KDA_OUT]		  = {"kda_out", gl_shader_kda_out_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_MLA_ABSORB_Q] = {"mla_absorb_q", gl_shader_mla_absorb_q_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_MLA_ABSORB_ATTN] = {"mla_absorb_attn", gl_shader_mla_absorb_attn_src, 0,
									 UINT32_MAX, 1, 0},
		[GL_PIPE_DEQUANT_Q8_0] = {"dequant_q8_0", gl_shader_dequant_q8_0_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_QK_NORM_ROPE] = {"qk_norm_rope", gl_shader_qk_norm_rope_src, 0, UINT32_MAX, 1, 0},
		[GL_PIPE_MATMUL_MM_IQ4_NL] = {"matmul_mm_iq4_nl", gl_shader_matmul_mm_src,
									  GL_PIPE_F_MM | GL_PIPE_F_USES_XQ, UINT32_MAX, GL_MM_TILE, 32},
		[GL_PIPE_MATMUL_MM_Q4_0]   = {"matmul_mm_q4_0", gl_shader_matmul_mm_src,
									  GL_PIPE_F_MM | GL_PIPE_F_USES_XQ, UINT32_MAX, GL_MM_TILE, 32},
		[GL_PIPE_MATMUL_MM_Q8_0]   = {"matmul_mm_q8_0", gl_shader_matmul_mm_src,
									  GL_PIPE_F_MM | GL_PIPE_F_USES_XQ, UINT32_MAX, GL_MM_TILE, 32},
		[GL_PIPE_MATMUL_F32_ACT] = {"matmul_f32_act", gl_shader_matmul_f32_src, GL_PIPE_F_FUSED_ACT,
									UINT32_MAX, 4, 0},
		[GL_PIPE_QUANTIZE_X_NORM] = {"rmsnorm_quantize", gl_shader_quantize_x_src, 0, UINT32_MAX, 1,
									 0},
	};
	_Static_assert(ARRAY_LEN(defs) == GL_PIPE_COUNT, "pipeline definition count mismatch");

	static const struct {
		int			pipe;
		const char *define;
	} variant_defs[] = {
		{GL_PIPE_MATMUL_BF16, "BF16"},
		{GL_PIPE_MATMUL_BF16_TILED, "BF16"},
		{GL_PIPE_MATMUL_Q5_K, "Q5K"},
		{GL_PIPE_MATMUL_Q5_K_REPACKED, "Q5K"},
		{GL_PIPE_QUANTIZE_X_NORM, "FUSED_NORM"},
		{GL_PIPE_MATMUL_MM_IQ4_NL, "IQ4_NL"},
		{GL_PIPE_MATMUL_MM_Q4_0, "Q4_0"},
		{GL_PIPE_MATMUL_MM_Q8_0, "Q8_0"},
	};
	for (int i = 0; i < GL_PIPE_COUNT; i++) {
		char *tuned_src = NULL;
		if (defs[i].flags & GL_PIPE_F_MATMUL_WG) {
			tuned_src = gl_shader_with_define(defs[i].src, "WG", p->matmul_wg);
			if (!tuned_src) {
				s = ERR_INTERNAL;
				break;
			}
		}
		if (defs[i].flags & (GL_PIPE_F_REPACKED | GL_PIPE_F_REPACKED_LANE)) {
			char *with_repacked =
				gl_shader_with_define(tuned_src ? tuned_src : defs[i].src, "REPACKED", 1);
			if (!with_repacked) {
				free(tuned_src);
				s = ERR_INTERNAL;
				break;
			}
			free(tuned_src);
			tuned_src = with_repacked;
		}
		if (defs[i].flags & GL_PIPE_F_TILED) {
			char *with_cols =
				gl_shader_with_define(tuned_src ? tuned_src : defs[i].src, "MT", GL_MATMUL_COLS);
			if (!with_cols) {
				free(tuned_src);
				s = ERR_INTERNAL;
				break;
			}
			free(tuned_src);
			tuned_src = with_cols;
		}
		if ((defs[i].flags & GL_PIPE_F_FUSED_ACT) && !tuned_src) {
			tuned_src = gl_shader_with_define(defs[i].src, "FUSED_ACT", 1);
			if (!tuned_src) {
				s = ERR_INTERNAL;
				break;
			}
		}
		for (size_t vi = 0; vi < ARRAY_LEN(variant_defs); vi++) {
			if (variant_defs[vi].pipe != i)
				continue;
			char *with_var = gl_shader_with_define(tuned_src ? tuned_src : defs[i].src,
												   variant_defs[vi].define, 1);
			if (!with_var) {
				free(tuned_src);
				s = ERR_INTERNAL;
				break;
			}
			free(tuned_src);
			tuned_src = with_var;
		}
		if (s != OK)
			break;
		if (!(defs[i].flags & GL_PIPE_F_MM) ||
			(p->ctx.max_shared_bytes >= GL_MM_SHARED_BYTES && p->ctx.max_wg_size >= GL_MM_THREADS &&
			 p->ctx.max_wg_invocations >= GL_MM_THREADS))
			s = gl_build_pipeline(tuned_src ? tuned_src : defs[i].src, defs[i].name, &p->pipes[i]);
		p->pipes[i].src			= defs[i].src;
		p->pipes[i].flags		= defs[i].flags;
		p->pipes[i].w_type		= defs[i].w_type;
		p->pipes[i].rows_per_wg = defs[i].rows_per_wg;
		p->pipes[i].block_size	= defs[i].block_size;
		free(tuned_src);
		if (s != OK) {
			for (int j = 0; j < i; j++)
				glDeleteProgram(p->pipes[j].program);
			free(p->attention_src);
			free(p->attention_batch_src);
			gl_context_free(&p->ctx);
			free(p);
			self->priv = NULL;
			return s;
		}
	}

	static const int8_t iq4nl_values[16] = {
		-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
	};
	uint32_t iq4nl_pairs[256];
	for (int b = 0; b < 256; b++)
		iq4nl_pairs[b] =
			(uint32_t)((int32_t)iq4nl_values[b & 15] + (int32_t)iq4nl_values[b >> 4] * 65536);
	p->iq3s_grid = gl_buf_alloc(p, sizeof(ggml_iq3s_grid), GL_STATIC_DRAW, ggml_iq3s_grid);
	{
		static uint8_t iqx_data[sizeof(ggml_iq2s_grid) + sizeof(ggml_iq2xxs_grid) +
								sizeof(ggml_iq3xxs_grid) + sizeof(ggml_ksigns_iq2xs)];
		uint8_t		  *dp = iqx_data;
		memcpy(dp, ggml_iq2s_grid, sizeof(ggml_iq2s_grid));
		dp += sizeof(ggml_iq2s_grid);
		memcpy(dp, ggml_iq2xxs_grid, sizeof(ggml_iq2xxs_grid));
		dp += sizeof(ggml_iq2xxs_grid);
		memcpy(dp, ggml_iq3xxs_grid, sizeof(ggml_iq3xxs_grid));
		dp += sizeof(ggml_iq3xxs_grid);
		memcpy(dp, ggml_ksigns_iq2xs, sizeof(ggml_ksigns_iq2xs));
		p->iqx_grid = gl_buf_alloc(p, sizeof(iqx_data), GL_STATIC_DRAW, iqx_data);
	}
	p->iq4nl_pairs = gl_buf_alloc(p, sizeof(iq4nl_pairs), GL_STATIC_DRAW, iq4nl_pairs);
	if (!p->iq3s_grid || !p->iq4nl_pairs || !p->iqx_grid) {
		if (p->iq3s_grid)
			gl_buf_free(p, p->iq3s_grid);
		if (p->iqx_grid)
			gl_buf_free(p, p->iqx_grid);
		if (p->iq4nl_pairs)
			gl_buf_free(p, p->iq4nl_pairs);
		for (int i = 0; i < GL_PIPE_COUNT; i++)
			glDeleteProgram(p->pipes[i].program);
		free(p->attention_src);
		free(p->attention_batch_src);
		gl_context_free(&p->ctx);
		free(p);
		self->priv = NULL;
		return ERR_OUT_OF_MEMORY;
	}

	if (p->ctx.unified_memory)
		self->caps |= BCAP_UNIFIED_MEMORY;
	p->device_local_total_estimate =
		p->ctx.video_memory_bytes ? p->ctx.video_memory_bytes : get_total_memory();

	log_tag("OGL", "%.*s", (int)backend_name_brief_len(p->ctx.renderer), p->ctx.renderer);
	DEBUG("opengl: %s (max_wg=%d, shared=%dKB, ssbo_bindings=%d, attn_wg=%d, matmul_wg=%d)",
		  p->ctx.renderer, p->ctx.max_wg_size, p->ctx.max_shared_bytes / 1024,
		  p->ctx.max_ssbo_bindings, GL_ATTENTION_WG, p->matmul_wg);

	return OK;
}

static void gl_gpuprof_report(gl_priv *p);

static void gl_free(backend *self) {
	gl_priv *p = self->priv;
	if (!p)
		return;
	gl_thread_current = NULL;

	if (gl_ensure_context(p) != OK) {
		WARN("gl: free could not make the GL context current; leaking GL objects");
		free(p->all_bufs);
		free(p->attention_src);
		free(p->attention_batch_src);
		gl_context_free(&p->ctx);
		gl_thread_current = NULL;
		free(p);
		self->priv = NULL;
		return;
	}

	INFO("gl: totals: %llu dispatches, %llu barriers, %llu flushes, %llu finishes, %llu readbacks",
		 (unsigned long long)p->stats.dispatches, (unsigned long long)p->stats.barriers,
		 (unsigned long long)p->stats.flushes, (unsigned long long)p->stats.finishes,
		 (unsigned long long)p->stats.readbacks);
	INFO("gl: host time: %.1f ms in dispatch/barrier calls, %.1f ms waiting in glFinish",
		 (double)p->stats.launch_us / 1000.0, (double)p->stats.wait_us / 1000.0);

	gl_gpuprof_report(p);

	for (int i = 0; i < BACKEND_TOKEN_REGS; i++)
		if (p->tok_fence[i])
			glDeleteSync(p->tok_fence[i]);
	for (int i = 0; i < GL_PIPE_COUNT; i++)
		glDeleteProgram(p->pipes[i].program);

	gl_buf_free(p, p->iq3s_grid);
	gl_buf_free(p, p->iq4nl_pairs);
	gl_buf_free(p, p->iqx_grid);
	for (int i = 0; i < GL_ROPE_CACHE_SLOTS; i++)
		free(p->rope_cache[i].valid);
	free(p->attention_src);
	free(p->attention_batch_src);
	free(p->fb_scores);
	free(p->fb_rowbuf);

	for (int i = 0; i < p->scratch_pool_count; i++)
		gl_buf_free(p, p->scratch_pool[i]);
	p->scratch_pool_count = 0;

	while (p->all_bufs_count > 0)
		gl_buf_free(p, p->all_bufs[0]);
	free(p->all_bufs);

	gl_context_free(&p->ctx);

	gl_thread_current = NULL;
	free(p);
	self->priv = NULL;
}

static int gl_type_is_repacked(uint32_t type) {
	return type == GGML_TYPE_Q4_0_REPACKED || type == GGML_TYPE_Q8_0_REPACKED ||
		   type == GGML_TYPE_IQ4_NL_REPACKED || type == GGML_TYPE_Q4_K_REPACKED ||
		   type == GGML_TYPE_Q5_K_REPACKED || type == GGML_TYPE_Q6_K_REPACKED ||
		   type == GGML_TYPE_IQ3_S_REPACKED;
}

static int gl_matmul_pipe_for_type(gl_priv *p, uint32_t w_type, gl_pipeline **out_pipe,
								   int *out_block_size);

static int gl_weight_needs_mirror(gl_priv *p, const tensor_desc *desc) {
	if (desc->type == GGML_TYPE_F32)
		return 1;
	gl_pipeline *pipe		= NULL;
	int			 block_size = 0;
	if (!gl_matmul_pipe_for_type(p, desc->type, &pipe, &block_size))
		return 1;
	if (block_size > 0 && (desc->dims[0] % (uint64_t)block_size) != 0)
		return 1;
	return 0;
}

static status_code gl_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	uint64_t d1 = (desc->n_dims == 1) ? 1 : desc->dims[1];
	if (gl_type_is_repacked(desc->type))
		d1 = (d1 + 7) & ~(uint64_t)7;
	size_t size = ggml_row_size(desc->type, desc->dims[0]) * (size_t)d1;

	bool	keep_host_mirror = gl_weight_needs_mirror(p, desc);
	gl_buf *b = gl_buf_alloc_ex(p, size, GL_STATIC_DRAW, desc->host_data, keep_host_mirror, desc);
	if (!b)
		return ERR_OUT_OF_MEMORY;

	out->handle	  = gl_buf_handle(b);
	out->size	  = size;
	out->host_ptr = desc->host_data;
	out->offset	  = 0;
	out->owner	  = self;
	return OK;
}

static status_code gl_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	gl_priv *p = self->priv;
	GL_ENTER(p);

	gl_buf *b = gl_scratch_pool_take(p, size);
	if (b) {
		b->host_dirty	= 0;
		b->device_dirty = 0;
		out->handle		= b->host_mirror;
		out->size		= b->size_bytes;
		out->host_ptr	= NULL;
		out->offset		= 0;
		out->owner		= self;
		return OK;
	}
	b = gl_buf_alloc(p, size, GL_DYNAMIC_COPY, NULL);
	if (!b)
		return ERR_OUT_OF_MEMORY;
	b->scratch	  = 1;
	out->handle	  = b->host_mirror;
	out->size	  = size;
	out->host_ptr = NULL;
	out->offset	  = 0;
	out->owner	  = self;
	return OK;
}

static status_code gl_buffer_alloc_from_host(backend *self, const void *host_data, size_t size,
											 buffer *out) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *b = gl_buf_alloc_ex(p, size, GL_DYNAMIC_COPY, host_data, false, NULL);
	if (!b)
		return ERR_OUT_OF_MEMORY;
	out->handle	  = gl_buf_handle(b);
	out->size	  = size;
	out->host_ptr = NULL;
	out->offset	  = 0;
	out->owner	  = self;
	return OK;
}

static void gl_kv_store_free(gl_priv *p, gl_kv_store *st);

static void gl_buffer_free(backend *self, buffer *buf) {
	gl_priv *p = self->priv;
	if (!buf || !buf->handle)
		return;
	GL_ENTER_VOID(p);
	gl_buf *b = as_glbuf(buf);

	if (b->usage == GL_KV_STORE_USAGE)
		gl_kv_store_free(p, buf->handle);
	else if (b->scratch)
		gl_scratch_pool_release(p, b);
	else
		gl_buf_free(p, b);

	buf->handle	  = NULL;
	buf->size	  = 0;
	buf->host_ptr = NULL;
	buf->offset	  = 0;
}

static status_code gl_read_region_f32(gl_priv *p, gl_buf *b, size_t byte_off, void *dst,
									  size_t bytes) {
	if (bytes == 0)
		return OK;
	if (!p->readback || p->readback_bytes < bytes) {
		size_t want = (bytes + (size_t)65535) & ~(size_t)65535;
		if (p->readback)
			gl_buf_free(p, p->readback);
		p->readback		  = gl_buf_alloc(p, want, GL_DYNAMIC_COPY, NULL);
		p->readback_bytes = p->readback ? want : 0;
		if (!p->readback)
			return ERR_OUT_OF_MEMORY;
	}
	gl_barrier_if_pending(p);
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	glBindBuffer(GL_COPY_READ_BUFFER, b->name);
	glBindBuffer(GL_COPY_WRITE_BUFFER, p->readback->name);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr)byte_off, 0,
						(GLsizeiptr)bytes);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	if (glGetError() != GL_NO_ERROR)
		return ERR_INTERNAL;
	p->stats.readbacks++;
	GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
	if (!fence)
		return ERR_INTERNAL;
	GLenum wr = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, (GLuint64)500000000);
	while (wr == GL_TIMEOUT_EXPIRED)
		wr = glClientWaitSync(fence, 0, (GLuint64)1000000000);
	glDeleteSync(fence);
	if (wr != GL_ALREADY_SIGNALED && wr != GL_CONDITION_SATISFIED)
		return ERR_INTERNAL;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->readback->name);
	void *src = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)bytes, GL_MAP_READ_BIT);
	if (!src) {
		glGetError();
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		return ERR_INTERNAL;
	}
	memcpy(dst, src, bytes);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	return OK;
}

static status_code gl_read_two_regions_f32(gl_priv *p, gl_buf *b0, size_t off0, void *dst0,
										   size_t bytes0, gl_buf *b1, size_t off1, void *dst1,
										   size_t bytes1) {
	size_t pad0	 = (bytes0 + (size_t)255) & ~(size_t)255;
	size_t total = pad0 + bytes1;
	if (!p->readback || p->readback_bytes < total) {
		size_t want = (total + (size_t)65535) & ~(size_t)65535;
		if (p->readback)
			gl_buf_free(p, p->readback);
		p->readback		  = gl_buf_alloc(p, want, GL_DYNAMIC_COPY, NULL);
		p->readback_bytes = p->readback ? want : 0;
		if (!p->readback)
			return ERR_OUT_OF_MEMORY;
	}
	gl_barrier_if_pending(p);
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	glBindBuffer(GL_COPY_WRITE_BUFFER, p->readback->name);
	glBindBuffer(GL_COPY_READ_BUFFER, b0->name);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr)off0, 0,
						(GLsizeiptr)bytes0);
	glBindBuffer(GL_COPY_READ_BUFFER, b1->name);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr)off1, (GLintptr)pad0,
						(GLsizeiptr)bytes1);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	if (glGetError() != GL_NO_ERROR)
		return ERR_INTERNAL;
	p->stats.readbacks++;
	GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
	if (!fence)
		return ERR_INTERNAL;
	GLenum wr = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, (GLuint64)500000000);
	while (wr == GL_TIMEOUT_EXPIRED)
		wr = glClientWaitSync(fence, 0, (GLuint64)1000000000);
	glDeleteSync(fence);
	if (wr != GL_ALREADY_SIGNALED && wr != GL_CONDITION_SATISFIED)
		return ERR_INTERNAL;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->readback->name);
	void *src = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)total, GL_MAP_READ_BIT);
	if (!src) {
		glGetError();
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		return ERR_INTERNAL;
	}
	memcpy(dst0, src, bytes0);
	memcpy(dst1, (const char *)src + pad0, bytes1);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	return OK;
}

static status_code gl_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n);

static status_code gl_buffer_read_f32_x2(backend *self, const buffer *buf_a, float *dst_a, int n_a,
										 const buffer *buf_b, float *dst_b, int n_b) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *ba		= as_glbuf(buf_a);
	gl_buf *bb		= as_glbuf(buf_b);
	size_t	bytes_a = (size_t)n_a * sizeof(float);
	size_t	bytes_b = (size_t)n_b * sizeof(float);
	int fused = n_a > 0 && n_b > 0 && ba->name && bb->name &&
				(ba->device_dirty || !ba->host_mirror) && (bb->device_dirty || !bb->host_mirror) &&
				buf_a->offset + bytes_a <= ba->size_bytes &&
				buf_b->offset + bytes_b <= bb->size_bytes;
	if (fused && gl_read_two_regions_f32(p, ba, buf_a->offset, dst_a, bytes_a, bb, buf_b->offset,
										 dst_b, bytes_b) == OK) {
		if (ba->host_mirror)
			memcpy((char *)ba->host_mirror + buf_a->offset, dst_a, bytes_a);
		if (bb->host_mirror)
			memcpy((char *)bb->host_mirror + buf_b->offset, dst_b, bytes_b);
		return OK;
	}
	status_code st = gl_buffer_read_f32(self, buf_a, dst_a, n_a);
	if (st == OK)
		st = gl_buffer_read_f32(self, buf_b, dst_b, n_b);
	return st;
}

static status_code gl_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *b	  = as_glbuf(buf);
	size_t	bytes = (size_t)n * sizeof(float);
	if (n <= 0)
		return OK;
	if ((b->device_dirty || !b->host_mirror) && b->name && buf->offset + bytes <= b->size_bytes) {
		if (gl_read_region_f32(p, b, buf->offset, host_dst, bytes) == OK) {
			if (b->host_mirror)
				memcpy((char *)b->host_mirror + buf->offset, host_dst, bytes);
			return OK;
		}
	}
	if (!b->host_mirror) {
		return gl_read_region_f32(p, b, buf->offset, host_dst, bytes);
	}
	gl_buf_sync_to_host(b);
	memcpy(host_dst, (char *)b->host_mirror + buf->offset, bytes);
	return OK;
}

static status_code gl_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *b	  = as_glbuf(buf);
	size_t	bytes = (size_t)n * sizeof(float);
	if (!b->host_mirror) {
		if (gl_ensure_context(b->owner) != OK)
			return ERR_INTERNAL;
		gl_barrier_if_pending(p);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)buf->offset, (GLsizeiptr)bytes,
						host_src);
		GLenum werr = glGetError();
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		if (werr != GL_NO_ERROR)
			return ERR_INTERNAL;
		gl_xq_cache_invalidate(p, b);
		return OK;
	}
	gl_buf_sync_to_host(b);
	memcpy((char *)b->host_mirror + buf->offset, host_src, bytes);
	b->host_dirty	= 1;
	b->device_dirty = 0;
	return OK;
}

static status_code gl_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *sb	  = as_glbuf(src);
	gl_buf *db	  = as_glbuf(dst);
	size_t	bytes = (size_t)n * sizeof(float);
	if (n <= 0 || bytes == 0)
		return OK;
	if (((src->offset | dst->offset | bytes) & 3u) != 0) {
		if (!sb->host_mirror || !db->host_mirror) {
			ERROR("gl: copy_buffer unaligned fallback needs host mirrors");
			return ERR_UNSUPPORTED;
		}
		gl_buf_sync_to_host(sb);
		gl_buf_sync_to_host(db);
		memcpy((char *)db->host_mirror + dst->offset, (char *)sb->host_mirror + src->offset, bytes);
		db->host_dirty	 = 1;
		db->device_dirty = 0;
		return OK;
	}
	gl_buf_sync_to_device(sb);
	gl_barrier_if_pending(p);
	glBindBuffer(GL_COPY_READ_BUFFER, sb->name);
	glBindBuffer(GL_COPY_WRITE_BUFFER, db->name);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr)src->offset,
						(GLintptr)dst->offset, (GLsizeiptr)bytes);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: copy_buffer device copy(%zu bytes) failed (glErr=0x%04x)", bytes, (unsigned)err);
		return ERR_INTERNAL;
	}
	db->device_dirty = 1;
	db->host_dirty	 = 0;
	gl_xq_cache_invalidate(p, db);
	return OK;
}

static size_t gl_mem_available(backend *self) {
	gl_priv *p	   = self->priv;
	size_t	 total = p->device_local_total_estimate;
	if (total < p->device_local_allocated)
		return 0;
	return total - p->device_local_allocated;
}

static size_t gl_mem_total(backend *self) {
	gl_priv *p = self->priv;
	return p->device_local_total_estimate;
}

static void gl_release_thread(backend *self) {
	gl_priv *p		  = self->priv;
	gl_thread_current = NULL;
	if (!p || !p->ctx.dpy || eglGetCurrentContext() != p->ctx.ctx)
		return;
	gl_finish(p);
	eglMakeCurrent(p->ctx.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

static void gl_synchronize(backend *self) {
	gl_priv *p = self->priv;
	GL_ENTER_VOID(p);
	gl_finish(p);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR && !p->device_lost) {
		WARN("gl: glFinish returned glErr=0x%04x; marking device as lost", (unsigned)err);
		p->device_lost = 1;
		return;
	}
}

static void gl_set_gpu_profile(backend *self, int enable) {
	gl_priv *p = self->priv;
	GL_ENTER_VOID(p);
	gl_gpuprof *g = &p->gpuprof;
	if (!enable || g->enabled)
		return;
	const char *ext = (const char *)glGetString(GL_EXTENSIONS);
	if (!ext || !strstr(ext, "GL_EXT_disjoint_timer_query")) {
		WARN("gl: GL_EXT_disjoint_timer_query unavailable; per-kernel GPU timing disabled");
		return;
	}
	g->gen	  = (void *)eglGetProcAddress("glGenQueriesEXT");
	g->del	  = (void *)eglGetProcAddress("glDeleteQueriesEXT");
	g->begin  = (void *)eglGetProcAddress("glBeginQueryEXT");
	g->end	  = (void *)eglGetProcAddress("glEndQueryEXT");
	g->result = (void *)eglGetProcAddress("glGetQueryObjectui64vEXT");
	if (!g->gen || !g->del || !g->begin || !g->end || !g->result) {
		WARN("gl: timer query entry points missing; per-kernel GPU timing disabled");
		return;
	}
	g->gen(GL_PROF_RING, g->q);
	g->enabled = 1;
}

static void gl_gpuprof_report(gl_priv *p) {
	gl_gpuprof *g = &p->gpuprof;
	if (!g->enabled)
		return;
	gl_finish(p);
	int		 order[GL_PIPE_COUNT];
	uint64_t total = 0;
	for (int i = 0; i < GL_PIPE_COUNT; i++) {
		order[i] = i;
		total += g->ns[i];
	}
	for (int i = 1; i < GL_PIPE_COUNT; i++) {
		int k = order[i];
		int j = i;
		while (j > 0 && g->ns[order[j - 1]] < g->ns[k]) {
			order[j] = order[j - 1];
			j--;
		}
		order[j] = k;
	}
	INFO("gl: GPU kernel time %.1f ms total (host waited %.1f ms in glFinish)", (double)total / 1e6,
		 (double)p->stats.wait_us / 1000.0);
	for (int i = 0; i < GL_PIPE_COUNT && i < 16 && g->ns[order[i]]; i++) {
		int k = order[i];
		INFO("gl:   %-28s %8.1f ms %5.1f%% %8llu calls %7.1f us/call", p->pipes[k].name,
			 (double)g->ns[k] / 1e6, 100.0 * (double)g->ns[k] / (double)total,
			 (unsigned long long)g->calls[k], (double)g->ns[k] / 1e3 / (double)g->calls[k]);
	}
	g->del(GL_PROF_RING, g->q);
	g->enabled = 0;
}

static void gl_submit(backend *self) {
	gl_priv *p = self->priv;
	GL_ENTER_VOID(p);
	p->stats.flushes++;
	glFlush();
	p->dispatches_since_flush = 0;
}

#define GL_ARGMAX_WG_CHUNK 8192
#define GL_ARGMAX_MAX_GROUPS 256

static status_code gl_argmax_stage1(gl_priv *p, const buffer *logits, int n, int *groups_out) {
	int groups = (n + GL_ARGMAX_WG_CHUNK - 1) / GL_ARGMAX_WG_CHUNK;
	groups	   = MIN(groups, GL_ARGMAX_MAX_GROUPS);
	groups	   = MAX(groups, 1);

	size_t partial_bytes = (size_t)groups * 2 * sizeof(uint32_t);
	if (!p->argmax_partial || p->argmax_partial_bytes < partial_bytes) {
		if (p->argmax_partial)
			gl_buf_free(p, p->argmax_partial);
		p->argmax_partial = gl_buf_alloc(p, partial_bytes, GL_DYNAMIC_COPY, NULL);
		if (!p->argmax_partial)
			return ERR_OUT_OF_MEMORY;
		p->argmax_partial_bytes			= partial_bytes;
		p->argmax_partial->device_dirty = 1;
	}

	gl_buf			  *s1_bufs[2]	  = {as_glbuf(logits), p->argmax_partial};
	int				   s1_uniforms[2] = {n, (n + groups - 1) / groups};
	static const char *s1_unames[2]	  = {"u_n", "u_chunk"};
	status_code s = gl_dispatch_all_dirty(p, &p->pipes[GL_PIPE_ARGMAX_STAGE1], s1_bufs, 2,
										  s1_unames, s1_uniforms, 2, (GLuint)groups, 1, 1, 1);
	*groups_out	  = groups;
	return s;
}

static status_code gl_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	gl_priv *p = self->priv;
	GL_ENTER(p);

	int			groups;
	status_code s = gl_argmax_stage1(p, logits, n, &groups);
	if (s != OK)
		return s;

	gl_buf_sync_to_host(p->argmax_partial);

	const uint32_t *pa		 = (const uint32_t *)p->argmax_partial->host_mirror;
	float			best_val = gl_u32_as_float(0xFF800000u);
	int32_t			best_idx = 0;
	for (int g = 0; g < groups; g++) {
		float v = gl_u32_as_float(pa[2 * g + 0]);
		if (v > best_val) {
			best_val = v;
			best_idx = (int32_t)pa[2 * g + 1];
		}
	}

	p->argmax_partial->device_dirty = 1;

	*out_idx = best_idx;
	return OK;
}

static status_code gl_ensure_token_regs(gl_priv *p) {
	if (p->tok_reg && p->tok_rb)
		return OK;
	p->tok_reg = gl_buf_alloc(p, 64, GL_DYNAMIC_COPY, NULL);
	p->tok_rb  = gl_buf_alloc(p, 64, GL_DYNAMIC_COPY, NULL);
	return p->tok_reg && p->tok_rb ? OK : ERR_OUT_OF_MEMORY;
}

static status_code gl_argmax_to_reg(backend *self, const buffer *logits, int n, int reg) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (reg < 0 || reg >= BACKEND_TOKEN_REGS)
		return ERR_INVALID_ARG;
	status_code s = gl_ensure_token_regs(p);
	if (s != OK)
		return s;

	int groups;
	s = gl_argmax_stage1(p, logits, n, &groups);
	if (s != OK)
		return s;

	gl_pipeline *pipe = &p->pipes[GL_PIPE_ARGMAX_STAGE2];
	gl_use_pipeline(p, pipe);
	gl_bind_raw(p, 0, p->argmax_partial->name, 0, 0, 0);
	gl_bind_raw(p, 1, p->tok_reg->name, 0, 0, 1);
	gl_pu(pipe, "u_groups", groups);
	gl_pu(pipe, "u_slot", reg);
	gl_launch(p, 1, 1, 1);

	gl_barrier_if_pending(p);
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	glBindBuffer(GL_COPY_READ_BUFFER, p->tok_reg->name);
	glBindBuffer(GL_COPY_WRITE_BUFFER, p->tok_rb->name);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr)(reg * 4),
						(GLintptr)(reg * 4), 4);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	if (p->tok_fence[reg])
		glDeleteSync(p->tok_fence[reg]);
	p->tok_fence[reg] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
	p->stats.flushes++;
	glFlush();
	return p->tok_fence[reg] ? OK : ERR_INTERNAL;
}

static status_code gl_read_token_reg(backend *self, int reg, int32_t *out_token) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (reg < 0 || reg >= BACKEND_TOKEN_REGS || !p->tok_fence[reg])
		return ERR_INVALID_ARG;
	uint64_t t0 = gl_now_us();
	GLenum	 wr =
		glClientWaitSync(p->tok_fence[reg], GL_SYNC_FLUSH_COMMANDS_BIT, (GLuint64)500000000);
	while (wr == GL_TIMEOUT_EXPIRED)
		wr = glClientWaitSync(p->tok_fence[reg], 0, (GLuint64)1000000000);
	p->stats.wait_us += gl_now_us() - t0;
	p->stats.readbacks++;
	glDeleteSync(p->tok_fence[reg]);
	p->tok_fence[reg] = NULL;
	if (wr != GL_ALREADY_SIGNALED && wr != GL_CONDITION_SATISFIED)
		return ERR_INTERNAL;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, p->tok_rb->name);
	const uint32_t *src =
		glMapBufferRange(GL_SHADER_STORAGE_BUFFER, (GLintptr)(reg * 4), 4, GL_MAP_READ_BIT);
	if (!src) {
		glGetError();
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		return ERR_INTERNAL;
	}
	*out_token = (int32_t)*src;
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	return OK;
}

static status_code gl_buf_read_range(gl_buf *b, size_t byte_off, size_t len, void *dst) {
	if (!b || byte_off + len > b->size_bytes)
		return ERR_INVALID_ARG;
	if (gl_ensure_context(b->owner) != OK)
		return ERR_INTERNAL;
	gl_finish(b->owner);
	gl_barrier(b->owner);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	void *ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, (GLintptr)byte_off, (GLsizeiptr)len,
								 GL_MAP_READ_BIT);
	if (!ptr)
		return ERR_INTERNAL;
	memcpy(dst, ptr, len);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	return OK;
}

static status_code gl_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
								  int token, int dim, buffer *x_out) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	int tok_slot = -1;
	if (TOKEN_IS_REG(token)) {
		tok_slot = TOKEN_REG_INDEX(token);
		if (tok_slot >= BACKEND_TOKEN_REGS || !p->tok_reg)
			return ERR_INVALID_ARG;
		token = 0;
	}
	gl_pipeline *pipe;
	switch (tok_embd_type) {
	case GGML_TYPE_F32:
		pipe = &p->pipes[GL_PIPE_EMBD_LOOKUP_F32];
		break;
	case GGML_TYPE_F16:
		pipe = &p->pipes[GL_PIPE_EMBD_LOOKUP_F16];
		break;
	case GGML_TYPE_BF16:
		pipe = &p->pipes[GL_PIPE_EMBD_LOOKUP_BF16];
		break;
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ2_S:
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ4_XS:
		pipe = &p->pipes[GL_PIPE_EMBD_LOOKUP_QUANT];
		break;
	case GGML_TYPE_Q4_0_REPACKED:
	case GGML_TYPE_Q8_0_REPACKED:
	case GGML_TYPE_IQ4_NL_REPACKED:
	case GGML_TYPE_IQ3_S_REPACKED:
	case GGML_TYPE_Q4_K_REPACKED:
	case GGML_TYPE_Q5_K_REPACKED:
	case GGML_TYPE_Q6_K_REPACKED:
		return ERR_UNSUPPORTED;
	default: {
		if (tok_slot >= 0)
			return ERR_UNSUPPORTED;
		gl_buf *eb = as_glbuf(tok_embd);
		gl_buf *ob = as_glbuf(x_out);
		if (!eb || !ob)
			return ERR_INVALID_ARG;
		gl_buf_sync_to_host(ob);
		size_t	 row_stride = ggml_row_size(tok_embd_type, (size_t)dim);
		size_t	 row_off	= tok_embd->offset + (size_t)token * row_stride;
		uint8_t *src		= xmalloc(row_stride);
		if (!src)
			return ERR_OUT_OF_MEMORY;
		status_code rr = gl_buf_read_range(eb, row_off, row_stride, src);
		if (rr != OK) {
			free(src);
			return rr;
		}
		float	*dst  = (float *)((uint8_t *)ob->host_mirror + x_out->offset);
		backend *host = backend_host();
		if (!host || !host->dequant_row) {
			free(src);
			return ERR_UNSUPPORTED;
		}
		status_code dq = host->dequant_row(host, tok_embd_type, src, dim, dst);
		free(src);
		if (dq != OK)
			return dq;
		backend_report_host_fallback(self, "embd_lookup", HFB_WEIGHT_TYPE,
									 "embedding type '%s' (type=%u) has no opengl shader; row "
									 "dequantized on host (cpu)",
									 ggml_type_name(tok_embd_type), tok_embd_type);
		ob->host_dirty	 = 1;
		ob->device_dirty = 0;
		return OK;
	}
	}

	size_t	row_stride = ggml_row_size(tok_embd_type, dim);
	gl_buf *bufs[2]	   = {as_glbuf(tok_embd), as_glbuf(x_out)};
	gl_buf_sync_to_device(bufs[0]);
	gl_buf_sync_to_device(bufs[1]);

	if (pipe == &p->pipes[GL_PIPE_EMBD_LOOKUP_QUANT]) {
		int uniforms[5]				 = {token, dim, (int)tok_embd_type, (int)row_stride, tok_slot};
		static const char *unames[5] = {"u_token", "u_dim", "u_type", "u_row_stride", "u_tok_slot"};
		gl_use_pipeline(p, pipe);
		gl_bind_ssbo(0, tok_embd);
		gl_bind_ssbo_out(1, x_out);
		if (tok_slot >= 0)
			gl_bind_raw(p, 5, p->tok_reg->name, 0, 0, 0);
		if ((tok_embd_type == GGML_TYPE_IQ2_S || tok_embd_type == GGML_TYPE_IQ2_XXS ||
			 tok_embd_type == GGML_TYPE_IQ3_XXS || tok_embd_type == GGML_TYPE_IQ4_XS) &&
			p->iqx_grid)
			gl_bind_raw(p, 2, p->iqx_grid->name, 0, 0, 0);
		else
			gl_bind_raw(p, 2, p->iq3s_grid->name, 0, 0, 0);
		gl_pu(pipe, unames[0], uniforms[0]);
		gl_pu(pipe, unames[1], uniforms[1]);
		gl_pu(pipe, unames[2], uniforms[2]);
		gl_pu(pipe, unames[3], uniforms[3]);
		gl_pu(pipe, unames[4], uniforms[4]);
		GLuint groups = (GLuint)((dim + 63) / 64);
		gl_launch(p, groups, 1, 1);
		bufs[1]->device_dirty = 1;
		return OK;
	}

	int				   uniforms[4] = {token, dim, (int)row_stride, tok_slot};
	static const char *unames[4]   = {"u_token", "u_dim", "u_row_stride", "u_tok_slot"};
	GLuint			   groups	   = (GLuint)((dim + 63) / 64);
	if (tok_slot >= 0)
		gl_bind_raw(p, 5, p->tok_reg->name, 0, 0, 0);
	status_code s = gl_dispatch(p, pipe, bufs, 2, unames, uniforms, 4, groups, 1, 1, 1);
	if (s == OK)
		bufs[1]->device_dirty = 1;
	return s;
}

static status_code gl_rmsnorm_quantize(gl_priv *p, const buffer *x, const buffer *w, buffer *y,
									   int n, float eps);

static status_code gl_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							  float eps) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *bufs[3] = {as_glbuf(x), as_glbuf(w), as_glbuf(y)};
	gl_buf_sync_to_device(bufs[0]);
	gl_buf_sync_to_device(bufs[1]);
	gl_buf_sync_to_device(bufs[2]);

	if (n <= 0)
		return OK;

	status_code fs = gl_rmsnorm_quantize(p, x, w, y, n, eps);
	if (fs == OK) {
		bufs[2]->device_dirty = 1;
		return OK;
	}
	if (fs != ERR_UNSUPPORTED)
		return fs;

	gl_use_pipeline(p, &p->pipes[GL_PIPE_RMSNORM]);
	gl_bind_ssbo(0, x);
	gl_bind_ssbo(1, w);
	gl_bind_ssbo_out(2, y);
	gl_pu(&p->pipes[GL_PIPE_RMSNORM], "u_n", n);
	gl_puf(&p->pipes[GL_PIPE_RMSNORM], "u_eps", eps);

	GLuint groups = 1;
	gl_launch(p, groups, 1, 1);
	bufs[2]->device_dirty = 1;
	return OK;
}

static status_code gl_ensure_dummy(gl_priv *p) {
	if (!p->dummy) {
		p->dummy = gl_buf_alloc(p, 16, GL_DYNAMIC_COPY, NULL);
		if (!p->dummy)
			return ERR_OUT_OF_MEMORY;
	}
	return OK;
}

static status_code gl_rmsnorm_ph_impl(gl_priv *p, const buffer *x, const buffer *w, buffer *y,
									  int n_heads, int head_dim, float eps, int has_weight,
									  int rows, const buffer *residual, float out_scale) {
	if (n_heads <= 0 || head_dim <= 0 || rows <= 0)
		return OK;
	if (!p->pipes[GL_PIPE_RMSNORM_PER_HEAD].program)
		return ERR_UNSUPPORTED;
	if (has_weight && !w)
		return ERR_INVALID_ARG;

	gl_buf *xb = as_glbuf(x);
	gl_buf *yb = as_glbuf(y);
	gl_buf *wb = has_weight ? as_glbuf(w) : NULL;
	gl_buf *rb = residual ? as_glbuf(residual) : NULL;

	status_code ds = gl_ensure_dummy(p);
	if (ds != OK)
		return ds;

	gl_buf_sync_to_device(xb);
	gl_buf_sync_to_device(yb);
	if (wb)
		gl_buf_sync_to_device(wb);
	if (rb)
		gl_buf_sync_to_device(rb);

	gl_use_pipeline(p, &p->pipes[GL_PIPE_RMSNORM_PER_HEAD]);
	gl_bind_ssbo(0, x);
	if (wb)
		gl_bind_ssbo(1, w);
	else
		gl_bind_raw(p, 1, p->dummy->name, 0, 0, 0);
	gl_bind_ssbo_out(2, y);
	if (rb)
		gl_bind_ssbo(3, residual);
	else
		gl_bind_raw(p, 3, p->dummy->name, 0, 0, 0);
	gl_pu(&p->pipes[GL_PIPE_RMSNORM_PER_HEAD], "u_n_heads", n_heads);
	gl_pu(&p->pipes[GL_PIPE_RMSNORM_PER_HEAD], "u_head_dim", head_dim);
	gl_puf(&p->pipes[GL_PIPE_RMSNORM_PER_HEAD], "u_eps", eps);
	gl_pu(&p->pipes[GL_PIPE_RMSNORM_PER_HEAD], "u_has_weight", has_weight);
	gl_pu(&p->pipes[GL_PIPE_RMSNORM_PER_HEAD], "u_rows", rows);
	gl_pu(&p->pipes[GL_PIPE_RMSNORM_PER_HEAD], "u_has_residual", rb ? 1 : 0);
	gl_puf(&p->pipes[GL_PIPE_RMSNORM_PER_HEAD], "u_out_scale", out_scale);

	gl_launch(p, (GLuint)n_heads, (GLuint)rows, 1);
	yb->device_dirty = 1;
	return OK;
}

static status_code gl_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
									   int n_heads, int head_dim, float eps) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_rmsnorm_ph_impl(p, x, w, y, n_heads, head_dim, eps, 1, 1, NULL, 1.0f);
}

static status_code gl_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n,
									   float eps) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_rmsnorm_ph_impl(p, x, NULL, y, 1, n, eps, 0, 1, NULL, 1.0f);
}

static status_code gl_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y,
												int n_heads, int head_dim, float eps) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_rmsnorm_ph_impl(p, x, NULL, y, n_heads, head_dim, eps, 0, 1, NULL, 1.0f);
}

static status_code gl_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
								  const buffer *residual, buffer *y, int n, float eps,
								  float out_scale) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_rmsnorm_ph_impl(p, x, w, y, 1, n, eps, 1, 1, residual, out_scale);
}

static status_code gl_rmsnorm_add_batch(backend *self, const buffer *x, const buffer *w,
										const buffer *residual, buffer *y, int n, float eps,
										float out_scale, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 1)
		return gl_rmsnorm_add(self, x, w, residual, y, n, eps, out_scale);
	if (!residual)
		return gl_rmsnorm_ph_impl(p, x, w, y, 1, n, eps, 1, m, NULL, out_scale);
	return gl_rmsnorm_ph_impl(p, x, w, y, 1, n, eps, 1, m, residual, out_scale);
}

static status_code gl_elementwise(gl_priv *p, buffer *x, const buffer *y, const buffer *z, int n,
								  int mode, float scale, int aux, int rows) {
	gl_buf *dummy = NULL;
	if (!p->dummy) {
		p->dummy = gl_buf_alloc(p, 16, GL_DYNAMIC_COPY, NULL);
		if (!p->dummy)
			return ERR_OUT_OF_MEMORY;
	}
	dummy = p->dummy;

	if (!x || !x->handle) {
		ERROR("gl: elementwise: x buffer has NULL handle (mode=%d, n=%d)", mode, n);
		return ERR_INVALID_ARG;
	}
	if (y && !y->handle) {
		ERROR("gl: elementwise: y buffer has NULL handle (mode=%d, n=%d)", mode, n);
		return ERR_INVALID_ARG;
	}
	if (z && !z->handle) {
		ERROR("gl: elementwise: z buffer has NULL handle (mode=%d, n=%d)", mode, n);
		return ERR_INVALID_ARG;
	}

	gl_buf *bufs[3] = {as_glbuf(x), y ? as_glbuf(y) : dummy, z ? as_glbuf(z) : dummy};

	for (int i = 0; i < 3; i++) {
		if (!bufs[i]) {
			ERROR("gl: elementwise: bufs[%d] is NULL (mode=%d, n=%d)", i, mode, n);
			return ERR_INVALID_ARG;
		}
		if (bufs[i]->size_bytes == 0 || bufs[i]->size_bytes > (size_t)(1ull << 40)) {
			ERROR("gl: elementwise: bufs[%d] has invalid size=%zu (handle=%p, owner=%s, "
				  "mode=%d, n=%d) — buffer handle is not a valid gl_buf",
				  i, bufs[i]->size_bytes,
				  (void *)(i == 0	? x
						   : i == 1 ? y
									: z)
					  ->handle,
				  (i == 0	? x
				   : i == 1 ? y
							: z)
						  ->owner
					  ? (i == 0	  ? x
						 : i == 1 ? y
								  : z)
							->owner->name
					  : "NULL",
				  mode, n);
			return ERR_INVALID_ARG;
		}
	}

	for (int i = 0; i < 3; i++)
		gl_buf_sync_to_device(bufs[i]);

	if (n <= 0 || rows <= 0) {
		bufs[0]->device_dirty = 1;
		return OK;
	}

	gl_use_pipeline(p, &p->pipes[GL_PIPE_ELEMENTWISE]);
	gl_bind_ssbo_out(0, x);
	if (y)
		gl_bind_ssbo(1, y);
	else
		gl_bind_raw(p, 1, dummy->name, 0, 0, 0);
	if (z)
		gl_bind_ssbo_out(2, z);
	else
		gl_bind_raw(p, 2, dummy->name, 0, 0, 0);
	gl_pu(&p->pipes[GL_PIPE_ELEMENTWISE], "u_n", n);
	gl_pu(&p->pipes[GL_PIPE_ELEMENTWISE], "u_mode", mode);
	gl_puf(&p->pipes[GL_PIPE_ELEMENTWISE], "u_scale", scale);
	gl_pu(&p->pipes[GL_PIPE_ELEMENTWISE], "u_aux", aux);
	gl_pu(&p->pipes[GL_PIPE_ELEMENTWISE], "u_m", rows);

	GLuint groups = (GLuint)((n + 127) / 128);
	GLuint gy	  = (GLuint)MIN(rows, 65535);
	GLuint gz	  = (GLuint)((rows + 65534) / 65535);
	gl_launch(p, groups, gy, gz);
	bufs[0]->device_dirty = 1;
	if (z)
		bufs[2]->device_dirty = 1;
	return OK;
}

static status_code gl_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	status_code s = gl_elementwise(p, x, y, NULL, n, 1, 0.0f, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: add_inplace: routing to CPU due to bad buffer");
		backend_report_host_fallback(self, "add_inplace", HFB_BUF_HOST_RESIDENT,
									 "buffer is not a device buffer; executed on host (cpu)");
		backend *host = backend_host();
		if (host && host->add_inplace)
			return host->add_inplace(host, x, y, n);
	}
	return s;
}

static status_code gl_scale_inplace(backend *self, buffer *x, float scale, int n) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	status_code s = gl_elementwise(p, x, NULL, NULL, n, 2, scale, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: scale_inplace: routing to CPU due to bad buffer");
		backend_report_host_fallback(self, "scale_inplace", HFB_BUF_HOST_RESIDENT,
									 "buffer is not a device buffer; executed on host (cpu)");
		backend *host = backend_host();
		if (host && host->scale_inplace)
			return host->scale_inplace(host, x, scale, n);
	}
	return s;
}

static status_code gl_scale_add_inplace(backend *self, buffer *x, buffer *y, float scale, int n) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	status_code s = gl_elementwise(p, x, y, NULL, n, 8, scale, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: scale_add_inplace: routing to CPU due to bad buffer");
		backend_report_host_fallback(self, "scale_add_inplace", HFB_BUF_HOST_RESIDENT,
									 "buffer is not a device buffer; executed on host (cpu)");
		backend *host = backend_host();
		if (host && host->add_inplace) {
			if (scale != 1.0f && host->scale_inplace)
				host->scale_inplace(host, y, scale, n);
			return host->add_inplace(host, x, y, n);
		}
	}
	return s;
}

static status_code gl_softcap(backend *self, buffer *x, float cap, int n) {
	if (cap <= 0.0f || n <= 0)
		return OK;
	gl_priv *p = self->priv;
	GL_ENTER(p);
	status_code s = gl_elementwise(p, x, NULL, NULL, n, 4, cap, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: softcap: routing to CPU due to bad buffer");
		backend_report_host_fallback(self, "softcap", HFB_BUF_HOST_RESIDENT,
									 "buffer is not a device buffer; executed on host (cpu)");
		backend *host = backend_host();
		if (host && host->softcap)
			return host->softcap(host, x, cap, n);
	}
	return s;
}

static status_code gl_split_qgate(backend *self, const buffer *mixed, buffer *q, buffer *gate,
								  int n_heads, int head_dim, int n_rows) {
	if (n_rows <= 0 || n_heads <= 0 || head_dim <= 0)
		return OK;
	gl_priv *p = self->priv;
	GL_ENTER(p);
	status_code s =
		gl_elementwise(p, q, mixed, gate, n_heads * head_dim, 5, 0.0f, head_dim, n_rows);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: split_qgate: routing to CPU due to bad buffer");
		backend_report_host_fallback(self, "split_qgate", HFB_BUF_HOST_RESIDENT,
									 "buffer is not a device buffer; executed on host (cpu)");
		backend *host = backend_host();
		if (host && host->split_qgate)
			return host->split_qgate(host, mixed, q, gate, n_heads, head_dim, n_rows);
	}
	return s;
}

static status_code gl_attn_output_gate(backend *self, buffer *out, const buffer *gate, int n,
									   int n_rows) {
	if (n <= 0 || n_rows <= 0)
		return OK;
	gl_priv *p = self->priv;
	GL_ENTER(p);
	status_code s = gl_elementwise(p, out, gate, NULL, n, 6, 0.0f, 0, n_rows);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: attn_output_gate: routing to CPU due to bad buffer");
		backend_report_host_fallback(self, "attn_output_gate", HFB_BUF_HOST_RESIDENT,
									 "buffer is not a device buffer; executed on host (cpu)");
		backend *host = backend_host();
		if (host && host->attn_output_gate)
			return host->attn_output_gate(host, out, gate, n, n_rows);
	}
	return s;
}

static status_code gl_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
								  float combine_scale) {
	if (n <= 0)
		return OK;
	gl_priv *p = self->priv;
	GL_ENTER(p);
	status_code s = gl_elementwise(p, ple, proj, NULL, n, 7, combine_scale, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: ple_combine: routing to CPU due to bad buffer");
		backend_report_host_fallback(self, "ple_combine", HFB_BUF_HOST_RESIDENT,
									 "buffer is not a device buffer; executed on host (cpu)");
		backend *host = backend_host();
		if (host && host->ple_combine)
			return host->ple_combine(host, ple, proj, n, combine_scale);
	}
	return s;
}

static status_code gl_ffn_activate_scaled(backend *self, const buffer *gate, const buffer *up,
										  buffer *out, int n, int activation, float gate_scale,
										  float up_scale) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *bufs[3] = {as_glbuf(gate), as_glbuf(up), as_glbuf(out)};
	for (int i = 0; i < 3; i++)
		gl_buf_sync_to_device(bufs[i]);

	if (n <= 0)
		return OK;

	gl_use_pipeline(p, &p->pipes[GL_PIPE_FFN_ACTIVATE]);
	gl_bind_ssbo(0, gate);
	gl_bind_ssbo(1, up);
	gl_bind_ssbo_out(2, out);
	gl_pu(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_n", n);
	gl_pu(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_act", activation);
	gl_puf(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_gs", gate_scale);
	gl_puf(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_us", up_scale);
	size_t fgroups = ((size_t)n + 127) / 128;
	GLuint fgx	   = fgroups > GL_MAX_DISPATCH_PER_DIM ? GL_MAX_DISPATCH_PER_DIM : (GLuint)fgroups;
	GLuint fgy	   = (GLuint)((fgroups + fgx - 1) / fgx);
	gl_launch(p, fgx, fgy, 1);
	bufs[2]->device_dirty = 1;
	return OK;
}

static status_code gl_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n) {
	return gl_ffn_activate_scaled(self, gate, up, out, n, 0, 1.0f, 1.0f);
}

static status_code gl_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
									  buffer *out, int n, int activation) {
	return gl_ffn_activate_scaled(self, gate, up, out, n, activation, 1.0f, 1.0f);
}

static status_code gl_rope_apply(backend *self, buffer *vec, int n_heads, int head_dim,
								 const float *cos_row, const float *sin_row) {
	gl_priv *p	  = self->priv;
	int		 half = head_dim / 2;
	if (n_heads <= 0 || half <= 0)
		return OK;

	size_t		half_bytes = (size_t)half * sizeof(float);
	buffer		cos_b = {0}, sin_b = {0};
	status_code s = buffer_ensure_scratch(self, &cos_b, half_bytes);
	if (s != OK)
		return s;
	s = buffer_ensure_scratch(self, &sin_b, half_bytes);
	if (s != OK) {
		self->buffer_free(self, &cos_b);
		return s;
	}
	self->buffer_write_f32(self, &cos_b, cos_row, half);
	self->buffer_write_f32(self, &sin_b, sin_row, half);

	gl_buf *bufs[3] = {as_glbuf(vec), as_glbuf(&cos_b), as_glbuf(&sin_b)};
	for (int i = 0; i < 3; i++)
		gl_buf_sync_to_device(bufs[i]);

	int total = n_heads * half;

	gl_use_pipeline(p, &p->pipes[GL_PIPE_ROPE]);
	gl_bind_ssbo_out(0, vec);
	gl_bind_ssbo(1, &cos_b);
	gl_bind_ssbo(2, &sin_b);
	gl_pu(&p->pipes[GL_PIPE_ROPE], "u_n_heads", n_heads);
	gl_pu(&p->pipes[GL_PIPE_ROPE], "u_head_dim", head_dim);
	gl_pu(&p->pipes[GL_PIPE_ROPE], "u_neox", self->rope_neox);
	gl_pu(&p->pipes[GL_PIPE_ROPE], "u_pos", 0);

	GLuint groups = (GLuint)((total + 127) / 128);
	gl_launch(p, groups, 1, 1);
	bufs[0]->device_dirty = 1;

	self->buffer_free(self, &cos_b);
	self->buffer_free(self, &sin_b);
	return OK;
}

static void gl_rope_cache_release(gl_priv *p, gl_rope_cache *slot) {
	if (slot->cos_buf)
		gl_buf_free(p, slot->cos_buf);
	if (slot->sin_buf)
		gl_buf_free(p, slot->sin_buf);
	free(slot->valid);
	*slot = (gl_rope_cache){0};
}

static void gl_upload_row(const gl_buf *b, size_t float_offset, const float *src, int count) {
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)(float_offset * sizeof(float)),
					(GLsizeiptr)((size_t)count * sizeof(float)), src);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

static gl_rope_cache *gl_rope_cache_slot(gl_priv *p, const float *cos_base, const float *sin_base,
										 int half, float theta) {
	for (int i = 0; i < GL_ROPE_CACHE_SLOTS; i++) {
		gl_rope_cache *slot = &p->rope_cache[i];
		if (slot->cos_base == cos_base && slot->sin_base == sin_base && slot->half == half &&
			slot->theta == theta)
			return slot;
	}
	gl_rope_cache *slot = &p->rope_cache[p->rope_cache_next++ % GL_ROPE_CACHE_SLOTS];
	gl_rope_cache_release(p, slot);
	slot->cos_base = cos_base;
	slot->sin_base = sin_base;
	slot->half	   = half;
	slot->theta	   = theta;
	return slot;
}

static void gl_rope_freq_rows(const float *freq_factors, int half, float theta, int pos0, int count,
							  float *cos_out, float *sin_out) {
	double theta_scale = pow((double)theta, -1.0 / (double)half);
	for (int i = 0; i < count; i++) {
		double base_freq = 1.0;
		for (int j = 0; j < half; j++) {
			float  fv = freq_factors[j];
			double ff = (fv >= 1e10f || fv == 0.0f) ? 0.0 : (double)fv;
			float  c  = 1.0f;
			float  sn = 0.0f;
			if (ff > 0.0) {
				double angle = base_freq * ff * (double)(pos0 + i);
				c			 = (float)cos(angle);
				sn			 = (float)sin(angle);
			}
			cos_out[(size_t)i * half + j] = c;
			sin_out[(size_t)i * half + j] = sn;
			base_freq *= theta_scale;
		}
	}
}

static gl_rope_cache *gl_rope_cache_range(gl_priv *p, const float *cos_base, const float *sin_base,
										  const float *freq_factors, int half, float theta, int pos,
										  int count) {
	if (pos < 0 || count <= 0)
		return NULL;
	int			   last = pos + count - 1;
	gl_rope_cache *slot = freq_factors ? gl_rope_cache_slot(p, freq_factors, NULL, half, theta)
									   : gl_rope_cache_slot(p, cos_base, sin_base, half, theta);
	if (last >= slot->capacity) {
		int capacity = GL_ROPE_CACHE_MIN_POSITIONS;
		while (capacity <= last)
			capacity *= 2;
		size_t	 bytes	 = (size_t)capacity * (size_t)half * sizeof(float);
		gl_buf	*cos_buf = gl_buf_alloc(p, bytes, GL_DYNAMIC_COPY, NULL);
		gl_buf	*sin_buf = gl_buf_alloc(p, bytes, GL_DYNAMIC_COPY, NULL);
		uint8_t *valid	 = xcalloc((size_t)capacity, 1);
		if (!cos_buf || !sin_buf || !valid) {
			if (cos_buf)
				gl_buf_free(p, cos_buf);
			if (sin_buf)
				gl_buf_free(p, sin_buf);
			free(valid);
			return NULL;
		}
		if (slot->cos_buf)
			gl_buf_free(p, slot->cos_buf);
		if (slot->sin_buf)
			gl_buf_free(p, slot->sin_buf);
		free(slot->valid);
		slot->cos_buf  = cos_buf;
		slot->sin_buf  = sin_buf;
		slot->valid	   = valid;
		slot->capacity = capacity;
	}
	int first_missing = -1;
	int last_missing  = -1;
	for (int i = pos; i <= last; i++) {
		if (slot->valid[i])
			continue;
		if (first_missing < 0)
			first_missing = i;
		last_missing = i;
	}
	if (first_missing >= 0) {
		size_t offset = (size_t)first_missing * (size_t)half;
		int	   floats = (last_missing - first_missing + 1) * half;
		if (freq_factors) {
			float *tbl = xmalloc((size_t)floats * 2 * sizeof(float));
			gl_rope_freq_rows(freq_factors, half, theta, first_missing,
							  last_missing - first_missing + 1, tbl, tbl + floats);
			gl_upload_row(slot->cos_buf, offset, tbl, floats);
			gl_upload_row(slot->sin_buf, offset, tbl + floats, floats);
			free(tbl);
		} else {
			gl_upload_row(slot->cos_buf, offset, cos_base + offset, floats);
			gl_upload_row(slot->sin_buf, offset, sin_base + offset, floats);
		}
		memset(slot->valid + first_missing, 1, (size_t)(last_missing - first_missing + 1));
	}
	return slot;
}

static gl_rope_cache *gl_rope_cache_row(gl_priv *p, const float *cos_base, const float *sin_base,
										int half, float theta, int pos) {
	return gl_rope_cache_range(p, cos_base, sin_base, NULL, half, theta, pos, 1);
}

static status_code gl_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						   const float *rope_cos_base, const float *rope_sin_base) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	int half = head_dim / 2;
	if (n_heads <= 0 || half <= 0)
		return OK;
	gl_rope_cache *slot =
		gl_rope_cache_row(p, rope_cos_base, rope_sin_base, half, self->rope_theta, pos);
	if (!slot || !p->pipes[GL_PIPE_ROPE].program)
		return gl_rope_apply(self, vec, n_heads, head_dim, rope_cos_base + (size_t)pos * half,
							 rope_sin_base + (size_t)pos * half);

	gl_buf *vgb = as_glbuf(vec);
	gl_buf_sync_to_device(vgb);
	gl_pipeline *pipe = &p->pipes[GL_PIPE_ROPE];
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo_out(0, vec);
	gl_bind_raw(p, 1, slot->cos_buf->name, 0, 0, 0);
	gl_bind_raw(p, 2, slot->sin_buf->name, 0, 0, 0);
	gl_pu(pipe, "u_n_heads", n_heads);
	gl_pu(pipe, "u_head_dim", head_dim);
	gl_pu(pipe, "u_neox", self->rope_neox);
	gl_pu(pipe, "u_pos", pos);
	gl_launch(p, (GLuint)((n_heads * half + 127) / 128), 1, 1);
	vgb->device_dirty = 1;
	return OK;
}

static status_code gl_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
							  int head_dim, int pos, const float *rope_cos_base,
							  const float *rope_sin_base) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	int half = head_dim / 2;
	if (n_heads <= 0 || n_kv_heads <= 0 || half <= 0 || !p->pipes[GL_PIPE_ROPE_QK].program)
		goto separate;
	{
		gl_rope_cache *slot =
			gl_rope_cache_row(p, rope_cos_base, rope_sin_base, half, self->rope_theta, pos);
		if (!slot)
			goto separate;
		gl_buf *qgb = as_glbuf(q);
		gl_buf *kgb = as_glbuf(k);
		gl_buf_sync_to_device(qgb);
		gl_buf_sync_to_device(kgb);
		gl_pipeline *pipe = &p->pipes[GL_PIPE_ROPE_QK];
		gl_use_pipeline(p, pipe);
		gl_bind_ssbo_out(0, q);
		gl_bind_ssbo_out(1, k);
		gl_bind_raw(p, 2, slot->cos_buf->name, 0, 0, 0);
		gl_bind_raw(p, 3, slot->sin_buf->name, 0, 0, 0);
		gl_pu(pipe, "u_n_heads", n_heads);
		gl_pu(pipe, "u_n_kv_heads", n_kv_heads);
		gl_pu(pipe, "u_head_dim", head_dim);
		gl_pu(pipe, "u_pos", pos);
		gl_pu(pipe, "u_neox", self->rope_neox);
		gl_launch(p, (GLuint)(((n_heads + n_kv_heads) * half + 127) / 128), 1, 1);
		qgb->device_dirty = 1;
		kgb->device_dirty = 1;
		return OK;
	}
separate:
	{
		status_code s = gl_rope(self, q, n_heads, head_dim, pos, rope_cos_base, rope_sin_base);
		if (s != OK)
			return s;
		return gl_rope(self, k, n_kv_heads, head_dim, pos, rope_cos_base, rope_sin_base);
	}
}

static status_code gl_rope_batch_dispatch(backend *self, buffer *vec, int n_heads, int head_dim,
										  int pos_start, const float *rope_cos_base,
										  const float *rope_sin_base, const float *freq_factors,
										  int m);

static status_code gl_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
							   const float *rope_cos_base, const float *rope_sin_base,
							   const float *freq_factors) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (!freq_factors)
		return gl_rope(self, vec, n_heads, head_dim, pos, rope_cos_base, rope_sin_base);
	return gl_rope_batch_dispatch(self, vec, n_heads, head_dim, pos, NULL, NULL, freq_factors, 1);
}

static status_code gl_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
									  int n_kv_heads, int head_dim, int rope_dim, int pos_start,
									  const float *rope_cos_base, const float *rope_sin_base,
									  int n_rows) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (n_rows <= 0 || rope_dim <= 0)
		return OK;

	int half = rope_dim / 2;
	if (half <= 0 || n_heads <= 0 || n_kv_heads <= 0)
		return OK;
	if (!p->pipes[GL_PIPE_PARTIAL_ROPE_QK].program)
		return ERR_UNSUPPORTED;

	size_t		tbl_floats = (size_t)n_rows * (size_t)half;
	buffer		cos_b = {0}, sin_b = {0};
	status_code s = buffer_ensure_scratch(self, &cos_b, tbl_floats * sizeof(float));
	if (s != OK)
		return s;
	s = buffer_ensure_scratch(self, &sin_b, tbl_floats * sizeof(float));
	if (s != OK) {
		self->buffer_free(self, &cos_b);
		return s;
	}
	self->buffer_write_f32(self, &cos_b, rope_cos_base + (size_t)pos_start * half, (int)tbl_floats);
	self->buffer_write_f32(self, &sin_b, rope_sin_base + (size_t)pos_start * half, (int)tbl_floats);

	gl_buf *qb = as_glbuf(q);
	gl_buf *kb = as_glbuf(k);
	gl_buf_sync_to_device(qb);
	gl_buf_sync_to_device(kb);
	gl_buf_sync_to_device(as_glbuf(&cos_b));
	gl_buf_sync_to_device(as_glbuf(&sin_b));

	gl_use_pipeline(p, &p->pipes[GL_PIPE_PARTIAL_ROPE_QK]);
	gl_bind_ssbo_out(0, q);
	gl_bind_ssbo_out(1, k);
	gl_bind_ssbo(2, &cos_b);
	gl_bind_ssbo(3, &sin_b);
	gl_pu(&p->pipes[GL_PIPE_PARTIAL_ROPE_QK], "u_n_heads", n_heads);
	gl_pu(&p->pipes[GL_PIPE_PARTIAL_ROPE_QK], "u_n_kv_heads", n_kv_heads);
	gl_pu(&p->pipes[GL_PIPE_PARTIAL_ROPE_QK], "u_head_dim", head_dim);
	gl_pu(&p->pipes[GL_PIPE_PARTIAL_ROPE_QK], "u_rope_dim", rope_dim);
	gl_pu(&p->pipes[GL_PIPE_PARTIAL_ROPE_QK], "u_pos0", pos_start);
	gl_pu(&p->pipes[GL_PIPE_PARTIAL_ROPE_QK], "u_n_rows", n_rows);

	int	   total_heads = n_heads + n_kv_heads;
	GLuint gx		   = (GLuint)((total_heads + 63) / 64);
	GLuint gy		   = (GLuint)MIN(n_rows, 65535);
	GLuint gz		   = (GLuint)((n_rows + 65534) / 65535);
	gl_launch(p, gx, gy, gz);
	qb->device_dirty = 1;
	kb->device_dirty = 1;

	self->buffer_free(self, &cos_b);
	self->buffer_free(self, &sin_b);
	return OK;
}

static status_code gl_repack_plan(backend *self, uint32_t type, uint64_t d0, uint64_t d1,
								  uint32_t *re_type_out) {
	(void)self;
	const uint64_t rows_group = 8;
	uint32_t	   re_type	  = 0;
	if (d1 % rows_group == 0) {
		if (type == GGML_TYPE_Q4_0 && d0 % 32 == 0)
			re_type = GGML_TYPE_Q4_0_REPACKED;
		else if (type == GGML_TYPE_Q8_0 && d0 % 32 == 0)
			re_type = GGML_TYPE_Q8_0_REPACKED;
		else if (type == GGML_TYPE_IQ4_NL && d0 % 32 == 0)
			re_type = GGML_TYPE_IQ4_NL_REPACKED;
		else if (type == GGML_TYPE_IQ3_S && d0 % 256 == 0)
			re_type = GGML_TYPE_IQ3_S_REPACKED;
		else if (type == GGML_TYPE_Q4_K && d0 % 256 == 0)
			re_type = GGML_TYPE_Q4_K_REPACKED;
		else if (type == GGML_TYPE_Q5_K && d0 % 256 == 0)
			re_type = GGML_TYPE_Q5_K_REPACKED;
		else if (type == GGML_TYPE_Q6_K && d0 % 256 == 0)
			re_type = GGML_TYPE_Q6_K_REPACKED;
	}
	if (!re_type)
		return ERR_UNSUPPORTED;
	*re_type_out = re_type;
	return OK;
}

static status_code gl_repack_weight(backend *self, uint32_t type, const void *src, void *dst,
									int n_rows, int k) {
	uint32_t	re_type = 0;
	status_code ps		= gl_repack_plan(self, type, (uint64_t)k, (uint64_t)n_rows, &re_type);
	if (ps != OK)
		return ps;
	if (!src || !dst || n_rows <= 0 || k <= 0)
		return ERR_INVALID_ARG;
	backend *host = backend_host();
	if (!host || !host->repack_weight)
		return ERR_UNSUPPORTED;
	return host->repack_weight(host, type, src, dst, n_rows, k);
}

static int gl_matmul_type_native(backend *self, uint32_t w_type) {
	const gl_priv *p = self->priv;
	for (int i = 0; i < GL_PIPE_COUNT; i++)
		if (p->pipes[i].w_type == w_type)
			return 1;
	return 0;
}

static int gl_matmul_pipe_for_type(gl_priv *p, uint32_t w_type, gl_pipeline **out_pipe,
								   int *out_block_size) {
	for (int i = 0; i < GL_PIPE_COUNT; i++) {
		if (p->pipes[i].w_type == UINT32_MAX || p->pipes[i].w_type != w_type)
			continue;
		if (!p->pipes[i].program)
			return 0;
		*out_pipe		= &p->pipes[i];
		*out_block_size = p->pipes[i].block_size;
		return 1;
	}
	return 0;
}

static int gl_matmul_buf_ok(const buffer *b) {
	gl_buf *gb = b ? as_glbuf(b) : NULL;
	return gb && gb->size_bytes != 0 && gb->size_bytes <= (size_t)(1ull << 40);
}

static int gl_matmul_f32_align_ok(const buffer *x, const buffer *y, const buffer *residual, int k) {
	if ((k & 3) != 0 || (x->offset & 15u) != 0 || (y->offset & 3u) != 0)
		return 0;
	if (residual && (residual->offset & 3u) != 0)
		return 0;
	return 1;
}

#define GL_XQ_BLOCK_WORDS 44u

static int gl_pipe_uses_xq(const gl_priv *p, const gl_pipeline *pipe) {
	(void)p;
	return (pipe->flags & GL_PIPE_F_USES_XQ) != 0;
}

static int gl_pipe_uses_iq4nl_lut(const gl_priv *p, const gl_pipeline *pipe) {
	(void)p;
	return (pipe->flags & GL_PIPE_F_IQ4NL_LUT) != 0;
}

static int gl_pipe_uses_xqk(const gl_priv *p, const gl_pipeline *pipe) {
	(void)p;
	return (pipe->flags & GL_PIPE_F_USES_XQK) != 0;
}

typedef struct {
	int		 plain_pipe;
	int		 act_pipe;
	int		 block;
	int		 check_block;
	unsigned words_per_block;
	unsigned gx_div;
} gl_quantize_spec;

typedef struct {
	const buffer *w;
	buffer		 *y;
	float		  eps;
} gl_quantize_norm;

static status_code gl_quantize_run(gl_priv *p, const gl_quantize_spec *spec, gl_buf **slot,
								   size_t *row_words, const buffer *x, int k, int m,
								   const buffer *up, int activation, float gs, float us,
								   int row_stride, const gl_quantize_norm *norm) {
	gl_pipeline *pipe =
		&p->pipes[norm ? GL_PIPE_QUANTIZE_X_NORM : (up ? spec->act_pipe : spec->plain_pipe)];
	if (!pipe->program || m > (int)GL_MAX_DISPATCH_PER_DIM)
		return ERR_UNSUPPORTED;
	const buffer *src = norm ? norm->y : x;
	if (spec->check_block && (k <= 0 || (k % spec->block) != 0))
		return ERR_UNSUPPORTED;
	int	   k_blocks	 = k / spec->block;
	size_t align	 = MAX(gl_ssbo_align(p), (size_t)4);
	size_t row_bytes = ALIGN_UP((size_t)k_blocks * spec->words_per_block * sizeof(uint32_t), align);
	size_t need		 = row_bytes * (size_t)m;
	*row_words		 = row_bytes / sizeof(uint32_t);

	int			 fam  = (slot == &p->qx) ? 0 : 1;
	int			 base = fam * 2;
	gl_xq_cache *hit  = NULL;
	for (int j = 0; j < 2 && !norm; j++) {
		gl_xq_cache *c = &p->xq_cache[base + j];
		if (!c->valid || !c->stage || c->stage->size_bytes < need)
			continue;
		gl_buf *upb = up ? as_glbuf(up) : NULL;
		if (c->src == as_glbuf(x) && c->src_offset == x->offset && c->k == k && c->m == m &&
			c->act == (up ? activation : -1) && c->up == upb && c->stride == row_stride &&
			(!up || (c->gs == gs && c->us == us))) {
			hit = c;
			break;
		}
	}
	if (hit) {
		*slot	 = hit->stage;
		hit->gen = ++p->xq_gen;
		return OK;
	}

	gl_xq_cache *ce = NULL;
	for (int j = 0; j < 2; j++) {
		if (!p->xq_cache[base + j].valid) {
			ce = &p->xq_cache[base + j];
			break;
		}
	}
	if (!ce) {
		uint64_t best_gen = UINT64_MAX;
		for (int j = 0; j < 2; j++) {
			gl_xq_cache *c = &p->xq_cache[base + j];
			if (c->gen < best_gen) {
				best_gen = c->gen;
				ce		 = c;
			}
		}
	}
	if (!ce->stage || ce->stage->size_bytes < need) {
		if (ce->stage)
			gl_buf_free(p, ce->stage);
		ce->stage = gl_buf_alloc(p, need, GL_DYNAMIC_COPY, NULL);
		if (!ce->stage)
			return ERR_OUT_OF_MEMORY;
	}
	*slot = ce->stage;

	gl_use_pipeline(p, pipe);
	gl_bind_ssbo(0, x);
	if (norm) {
		gl_bind_ssbo(3, norm->w);
		gl_bind_ssbo_out(4, norm->y);
		gl_puf(pipe, "u_eps", norm->eps);
	}
	if (up) {
		gl_bind_ssbo(2, up);
		gl_pu(pipe, "u_act", activation);
		gl_puf(pipe, "u_gs", gs);
		gl_puf(pipe, "u_us", us);
		gl_pu(pipe, "u_stride", row_stride > 0 ? row_stride : k);
		gl_pu(pipe, "u_up_off", row_stride > 0 ? k : 0);
	}
	gl_bind_raw(p, 1, (*slot)->name, 0, 0, 1);
	gl_pu(pipe, "u_k_blocks", k_blocks);
	gl_pu(pipe, "u_k", k);
	gl_pu(pipe, "u_row_words", (int)*row_words);
	GLuint gx		 = (GLuint)(((unsigned)k_blocks + spec->gx_div - 1u) / spec->gx_div);
	int	   row_chunk = 65536 / (k_blocks > 0 ? k_blocks : 1);
	if (row_chunk < 1)
		row_chunk = 1;
	for (int r0 = 0; r0 < m; r0 += row_chunk) {
		int rows = m - r0;
		if (rows > row_chunk)
			rows = row_chunk;
		gl_pu(pipe, "u_row0", r0);
		gl_launch(p, gx, (GLuint)rows, 1);
	}
	ce->valid	   = 1;
	ce->src		   = as_glbuf(src);
	ce->src_name   = ce->src->name;
	ce->src_offset = src->offset;
	ce->k		   = k;
	ce->m		   = m;
	ce->act		   = up ? activation : -1;
	ce->up		   = up ? as_glbuf(up) : NULL;
	ce->gs		   = gs;
	ce->us		   = us;
	ce->stride	   = row_stride;
	ce->gen		   = ++p->xq_gen;
	return OK;
}

static status_code gl_quantize_x(gl_priv *p, const buffer *x, int k, int m, const buffer *up,
								 int activation, float gs, float us, int row_stride) {
	static const gl_quantize_spec spec = {
		GL_PIPE_QUANTIZE_X, GL_PIPE_QUANTIZE_X_ACT, 32, 0, GL_XQ_BLOCK_WORDS, 64};
	return gl_quantize_run(p, &spec, &p->qx, &p->qx_row_words, x, k, m, up, activation, gs, us,
						   row_stride, NULL);
}

static status_code gl_rmsnorm_quantize(gl_priv *p, const buffer *x, const buffer *w, buffer *y,
									   int n, float eps) {
	static const gl_quantize_spec spec = {
		GL_PIPE_QUANTIZE_X, GL_PIPE_QUANTIZE_X_ACT, 32, 0, GL_XQ_BLOCK_WORDS, 64};
	if (n % spec.block != 0 || n / spec.block > (int)spec.gx_div)
		return ERR_UNSUPPORTED;
	gl_quantize_norm norm = {w, y, eps};
	return gl_quantize_run(p, &spec, &p->qx, &p->qx_row_words, x, n, 1, NULL, 0, 1.0f, 1.0f, 0,
						   &norm);
}

#define GL_XQK_WORDS_PER_BLOCK 212u

static status_code gl_quantize_q8_k(gl_priv *p, const buffer *x, int k, int m, const buffer *up,
									int activation, float gs, float us, int row_stride) {
	static const gl_quantize_spec spec = {
		GL_PIPE_QUANTIZE_Q8_K, GL_PIPE_QUANTIZE_Q8_K_ACT, 256, 1, GL_XQK_WORDS_PER_BLOCK, 1};
	return gl_quantize_run(p, &spec, &p->qxk, &p->qxk_row_words, x, k, m, up, activation, gs, us,
						   row_stride, NULL);
}

static int gl_matmul_rows_per_wg(const gl_priv *p, const gl_pipeline *pipe) {
	(void)p;
	return pipe->rows_per_wg;
}

static gl_pipeline *gl_matmul_batch_pipe(gl_priv *p, gl_pipeline *pipe, int m) {
	static const int mm_pipes[][2] = {
		{GL_PIPE_MATMUL_IQ4_NL, GL_PIPE_MATMUL_MM_IQ4_NL},
		{GL_PIPE_MATMUL_Q4_0, GL_PIPE_MATMUL_MM_Q4_0},
		{GL_PIPE_MATMUL_Q8_0, GL_PIPE_MATMUL_MM_Q8_0},
	};
	if (m < 2 || (pipe->flags & (GL_PIPE_F_TILED | GL_PIPE_F_MM)))
		return pipe;
	for (size_t i = 0; m >= GL_MM_MIN_M && i < ARRAY_LEN(mm_pipes); i++)
		if (pipe == &p->pipes[mm_pipes[i][0]] && p->pipes[mm_pipes[i][1]].program)
			return &p->pipes[mm_pipes[i][1]];
	for (int i = 0; i < GL_PIPE_COUNT; i++) {
		if (!(p->pipes[i].flags & GL_PIPE_F_TILED) || p->pipes[i].src != pipe->src)
			continue;
		const uint32_t variant_mask = GL_PIPE_F_REPACKED | GL_PIPE_F_REPACKED_LANE | GL_PIPE_F_BF16;
		if ((p->pipes[i].flags & variant_mask) != (pipe->flags & variant_mask))
			continue;
		return &p->pipes[i];
	}
	return pipe;
}

static long gl_matmul_chunk_wg(gl_priv *p, const gl_pipeline *pipe, int k_blocks_eff) {
	static long override = -1;
	if (override < 0) {
		const char *env = getenv("KAPPAI_GL_CHUNK_WG");
		long		c	= env ? atol(env) : 0;
		override		= (c >= 1 && c <= (long)GL_MAX_DISPATCH_PER_DIM) ? c : 0;
	}
	if (override > 0)
		return override;
	int	 threads = (pipe->flags & GL_PIPE_F_MATMUL_WG) ? p->matmul_wg : 32;
	long units	 = (long)threads * (long)(k_blocks_eff > 0 ? k_blocks_eff : 1);
	long c		 = (long)3145728 / (units > 0 ? units : 1);
	if (c < 64)
		c = 64;
	if (c > (long)GL_MAX_DISPATCH_PER_DIM)
		c = (long)GL_MAX_DISPATCH_PER_DIM;
	return c;
}

static status_code gl_matmul_launch_chunked(gl_priv *p, gl_pipeline *pipe, long total_wg,
											int k_blocks_eff) {
	if (total_wg <= 0)
		return OK;
	long chunk_max = gl_matmul_chunk_wg(p, pipe, k_blocks_eff);
	long done	   = 0;
	while (done < total_wg) {
		long chunk = total_wg - done;
		if (chunk > chunk_max)
			chunk = chunk_max;
		gl_pu(pipe, "u_wg_base", (int)done);
		gl_launch(p, (GLuint)chunk, 1, 1);
		done += chunk;
	}
	return OK;
}

static status_code gl_matmul_dispatch(gl_priv *p, gl_pipeline *pipe, int block_size,
									  const buffer *w, const buffer *x, buffer *y,
									  const buffer *residual, int n, int k, int m, int xq_ready,
									  const buffer *act_up, int activation) {
	if (n <= 0 || k <= 0 || m <= 0)
		return OK;
	pipe = act_up ? &p->pipes[GL_PIPE_MATMUL_F32_ACT] : gl_matmul_batch_pipe(p, pipe, m);

	gl_buf *bufs[3] = {as_glbuf(w), as_glbuf(x), as_glbuf(y)};
	for (int i = 0; i < 3; i++) {
		if (!gl_matmul_buf_ok(i == 0 ? w : i == 1 ? x : y)) {
			WARN("gl: matmul: buffer %d has invalid gl_buf (size=%zu), "
				 "routing to CPU",
				 i, bufs[i] ? bufs[i]->size_bytes : 0);
			return ERR_UNSUPPORTED;
		}
	}

	int mode = 0;
	if (residual) {
		gl_buf *rb = as_glbuf(residual);
		if (!rb || rb->size_bytes == 0 || rb->size_bytes > (size_t)(1ull << 40))
			return ERR_UNSUPPORTED;
		if (rb == bufs[2] && residual->offset == y->offset)
			mode = 2;
		else if (rb != bufs[2])
			mode = 1;
		else
			return ERR_UNSUPPORTED;
	}

	if (gl_ensure_dummy(p) != OK)
		return ERR_OUT_OF_MEMORY;

	gl_buf_sync_to_device(bufs[0]);
	gl_buf_sync_to_device(bufs[1]);
	gl_buf_sync_to_device(bufs[2]);
	if (mode == 1)
		gl_buf_sync_to_device(as_glbuf(residual));

	const int use_xq = gl_pipe_uses_xq(p, pipe);
	if (use_xq && !xq_ready) {
		status_code qs = gl_quantize_x(p, x, k, m, NULL, 0, 1.0f, 1.0f, 0);
		if (qs != OK)
			return qs;
	}

	const int use_xqk = gl_pipe_uses_xqk(p, pipe);
	if (use_xqk && !xq_ready) {
		status_code qs = gl_quantize_q8_k(p, x, k, m, NULL, 0, 1.0f, 1.0f, 0);
		if (qs != OK)
			return qs == ERR_UNSUPPORTED ? ERR_UNSUPPORTED : qs;
	}

	gl_use_pipeline(p, pipe);
	gl_bind_ssbo(0, w);
	if (use_xq)
		gl_bind_raw(p, 1, p->qx->name, 0, 0, 0);
	else if (use_xqk)
		gl_bind_raw(p, 1, p->qxk->name, 0, 0, 0);
	else
		gl_bind_ssbo(1, x);
	gl_bind_ssbo_out(2, y);
	if (mode == 1)
		gl_bind_ssbo(3, residual);
	else
		gl_bind_raw(p, 3, p->dummy->name, 0, 0, 0);
	if ((pipe->flags & GL_PIPE_F_IQX_GRID) != 0 && p->iqx_grid)
		gl_bind_raw(p, 4, p->iqx_grid->name, 0, 0, 0);
	else if (use_xqk)
		gl_bind_raw(p, 4, p->iq3s_grid->name, 0, 0, 0);
	else if (gl_pipe_uses_iq4nl_lut(p, pipe))
		gl_bind_raw(p, 4, p->iq4nl_pairs->name, 0, 0, 0);
	gl_pu(pipe, "u_n", n);

	if (block_size == 0) {
		gl_pu(pipe, "u_k", k);
	} else {
		int k_blocks = k / block_size;
		gl_pu(pipe, "u_k_blocks", k_blocks);
	}
	if (act_up) {
		gl_buf_sync_to_device(as_glbuf(act_up));
		gl_bind_ssbo(4, act_up);
		gl_pu(pipe, "u_act", activation);
	}
	gl_pu(pipe, "u_m", m);
	if (use_xq)
		gl_pu(pipe, "u_x_stride", (int)(p->qx_row_words / 4));
	else if (use_xqk)
		gl_pu(pipe, "u_x_stride", (int)p->qxk_row_words);
	gl_pu(pipe, "u_add_residual", mode);

	int	 rows_per_wg = gl_matmul_rows_per_wg(p, pipe);
	int	 cols_per_wg = (pipe->flags & GL_PIPE_F_MM)		 ? GL_MM_TILE
					   : (pipe->flags & GL_PIPE_F_TILED) ? GL_MATMUL_COLS
														 : 1;
	long total_wg =
		(long)((n + rows_per_wg - 1) / rows_per_wg) * (long)((m + cols_per_wg - 1) / cols_per_wg);
	int			k_eff	  = block_size > 0 ? k / block_size : (k + 31) / 32;
	status_code ls		  = gl_matmul_launch_chunked(p, pipe, total_wg, k_eff);
	bufs[2]->device_dirty = 1;
	return ls;
}

static status_code gl_host_weight_view(gl_priv *p, const buffer *w, size_t bytes, buffer *view,
									   void **heap) {
	(void)p;
	*heap	   = NULL;
	*view	   = *w;
	gl_buf *gb = as_glbuf(w);
	if (!gb || !w->handle)
		return ERR_INVALID_ARG;
	if (gb->host_mirror) {
		gl_buf_sync_to_host(gb);
		return OK;
	}
	if (bytes == 0 || w->offset + bytes > gb->size_bytes)
		return ERR_INVALID_ARG;
	uint8_t *tmp = xmalloc(bytes);
	if (!tmp)
		return ERR_OUT_OF_MEMORY;
	status_code rr = gl_buf_read_range(gb, w->offset, bytes, tmp);
	if (rr != OK) {
		free(tmp);
		return rr;
	}
	view->handle   = tmp;
	view->host_ptr = tmp;
	*heap		   = tmp;
	return OK;
}

static status_code gl_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							 buffer *y, int n, int k) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (n <= 0 || k <= 0)
		return OK;

	gl_pipeline *pipe		= NULL;
	int			 block_size = 0;
	if (!gl_matmul_pipe_for_type(p, w_type, &pipe, &block_size))
		goto host_fallback;
	if (w_type == GGML_TYPE_F32 && !gl_matmul_f32_align_ok(x, y, NULL, k))
		goto host_fallback;
	if (block_size > 0 && (k % block_size) != 0)
		goto host_fallback;

	{
		status_code s = gl_matmul_dispatch(p, pipe, block_size, w, x, y, NULL, n, k, 1, 0, NULL, 0);
		if (s == ERR_UNSUPPORTED)
			goto host_fallback;
		return s;
	}

host_fallback:
	{
		buffer w_view;
		void  *w_heap = NULL;
		if (gl_host_weight_view(p, w, w->size, &w_view, &w_heap) != OK)
			return ERR_UNSUPPORTED;
		gl_buffer_sync_to_host(x);
		{
			backend_report_host_fallback(
				self, "matmul",
				gl_matmul_type_native(self, w_type) ? HFB_OP_NOT_NATIVE : HFB_WEIGHT_TYPE,
				"weight type '%s' (type=%u)", ggml_type_name(w_type), w_type);
			backend *host = backend_host();
			if (!host || !host->matmul) {
				free(w_heap);
				return ERR_UNSUPPORTED;
			}
			status_code hs = host->matmul(host, &w_view, w_type, x, y, n, k);
			free(w_heap);
			gl_buffer_mark_host_written(y);
			return hs;
		}
	}
}

static status_code gl_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								   buffer *y, int n, int k, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (n <= 0 || k <= 0 || m <= 0)
		return OK;
	if (m == 1)
		return gl_matmul(self, w, w_type, x, y, n, k);

	gl_pipeline *pipe		= NULL;
	int			 block_size = 0;
	if (!gl_matmul_pipe_for_type(p, w_type, &pipe, &block_size))
		goto host_fallback;
	if (block_size > 0 && (k % block_size) != 0)
		goto host_fallback;
	if ((size_t)m * (size_t)k * sizeof(float) > x->size ||
		(size_t)m * (size_t)n * sizeof(float) > y->size)
		goto host_fallback;
	if (w_type == GGML_TYPE_F32 && (k & 3) != 0)
		goto host_fallback;
	{
		size_t align = gl_ssbo_align(p);
		if ((x->offset % align) != 0 || (y->offset % align) != 0 ||
			((size_t)k * sizeof(float)) % align != 0 || ((size_t)n * sizeof(float)) % align != 0)
			goto host_fallback;
	}

	{
		status_code ds =
			gl_matmul_dispatch(p, pipe, block_size, w, x, y, NULL, n, k, m, 0, NULL, 0);
		if (ds == ERR_UNSUPPORTED)
			goto host_fallback;
		return ds;
	}

host_fallback:
	{
		buffer w_view;
		void  *w_heap = NULL;
		if (gl_host_weight_view(p, w, w->size, &w_view, &w_heap) != OK)
			return ERR_UNSUPPORTED;
		gl_buffer_sync_to_host(x);
		gl_buffer_sync_to_host(y);
		{
			backend_report_host_fallback(
				self, "matmul_batch",
				gl_matmul_type_native(self, w_type) ? HFB_OP_NOT_NATIVE : HFB_WEIGHT_TYPE,
				"weight type '%s' (type=%u) m=%d", ggml_type_name(w_type), w_type, m);
			backend *host = backend_host();
			if (!host || !host->matmul_batch) {
				free(w_heap);
				return ERR_UNSUPPORTED;
			}
			status_code hs = host->matmul_batch(host, &w_view, w_type, x, y, n, k, m);
			free(w_heap);
			gl_buffer_mark_host_written(y);
			return hs;
		}
	}
}

static status_code gl_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *x, const buffer *residual, buffer *y, int n,
									  int k) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (n <= 0 || k <= 0)
		return OK;

	status_code	 s			= ERR_UNSUPPORTED;
	gl_pipeline *pipe		= NULL;
	int			 block_size = 0;
	if (gl_matmul_pipe_for_type(p, w_type, &pipe, &block_size) && gl_matmul_buf_ok(w) &&
		gl_matmul_buf_ok(x) && gl_matmul_buf_ok(y) && gl_matmul_buf_ok(residual) &&
		(w_type != GGML_TYPE_F32 || gl_matmul_f32_align_ok(x, y, residual, k)) &&
		(block_size == 0 || (k % block_size) == 0))
		s = gl_matmul_dispatch(p, pipe, block_size, w, x, y, residual, n, k, 1, 0, NULL, 0);

	if (s != ERR_UNSUPPORTED)
		return s;

	buffer w_view;
	void  *w_heap = NULL;
	if (gl_host_weight_view(p, w, w->size, &w_view, &w_heap) != OK)
		return ERR_UNSUPPORTED;
	gl_buffer_sync_to_host(x);
	gl_buffer_sync_to_host(residual);
	backend_report_host_fallback(self, "matmul_residual",
								 gl_matmul_type_native(self, w_type) ? HFB_OP_NOT_NATIVE
																	 : HFB_WEIGHT_TYPE,
								 "weight type '%s' (type=%u)", ggml_type_name(w_type), w_type);
	backend *host = backend_host();
	if (!host || !host->matmul_residual) {
		free(w_heap);
		return ERR_UNSUPPORTED;
	}
	status_code hs = host->matmul_residual(host, &w_view, w_type, x, residual, y, n, k);
	free(w_heap);
	gl_buffer_mark_host_written(y);
	return hs;
}

#define GL_MATMUL_MULTI_MAX 8

static status_code gl_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
								   int n_matmuls) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (n_matmuls < 1)
		return ERR_INVALID_ARG;
	if (n_matmuls == 1)
		return gl_matmul(self, w[0], w_types[0], x, y[0], n_list[0], k);
	if (k <= 0)
		return OK;
	if (n_matmuls > GL_MATMUL_MULTI_MAX)
		goto host_fallback;

	{
		gl_pipeline *pipes[GL_MATMUL_MULTI_MAX];
		int			 block_sizes[GL_MATMUL_MULTI_MAX];

		for (int i = 0; i < n_matmuls; i++) {
			if (n_list[i] <= 0)
				continue;
			if (!gl_matmul_pipe_for_type(p, w_types[i], &pipes[i], &block_sizes[i]))
				goto host_fallback;
			if (!gl_matmul_buf_ok(w[i]) || !gl_matmul_buf_ok(x) || !gl_matmul_buf_ok(y[i]))
				goto host_fallback;
			if (w_types[i] == GGML_TYPE_F32) {
				if (!gl_matmul_f32_align_ok(x, y[i], NULL, k))
					goto host_fallback;
			} else if (block_sizes[i] > 0 && (k % block_sizes[i]) != 0) {
				goto host_fallback;
			}
		}

		if (gl_ensure_dummy(p) != OK)
			return ERR_OUT_OF_MEMORY;

		int xq_ready  = 0;
		int xqk_ready = 0;
		for (int i = 0; i < n_matmuls; i++) {
			if (n_list[i] <= 0)
				continue;
			int use_xqk = gl_pipe_uses_xqk(p, pipes[i]);
			if (use_xqk ? xqk_ready : (xq_ready || !gl_pipe_uses_xq(p, pipes[i])))
				continue;
			gl_buf_sync_to_device(as_glbuf(x));
			status_code qs = use_xqk ? gl_quantize_q8_k(p, x, k, 1, NULL, 0, 1.0f, 1.0f, 0)
									 : gl_quantize_x(p, x, k, 1, NULL, 0, 1.0f, 1.0f, 0);
			if (qs == ERR_UNSUPPORTED)
				goto host_fallback;
			if (qs != OK)
				return qs;
			if (use_xqk)
				xqk_ready = 1;
			else
				xq_ready = 1;
		}

		for (int i = 0; i < n_matmuls; i++) {
			if (n_list[i] <= 0)
				continue;
			status_code s = gl_matmul_dispatch(
				p, pipes[i], block_sizes[i], w[i], x, y[i], NULL, n_list[i], k, 1,
				gl_pipe_uses_xqk(p, pipes[i]) ? xqk_ready : xq_ready, NULL, 0);
			if (s == ERR_UNSUPPORTED)
				goto host_fallback;
			if (s != OK)
				return s;
		}
		return OK;
	}

host_fallback:
	{
		buffer		  w_view[GL_MATMUL_MULTI_MAX];
		const buffer *w_ptrs[GL_MATMUL_MULTI_MAX];
		void		 *w_heap[GL_MATMUL_MULTI_MAX] = {0};
		for (int i = 0; i < n_matmuls; i++) {
			if (gl_host_weight_view(p, w[i], w[i]->size, &w_view[i], &w_heap[i]) != OK) {
				for (int j = 0; j < i; j++)
					free(w_heap[j]);
				return ERR_UNSUPPORTED;
			}
			w_ptrs[i] = &w_view[i];
		}
		gl_buffer_sync_to_host(x);
		{
			backend_report_host_fallback(self, "matmul_multi", HFB_OP_NOT_NATIVE,
										 "n_matmuls=%d k=%d", n_matmuls, k);
			backend *host = backend_host();
			if (!host || !host->matmul_multi) {
				for (int i = 0; i < n_matmuls; i++)
					free(w_heap[i]);
				return ERR_UNSUPPORTED;
			}
			status_code hs = host->matmul_multi(host, w_ptrs, w_types, x, y, n_list, k, n_matmuls);
			for (int i = 0; i < n_matmuls; i++)
				free(w_heap[i]);
			for (int i = 0; i < n_matmuls; i++)
				gl_buffer_mark_host_written(y[i]);
			return hs;
		}
	}
}

static void gl_kv_store_free(gl_priv *p, gl_kv_store *st) {
	if (!st)
		return;
	for (int i = 0; i < st->n_layers; i++)
		gl_buf_free(p, st->layers[i]);
	free(st->layers);
	free((char *)st - offsetof(gl_kv_wrap, st));
}

static gl_kv_store *gl_kv_store_new(gl_priv *p) {
	gl_kv_wrap *w = xcalloc(1, sizeof(gl_kv_wrap));
	w->hdr.owner  = p;
	w->hdr.usage  = GL_KV_STORE_USAGE;
	return &w->st;
}

#define GL_KV_ZERO_CHUNK_BYTES ((size_t)1 << 20)

static void gl_buf_zero(gl_buf *b) {
	static const char zeros[GL_KV_ZERO_CHUNK_BYTES];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	for (size_t off = 0; off < b->store_bytes; off += GL_KV_ZERO_CHUNK_BYTES) {
		size_t n = MIN(GL_KV_ZERO_CHUNK_BYTES, b->store_bytes - off);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)off, (GLsizeiptr)n, zeros);
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

static status_code gl_kv_store_alloc(gl_priv *p, gl_kv_store *st, int n_layers,
									 size_t layer_bytes) {
	st->layers = xcalloc((size_t)n_layers, sizeof(gl_buf *));
	for (int i = 0; i < n_layers; i++) {
		st->layers[i] = gl_buf_alloc_ex(p, layer_bytes, GL_DYNAMIC_COPY, NULL, false, NULL);
		if (!st->layers[i])
			return ERR_OUT_OF_MEMORY;
		st->n_layers = i + 1;
		gl_buf_zero(st->layers[i]);
	}
	return OK;
}

static gl_buf *gl_kv_layer_buf(const buffer *kv, int layer) {
	const gl_kv_store *st = kv->handle;
	if (layer < 0)
		layer = 0;
	if (layer >= st->n_layers)
		layer = st->n_layers - 1;
	return st->layers[layer];
}

static void *gl_kv_layer_read(gl_priv *p, const gl_buf *b) {
	void *copy = xmalloc(b->size_bytes);
	gl_barrier_if_pending(p);
	gl_finish(p);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	void *mapped =
		glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)b->size_bytes, GL_MAP_READ_BIT);
	if (mapped) {
		memcpy(copy, mapped, b->size_bytes);
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	} else {
		ERROR("gl: kv layer readback map(%zu bytes) failed (glErr=0x%04x)", b->size_bytes,
			  (unsigned)glGetError());
		memset(copy, 0, b->size_bytes);
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	return copy;
}

static status_code gl_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	int n_kv_layers = desc->n_kv_layers > 0 ? desc->n_kv_layers : 1;
	int n_kv_heads	= desc->n_kv_heads;
	int head_dim	= desc->head_dim;
	int n_ctx		= desc->n_ctx;

	int	   kv_q8	   = desc->kv_quant == KV_QUANT_Q8_0;
	int	   kv_n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	size_t total_bytes;
	if (kv_q8) {
		p->kv_quant		   = KV_QUANT_Q8_0;
		p->kv_row_bytes	   = (size_t)kv_n_blocks * KV_Q8_0_BLOCK_BYTES;
		total_bytes		   = (size_t)n_kv_layers * n_kv_heads * n_ctx * p->kv_row_bytes;
		p->kv_layer_stride = (size_t)n_kv_heads * n_ctx * p->kv_row_bytes;
		p->kv_kvh_stride   = (size_t)n_ctx * p->kv_row_bytes;
	} else {
		p->kv_quant		   = KV_QUANT_F16;
		p->kv_row_bytes	   = (size_t)head_dim * sizeof(uint16_t);
		total_bytes		   = (size_t)n_kv_layers * n_kv_heads * n_ctx * head_dim * sizeof(uint16_t);
		p->kv_layer_stride = (size_t)n_kv_heads * n_ctx * head_dim;
		p->kv_kvh_stride   = (size_t)n_ctx * head_dim;
	}

	size_t layer_bytes = kv_q8 ? p->kv_layer_stride : p->kv_layer_stride * sizeof(uint16_t);

	gl_kv_store *ks = gl_kv_store_new(p);
	gl_kv_store *vs = gl_kv_store_new(p);
	if (gl_kv_store_alloc(p, ks, n_kv_layers, layer_bytes) != OK ||
		gl_kv_store_alloc(p, vs, n_kv_layers, layer_bytes) != OK) {
		ERROR("gl: kv_alloc: failed to allocate %d per-layer KV buffers of %zu bytes", n_kv_layers,
			  layer_bytes);
		gl_kv_store_free(p, ks);
		gl_kv_store_free(p, vs);
		return ERR_OUT_OF_MEMORY;
	}

	p->kv_head_dim	 = head_dim;
	p->kv_n_kv_heads = n_kv_heads;
	p->kv_n_ctx		 = n_ctx;
	p->kv_n_layers	 = n_kv_layers;

	k_out->handle	= ks;
	k_out->size		= total_bytes;
	k_out->host_ptr = NULL;
	k_out->offset	= 0;
	k_out->owner	= self;

	v_out->handle	= vs;
	v_out->size		= total_bytes;
	v_out->host_ptr = NULL;
	v_out->offset	= 0;
	v_out->owner	= self;
	return OK;
}

static void gl_kv_free(backend *self, buffer *k, buffer *v) {
	gl_priv *p = self->priv;
	GL_ENTER_VOID(p);
	buffer *both[2] = {k, v};
	for (int i = 0; i < 2; i++) {
		if (!both[i] || !both[i]->handle)
			continue;
		gl_kv_store_free(p, both[i]->handle);
		both[i]->handle	  = NULL;
		both[i]->size	  = 0;
		both[i]->host_ptr = NULL;
		both[i]->offset	  = 0;
	}
}

static void gl_kv_row_dequant_q8_0(const uint8_t *row, float *out, int head_dim) {
	int n_blocks = (head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK;
	for (int b = 0; b < n_blocks; b++) {
		const uint8_t *blk = row + (size_t)b * KV_Q8_0_BLOCK_BYTES;
		uint16_t	   d16;
		memcpy(&d16, blk, sizeof(d16));
		float d	   = gl_f16_to_f32(d16);
		int	  base = b * KV_Q8_0_BLOCK;
		int	  n	   = head_dim - base;
		if (n > KV_Q8_0_BLOCK)
			n = KV_Q8_0_BLOCK;
		for (int j = 0; j < n; j++)
			out[base + j] = d * (float)(int8_t)blk[2 + j];
	}
}

static status_code gl_kv_put_impl(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								  const buffer *k_in, const buffer *v_in, int in_row_stride,
								  int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								  int m, int in_row0) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (m <= 0 || n_active <= 0 || head_dim <= 0)
		return OK;
	if (!p->pipes[GL_PIPE_KV_PUT].program)
		return ERR_UNSUPPORTED;

	gl_buf *kin = as_glbuf(k_in);
	gl_buf *vin = as_glbuf(v_in);
	gl_buf *kb	= gl_kv_layer_buf(k, layer);
	gl_buf *vb	= gl_kv_layer_buf(v, layer);

	gl_buf_sync_to_device(kin);
	gl_buf_sync_to_device(vin);

	int kv_q8 = (p->kv_quant == KV_QUANT_Q8_0);

	size_t kvh_stride = p->kv_kvh_stride
							? p->kv_kvh_stride
							: (kv_q8 ? (size_t)n_ctx * p->kv_row_bytes : (size_t)n_ctx * head_dim);
	int	   row_stride_u = kv_q8 ? (int)p->kv_row_bytes : head_dim;

	gl_use_pipeline(p, &p->pipes[GL_PIPE_KV_PUT]);
	gl_bind_ssbo(0, k_in);
	gl_bind_ssbo(1, v_in);
	gl_bind_raw(p, 2, kb->name, 0, 0, 1);
	gl_bind_raw(p, 3, vb->name, 0, 0, 1);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_layer_off", 0);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_pos_start", pos_start);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_kvh_stride", (int)kvh_stride);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_row_stride", row_stride_u);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_head_dim", head_dim);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_kv_q8", kv_q8 ? 1 : 0);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_in_row_stride", in_row_stride);
	gl_pu(&p->pipes[GL_PIPE_KV_PUT], "u_in_row0", in_row0);

	gl_launch(p, (GLuint)m, (GLuint)n_active, 1);
	return OK;
}

static status_code gl_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
							 const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
							 int n_ctx, int n_kv_heads_active) {
	return gl_kv_put_impl(self, k, v, layer, pos, k_in, v_in, 0, n_kv_heads, head_dim, n_ctx,
						  n_kv_heads_active, 1, 0);
}

static status_code gl_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								   const buffer *k_in, const buffer *v_in, int in_row_stride,
								   int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								   int m) {
	return gl_kv_put_impl(self, k, v, layer, pos_start, k_in, v_in, in_row_stride, n_kv_heads,
						  head_dim, n_ctx, n_kv_heads_active, m, 0);
}

static status_code gl_kv_shift(backend *self, buffer *k, buffer *v, int layer,
							   const kv_shift_chunk *chunks, int n_chunks, const float *cos_tbl,
							   const float *sin_tbl, int n_kv_heads, int head_dim, int n_ctx,
							   int n_kv_heads_active) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int half	 = head_dim / 2;
	if (n_chunks <= 0 || n_active <= 0 || half <= 0)
		return OK;
	if (!p->pipes[GL_PIPE_KV_SHIFT_GATHER].program || !p->pipes[GL_PIPE_KV_PUT].program)
		return ERR_UNSUPPORTED;

	int total_rows = 0;
	for (int i = 0; i < n_chunks; i++)
		total_rows += chunks[i].n_rows;
	int	   tmp_stride = n_active * head_dim;
	size_t tmp_bytes  = (size_t)total_rows * (size_t)tmp_stride * sizeof(float);
	size_t tbl_floats = (size_t)n_chunks * (size_t)half;

	buffer		k_tmp = {0}, v_tmp = {0}, cos_b = {0}, sin_b = {0};
	status_code s = buffer_ensure_scratch(self, &k_tmp, tmp_bytes);
	if (s == OK)
		s = buffer_ensure_scratch(self, &v_tmp, tmp_bytes);
	if (s == OK)
		s = buffer_ensure_scratch(self, &cos_b, tbl_floats * sizeof(float));
	if (s == OK)
		s = buffer_ensure_scratch(self, &sin_b, tbl_floats * sizeof(float));
	if (s != OK)
		goto done;
	self->buffer_write_f32(self, &cos_b, cos_tbl, (int)tbl_floats);
	self->buffer_write_f32(self, &sin_b, sin_tbl, (int)tbl_floats);
	gl_buf_sync_to_device(as_glbuf(&cos_b));
	gl_buf_sync_to_device(as_glbuf(&sin_b));

	gl_buf *kb		   = gl_kv_layer_buf(k, layer);
	gl_buf *vb		   = gl_kv_layer_buf(v, layer);
	int		kv_q8	   = (p->kv_quant == KV_QUANT_Q8_0);
	size_t	kvh_stride = p->kv_kvh_stride
							 ? p->kv_kvh_stride
							 : (kv_q8 ? (size_t)n_ctx * p->kv_row_bytes : (size_t)n_ctx * head_dim);
	int		row_stride_u = kv_q8 ? (int)p->kv_row_bytes : head_dim;

	gl_barrier(p);
	int row0 = 0;
	for (int i = 0; i < n_chunks; i++) {
		gl_pipeline *pipe = &p->pipes[GL_PIPE_KV_SHIFT_GATHER];
		gl_use_pipeline(p, pipe);
		gl_bind_raw(p, 0, kb->name, 0, 0, 0);
		gl_bind_raw(p, 1, vb->name, 0, 0, 0);
		gl_bind_ssbo_out(2, &k_tmp);
		gl_bind_ssbo_out(3, &v_tmp);
		gl_bind_ssbo(4, &cos_b);
		gl_bind_ssbo(5, &sin_b);
		gl_pu(pipe, "u_src_pos", chunks[i].src_pos);
		gl_pu(pipe, "u_kvh_stride", (int)kvh_stride);
		gl_pu(pipe, "u_row_stride", row_stride_u);
		gl_pu(pipe, "u_head_dim", head_dim);
		gl_pu(pipe, "u_kv_q8", kv_q8 ? 1 : 0);
		gl_pu(pipe, "u_neox", self->rope_neox);
		gl_pu(pipe, "u_tmp_row0", row0);
		gl_pu(pipe, "u_tmp_row_stride", tmp_stride);
		gl_pu(pipe, "u_tbl_off", i * half);
		gl_launch(p, (GLuint)chunks[i].n_rows, (GLuint)n_active, 1);
		row0 += chunks[i].n_rows;
	}
	as_glbuf(&k_tmp)->device_dirty = 1;
	as_glbuf(&v_tmp)->device_dirty = 1;
	gl_barrier(p);

	row0 = 0;
	for (int i = 0; i < n_chunks && s == OK; i++) {
		s = gl_kv_put_impl(self, k, v, layer, chunks[i].dst_pos, &k_tmp, &v_tmp, tmp_stride,
						   n_kv_heads, head_dim, n_ctx, n_kv_heads_active, chunks[i].n_rows, row0);
		row0 += chunks[i].n_rows;
	}
	gl_barrier(p);

done:
	if (k_tmp.owner)
		self->buffer_free(self, &k_tmp);
	if (v_tmp.owner)
		self->buffer_free(self, &v_tmp);
	if (cos_b.owner)
		self->buffer_free(self, &cos_b);
	if (sin_b.owner)
		self->buffer_free(self, &sin_b);
	return s;
}

static uint32_t gl_mla_wtype_code(uint32_t type) {
	if (type == GGML_TYPE_F32)
		return 0;
	if (type == GGML_TYPE_F16)
		return 1;
	if (type == GGML_TYPE_BF16)
		return 2;
	return UINT32_MAX;
}

static status_code gl_kv_alloc_mla(backend *self, int n_layers, int n_ctx, int kv_lora, int qk_rope,
								   buffer *kv_out) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	size_t		 layer_bytes = (size_t)(kv_lora + qk_rope) * (size_t)n_ctx * sizeof(float);
	gl_kv_store *ks			 = gl_kv_store_new(p);
	if (gl_kv_store_alloc(p, ks, n_layers, layer_bytes) != OK) {
		ERROR("gl: kv_alloc_mla: failed to allocate %d MLA layer buffers of %zu bytes", n_layers,
			  layer_bytes);
		gl_kv_store_free(p, ks);
		return ERR_OUT_OF_MEMORY;
	}
	kv_out->handle	 = ks;
	kv_out->size	 = layer_bytes * (size_t)n_layers;
	kv_out->host_ptr = NULL;
	kv_out->offset	 = 0;
	kv_out->owner	 = self;
	return OK;
}

static status_code gl_kv_put_mla(backend *self, buffer *kv_cache, int layer, int pos,
								 const buffer *kv_a_in, const buffer *kv_a_norm_w, int kv_lora,
								 int qk_rope, int n_ctx, float eps) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	(void)n_ctx;
	if (kv_lora <= 0 && qk_rope <= 0)
		return OK;
	if (!p->pipes[GL_PIPE_MLA_KV_PUT].program)
		return ERR_UNSUPPORTED;
	gl_buf *cb = gl_kv_layer_buf(kv_cache, layer);
	gl_buf_sync_to_device(as_glbuf(kv_a_in));

	bool   norm_device = kv_lora > 0 && kv_a_norm_w->owner == self && kv_a_norm_w->handle;
	buffer norm_staged = {0};
	if (kv_lora > 0) {
		if (norm_device) {
			gl_buf_sync_to_device(as_glbuf(kv_a_norm_w));
		} else {
			const float *nd = kv_a_norm_w->host_ptr ? (const float *)kv_a_norm_w->host_ptr
													: (const float *)kv_a_norm_w->handle;
			if (!nd)
				return ERR_INVALID_ARG;
			status_code s =
				self->buffer_alloc_scratch(self, (size_t)kv_lora * sizeof(float), &norm_staged);
			if (s != OK)
				return s;
			s = self->buffer_write_f32(self, &norm_staged, nd, kv_lora);
			if (s != OK) {
				self->buffer_free(self, &norm_staged);
				return s;
			}
		}
	}

	gl_use_pipeline(p, &p->pipes[GL_PIPE_MLA_KV_PUT]);
	gl_bind_ssbo(0, kv_a_in);
	if (kv_lora > 0)
		gl_bind_ssbo(1, norm_device ? kv_a_norm_w : &norm_staged);
	gl_bind_raw(p, 2, cb->name, 0, 0, 1);
	gl_pu(&p->pipes[GL_PIPE_MLA_KV_PUT], "u_kv_lora", kv_lora);
	gl_pu(&p->pipes[GL_PIPE_MLA_KV_PUT], "u_qk_rope", qk_rope);
	gl_pu(&p->pipes[GL_PIPE_MLA_KV_PUT], "u_pos", pos);
	gl_puf(&p->pipes[GL_PIPE_MLA_KV_PUT], "u_eps", eps);
	gl_launch(p, 1, 1, 1);
	if (!norm_device)
		self->buffer_free(self, &norm_staged);
	return OK;
}

static status_code gl_attention_mla(backend *self, const buffer *q, const buffer *kv_cache,
									const buffer *k_b_w, const buffer *v_b_w, buffer *out,
									int layer, int pos, int n_heads, int qk_head, int qk_rope,
									int qk_nope, int v_head, int kv_lora, int n_ctx,
									const float *rope_cos_base, const float *rope_sin_base,
									uint32_t k_b_type, uint32_t v_b_type, float scale) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	(void)n_ctx;
	if (qk_nope < 0 || kv_lora < 0 || kv_lora > GL_MLA_MAX_KV_LORA || (qk_rope & 1) ||
		qk_head != qk_nope + qk_rope || qk_head > GL_MLA_MAX_QK_HEAD || pos < 0 || n_heads <= 0 ||
		qk_head < 0) {
		return ERR_INVALID_ARG;
	}
	uint32_t kw = gl_mla_wtype_code(k_b_type);
	uint32_t vw = gl_mla_wtype_code(v_b_type);
	if (kw == UINT32_MAX || vw == UINT32_MAX) {
		ERROR("gl: attention_mla: unsupported k_b/v_b weight type %s/%s", ggml_type_name(k_b_type),
			  ggml_type_name(v_b_type));
		return ERR_UNSUPPORTED;
	}
	int			   half_rope = qk_rope / 2;
	gl_rope_cache *rsp		 = NULL;
	if (half_rope > 0 && rope_cos_base && rope_sin_base) {
		rsp = gl_rope_cache_range(p, rope_cos_base, rope_sin_base, NULL, half_rope,
								  self->rope_theta, 0, pos + 1);
		if (!rsp)
			return ERR_OUT_OF_MEMORY;
	}

	gl_buf *kb = gl_kv_layer_buf(kv_cache, layer);
	gl_buf_sync_to_device(as_glbuf(q));
	gl_buf_sync_to_device(as_glbuf(out));
	gl_buffer_sync_to_device(k_b_w);
	gl_buffer_sync_to_device(v_b_w);

	gl_use_pipeline(p, &p->pipes[GL_PIPE_MLA_ATTENTION]);
	if (qk_head > 0)
		gl_bind_ssbo(0, q);
	gl_bind_raw(p, 1, kb->name, 0, 0, 0);
	if (qk_nope > 0 && kv_lora > 0)
		gl_bind_ssbo(2, k_b_w);
	if (kv_lora > 0 && v_head > 0)
		gl_bind_ssbo(3, v_b_w);
	gl_bind_raw(p, 4, rsp ? rsp->cos_buf->name : kb->name, 0, 0, 0);
	gl_bind_raw(p, 5, rsp ? rsp->sin_buf->name : kb->name, 0, 0, 0);
	if (v_head > 0)
		gl_bind_ssbo_out(6, out);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_pos", pos);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_qk_head", qk_head);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_qk_nope", qk_nope);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_qk_rope", qk_rope);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_v_head", v_head);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_kv_lora", kv_lora);
	gl_puf(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_scale", scale);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_kwtype", (int)kw);
	gl_pu(&p->pipes[GL_PIPE_MLA_ATTENTION], "u_vwtype", (int)vw);
	gl_launch(p, (GLuint)n_heads, 1, 1);
	as_glbuf(out)->device_dirty = 1;
	return OK;
}

static status_code gl_attention_host_fallback(backend *self, const buffer *q, const buffer *k_cache,
											  const buffer *v_cache, buffer *out, int layer,
											  int pos, int n_heads, int n_kv_heads, int n_active,
											  int head_dim, int n_ctx, float scale,
											  int attn_start) {
	(void)n_kv_heads;
	gl_priv *p = self->priv;

	gl_buffer_sync_to_host(q);

	int n_pos = pos + 1 - attn_start;
	if (n_pos <= 0)
		return ERR_INVALID_ARG;

	size_t kvh_stride = p->kv_kvh_stride ? p->kv_kvh_stride : (size_t)n_ctx * head_dim;

	void		   *k_host	= gl_kv_layer_read(p, gl_kv_layer_buf(k_cache, layer));
	void		   *v_host	= gl_kv_layer_read(p, gl_kv_layer_buf(v_cache, layer));
	const uint16_t *kd_base = k_host;
	const uint16_t *vd_base = v_host;
	const float	   *qf		= (const float *)((char *)as_glbuf(q)->host_mirror + q->offset);
	float		   *outf	= (float *)((char *)as_glbuf(out)->host_mirror + out->offset);

	int n_groups = (n_heads + n_active - 1) / n_active;
	if (n_groups <= 0) {
		free(k_host);
		free(v_host);
		return ERR_INVALID_ARG;
	}

	if (p->fb_scores_cap < (size_t)n_pos) {
		free(p->fb_scores);
		p->fb_scores = xmalloc((size_t)n_pos * sizeof(float));
		if (!p->fb_scores) {
			p->fb_scores_cap = 0;
			free(k_host);
			free(v_host);
			return ERR_OUT_OF_MEMORY;
		}
		p->fb_scores_cap = (size_t)n_pos;
	}
	float *scores = p->fb_scores;

	if (p->kv_quant == KV_QUANT_Q8_0) {
		const uint8_t *kdb = k_host;
		const uint8_t *vdb = v_host;
		size_t		   row_bytes =
			p->kv_row_bytes
				? p->kv_row_bytes
				: (size_t)((head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK * KV_Q8_0_BLOCK_BYTES);
		if (p->fb_rowbuf_cap < (size_t)head_dim) {
			free(p->fb_rowbuf);
			p->fb_rowbuf = xmalloc((size_t)head_dim * sizeof(float));
			if (!p->fb_rowbuf) {
				p->fb_rowbuf_cap = 0;
				free(k_host);
				free(v_host);
				return ERR_OUT_OF_MEMORY;
			}
			p->fb_rowbuf_cap = (size_t)head_dim;
		}
		float *rowbuf = p->fb_rowbuf;
		for (int h = 0; h < n_heads; h++) {
			int			   kvh	 = h / n_groups;
			const uint8_t *kh	 = kdb + (size_t)kvh * kvh_stride;
			const uint8_t *vh	 = vdb + (size_t)kvh * kvh_stride;
			const float	  *qh	 = qf + (size_t)h * head_dim;
			float		  *out_h = outf + (size_t)h * head_dim;
			for (int t = 0; t < n_pos; t++) {
				gl_kv_row_dequant_q8_0(kh + (size_t)(attn_start + t) * row_bytes, rowbuf, head_dim);
				float dot = 0.0f;
				for (int d = 0; d < head_dim; d++)
					dot += qh[d] * rowbuf[d];
				scores[t] = dot * scale;
			}
			gl_softmax_masked(scores, n_pos);
			for (int d = 0; d < head_dim; d++)
				out_h[d] = 0.0f;
			for (int t = 0; t < n_pos; t++) {
				float w = scores[t];
				gl_kv_row_dequant_q8_0(vh + (size_t)(attn_start + t) * row_bytes, rowbuf, head_dim);
				for (int d = 0; d < head_dim; d++)
					out_h[d] += w * rowbuf[d];
			}
		}
		as_glbuf(out)->host_dirty	= 1;
		as_glbuf(out)->device_dirty = 0;
		free(k_host);
		free(v_host);
		return OK;
	}

	for (int h = 0; h < n_heads; h++) {
		int				kvh	  = h / n_groups;
		const uint16_t *kh	  = kd_base + ((size_t)kvh * kvh_stride);
		const uint16_t *vh	  = vd_base + ((size_t)kvh * kvh_stride);
		const float	   *qh	  = qf + ((size_t)h * head_dim);
		float		   *out_h = outf + ((size_t)h * head_dim);

		for (int t = 0; t < n_pos; t++) {
			const uint16_t *kt	= kh + ((size_t)(attn_start + t) * head_dim);
			float			dot = 0.0f;
			for (int d = 0; d < head_dim; d++)
				dot += qh[d] * gl_f16_to_f32(kt[d]);
			scores[t] = dot * scale;
		}
		gl_softmax_masked(scores, n_pos);

		for (int d = 0; d < head_dim; d++)
			out_h[d] = 0.0f;
		for (int t = 0; t < n_pos; t++) {
			float			w  = scores[t];
			const uint16_t *vt = vh + ((size_t)(attn_start + t) * head_dim);
			for (int d = 0; d < head_dim; d++)
				out_h[d] += w * gl_f16_to_f32(vt[d]);
		}
	}

	as_glbuf(out)->host_dirty	= 1;
	as_glbuf(out)->device_dirty = 0;
	free(k_host);
	free(v_host);
	return OK;
}

static status_code gl_attention_impl(backend *self, const buffer *q, const buffer *k_cache,
									 const buffer *v_cache, buffer *out, int layer, int pos,
									 int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									 float scale, int attn_start) {
	gl_priv *p = self->priv;

	if (head_dim > GL_ATTENTION_MAX_HEAD_DIM) {
		WARN("gl: attention: head_dim=%d exceeds shader MAX_HEAD_DIM=%d; "
			 "computing on host",
			 head_dim, GL_ATTENTION_MAX_HEAD_DIM);
		return gl_attention_host_fallback(self, q, k_cache, v_cache, out, layer, pos, n_heads,
										  n_kv_heads, n_kv_heads, head_dim, n_ctx, scale,
										  attn_start);
	}

	int kv_q8		 = (p->kv_quant == KV_QUANT_Q8_0);
	int row_stride_u = kv_q8 ? (int)p->kv_row_bytes : head_dim;

	const int n_pos	   = MAX(pos + 1 - attn_start, 1);
	int		  n_splits = (GL_ATTENTION_TARGET_WORKGROUPS + n_heads - 1) / n_heads;
	n_splits		   = MIN(n_splits, GL_ATTENTION_MAX_SPLITS);
	n_splits		= MIN(n_splits, (n_pos + GL_ATTENTION_MIN_CHUNK - 1) / GL_ATTENTION_MIN_CHUNK);
	n_splits		= MAX(n_splits, 1);
	const int chunk = (n_pos + n_splits - 1) / n_splits;
	n_splits		= (n_pos + chunk - 1) / chunk;

	size_t part_bytes = (size_t)n_heads * n_splits * (size_t)(head_dim + 2) * sizeof(float);
	if (!p->attn_part || p->attn_part_cap < part_bytes) {
		if (p->attn_part)
			gl_buf_free(p, p->attn_part);
		p->attn_part = gl_buf_alloc(p, part_bytes, GL_DYNAMIC_COPY, NULL);
		if (!p->attn_part)
			return ERR_OUT_OF_MEMORY;
		p->attn_part_cap = part_bytes;
	}

	gl_buf *qbuf = as_glbuf(q);
	gl_buf *kbuf = gl_kv_layer_buf(k_cache, layer);
	gl_buf *vbuf = gl_kv_layer_buf(v_cache, layer);
	gl_buf *obuf = as_glbuf(out);

	gl_buf_sync_to_device(qbuf);
	gl_buf_sync_to_device(obuf);

	gl_pipeline *attn = &p->pipes[GL_PIPE_ATTENTION];
	gl_use_pipeline(p, attn);
	gl_bind_ssbo(0, q);
	gl_bind_raw(p, 1, kbuf->name, 0, 0, 0);
	gl_bind_raw(p, 2, vbuf->name, 0, 0, 0);
	gl_bind_raw(p, 3, p->attn_part->name, 0, 0, 1);
	gl_bind_ssbo_out(4, out);

	gl_pu(attn, "u_layer_off", 0);
	gl_pu(attn, "u_pos", pos);
	gl_pu(attn, "u_n_heads", n_heads);
	gl_pu(attn, "u_n_kv_heads", n_kv_heads);
	gl_pu(attn, "u_head_dim", head_dim);
	gl_pu(attn, "u_n_ctx", n_ctx);
	gl_puf(attn, "u_scale", scale);
	gl_pu(attn, "u_stride_head_dim", row_stride_u);
	gl_pu(attn, "u_attn_start", attn_start);
	gl_pu(attn, "u_kv_q8", kv_q8 ? 1 : 0);
	gl_pu(attn, "u_kvh_stride", p->kv_kvh_stride ? (int)p->kv_kvh_stride : n_ctx * row_stride_u);
	gl_pu(attn, "u_chunk", chunk);
	gl_pu(attn, "u_n_splits", n_splits);
	gl_launch(p, (GLuint)n_heads, (GLuint)n_splits, 1);

	if (n_splits > 1) {
		gl_pipeline *merge = &p->pipes[GL_PIPE_ATTENTION_MERGE];
		gl_use_pipeline(p, merge);
		gl_bind_raw(p, 3, p->attn_part->name, 0, 0, 0);
		gl_bind_ssbo_out(4, out);
		gl_pu(merge, "u_head_dim", head_dim);
		gl_pu(merge, "u_n_splits", n_splits);
		gl_launch(p, (GLuint)n_heads, 1, 1);
	}
	obuf->device_dirty = 1;
	return OK;
}

static status_code gl_attention_batch_impl(backend *self, const buffer *q, const buffer *k_cache,
										   const buffer *v_cache, buffer *out, int layer,
										   int pos_start, int n_heads, int n_kv_heads, int head_dim,
										   int n_ctx, float scale, int attn_start, int m,
										   int window) {
	gl_priv *p = self->priv;

	if (head_dim > GL_ATTENTION_MAX_HEAD_DIM) {
		WARN("gl: attention_batch: head_dim=%d exceeds shader MAX_HEAD_DIM=%d; "
			 "computing on host",
			 head_dim, GL_ATTENTION_MAX_HEAD_DIM);
		status_code s = OK;
		for (int i = 0; i < m; i++) {
			size_t qrow = (size_t)n_heads * (size_t)head_dim * sizeof(float);
			buffer qv	= buffer_slice(q, (size_t)i * qrow, qrow);
			buffer ov	= buffer_slice(out, (size_t)i * qrow, qrow);
			s = gl_attention_host_fallback(self, &qv, k_cache, v_cache, &ov, layer, pos_start + i,
										   n_heads, n_kv_heads, n_kv_heads, head_dim, n_ctx, scale,
										   attn_start);
			if (s != OK)
				return s;
		}
		return s;
	}

	int kv_q8		 = (p->kv_quant == KV_QUANT_Q8_0);
	int row_stride_u = kv_q8 ? (int)p->kv_row_bytes : head_dim;

	gl_buf *qbuf = as_glbuf(q);
	gl_buf *kbuf = gl_kv_layer_buf(k_cache, layer);
	gl_buf *vbuf = gl_kv_layer_buf(v_cache, layer);
	gl_buf *obuf = as_glbuf(out);

	gl_buf_sync_to_device(qbuf);
	gl_buf_sync_to_device(obuf);

	gl_use_pipeline(p, &p->pipes[GL_PIPE_ATTENTION_BATCH]);
	gl_bind_ssbo(0, q);
	gl_bind_raw(p, 1, kbuf->name, 0, 0, 0);
	gl_bind_raw(p, 2, vbuf->name, 0, 0, 0);
	gl_bind_raw(p, 3, kbuf->name, 0, 0, 0);
	gl_bind_ssbo_out(4, out);

	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_layer_off", 0);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_pos_start", pos_start);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_n_heads", n_heads);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_n_kv_heads", n_kv_heads);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_head_dim", head_dim);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_n_ctx", n_ctx);
	gl_puf(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_scale", scale);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_stride_head_dim", row_stride_u);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_attn_start", attn_start);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_kv_q8", kv_q8 ? 1 : 0);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_m", m);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_window", window);
	gl_pu(&p->pipes[GL_PIPE_ATTENTION_BATCH], "u_kvh_stride",
		  p->kv_kvh_stride ? (int)p->kv_kvh_stride : n_ctx * row_stride_u);

	GLuint groups_y = (GLuint)((m + GL_ATTENTION_Q_CHUNK - 1) / GL_ATTENTION_Q_CHUNK);
	gl_launch(p, (GLuint)n_heads, groups_y, 1);
	obuf->device_dirty = 1;
	return OK;
}

static status_code gl_attention(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
								int n_kv_heads, int head_dim, int n_ctx, int flash_attn,
								float scale, int n_kv_heads_active) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	(void)flash_attn;
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (n_active != n_kv_heads)
		return gl_attention_host_fallback(self, q, k_cache, v_cache, out, layer, pos, n_heads,
										  n_kv_heads, n_active, head_dim, n_ctx, scale, 0);
	return gl_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							 head_dim, n_ctx, scale, 0);
}

static status_code gl_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
									const buffer *v_cache, buffer *out, int layer, int pos,
									int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									int flash_attn, float scale, int sliding_window,
									int n_kv_heads_active) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	(void)flash_attn;
	int n_active   = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int n_pos	   = pos + 1;
	int attn_start = 0;
	if (sliding_window > 0 && n_pos > sliding_window)
		attn_start = n_pos - sliding_window;
	if (n_active != n_kv_heads)
		return gl_attention_host_fallback(self, q, k_cache, v_cache, out, layer, pos, n_heads,
										  n_kv_heads, n_active, head_dim, n_ctx, scale, attn_start);
	return gl_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							 head_dim, n_ctx, scale, attn_start);
}

static status_code gl_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									int n, float eps, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 0 || n <= 0)
		return OK;
	if (m == 1)
		return gl_rmsnorm(self, x, w, y, n, eps);
	return gl_rmsnorm_ph_impl(p, x, w, y, 1, n, eps, 1, m, NULL, 1.0f);
}

static status_code gl_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w,
											 buffer *y, int n_heads, int head_dim, float eps,
											 int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_rmsnorm_ph_impl(p, x, w, y, n_heads, head_dim, eps, 1, m, NULL, 1.0f);
}

static status_code gl_rmsnorm_noweight_batch(backend *self, const buffer *x, buffer *y, int n,
											 float eps, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_rmsnorm_ph_impl(p, x, NULL, y, 1, n, eps, 0, m, NULL, 1.0f);
}

static status_code gl_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x, buffer *y,
													  int n_heads, int head_dim, float eps, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_rmsnorm_ph_impl(p, x, NULL, y, n_heads, head_dim, eps, 0, m, NULL, 1.0f);
}

static status_code gl_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	return gl_elementwise(p, x, y, NULL, n, 1, 0.0f, 0, m);
}

static status_code gl_rope_batch_dispatch(backend *self, buffer *vec, int n_heads, int head_dim,
										  int pos_start, const float *rope_cos_base,
										  const float *rope_sin_base, const float *freq_factors,
										  int m) {
	gl_priv *p	  = self->priv;
	int		 half = head_dim / 2;
	if (n_heads <= 0 || half <= 0 || m <= 0)
		return OK;

	gl_rope_cache *slot = gl_rope_cache_range(p, rope_cos_base, rope_sin_base, freq_factors, half,
											  self->rope_theta, pos_start, m);
	if (!slot)
		return ERR_OUT_OF_MEMORY;

	gl_buf *vb = as_glbuf(vec);
	if (!vb || vb->size_bytes == 0)
		return ERR_INVALID_ARG;
	gl_buf_sync_to_device(vb);

	gl_use_pipeline(p, &p->pipes[GL_PIPE_ROPE_BATCH]);
	gl_bind_ssbo_out(0, vec);
	gl_bind_raw(p, 1, slot->cos_buf->name, 0, 0, 0);
	gl_bind_raw(p, 2, slot->sin_buf->name, 0, 0, 0);
	gl_pu(&p->pipes[GL_PIPE_ROPE_BATCH], "u_n_heads", n_heads);
	gl_pu(&p->pipes[GL_PIPE_ROPE_BATCH], "u_head_dim", head_dim);
	gl_pu(&p->pipes[GL_PIPE_ROPE_BATCH], "u_m", m);
	gl_pu(&p->pipes[GL_PIPE_ROPE_BATCH], "u_neox", self->rope_neox);
	gl_pu(&p->pipes[GL_PIPE_ROPE_BATCH], "u_row_stride", n_heads * head_dim);
	gl_pu(&p->pipes[GL_PIPE_ROPE_BATCH], "u_pos0", pos_start);

	int total = m * n_heads * half;
	gl_launch(p, (GLuint)((total + 127) / 128), 1, 1);
	vb->device_dirty = 1;
	return OK;
}

static status_code gl_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim,
								 int pos_start, const float *rope_cos_base,
								 const float *rope_sin_base, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 0)
		return OK;
	if (m == 1)
		return gl_rope(self, vec, n_heads, head_dim, pos_start, rope_cos_base, rope_sin_base);
	return gl_rope_batch_dispatch(self, vec, n_heads, head_dim, pos_start, rope_cos_base,
								  rope_sin_base, NULL, m);
}

static status_code gl_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads,
									int n_kv_heads, int head_dim, int pos_start,
									const float *rope_cos_base, const float *rope_sin_base, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 0)
		return OK;
	if (m == 1)
		return gl_rope_qk(self, q, k, n_heads, n_kv_heads, head_dim, pos_start, rope_cos_base,
						  rope_sin_base);
	status_code s = gl_rope_batch_dispatch(self, q, n_heads, head_dim, pos_start, rope_cos_base,
										   rope_sin_base, NULL, m);
	if (s != OK)
		return s;
	return gl_rope_batch_dispatch(self, k, n_kv_heads, head_dim, pos_start, rope_cos_base,
								  rope_sin_base, NULL, m);
}

static status_code gl_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim,
									 int pos_start, const float *rope_cos_base,
									 const float *rope_sin_base, const float *freq_factors, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 0)
		return OK;
	if (!freq_factors)
		return gl_rope_batch(self, vec, n_heads, head_dim, pos_start, rope_cos_base, rope_sin_base,
							 m);
	return gl_rope_batch_dispatch(self, vec, n_heads, head_dim, pos_start, NULL, NULL, freq_factors,
								  m);
}

static status_code gl_qk_norm_rope(backend *self, buffer *q, buffer *k, const buffer *q_w,
								   const buffer *k_w, int n_heads, int n_kv_heads, int head_dim,
								   float eps, int pos_start, const float *rope_cos_base,
								   const float *rope_sin_base, const float *freq_factors, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 0)
		return OK;
	if (head_dim <= 0 || (head_dim & 1) || head_dim > GL_QK_NORM_ROPE_MAX_HEAD_DIM)
		return ERR_UNSUPPORTED;
	if (!p->pipes[GL_PIPE_QK_NORM_ROPE].program)
		return ERR_UNSUPPORTED;
	gl_buf *qb = as_glbuf(q);
	gl_buf *kb = n_kv_heads > 0 ? as_glbuf(k) : NULL;
	gl_buf *wq = as_glbuf(q_w);
	gl_buf *wk = n_kv_heads > 0 ? as_glbuf(k_w) : NULL;
	if (!qb || !wq || (n_kv_heads > 0 && (!kb || !wk)))
		return ERR_INVALID_ARG;
	if (gl_ensure_dummy(p) != OK)
		return ERR_OUT_OF_MEMORY;

	int			   half = head_dim / 2;
	gl_rope_cache *slot = gl_rope_cache_range(p, rope_cos_base, rope_sin_base, freq_factors, half,
											  self->rope_theta, pos_start, m);
	if (!slot)
		return ERR_OUT_OF_MEMORY;

	gl_buf_sync_to_device(qb);
	gl_buf_sync_to_device(wq);
	if (kb) {
		gl_buf_sync_to_device(kb);
		gl_buf_sync_to_device(wk);
	}

	gl_pipeline *pipe = &p->pipes[GL_PIPE_QK_NORM_ROPE];
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo_out(0, q);
	if (kb) {
		gl_bind_ssbo_out(1, k);
		gl_bind_ssbo(3, k_w);
	} else {
		gl_bind_raw(p, 1, p->dummy->name, 0, 0, 0);
		gl_bind_raw(p, 3, p->dummy->name, 0, 0, 0);
	}
	gl_bind_ssbo(2, q_w);
	gl_bind_raw(p, 4, slot->cos_buf->name, 0, 0, 0);
	gl_bind_raw(p, 5, slot->sin_buf->name, 0, 0, 0);
	gl_pu(pipe, "u_n_heads", n_heads);
	gl_pu(pipe, "u_n_kv_heads", n_kv_heads);
	gl_pu(pipe, "u_head_dim", head_dim);
	gl_puf(pipe, "u_eps", eps);
	gl_pu(pipe, "u_neox", self->rope_neox);
	gl_pu(pipe, "u_pos0", pos_start);
	gl_launch(p, (GLuint)(n_heads + n_kv_heads), (GLuint)m, 1);
	qb->device_dirty = 1;
	if (kb)
		kb->device_dirty = 1;
	return OK;
}

static status_code gl_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
									  const buffer *v_cache, buffer *out, int layer, int pos_start,
									  int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									  int flash_attn, float scale, int n_kv_heads_active, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	(void)flash_attn;
	if (m <= 0)
		return OK;
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (n_active != n_kv_heads || m == 1) {
		size_t qrow = (size_t)n_heads * (size_t)head_dim * sizeof(float);
		for (int i = 0; i < m; i++) {
			buffer		qv = buffer_slice(q, (size_t)i * qrow, qrow);
			buffer		ov = buffer_slice(out, (size_t)i * qrow, qrow);
			status_code s =
				gl_attention(self, &qv, k_cache, v_cache, &ov, layer, pos_start + i, n_heads,
							 n_kv_heads, head_dim, n_ctx, 0, scale, n_kv_heads_active);
			if (s != OK)
				return s;
		}
		return OK;
	}
	return gl_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
								   n_kv_heads, head_dim, n_ctx, scale, 0, m, 0);
}

static status_code gl_attention_swa_batch(backend *self, const buffer *q, const buffer *k_cache,
										  const buffer *v_cache, buffer *out, int layer,
										  int pos_start, int n_heads, int n_kv_heads, int head_dim,
										  int n_ctx, int flash_attn, float scale,
										  int sliding_window, int n_kv_heads_active, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	(void)flash_attn;
	if (m <= 0)
		return OK;
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (n_active != n_kv_heads || m == 1) {
		size_t qrow = (size_t)n_heads * (size_t)head_dim * sizeof(float);
		for (int i = 0; i < m; i++) {
			buffer		qv = buffer_slice(q, (size_t)i * qrow, qrow);
			buffer		ov = buffer_slice(out, (size_t)i * qrow, qrow);
			status_code s = gl_attention_swa(self, &qv, k_cache, v_cache, &ov, layer, pos_start + i,
											 n_heads, n_kv_heads, head_dim, n_ctx, 0, scale,
											 sliding_window, n_kv_heads_active);
			if (s != OK)
				return s;
		}
		return OK;
	}
	int n_pos	   = pos_start + 1;
	int attn_start = 0;
	if (sliding_window > 0 && n_pos > sliding_window)
		attn_start = n_pos - sliding_window;
	return gl_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
								   n_kv_heads, head_dim, n_ctx, scale, attn_start, m,
								   sliding_window);
}

static status_code gl_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up,
										 buffer *out, int n, int activation, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 0 || n <= 0)
		return OK;
	if (m == 1)
		return gl_ffn_activate_ex(self, gate, up, out, n, activation);
	size_t total = (size_t)n * (size_t)m;
	if (total > (size_t)INT_MAX)
		goto rowwise;
	{
		gl_buf *bufs[3] = {as_glbuf(gate), as_glbuf(up), as_glbuf(out)};
		for (int i = 0; i < 3; i++)
			gl_buf_sync_to_device(bufs[i]);
		gl_use_pipeline(p, &p->pipes[GL_PIPE_FFN_ACTIVATE]);
		gl_bind_ssbo(0, gate);
		gl_bind_ssbo(1, up);
		gl_bind_ssbo_out(2, out);
		gl_pu(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_n", (int)total);
		gl_pu(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_act", activation);
		gl_puf(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_gs", 1.0f);
		gl_puf(&p->pipes[GL_PIPE_FFN_ACTIVATE], "u_us", 1.0f);
		size_t bgroups = (total + 127) / 128;
		GLuint bgx = bgroups > GL_MAX_DISPATCH_PER_DIM ? GL_MAX_DISPATCH_PER_DIM : (GLuint)bgroups;
		GLuint bgy = (GLuint)((bgroups + bgx - 1) / bgx);
		gl_launch(p, bgx, bgy, 1);
		bufs[2]->device_dirty = 1;
		return OK;
	}
rowwise:
	for (int i = 0; i < m; i++) {
		buffer gv =
			buffer_slice(gate, (size_t)i * (size_t)n * sizeof(float), (size_t)n * sizeof(float));
		buffer uv =
			buffer_slice(up, (size_t)i * (size_t)n * sizeof(float), (size_t)n * sizeof(float));
		buffer ov =
			buffer_slice(out, (size_t)i * (size_t)n * sizeof(float), (size_t)n * sizeof(float));
		status_code s = gl_ffn_activate_ex(self, &gv, &uv, &ov, n, activation);
		if (s != OK)
			return s;
	}
	return OK;
}

static status_code gl_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *gate, const buffer *up, buffer *y, int n, int k,
									  int activation) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (n <= 0 || k <= 0)
		return OK;
	if (!gl_matmul_type_native(self, w_type)) {
		gl_buffer_sync_to_host(w);
		gl_buffer_sync_to_host(gate);
		gl_buffer_sync_to_host(up);
		backend_report_host_fallback(self, "matmul_ffn_down", HFB_WEIGHT_TYPE,
									 "weight type '%s' (type=%u)", ggml_type_name(w_type), w_type);
		backend *host = backend_host();
		if (!host || !host->matmul_ffn_down)
			return ERR_UNSUPPORTED;
		status_code hs = host->matmul_ffn_down(host, w, w_type, gate, up, y, n, k, activation);
		gl_buffer_mark_host_written(y);
		return hs;
	}
	{
		gl_pipeline *pipe		= NULL;
		int			 block_size = 0;
		if (gl_matmul_pipe_for_type(p, w_type, &pipe, &block_size) && block_size > 0 &&
			(k % block_size) == 0 && gl_matmul_buf_ok(gate) && gl_matmul_buf_ok(up) &&
			(gl_pipe_uses_xq(p, pipe) || gl_pipe_uses_xqk(p, pipe))) {
			gl_buf_sync_to_device(as_glbuf(gate));
			gl_buf_sync_to_device(as_glbuf(up));
			status_code qs = gl_pipe_uses_xqk(p, pipe)
								 ? gl_quantize_q8_k(p, gate, k, 1, up, activation, 1.0f, 1.0f, 0)
								 : gl_quantize_x(p, gate, k, 1, up, activation, 1.0f, 1.0f, 0);
			if (qs == OK) {
				qs = gl_matmul_dispatch(p, pipe, block_size, w, gate, y, NULL, n, k, 1, 1, NULL, 0);
				if (qs != ERR_UNSUPPORTED)
					return qs;
			} else if (qs != ERR_UNSUPPORTED) {
				return qs;
			}
		}
	}
	if (w_type == GGML_TYPE_F32 && gl_matmul_f32_align_ok(gate, y, NULL, k) &&
		(up->offset & 15u) == 0 && gl_matmul_buf_ok(up)) {
		gl_pipeline *pipe		= NULL;
		int			 block_size = 0;
		if (gl_matmul_pipe_for_type(p, w_type, &pipe, &block_size)) {
			status_code fs = gl_matmul_dispatch(p, pipe, block_size, w, gate, y, NULL, n, k, 1, 0,
												up, activation);
			if (fs != ERR_UNSUPPORTED)
				return fs;
		}
	}
	buffer		act = {0};
	status_code s	= self->buffer_alloc_scratch(self, (size_t)k * sizeof(float), &act);
	if (s != OK)
		return s;
	s = gl_ffn_activate_ex(self, gate, up, &act, k, activation);
	if (s == OK)
		s = gl_matmul(self, w, w_type, &act, y, n, k);
	self->buffer_free(self, &act);
	(void)p;
	return s;
}

static status_code gl_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
											const buffer *gate, const buffer *up, buffer *y, int n,
											int k, int activation, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (n <= 0 || k <= 0 || m <= 0)
		return OK;
	if (m == 1)
		return gl_matmul_ffn_down(self, w, w_type, gate, up, y, n, k, activation);
	if (!gl_matmul_type_native(self, w_type)) {
		gl_buffer_sync_to_host(w);
		gl_buffer_sync_to_host(gate);
		gl_buffer_sync_to_host(up);
		backend_report_host_fallback(self, "matmul_ffn_down_batch", HFB_WEIGHT_TYPE,
									 "weight type '%s' (type=%u) m=%d", ggml_type_name(w_type),
									 w_type, m);
		backend *host = backend_host();
		if (!host || !host->matmul_ffn_down_batch)
			return ERR_UNSUPPORTED;
		status_code hs =
			host->matmul_ffn_down_batch(host, w, w_type, gate, up, y, n, k, activation, m);
		gl_buffer_mark_host_written(y);
		return hs;
	}
	buffer		act = {0};
	status_code s	= self->buffer_alloc_scratch(self, (size_t)m * (size_t)k * sizeof(float), &act);
	if (s != OK)
		return s;
	s = gl_ffn_activate_batch(self, gate, up, &act, k, activation, m);
	if (s == OK)
		s = gl_matmul_batch(self, w, w_type, &act, y, n, k, m);
	self->buffer_free(self, &act);
	(void)p;
	return s;
}

static status_code gl_ffn_activate_fused_batch(backend *self, const buffer *fused, buffer *out,
											   int n, int activation, int m) {
	if (m <= 0 || n <= 0)
		return OK;
	gl_priv *p = self->priv;
	GL_ENTER(p);
	gl_buf *fb = as_glbuf(fused);
	gl_buf *ob = as_glbuf(out);
	if (!fb || !ob || fb->size_bytes == 0 || ob->size_bytes == 0)
		return ERR_INVALID_ARG;
	gl_buf_sync_to_device(fb);
	gl_buf_sync_to_device(ob);

	gl_use_pipeline(p, &p->pipes[GL_PIPE_FFN_ACTIVATE_FUSED_BATCH]);
	gl_bind_ssbo(0, fused);
	gl_bind_ssbo_out(1, out);
	gl_pu(&p->pipes[GL_PIPE_FFN_ACTIVATE_FUSED_BATCH], "u_n", n);
	gl_pu(&p->pipes[GL_PIPE_FFN_ACTIVATE_FUSED_BATCH], "u_m", m);
	gl_pu(&p->pipes[GL_PIPE_FFN_ACTIVATE_FUSED_BATCH], "u_act", activation);

	size_t total  = (size_t)n * (size_t)m;
	size_t groups = (total + 127) / 128;
	GLuint gx	  = groups > GL_MAX_DISPATCH_PER_DIM ? GL_MAX_DISPATCH_PER_DIM : (GLuint)groups;
	GLuint gy	  = (GLuint)((groups + gx - 1) / gx);
	gl_launch(p, gx, gy, 1);
	ob->device_dirty = 1;
	return OK;
}

static status_code gl_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
										 const buffer *x, buffer **y, const int *n_list, int k,
										 int n_matmuls, int m) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (m <= 0)
		return OK;
	if (m == 1)
		return gl_matmul_multi(self, w, w_types, x, y, n_list, k, n_matmuls);
	for (int r = 0; r < m; r++) {
		buffer xv =
			buffer_slice(x, (size_t)r * (size_t)k * sizeof(float), (size_t)k * sizeof(float));
		buffer *yv[GL_MATMUL_MULTI_MAX];
		buffer	yvs[GL_MATMUL_MULTI_MAX];
		if (n_matmuls > GL_MATMUL_MULTI_MAX)
			return ERR_INVALID_ARG;
		for (int i = 0; i < n_matmuls; i++) {
			yvs[i] = buffer_slice(y[i], (size_t)r * (size_t)n_list[i] * sizeof(float),
								  (size_t)n_list[i] * sizeof(float));
			yv[i]  = &yvs[i];
		}
		status_code s = gl_matmul_multi(self, w, w_types, &xv, yv, n_list, k, n_matmuls);
		if (s != OK)
			return s;
	}
	return OK;
}

static status_code gl_moe_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n, float gate_scale, float up_scale, int use_gelu) {
	if (n <= 0)
		return OK;
	int act = use_gelu ? 1 : 0;
	return gl_ffn_activate_scaled(self, gate, up, out, n, act, gate_scale, up_scale);
}

static status_code gl_dequant_row(backend *self, uint32_t type, const void *src, int n_elems,
								  float *dst) {
	(void)self;
	if (!src || !dst || n_elems <= 0)
		return ERR_INVALID_ARG;
	backend_report_host_fallback(self, "dequant_row", HFB_OP_NOT_NATIVE,
								 "dequant_row is host-only by design (type=%u)", type);
	backend *host = backend_host();
	if (!host || !host->dequant_row)
		return ERR_UNSUPPORTED;
	return host->dequant_row(host, type, src, n_elems, dst);
}

static int gl_dequant_type_native(backend *self, uint32_t type, int k) {
	gl_priv *p = self->priv;
	return type == GGML_TYPE_Q8_0 && k > 0 && k % 32 == 0 && p->pipes[GL_PIPE_DEQUANT_Q8_0].program;
}

static status_code gl_dequant_weight(backend *self, uint32_t type, const void *src, int n_rows,
									 int k, buffer *out) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (!gl_dequant_type_native(self, type, k) || n_rows <= 0 || !src)
		return ERR_UNSUPPORTED;
	const size_t src_bytes = ggml_row_size(type, (size_t)k) * (size_t)n_rows;
	const size_t total	   = (size_t)n_rows * (size_t)k;
	const size_t dst_bytes = total * sizeof(float);

	gl_buf *sb = gl_buf_alloc_ex(p, src_bytes, GL_STATIC_DRAW, src, false, NULL);
	if (!sb)
		return ERR_OUT_OF_MEMORY;
	gl_buf *db = gl_buf_alloc_ex(p, dst_bytes, GL_STATIC_DRAW, NULL, false, NULL);
	if (!db) {
		gl_buf_free(p, sb);
		return ERR_OUT_OF_MEMORY;
	}
	buffer sbuf = {.handle = gl_buf_handle(sb), .size = src_bytes, .owner = self};
	buffer dbuf = {.handle = gl_buf_handle(db), .size = dst_bytes, .owner = self};

	const GLuint groups = (GLuint)MIN((total + 63) / 64, (size_t)GL_MAX_DISPATCH_PER_DIM);
	gl_pipeline *pipe	= &p->pipes[GL_PIPE_DEQUANT_Q8_0];
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo(0, &sbuf);
	gl_bind_ssbo_out(1, &dbuf);
	gl_pu(pipe, "u_total", (int)total);
	gl_pu(pipe, "u_stride", (int)(groups * 64u));
	gl_launch(p, groups, 1, 1);
	db->device_dirty = 1;
	gl_buf_free(p, sb);
	*out = dbuf;
	return OK;
}

static status_code gl_moe_expert_rows_batched(backend *self, gl_priv *p, const buffer *xb,
											  buffer *out, const buffer *idx_buf,
											  const buffer *w_buf, int idx_off, int cnt, int dim,
											  int inter, int use_gelu,
											  const moe_resident_expert *e) {
	gl_pipeline *gate_pipe = NULL, *up_pipe = NULL, *down_pipe = NULL;
	int			 gate_block = 0, up_block = 0, down_block = 0;
	if (!gl_matmul_pipe_for_type(p, e->gate_type, &gate_pipe, &gate_block) ||
		(!e->gate_up_fused && !gl_matmul_pipe_for_type(p, e->up_type, &up_pipe, &up_block)) ||
		!gl_matmul_pipe_for_type(p, e->down_type, &down_pipe, &down_block))
		return ERR_UNSUPPORTED;
	if (!gl_pipe_uses_xq(p, down_pipe) && !gl_pipe_uses_xqk(p, down_pipe))
		return ERR_UNSUPPORTED;
	if (gate_block > 0 && (dim % gate_block) != 0)
		return ERR_UNSUPPORTED;
	if (!e->gate_up_fused && up_block > 0 && (dim % up_block) != 0)
		return ERR_UNSUPPORTED;
	if (down_block > 0 && (inter % down_block) != 0)
		return ERR_UNSUPPORTED;
	if ((e->gate_w->owner && e->gate_w->owner != self) ||
		(e->down_w->owner && e->down_w->owner != self) ||
		(e->up_w && e->up_w->owner && e->up_w->owner != self))
		return ERR_UNSUPPORTED;

	buffer		x_pk = {0}, gu = {0}, g = {0}, u = {0}, y = {0};
	status_code s =
		self->buffer_alloc_scratch(self, (size_t)cnt * (size_t)dim * sizeof(float), &x_pk);
	if (s == OK && e->gate_up_fused)
		s = self->buffer_alloc_scratch(self, (size_t)cnt * (size_t)inter * 2 * sizeof(float), &gu);
	else if (s == OK)
		s = self->buffer_alloc_scratch(self, (size_t)cnt * (size_t)inter * sizeof(float), &g);
	if (s == OK && !e->gate_up_fused)
		s = self->buffer_alloc_scratch(self, (size_t)cnt * (size_t)inter * sizeof(float), &u);
	if (s == OK)
		s = self->buffer_alloc_scratch(self, (size_t)cnt * (size_t)dim * sizeof(float), &y);
	if (s != OK)
		goto done;

	{
		gl_pipeline *gp		= &p->pipes[GL_PIPE_MOE_GATHER];
		GLuint		 groups = (GLuint)(((size_t)cnt * (size_t)dim + 127) / 128);
		GLuint		 gx		= groups > 65535u ? 65535u : groups;
		GLuint		 gy		= (groups + 65534u) / 65535u;
		gl_use_pipeline(p, gp);
		gl_bind_ssbo(0, idx_buf);
		gl_bind_ssbo(1, xb);
		gl_bind_ssbo_out(2, &x_pk);
		gl_pu(gp, "u_cnt", cnt);
		gl_pu(gp, "u_dim", dim);
		gl_pu(gp, "u_idx_off", idx_off);
		gl_launch(p, gx, gy, 1);
		as_glbuf(&x_pk)->device_dirty = 1;
	}

	buffer gate_b = {0}, up_b = {0};
	if (e->gate_up_fused) {
		gate_b = gu;
		up_b   = gu;
		s = gl_matmul_dispatch(p, gate_pipe, gate_block, e->gate_w, &x_pk, &gu, NULL, inter * 2,
							   dim, cnt, 0, NULL, 0);
	} else {
		s = gl_matmul_dispatch(p, gate_pipe, gate_block, e->gate_w, &x_pk, &g, NULL, inter, dim,
							   cnt, 0, NULL, 0);
		if (s == OK)
			s = gl_matmul_dispatch(p, up_pipe, up_block, e->up_w, &x_pk, &u, NULL, inter, dim, cnt,
								   0, NULL, 0);
		gate_b = g;
		up_b   = u;
	}
	if (s != OK)
		goto done;

	{
		int act_mode   = use_gelu ? 1 : 0;
		int row_stride = e->gate_up_fused ? inter * 2 : 0;
		s = gl_pipe_uses_xqk(p, down_pipe)
				? gl_quantize_q8_k(p, &gate_b, inter, cnt, &up_b, act_mode, e->gate_scale,
								   e->up_scale, row_stride)
				: gl_quantize_x(p, &gate_b, inter, cnt, &up_b, act_mode, e->gate_scale, e->up_scale,
								row_stride);
		if (s != OK)
			goto done;
	}

	s = gl_matmul_dispatch(p, down_pipe, down_block, e->down_w, &gate_b, &y, NULL, dim, inter, cnt,
						   1, NULL, 0);
	if (s != OK)
		goto done;

	{
		gl_pipeline *cp		= &p->pipes[GL_PIPE_MOE_COMBINE];
		GLuint		 groups = (GLuint)(((size_t)cnt * (size_t)dim + 127) / 128);
		GLuint		 gx		= groups > 65535u ? 65535u : groups;
		GLuint		 gy		= (groups + 65534u) / 65535u;
		gl_use_pipeline(p, cp);
		gl_bind_ssbo(0, idx_buf);
		gl_bind_ssbo(1, w_buf);
		gl_bind_ssbo(2, &y);
		gl_bind_ssbo_out(3, out);
		gl_pu(cp, "u_cnt", cnt);
		gl_pu(cp, "u_dim", dim);
		gl_pu(cp, "u_idx_off", idx_off);
		gl_puf(cp, "u_scale", e->down_scale);
		gl_launch(p, gx, gy, 1);
		as_glbuf(out)->device_dirty = 1;
	}

done:
	self->buffer_free(self, &x_pk);
	self->buffer_free(self, &gu);
	self->buffer_free(self, &g);
	self->buffer_free(self, &u);
	self->buffer_free(self, &y);
	return s;
}

static status_code gl_moe_expert_ffn(backend *self, const buffer *x, buffer *out,
									 const moe_resident_expert *e, int dim, int inter) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (!e || !e->gate_w || !e->down_w || dim <= 0 || inter <= 0)
		return ERR_INVALID_ARG;
	if (e->up_w == NULL && !e->gate_up_fused)
		return ERR_INVALID_ARG;
	if ((e->up_w && !gl_matmul_type_native(self, e->up_type)) ||
		!gl_matmul_type_native(self, e->gate_type) || !gl_matmul_type_native(self, e->down_type)) {
		backend_report_host_fallback(
			self, "moe_expert_ffn", HFB_WEIGHT_TYPE,
			"no native matmul for expert weight types gate=%s up=%s down=%s",
			ggml_type_name(e->gate_type), e->up_w ? ggml_type_name(e->up_type) : "fused",
			ggml_type_name(e->down_type));
		return ERR_UNSUPPORTED;
	}
	if ((e->gate_w->owner && e->gate_w->owner != self) ||
		(e->down_w->owner && e->down_w->owner != self) ||
		(e->up_w && e->up_w->owner && e->up_w->owner != self)) {
		backend_report_host_fallback(self, "moe_expert_ffn", HFB_BUF_HOST_RESIDENT,
									 "expert weights owned by another backend");
		return ERR_UNSUPPORTED;
	}

	gl_pipeline *down_pipe	= NULL;
	int			 down_block = 0;
	float		 ws			= e->weight * e->down_scale;
	int			 fused_tail = gl_matmul_pipe_for_type(p, e->down_type, &down_pipe, &down_block) &&
							  (gl_pipe_uses_xq(p, down_pipe) || gl_pipe_uses_xqk(p, down_pipe)) &&
							  down_block > 0 && (inter % down_block) == 0 && ws >= 1e-2f;

	buffer		gu = {0}, act = {0}, y = {0};
	buffer		gate_buf = {0}, up_buf = {0};
	bool		fused = e->gate_up_fused;
	status_code s	  = OK;
	if (!fused_tail) {
		s = self->buffer_alloc_scratch(self, (size_t)inter * sizeof(float), &act);
		if (s == OK)
			s = self->buffer_alloc_scratch(self, (size_t)dim * sizeof(float), &y);
	}
	if (s == OK && fused) {
		s = self->buffer_alloc_scratch(self, (size_t)inter * 2 * sizeof(float), &gu);
		if (s == OK) {
			gate_buf = buffer_slice(&gu, 0, (size_t)inter * sizeof(float));
			up_buf =
				buffer_slice(&gu, (size_t)inter * sizeof(float), (size_t)inter * sizeof(float));
		}
	} else if (s == OK) {
		s = self->buffer_alloc_scratch(self, (size_t)inter * sizeof(float), &gate_buf);
		if (s == OK)
			s = self->buffer_alloc_scratch(self, (size_t)inter * sizeof(float), &up_buf);
	}
	if (s == OK) {
		if (fused)
			s = self->matmul(self, e->gate_w, e->gate_type, x, &gu, inter * 2, dim);
		else
			s = self->matmul(self, e->gate_w, e->gate_type, x, &gate_buf, inter, dim);
	}
	if (s == OK && !fused)
		s = self->matmul(self, e->up_w, e->up_type, x, &up_buf, inter, dim);
	if (s == OK && fused_tail) {
		gl_buf_sync_to_device(as_glbuf(&gate_buf));
		gl_buf_sync_to_device(as_glbuf(&up_buf));
		int			act_mode = e->use_gelu ? 1 : 0;
		float		uscale	 = e->up_scale * ws;
		status_code qs		 = gl_pipe_uses_xqk(p, down_pipe)
								   ? gl_quantize_q8_k(p, &gate_buf, inter, 1, &up_buf, act_mode,
													  e->gate_scale, uscale, 0)
								   : gl_quantize_x(p, &gate_buf, inter, 1, &up_buf, act_mode,
												   e->gate_scale, uscale, 0);
		if (qs == OK) {
			qs = gl_matmul_dispatch(p, down_pipe, down_block, e->down_w, &gate_buf, out, out, dim,
									inter, 1, 1, NULL, 0);
			if (qs != ERR_UNSUPPORTED) {
				if (qs != OK)
					return qs;
				self->buffer_free(self, &gu);
				if (!fused) {
					self->buffer_free(self, &gate_buf);
					self->buffer_free(self, &up_buf);
				}
				return OK;
			}
			fused_tail = 0;
		} else if (qs != ERR_UNSUPPORTED) {
			return qs;
		} else {
			fused_tail = 0;
		}
	}
	if (s == OK && !fused_tail && !act.handle) {
		s = self->buffer_alloc_scratch(self, (size_t)inter * sizeof(float), &act);
		if (s == OK)
			s = self->buffer_alloc_scratch(self, (size_t)dim * sizeof(float), &y);
	}
	if (s == OK && !fused_tail) {
		s = gl_moe_activate(self, &gate_buf, &up_buf, &act, inter, e->gate_scale, e->up_scale,
							e->use_gelu);
		if (s == OK)
			s = self->matmul(self, e->down_w, e->down_type, &act, &y, dim, inter);
		if (s == OK) {
			if (ws == 1.0f)
				s = gl_add_inplace(self, out, &y, dim);
			else
				s = gl_scale_add_inplace(self, out, &y, ws, dim);
		}
	}
	self->buffer_free(self, &act);
	self->buffer_free(self, &y);
	self->buffer_free(self, &gu);
	if (!fused) {
		self->buffer_free(self, &gate_buf);
		self->buffer_free(self, &up_buf);
	}
	return s;
}

static status_code gl_moe_experts_batch(backend *self, const buffer *xb, buffer *out, int n_rows,
										int dim, int inter, int use_gelu, int n_experts,
										const moe_resident_expert *experts, const int *counts,
										const int *rows_packed, const float *weights_packed) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (!xb || !out || n_rows <= 0 || dim <= 0 || inter <= 0 || n_experts < 0)
		return ERR_INVALID_ARG;
	if (n_experts > 0 && (!counts || !rows_packed || !weights_packed || !experts))
		return ERR_INVALID_ARG;
	int total_packed = 0;
	int max_cnt		 = 0;
	for (int i = 0; i < n_experts; i++) {
		if (counts[i] > max_cnt)
			max_cnt = counts[i];
		total_packed += counts[i];
	}

	buffer idx_buf = {0}, w_buf = {0};
	int	   idx_uploaded = 0;
	if (max_cnt >= 2) {
		status_code us =
			self->buffer_alloc_scratch(self, (size_t)total_packed * sizeof(int), &idx_buf);
		if (us == OK)
			us = self->buffer_alloc_scratch(self, (size_t)total_packed * sizeof(float), &w_buf);
		if (us == OK && idx_buf.handle && w_buf.handle) {
			memcpy(idx_buf.handle, rows_packed, (size_t)total_packed * sizeof(int));
			memcpy(w_buf.handle, weights_packed, (size_t)total_packed * sizeof(float));
			as_glbuf(&idx_buf)->device_dirty = 0;
			as_glbuf(&idx_buf)->host_dirty	 = 1;
			as_glbuf(&w_buf)->device_dirty	 = 0;
			as_glbuf(&w_buf)->host_dirty	 = 1;
			gl_buf_sync_to_device(as_glbuf(&idx_buf));
			gl_buf_sync_to_device(as_glbuf(&w_buf));
			idx_uploaded = 1;
		}
		if (!idx_uploaded) {
			self->buffer_free(self, &idx_buf);
			self->buffer_free(self, &w_buf);
			idx_buf = (buffer){0};
			w_buf	= (buffer){0};
		}
	}

	int			offset = 0;
	status_code st	   = OK;
	for (int i = 0; st == OK && i < n_experts; i++) {
		if (counts[i] <= 0)
			continue;
		const moe_resident_expert *e = &experts[i];
		if (!e->gate_w || !e->down_w) {
			st = ERR_INVALID_ARG;
			break;
		}
		if (!gl_matmul_type_native(self, e->gate_type) ||
			!gl_matmul_type_native(self, e->down_type)) {
			st = ERR_UNSUPPORTED;
			break;
		}
		moe_resident_expert one = *e;
		one.use_gelu			= use_gelu;
		int cnt					= counts[i];
		int batched				= idx_uploaded && cnt >= 2;
		if (batched) {
			for (int c = 0; c < cnt; c++) {
				int row = rows_packed[offset + c];
				if (row < 0 || row >= n_rows) {
					batched = 0;
					break;
				}
			}
		}
		if (batched)
			st = gl_moe_expert_rows_batched(self, p, xb, out, &idx_buf, &w_buf, offset, cnt, dim,
											inter, use_gelu, e);
		if (!batched || st != OK) {
			st = OK;
			for (int c = 0; c < cnt; c++) {
				int row = rows_packed[offset + c];
				if (row < 0 || row >= n_rows) {
					st = ERR_INVALID_ARG;
					break;
				}
				buffer xv  = buffer_slice(xb, (size_t)row * (size_t)dim * sizeof(float),
										  (size_t)dim * sizeof(float));
				buffer ov  = buffer_slice(out, (size_t)row * (size_t)dim * sizeof(float),
										  (size_t)dim * sizeof(float));
				one.weight = weights_packed[offset + c];
				st		   = gl_moe_expert_ffn(self, &xv, &ov, &one, dim, inter);
				if (st != OK)
					break;
			}
		}
		offset += counts[i];
	}
	self->buffer_free(self, &idx_buf);
	self->buffer_free(self, &w_buf);
	return st;
}

static status_code gl_require_resident(backend *self, const char *op, const buffer *const *bufs,
									   int n) {
	for (int i = 0; i < n; i++) {
		const buffer *b = bufs[i];
		if (b && b->handle && b->owner == self)
			continue;
		ERROR("gl: %s: operand %d is not resident on '%s'", op, i, self->name);
		return ERR_INVALID_ARG;
	}
	return OK;
}

static void gl_sync_operands(const buffer *const *bufs, int n) {
	for (int i = 0; i < n; i++)
		gl_buf_sync_to_device(as_glbuf(bufs[i]));
}

static buffer gl_float_slice(const buffer *parent, size_t float_off, size_t n_floats) {
	return buffer_slice(parent, float_off * sizeof(float), n_floats * sizeof(float));
}

static GLuint gl_groups_for(int n, int per_group) {
	return (GLuint)((n + per_group - 1) / per_group);
}

static status_code gl_hc_init(backend *self, const hc_init_desc *d) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	const buffer *ops[] = {d->x, d->hc};
	status_code	  s		= gl_require_resident(self, "hc_init", ops, 2);
	if (s != OK)
		return s;
	gl_sync_operands(ops, 2);
	gl_pipeline *pipe  = &p->pipes[GL_PIPE_HC_INIT];
	int			 total = d->dim * d->n_hc;
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo(0, d->x);
	gl_bind_ssbo_out(1, d->hc);
	gl_pu(pipe, "u_dim", d->dim);
	gl_pu(pipe, "u_total", total);
	gl_launch(p, gl_groups_for(total, 64), 1, 1);
	as_glbuf(d->hc)->device_dirty = 1;
	return OK;
}

static status_code gl_hc_final(backend *self, const hc_final_desc *d) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	const buffer *ops[] = {d->hc, d->x};
	status_code	  s		= gl_require_resident(self, "hc_final", ops, 2);
	if (s != OK)
		return s;
	gl_sync_operands(ops, 2);
	gl_pipeline *pipe = &p->pipes[GL_PIPE_HC_FINAL];
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo(0, d->hc);
	gl_bind_ssbo_out(1, d->x);
	gl_pu(pipe, "u_dim", d->dim);
	gl_pu(pipe, "u_n_hc", d->n_hc);
	gl_launch(p, gl_groups_for(d->dim, 64), 1, 1);
	as_glbuf(d->x)->device_dirty = 1;
	return OK;
}

static status_code gl_hc_post(backend *self, const hc_post_desc *d) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (d->n_hc > GL_HC_MAX) {
		ERROR("gl: hc_post: n_hc=%d exceeds %d", d->n_hc, GL_HC_MAX);
		return ERR_UNSUPPORTED;
	}
	const buffer *ops[] = {d->sub, d->coef, d->hc};
	status_code	  s		= gl_require_resident(self, "hc_post", ops, 3);
	if (s != OK)
		return s;
	gl_sync_operands(ops, 3);
	gl_pipeline *pipe = &p->pipes[GL_PIPE_HC_POST];
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo(0, d->sub);
	gl_bind_ssbo(1, d->coef);
	gl_bind_ssbo_out(2, d->hc);
	gl_pu(pipe, "u_dim", d->dim);
	gl_pu(pipe, "u_n_hc", d->n_hc);
	gl_launch(p, gl_groups_for(d->dim, 64), 1, 1);
	as_glbuf(d->hc)->device_dirty = 1;
	return OK;
}

static status_code gl_hc_pre(backend *self, const hc_pre_desc *d) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	if (d->n_hc > GL_HC_MAX) {
		ERROR("gl: hc_pre: n_hc=%d exceeds %d", d->n_hc, GL_HC_MAX);
		return ERR_UNSUPPORTED;
	}
	const buffer *ops[] = {d->stream, d->fn.buf, d->base, d->scale, d->x, d->coef};
	status_code	  s		= gl_require_resident(self, "hc_pre", ops, 6);
	if (s != OK)
		return s;

	const int	 flat	 = d->n_hc * d->dim;
	const int	 mix_dim = hc_mix_dim(d->n_hc);
	const size_t flat_p	 = ALIGN_UP((size_t)flat, 64);
	buffer		 scratch = {0};
	s = self->buffer_alloc_scratch(self, (flat_p + ALIGN_UP((size_t)mix_dim, 64)) * sizeof(float),
								   &scratch);
	if (s != OK)
		return s;
	buffer normed = gl_float_slice(&scratch, 0, (size_t)flat);
	buffer mixes  = gl_float_slice(&scratch, flat_p, (size_t)mix_dim);

	s = gl_rmsnorm_noweight(self, d->stream, &normed, flat, d->norm_eps);
	if (s == OK)
		s = gl_matmul(self, d->fn.buf, d->fn.type, &normed, &mixes, mix_dim, flat);
	if (s != OK) {
		self->buffer_free(self, &scratch);
		return s;
	}

	const buffer *params[] = {d->base, d->scale, d->x, d->coef};
	gl_sync_operands(params, 4);
	gl_buf_sync_to_device(as_glbuf(&mixes));

	gl_pipeline *coef_pipe = &p->pipes[GL_PIPE_HC_PRE_COEF];
	gl_use_pipeline(p, coef_pipe);
	gl_bind_ssbo(0, &mixes);
	gl_bind_ssbo(1, d->base);
	gl_bind_ssbo(2, d->scale);
	gl_bind_ssbo_out(3, d->coef);
	gl_pu(coef_pipe, "u_n_hc", d->n_hc);
	gl_pu(coef_pipe, "u_iters", d->sinkhorn_iters);
	gl_puf(coef_pipe, "u_eps", d->eps);
	gl_launch(p, 1, 1, 1);

	gl_pipeline *mix_pipe = &p->pipes[GL_PIPE_HC_PRE_MIX];
	gl_use_pipeline(p, mix_pipe);
	gl_bind_ssbo(0, &mixes);
	gl_bind_ssbo(1, d->base);
	gl_bind_ssbo(2, d->scale);
	gl_bind_ssbo(3, d->stream);
	gl_bind_ssbo_out(4, d->x);
	gl_pu(mix_pipe, "u_dim", d->dim);
	gl_pu(mix_pipe, "u_n_hc", d->n_hc);
	gl_puf(mix_pipe, "u_eps", d->eps);
	gl_launch(p, gl_groups_for(d->dim, 64), 1, 1);

	as_glbuf(d->coef)->device_dirty = 1;
	as_glbuf(d->x)->device_dirty	= 1;
	self->buffer_free(self, &scratch);
	return OK;
}

static void gl_kda_conv(gl_priv *p, buffer *x, const buffer *w, buffer *state, int state_base,
						int channels, int kernel, int reset) {
	gl_pipeline *pipe = &p->pipes[GL_PIPE_KDA_CONV];
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo_out(0, x);
	gl_bind_ssbo(1, w);
	gl_bind_ssbo_out(2, state);
	gl_pu(pipe, "u_channels", channels);
	gl_pu(pipe, "u_kernel", kernel);
	gl_pu(pipe, "u_reset", reset);
	gl_pu(pipe, "u_state_base", state_base);
	gl_launch(p, gl_groups_for(channels, 64), 1, 1);
	as_glbuf(x)->device_dirty	  = 1;
	as_glbuf(state)->device_dirty = 1;
}

static void gl_kda_l2norm(gl_priv *p, buffer *x, int n_heads, int head_dim) {
	gl_pipeline *pipe = &p->pipes[GL_PIPE_KDA_L2NORM];
	gl_use_pipeline(p, pipe);
	gl_bind_ssbo_out(0, x);
	gl_pu(pipe, "u_head_dim", head_dim);
	gl_puf(pipe, "u_eps", GL_KDA_L2_EPS);
	gl_launch(p, (GLuint)n_heads, 1, 1);
	as_glbuf(x)->device_dirty = 1;
}

static status_code gl_kda_project(backend *self, const backend_weight *w, const buffer *x,
								  buffer *y, int n, int k) {
	return gl_matmul(self, w->buf, w->type, x, y, n, k);
}

typedef struct {
	buffer q, k, v, g, o, g2, tmp_f, tmp_g, beta;
} gl_kda_scratch;

static status_code gl_kda_scratch_alloc(backend *self, const kda_desc *d, buffer *root,
										gl_kda_scratch *out) {
	const size_t D	= (size_t)d->n_heads * d->head_dim;
	const size_t S	= ALIGN_UP((size_t)d->head_dim, 64);
	const size_t H	= ALIGN_UP((size_t)d->n_heads, 64);
	const size_t Dp = ALIGN_UP(D, 64);
	status_code	 s	= self->buffer_alloc_scratch(self, (6 * Dp + 2 * S + H) * sizeof(float), root);
	if (s != OK)
		return s;
	buffer *const slots[] = {&out->q, &out->k, &out->v, &out->g, &out->o, &out->g2};
	size_t		  off	  = 0;
	for (int i = 0; i < 6; i++, off += Dp)
		*slots[i] = gl_float_slice(root, off, D);
	out->tmp_f = gl_float_slice(root, off, (size_t)d->head_dim);
	off += S;
	out->tmp_g = gl_float_slice(root, off, (size_t)d->head_dim);
	off += S;
	out->beta = gl_float_slice(root, off, (size_t)d->n_heads);
	return OK;
}

static status_code gl_kda(backend *self, const kda_desc *d) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	const int dim = d->dim, H = d->n_heads, S = d->head_dim, D = H * S;
	const int hist = d->conv_kernel - 1;
	if (hist < 1 || hist > GL_KDA_MAX_HIST || (unsigned)D > GL_MAX_DISPATCH_PER_DIM) {
		ERROR("gl: kda: unsupported shape heads=%d head_dim=%d conv_kernel=%d", H, S,
			  d->conv_kernel);
		return ERR_UNSUPPORTED;
	}
	const buffer *ops[] = {d->x,		d->wq.buf,	   d->wk.buf,  d->wv.buf,  d->wo.buf,
						   d->f_a.buf,	d->f_b.buf,	   d->g_a.buf, d->g_b.buf, d->beta.buf,
						   d->conv_q,	d->conv_k,	   d->conv_v,  d->dt_bias, d->ssm_a,
						   d->out_norm, d->conv_state, d->state,   d->out};
	status_code	  s		= gl_require_resident(self, "kda", ops, (int)ARRAY_LEN(ops));
	if (s != OK)
		return s;
	const int conv_stride = D * hist;

	buffer		   root = {0};
	gl_kda_scratch sc;
	s = gl_kda_scratch_alloc(self, d, &root, &sc);
	if (s != OK)
		return s;

	s = gl_kda_project(self, &d->wq, d->x, &sc.q, D, dim);
	if (s == OK)
		s = gl_kda_project(self, &d->wk, d->x, &sc.k, D, dim);
	if (s == OK)
		s = gl_kda_project(self, &d->wv, d->x, &sc.v, D, dim);
	if (s == OK)
		s = gl_kda_project(self, &d->f_a, d->x, &sc.tmp_f, S, dim);
	if (s == OK)
		s = gl_kda_project(self, &d->f_b, &sc.tmp_f, &sc.g, D, S);
	if (s == OK)
		s = gl_kda_project(self, &d->beta, d->x, &sc.beta, H, dim);
	if (s == OK)
		s = gl_kda_project(self, &d->g_a, d->x, &sc.tmp_g, S, dim);
	if (s == OK)
		s = gl_kda_project(self, &d->g_b, &sc.tmp_g, &sc.g2, D, S);
	if (s != OK) {
		self->buffer_free(self, &root);
		return s;
	}

	const buffer *state_ops[] = {d->conv_state, d->state, d->out, &root};
	gl_sync_operands(state_ops, 4);
	gl_sync_operands(&ops[10], 6);

	gl_kda_conv(p, &sc.q, d->conv_q, d->conv_state, 0, D, d->conv_kernel, d->reset);
	gl_kda_conv(p, &sc.k, d->conv_k, d->conv_state, conv_stride, D, d->conv_kernel, d->reset);
	gl_kda_conv(p, &sc.v, d->conv_v, d->conv_state, 2 * conv_stride, D, d->conv_kernel, d->reset);
	gl_kda_l2norm(p, &sc.q, H, S);
	gl_kda_l2norm(p, &sc.k, H, S);

	gl_pipeline *gate = &p->pipes[GL_PIPE_KDA_GATE];
	gl_use_pipeline(p, gate);
	gl_bind_ssbo_out(0, &sc.g);
	gl_bind_ssbo(1, d->dt_bias);
	gl_bind_ssbo(2, d->ssm_a);
	gl_bind_ssbo_out(3, &sc.beta);
	gl_pu(gate, "u_inner", D);
	gl_pu(gate, "u_n_heads", H);
	gl_pu(gate, "u_head_dim", S);
	gl_pu(gate, "u_use_lower_bound", d->gate_lower_bound > -INFINITY ? 1 : 0);
	gl_puf(gate, "u_lower_bound", d->gate_lower_bound > -INFINITY ? d->gate_lower_bound : 0.0f);
	gl_launch(p, gl_groups_for(D + H, 64), 1, 1);

	gl_pipeline *delta = &p->pipes[GL_PIPE_KDA_DELTA];
	gl_use_pipeline(p, delta);
	gl_bind_ssbo_out(0, d->state);
	gl_bind_ssbo(1, &sc.q);
	gl_bind_ssbo(2, &sc.k);
	gl_bind_ssbo(3, &sc.v);
	gl_bind_ssbo(4, &sc.g);
	gl_bind_ssbo(5, &sc.beta);
	gl_bind_ssbo_out(6, &sc.o);
	gl_pu(delta, "u_head_dim", S);
	gl_pu(delta, "u_reset", d->reset);
	gl_launch(p, (GLuint)D, 1, 1);

	gl_pipeline *post = &p->pipes[GL_PIPE_KDA_OUT];
	gl_use_pipeline(p, post);
	gl_bind_ssbo_out(0, &sc.o);
	gl_bind_ssbo(1, &sc.g2);
	gl_bind_ssbo(2, d->out_norm);
	gl_pu(post, "u_head_dim", S);
	gl_puf(post, "u_eps", d->norm_eps);
	gl_launch(p, (GLuint)H, 1, 1);

	as_glbuf(d->state)->device_dirty = 1;
	as_glbuf(&root)->device_dirty	 = 1;
	s								 = gl_kda_project(self, &d->wo, &sc.o, d->out, dim, D);
	self->buffer_free(self, &root);
	return s;
}

static status_code gl_mla_absorbed(backend *self, const mla_absorbed_desc *d) {
	gl_priv *p = self->priv;
	GL_ENTER(p);
	const int dim = d->dim, H = d->n_heads, R = d->kv_lora, QK = d->qk_head, VH = d->v_head;
	const int QL = d->q_lora;
	if (R > GL_MLA_MAX_LORA || (unsigned)H > GL_MAX_DISPATCH_PER_DIM) {
		ERROR("gl: mla_absorbed: unsupported shape heads=%d kv_lora=%d", H, R);
		return ERR_UNSUPPORTED;
	}
	const buffer *ops[] = {d->x,	  d->q_a.buf,	   d->q_b.buf,	 d->kv_a.buf,
						   d->wo.buf, d->q_a_norm,	   d->kv_a_norm, d->k_b,
						   d->v_b,	  d->latent_cache, d->out};
	status_code	  s		= gl_require_resident(self, "mla_absorbed", ops, (int)ARRAY_LEN(ops));
	if (s != OK)
		return s;

	const size_t ql_p = ALIGN_UP((size_t)QL, 64), q_p = ALIGN_UP((size_t)H * QK, 64);
	const size_t qa_p = ALIGN_UP((size_t)H * R, 64), kv_p = ALIGN_UP((size_t)R, 64);
	const size_t ho_p = ALIGN_UP((size_t)H * VH, 64);
	buffer		 root = {0};
	s = self->buffer_alloc_scratch(self, (2 * ql_p + q_p + qa_p + 2 * kv_p + ho_p) * sizeof(float),
								   &root);
	if (s != OK)
		return s;
	size_t off = 0;
	buffer qa  = gl_float_slice(&root, off, (size_t)QL);
	off += ql_p;
	buffer qn = gl_float_slice(&root, off, (size_t)QL);
	off += ql_p;
	buffer q = gl_float_slice(&root, off, (size_t)H * QK);
	off += q_p;
	buffer q_abs = gl_float_slice(&root, off, (size_t)H * R);
	off += qa_p;
	buffer kv = gl_float_slice(&root, off, (size_t)R);
	off += kv_p;
	buffer kvn = gl_float_slice(&root, off, (size_t)R);
	off += kv_p;
	buffer ho	  = gl_float_slice(&root, off, (size_t)H * VH);
	buffer latent = gl_float_slice(d->latent_cache, (size_t)d->pos * R, (size_t)R);

	s = gl_matmul(self, d->q_a.buf, d->q_a.type, d->x, &qa, QL, dim);
	if (s == OK)
		s = gl_rmsnorm(self, &qa, d->q_a_norm, &qn, QL, d->norm_eps);
	if (s == OK)
		s = gl_matmul(self, d->q_b.buf, d->q_b.type, &qn, &q, H * QK, QL);
	if (s == OK)
		s = gl_matmul(self, d->kv_a.buf, d->kv_a.type, d->x, &kv, R, dim);
	if (s == OK)
		s = gl_rmsnorm(self, &kv, d->kv_a_norm, &kvn, R, d->norm_eps);
	if (s == OK)
		s = gl_copy_buffer(self, &kvn, &latent, R);
	if (s != OK) {
		self->buffer_free(self, &root);
		return s;
	}

	const buffer *rest[] = {d->k_b, d->v_b, d->latent_cache, d->out, &q, &q_abs, &ho};
	gl_sync_operands(rest, (int)ARRAY_LEN(rest));

	gl_pipeline *absorb = &p->pipes[GL_PIPE_MLA_ABSORB_Q];
	gl_use_pipeline(p, absorb);
	gl_bind_ssbo(0, d->k_b);
	gl_bind_ssbo(1, &q);
	gl_bind_ssbo_out(2, &q_abs);
	gl_pu(absorb, "u_total", H * R);
	gl_pu(absorb, "u_kv_lora", R);
	gl_pu(absorb, "u_qk_head", QK);
	gl_launch(p, gl_groups_for(H * R, 64), 1, 1);

	gl_pipeline *attn = &p->pipes[GL_PIPE_MLA_ABSORB_ATTN];
	gl_use_pipeline(p, attn);
	gl_bind_ssbo(0, &q_abs);
	gl_bind_ssbo(1, d->latent_cache);
	gl_bind_ssbo(2, d->v_b);
	gl_bind_ssbo_out(3, &ho);
	gl_pu(attn, "u_n_pos", d->pos + 1);
	gl_pu(attn, "u_kv_lora", R);
	gl_pu(attn, "u_v_head", VH);
	gl_puf(attn, "u_scale", 1.0f / sqrtf((float)QK));
	gl_launch(p, (GLuint)H, 1, 1);

	as_glbuf(&q_abs)->device_dirty = 1;
	as_glbuf(&ho)->device_dirty	   = 1;
	s = gl_matmul(self, d->wo.buf, d->wo.type, &ho, d->out, dim, H * VH);
	self->buffer_free(self, &root);
	return s;
}

static status_code gl_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name	  = "opengl";
	out->priority = 80;
	out->caps	  = BCAP_HOST_VISIBLE_BUFFERS | BCAP_RMSNORM_ADD | BCAP_KV_QUANT_Q8_0 |
					BCAP_MATMUL_RESIDUAL | BCAP_MULTI_MATMUL | BCAP_MATMUL_FFN_DOWN |
					BCAP_ROPE_QK_FUSED | BCAP_QK_NORM_ROPE | BCAP_MOE_EXPERT_RESIDENT |
					BCAP_IN_ORDER_QUEUE | BCAP_KV_SHIFT;

	out->probe							 = gl_probe;
	out->device_count					 = gl_device_count;
	out->device_identity				 = gl_device_identity;
	out->init							 = gl_init;
	out->free							 = gl_free;
	out->buffer_alloc_weight			 = gl_buffer_alloc_weight;
	out->buffer_alloc_scratch			 = gl_buffer_alloc_scratch;
	out->buffer_alloc_from_host			 = gl_buffer_alloc_from_host;
	out->buffer_free					 = gl_buffer_free;
	out->buffer_read_f32				 = gl_buffer_read_f32;
	out->buffer_read_f32_x2				 = gl_buffer_read_f32_x2;
	out->buffer_write_f32				 = gl_buffer_write_f32;
	out->copy_buffer					 = gl_copy_buffer;
	out->ple_combine					 = gl_ple_combine;
	out->mem_available					 = gl_mem_available;
	out->mem_total						 = gl_mem_total;
	out->release_thread					 = gl_release_thread;
	out->synchronize					 = gl_synchronize;
	out->set_gpu_profile				 = gl_set_gpu_profile;
	out->submit							 = gl_submit;
	out->argmax							 = gl_argmax;
	out->argmax_to_reg					 = gl_argmax_to_reg;
	out->read_token_reg					 = gl_read_token_reg;
	out->embd_lookup					 = gl_embd_lookup;
	out->rmsnorm						 = gl_rmsnorm;
	out->rmsnorm_per_head				 = gl_rmsnorm_per_head;
	out->rmsnorm_noweight				 = gl_rmsnorm_noweight;
	out->rmsnorm_noweight_per_head		 = gl_rmsnorm_noweight_per_head;
	out->rmsnorm_add					 = gl_rmsnorm_add;
	out->rmsnorm_add_batch				 = gl_rmsnorm_add_batch;
	out->matmul							 = gl_matmul;
	out->matmul_type_native				 = gl_matmul_type_native;
	out->repack_plan					 = gl_repack_plan;
	out->repack_weight					 = gl_repack_weight;
	out->matmul_residual				 = gl_matmul_residual;
	out->matmul_multi					 = gl_matmul_multi;
	out->matmul_batch					 = gl_matmul_batch;
	out->add_inplace					 = gl_add_inplace;
	out->scale_inplace					 = gl_scale_inplace;
	out->softcap						 = gl_softcap;
	out->ffn_activate					 = gl_ffn_activate;
	out->ffn_activate_ex				 = gl_ffn_activate_ex;
	out->rope							 = gl_rope;
	out->rope_qk						 = gl_rope_qk;
	out->rope_ext						 = gl_rope_ext;
	out->split_qgate					 = gl_split_qgate;
	out->attn_output_gate				 = gl_attn_output_gate;
	out->partial_rope_qk				 = gl_partial_rope_qk;
	out->kv_alloc						 = gl_kv_alloc;
	out->kv_free						 = gl_kv_free;
	out->kv_put							 = gl_kv_put;
	out->kv_put_batch					 = gl_kv_put_batch;
	out->kv_shift						 = gl_kv_shift;
	out->attention_mla					 = gl_attention_mla;
	out->kv_alloc_mla					 = gl_kv_alloc_mla;
	out->kv_put_mla						 = gl_kv_put_mla;
	out->attention						 = gl_attention;
	out->attention_swa					 = gl_attention_swa;
	out->rmsnorm_batch					 = gl_rmsnorm_batch;
	out->rmsnorm_per_head_batch			 = gl_rmsnorm_per_head_batch;
	out->rmsnorm_noweight_batch			 = gl_rmsnorm_noweight_batch;
	out->rmsnorm_noweight_per_head_batch = gl_rmsnorm_noweight_per_head_batch;
	out->add_batch						 = gl_add_batch;
	out->rope_batch						 = gl_rope_batch;
	out->rope_qk_batch					 = gl_rope_qk_batch;
	out->qk_norm_rope					 = gl_qk_norm_rope;
	out->rope_ext_batch					 = gl_rope_ext_batch;
	out->attention_batch				 = gl_attention_batch;
	out->attention_swa_batch			 = gl_attention_swa_batch;
	out->ffn_activate_batch				 = gl_ffn_activate_batch;
	out->ffn_activate_fused_batch		 = gl_ffn_activate_fused_batch;
	out->matmul_multi_batch				 = gl_matmul_multi_batch;
	out->matmul_ffn_down				 = gl_matmul_ffn_down;
	out->matmul_ffn_down_batch			 = gl_matmul_ffn_down_batch;
	out->moe_activate					 = gl_moe_activate;
	out->moe_expert_ffn					 = gl_moe_expert_ffn;
	out->moe_experts_batch				 = gl_moe_experts_batch;
	out->dequant_row					 = gl_dequant_row;
	out->dequant_type_native			 = gl_dequant_type_native;
	out->dequant_weight					 = gl_dequant_weight;
	out->hc_init						 = gl_hc_init;
	out->hc_pre							 = gl_hc_pre;
	out->hc_post						 = gl_hc_post;
	out->hc_final						 = gl_hc_final;
	out->kda							 = gl_kda;
	out->mla_absorbed					 = gl_mla_absorbed;

	return OK;
}

BACKEND_REGISTER("opengl", gl_ctor)
