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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GL_MAX_BINDINGS 8
#define GL_SCRATCH_POOL_CAP 32
#define GL_MAX_DISPATCH_PER_DIM 65535u
#define GL_PIPELINE_MAX_UNIFORMS 16
#define GL_UNIFORM_NAME_CAP 32

typedef struct gl_buf {
	void  *host_mirror;
	GLuint name;
	size_t size_bytes;
	size_t store_bytes;
	GLenum usage;
	int	   device_dirty;
	int	   host_dirty;
} gl_buf;

typedef struct {
	GLuint		program;
	GLint		n_bindings;
	const char *name;
	int			n_uni;
	GLint		uloc[GL_PIPELINE_MAX_UNIFORMS];
	char		uname[GL_PIPELINE_MAX_UNIFORMS][GL_UNIFORM_NAME_CAP];
} gl_pipeline;

typedef struct {
	gl_context ctx;

	gl_pipeline p_argmax_stage1;
	gl_pipeline p_argmax_stage2;
	gl_pipeline p_rmsnorm;
	gl_pipeline p_rmsnorm_per_head;
	gl_pipeline p_elementwise;
	gl_pipeline p_embd_lookup_f16;
	gl_pipeline p_embd_lookup_f32;
	gl_pipeline p_embd_lookup_bf16;
	gl_pipeline p_matmul_f32;
	gl_pipeline p_matmul_iq4_nl;
	gl_pipeline p_matmul_q4_0;
	gl_pipeline p_matmul_q4_1;
	gl_pipeline p_matmul_q5_0;
	gl_pipeline p_matmul_q5_1;
	gl_pipeline p_matmul_q8_0;
	gl_pipeline p_matmul_q4_k;
	gl_pipeline p_matmul_f16;
	gl_pipeline p_matmul_bf16;
	gl_pipeline p_matmul_q5_k;
	gl_pipeline p_matmul_q6_k;
	gl_pipeline p_rope;
	gl_pipeline p_rope_batch;
	gl_pipeline p_ffn_activate;
	gl_pipeline p_ffn_activate_fused_batch;
	gl_pipeline p_attention;
	gl_pipeline p_attention_batch;
	gl_pipeline p_partial_rope_qk;
	gl_pipeline p_kv_put;

	gl_buf *argmax_partial;
	size_t	argmax_partial_bytes;

	size_t kv_layer_stride;
	size_t kv_kvh_stride;
	int	   kv_head_dim;
	int	   kv_n_kv_heads;
	int	   kv_n_ctx;
	int	   kv_n_layers;
	int	   kv_quant;
	size_t kv_row_bytes;

	gl_buf *attn_scores;
	size_t	attn_scores_cap;

	gl_buf *dummy;

	gl_buf *scratch_pool[GL_SCRATCH_POOL_CAP];
	size_t	scratch_pool_sizes[GL_SCRATCH_POOL_CAP];
	int		scratch_pool_count;

	size_t device_local_allocated;
	size_t device_local_total_estimate;

	int device_lost;

	buffer rope_cos_tbl;
	buffer rope_sin_tbl;

	char *attention_src;
	int	  attention_tile_t;
	int	  attention_batch_tile_t;
	char *attention_batch_src;

	float *fb_scores;
	size_t fb_scores_cap;
	float *fb_rowbuf;
	size_t fb_rowbuf_cap;

	gl_buf **all_bufs;
	int		 all_bufs_count;
	int		 all_bufs_cap;
} gl_priv;

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

static gl_priv *g_active_gl;

static status_code gl_ensure_context(gl_priv *p) {
	if (!p || !p->ctx.dpy || p->device_lost)
		return ERR_INTERNAL;
	if (eglGetCurrentContext() == p->ctx.ctx)
		return OK;
	if (!eglMakeCurrent(p->ctx.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, p->ctx.ctx)) {
		ERROR("gl: failed to make the GL context current on the calling thread "
			  "(eglErr=0x%04x); context is likely owned by another thread",
			  (unsigned)eglGetError());
		return ERR_INTERNAL;
	}
	return OK;
}

static status_code gl_probe(void);
static int		   gl_device_count(void);
static status_code gl_init(backend *self, int device_index);
static void		   gl_free(backend *self);

static status_code gl_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out);
static status_code gl_buffer_alloc_scratch(backend *self, size_t size, buffer *out);
static status_code gl_buffer_alloc_from_host(backend *self, const void *host_data, size_t size,
											 buffer *out);
static void		   gl_buffer_free(backend *self, buffer *buf);
static status_code gl_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n);
static status_code gl_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n);
static status_code gl_copy_buffer(backend *self, const buffer *src, buffer *dst, int n);

static size_t gl_mem_available(backend *self);
static size_t gl_mem_total(backend *self);
static void	  gl_synchronize(backend *self);
static void	  gl_begin_batch(backend *self);
static void	  gl_end_batch(backend *self);

static status_code gl_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx);
static status_code gl_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
								  int token, int dim, buffer *x_out);
static status_code gl_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							  float eps);
static status_code gl_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
									   int n_heads, int head_dim, float eps);
static status_code gl_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n, float eps);
static status_code gl_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y,
												int n_heads, int head_dim, float eps);
static status_code gl_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
								  const buffer *residual, buffer *y, int n, float eps,
								  float out_scale);
static status_code gl_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
								  float combine_scale);
static status_code gl_elementwise(gl_priv *p, buffer *x, const buffer *y, const buffer *z, int n,
								  int mode, float scale, int aux, int rows);
static status_code gl_add_inplace(backend *self, buffer *x, const buffer *y, int n);
static status_code gl_scale_inplace(backend *self, buffer *x, float scale, int n);
static status_code gl_softcap(backend *self, buffer *x, float cap, int n);
static status_code gl_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n);
static status_code gl_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
									  buffer *out, int n, int activation);
static status_code gl_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						   const float *rope_cos_base, const float *rope_sin_base);
static status_code gl_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
							  int head_dim, int pos, const float *rope_cos_base,
							  const float *rope_sin_base);
static status_code gl_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
							   const float *rope_cos_base, const float *rope_sin_base,
							   const float *freq_factors);
static status_code gl_split_qgate(backend *self, const buffer *mixed, buffer *q, buffer *gate,
								  int n_heads, int head_dim, int n_rows);
static status_code gl_attn_output_gate(backend *self, buffer *out, const buffer *gate, int n,
									   int n_rows);
static status_code gl_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
									  int n_kv_heads, int head_dim, int rope_dim, int pos_start,
									  const float *rope_cos_base, const float *rope_sin_base,
									  int n_rows);
static status_code gl_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							 buffer *y, int n, int k);
static int		   gl_matmul_type_native(backend *self, uint32_t w_type);
static status_code gl_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *x, const buffer *residual, buffer *y, int n,
									  int k);
static status_code gl_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
								   int n_matmuls);

static status_code gl_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out);
static void		   gl_kv_free(backend *self, buffer *k, buffer *v);
static status_code gl_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
							 const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
							 int n_ctx, int n_kv_heads_active);
static status_code gl_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								   const buffer *k_in, const buffer *v_in, int in_row_stride,
								   int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								   int m);
static status_code gl_attention(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
								int n_kv_heads, int head_dim, int n_ctx, int flash_attn,
								float scale, int n_kv_heads_active);
static status_code gl_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
									const buffer *v_cache, buffer *out, int layer, int pos,
									int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									int flash_attn, float scale, int sliding_window,
									int n_kv_heads_active);

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
#define GL_ATTENTION_TILE_T_MAX 16
#define GL_ATTENTION_Q_CHUNK 4
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

static int gl_attention_tile_t(const gl_context *ctx) {
	int shared_budget = ctx->max_shared_bytes > 0 ? ctx->max_shared_bytes : 16384;
	int per_t		  = GL_ATTENTION_KV_ROW_BYTES + GL_ATTENTION_Q_CHUNK * 2 * sizeof(float);
	int tile_t		  = (shared_budget - GL_ATTENTION_FIXED_SHARED_BYTES) / per_t;
	if (tile_t > GL_ATTENTION_TILE_T_MAX)
		tile_t = GL_ATTENTION_TILE_T_MAX;
	if (tile_t < 1)
		tile_t = 1;
	return tile_t;
}

static char *gl_shader_with_tile_t(const char *src, int tile_t) {
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
	char   tile_define[64];
	int	   define_len = snprintf(tile_define, sizeof(tile_define), "#define TILE_T %d\n", tile_t);
	size_t src_len	  = strlen(src);
	char  *out		  = xmalloc(src_len + (size_t)define_len + 1);
	memcpy(out, src, head_len);
	memcpy(out + head_len, tile_define, (size_t)define_len);
	memcpy(out + head_len + define_len, src + head_len, src_len - head_len);
	return out;
}

static void gl_pipeline_cache_uniforms(gl_pipeline *p) {
	p->n_uni	= 0;
	GLint count = 0;
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

static GLint gl_pipeline_loc(const gl_pipeline *p, const char *name) {
	for (int i = 0; i < p->n_uni; i++)
		if (strcmp(p->uname[i], name) == 0)
			return p->uloc[i];
	return -1;
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

static void gl_pu(const gl_pipeline *p, const char *name, int v) {
	GLint loc = gl_pipeline_loc(p, name);
	if (loc < 0) {
		gl_pipeline_warn_missing(p, name);
		return;
	}
	glUniform1i(loc, v);
}

static void gl_puf(const gl_pipeline *p, const char *name, float v) {
	GLint loc = gl_pipeline_loc(p, name);
	if (loc < 0) {
		gl_pipeline_warn_missing(p, name);
		return;
	}
	glUniform1f(loc, (GLfloat)v);
}

static void gl_pui(const gl_pipeline *p, const char *const *names, const int *vals, int count) {
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
	out->name		= name;
	out->n_bindings = 0;
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

static inline void gl_bind_ssbo(int binding, const buffer *b) {
	gl_buf *gb = as_glbuf(b);
	if (b->offset == 0) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)binding, gb->name);
	} else {
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, (GLuint)binding, gb->name, (GLintptr)b->offset,
						  (GLsizeiptr)b->size);
	}
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

static gl_buf *gl_buf_alloc(gl_priv *p, size_t size, GLenum usage, const void *initial_data) {
	if (size == 0)
		size = 1;

	if (gl_ensure_context(p) != OK) {
		ERROR("gl: buf_alloc(%zu bytes) without a usable GL context", size);
		return NULL;
	}

	gl_buf *b		= xcalloc(1, sizeof(gl_buf) + size);
	b->size_bytes	= size;
	b->usage		= usage;
	b->host_mirror	= (char *)b + sizeof(gl_buf);
	b->device_dirty = 0;
	b->host_dirty	= 0;

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
	if (initial_data && alloc_size != size) {
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

	if (initial_data)
		memcpy(b->host_mirror, initial_data, size);

	gl_buf_register(p, b);
	p->device_local_allocated += size;
	return b;
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

	gl_ensure_context(g_active_gl);

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
}

static void gl_buf_sync_to_host(gl_buf *b) {
	if (!b || !b->device_dirty)
		return;
	glFinish();
	gl_ensure_context(g_active_gl);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b->name);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	void *ptr =
		glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)b->size_bytes, GL_MAP_READ_BIT);
	if (!ptr) {
		ERROR("gl: glMapBufferRange(%zu bytes) failed (glErr=0x%04x), retrying", b->size_bytes,
			  (unsigned)glGetError());
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		glFinish();
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

static void gl_buf_free(gl_priv *p, gl_buf *b) {
	if (!b)
		return;
	gl_buf_unregister(p, b);
	if (b->name) {
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
							   int n_uniforms, GLuint groups_x, GLuint groups_y, GLuint groups_z) {
	if (!pipe->program) {
		ERROR("gl: dispatch on '%s' with no compiled program", pipe->name);
		return ERR_INTERNAL;
	}
	if (p->device_lost)
		return ERR_INTERNAL;
	{
		status_code ecs = gl_ensure_context(p);
		if (ecs != OK)
			return ecs;
	}

	for (int i = 0; i < n_bufs && i < GL_MAX_BINDINGS; i++)
		gl_buf_sync_to_device(bufs[i]);

	glUseProgram(pipe->program);
	for (int i = 0; i < n_bufs && i < GL_MAX_BINDINGS; i++) {
		if (!bufs[i])
			continue;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, bufs[i]->name);
	}
	gl_pui(pipe, uniform_names, uniforms, n_uniforms);

	glDispatchCompute(groups_x, groups_y, groups_z);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: glDispatchCompute('%s') failed (glErr=0x%04x, dim=%ux%ux%u)", pipe->name,
			  (unsigned)err, groups_x, groups_y, groups_z);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	glFlush();
	return OK;
}

static status_code gl_dispatch_all_dirty(gl_priv *p, gl_pipeline *pipe, gl_buf **bufs, int n_bufs,
										 const char *const *uniform_names, const int *uniforms,
										 int n_uniforms, GLuint gx, GLuint gy, GLuint gz,
										 int n_outputs) {
	status_code s =
		gl_dispatch(p, pipe, bufs, n_bufs, uniform_names, uniforms, n_uniforms, gx, gy, gz);
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

	p->attention_tile_t		  = gl_attention_tile_t(&p->ctx);
	p->attention_src		  = gl_shader_with_tile_t(gl_shader_attention_src, p->attention_tile_t);
	p->attention_batch_tile_t = gl_attention_batch_tile_t(&p->ctx);
	p->attention_batch_src =
		gl_shader_with_tile_t(gl_shader_attention_batch_src, p->attention_batch_tile_t);
	if (!p->attention_src || !p->attention_batch_src) {
		free(p->attention_src);
		free(p->attention_batch_src);
		gl_context_free(&p->ctx);
		free(p);
		self->priv = NULL;
		return ERR_INTERNAL;
	}

	gl_pipeline pipelines[] = {
		{0, 0, "argmax_stage1"},	{0, 0, "argmax_stage2"},   {0, 0, "rmsnorm"},
		{0, 0, "elementwise"},		{0, 0, "embd_lookup_f16"}, {0, 0, "embd_lookup_f32"},
		{0, 0, "embd_lookup_bf16"}, {0, 0, "matmul_f32"},	   {0, 0, "matmul_iq4_nl"},
		{0, 0, "matmul_q5_k"},		{0, 0, "matmul_q6_k"},	   {0, 0, "rope"},
		{0, 0, "ffn_activate"},		{0, 0, "attention"},	   {0, 0, "rmsnorm_per_head"},
		{0, 0, "partial_rope_qk"},	{0, 0, "matmul_q4_0"},	   {0, 0, "matmul_q4_1"},
		{0, 0, "matmul_q5_0"},		{0, 0, "matmul_q5_1"},	   {0, 0, "matmul_q8_0"},
		{0, 0, "matmul_q4_k"},		{0, 0, "matmul_f16"},	   {0, 0, "matmul_bf16"},
		{0, 0, "kv_put"},			{0, 0, "rope_batch"},	   {0, 0, "ffn_activate_fused_batch"},
		{0, 0, "attention_batch"},
	};
	const char *sources[] = {
		gl_shader_argmax_stage1_src,
		gl_shader_argmax_stage2_src,
		gl_shader_rmsnorm_src,
		gl_shader_elementwise_src,
		gl_shader_embd_lookup_f16_src,
		gl_shader_embd_lookup_f32_src,
		gl_shader_embd_lookup_bf16_src,
		gl_shader_matmul_f32_src,
		gl_shader_matmul_iq4_nl_src,
		gl_shader_matmul_q5_k_src,
		gl_shader_matmul_q6_k_src,
		gl_shader_rope_src,
		gl_shader_ffn_activate_src,
		p->attention_src,
		gl_shader_rmsnorm_per_head_src,
		gl_shader_partial_rope_qk_src,
		gl_shader_matmul_q4_0_src,
		gl_shader_matmul_q4_1_src,
		gl_shader_matmul_q5_0_src,
		gl_shader_matmul_q5_1_src,
		gl_shader_matmul_q8_0_src,
		gl_shader_matmul_q4_k_src,
		gl_shader_matmul_f16_src,
		gl_shader_matmul_bf16_src,
		gl_shader_kv_put_src,
		gl_shader_rope_batch_src,
		gl_shader_ffn_activate_fused_batch_src,
		p->attention_batch_src,
	};
	_Static_assert(ARRAY_LEN(pipelines) == ARRAY_LEN(sources), "pipeline/source count mismatch");

	for (int i = 0; i < (int)ARRAY_LEN(pipelines); i++) {
		s = gl_build_pipeline(sources[i], pipelines[i].name, &pipelines[i]);
		if (s != OK) {
			for (int j = 0; j < i; j++)
				glDeleteProgram(pipelines[j].program);
			free(p->attention_src);
			free(p->attention_batch_src);
			gl_context_free(&p->ctx);
			free(p);
			self->priv = NULL;
			return s;
		}
	}
	p->p_argmax_stage1			  = pipelines[0];
	p->p_argmax_stage2			  = pipelines[1];
	p->p_rmsnorm				  = pipelines[2];
	p->p_elementwise			  = pipelines[3];
	p->p_embd_lookup_f16		  = pipelines[4];
	p->p_embd_lookup_f32		  = pipelines[5];
	p->p_embd_lookup_bf16		  = pipelines[6];
	p->p_matmul_f32				  = pipelines[7];
	p->p_matmul_iq4_nl			  = pipelines[8];
	p->p_matmul_q5_k			  = pipelines[9];
	p->p_matmul_q6_k			  = pipelines[10];
	p->p_rope					  = pipelines[11];
	p->p_ffn_activate			  = pipelines[12];
	p->p_attention				  = pipelines[13];
	p->p_rmsnorm_per_head		  = pipelines[14];
	p->p_partial_rope_qk		  = pipelines[15];
	p->p_matmul_q4_0			  = pipelines[16];
	p->p_matmul_q4_1			  = pipelines[17];
	p->p_matmul_q5_0			  = pipelines[18];
	p->p_matmul_q5_1			  = pipelines[19];
	p->p_matmul_q8_0			  = pipelines[20];
	p->p_matmul_q4_k			  = pipelines[21];
	p->p_matmul_f16				  = pipelines[22];
	p->p_matmul_bf16			  = pipelines[23];
	p->p_kv_put					  = pipelines[24];
	p->p_rope_batch				  = pipelines[25];
	p->p_ffn_activate_fused_batch = pipelines[26];
	p->p_attention_batch		  = pipelines[27];

	p->device_local_total_estimate = get_total_memory();

	log_tag("GL", "%s (max_wg=%d, shared=%dKB, ssbo_bindings=%d, attn_tile=%d)", p->ctx.renderer,
			p->ctx.max_wg_size, p->ctx.max_shared_bytes / 1024, p->ctx.max_ssbo_bindings,
			p->attention_tile_t);

	g_active_gl = p;
	return OK;
}

static void gl_free(backend *self) {
	gl_priv *p = self->priv;
	if (!p)
		return;

	if (gl_ensure_context(p) != OK)
		WARN("gl: freeing backend without a current GL context; GL objects may leak");

	glDeleteProgram(p->p_argmax_stage1.program);
	glDeleteProgram(p->p_argmax_stage2.program);
	glDeleteProgram(p->p_rmsnorm.program);
	glDeleteProgram(p->p_elementwise.program);
	glDeleteProgram(p->p_embd_lookup_f16.program);
	glDeleteProgram(p->p_embd_lookup_f32.program);
	glDeleteProgram(p->p_embd_lookup_bf16.program);
	glDeleteProgram(p->p_matmul_f32.program);
	glDeleteProgram(p->p_matmul_iq4_nl.program);
	glDeleteProgram(p->p_matmul_q4_0.program);
	glDeleteProgram(p->p_matmul_q4_1.program);
	glDeleteProgram(p->p_matmul_q5_0.program);
	glDeleteProgram(p->p_matmul_q5_1.program);
	glDeleteProgram(p->p_matmul_q8_0.program);
	glDeleteProgram(p->p_matmul_q4_k.program);
	glDeleteProgram(p->p_matmul_f16.program);
	glDeleteProgram(p->p_matmul_bf16.program);
	glDeleteProgram(p->p_matmul_q5_k.program);
	glDeleteProgram(p->p_matmul_q6_k.program);
	glDeleteProgram(p->p_rope.program);
	glDeleteProgram(p->p_ffn_activate.program);
	glDeleteProgram(p->p_attention.program);
	glDeleteProgram(p->p_rmsnorm_per_head.program);
	glDeleteProgram(p->p_partial_rope_qk.program);
	glDeleteProgram(p->p_kv_put.program);
	glDeleteProgram(p->p_rope_batch.program);
	glDeleteProgram(p->p_ffn_activate_fused_batch.program);
	glDeleteProgram(p->p_attention_batch.program);

	if (p->rope_cos_tbl.handle)
		gl_buf_free(p, as_glbuf(&p->rope_cos_tbl));
	if (p->rope_sin_tbl.handle)
		gl_buf_free(p, as_glbuf(&p->rope_sin_tbl));
	free(p->attention_src);
	free(p->attention_batch_src);
	free(p->fb_scores);
	free(p->fb_rowbuf);

	for (int i = 0; i < p->scratch_pool_count; i++)
		gl_buf_free(p, p->scratch_pool[i]);
	p->scratch_pool_count = 0;

	if (p->argmax_partial)
		gl_buf_free(p, p->argmax_partial);
	if (p->dummy)
		gl_buf_free(p, p->dummy);
	if (p->attn_scores)
		gl_buf_free(p, p->attn_scores);

	while (p->all_bufs_count > 0) {
		gl_buf *b = p->all_bufs[0];
		gl_buf_free(p, b);
	}
	free(p->all_bufs);

	if (g_active_gl == p)
		g_active_gl = NULL;

	gl_context_free(&p->ctx);

	free(p);
	self->priv = NULL;
}

static status_code gl_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	gl_priv *p	  = self->priv;
	size_t	 size = (desc->n_dims == 1) ? ggml_row_size(desc->type, desc->dims[0])
										: ggml_row_size(desc->type, desc->dims[0]) * desc->dims[1];

	gl_buf *b = gl_buf_alloc(p, size, GL_STATIC_DRAW, desc->host_data);
	if (!b)
		return ERR_OUT_OF_MEMORY;

	out->handle	  = b->host_mirror;
	out->size	  = size;
	out->host_ptr = desc->host_data;
	out->offset	  = 0;
	out->owner	  = self;
	return OK;
}

static status_code gl_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	gl_priv *p = self->priv;

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
	b = gl_buf_alloc(p, size, GL_DYNAMIC_DRAW, NULL);
	if (!b)
		return ERR_OUT_OF_MEMORY;
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
	gl_buf	*b = gl_buf_alloc(p, size, GL_DYNAMIC_COPY, host_data);
	if (!b)
		return ERR_OUT_OF_MEMORY;
	out->handle	  = b->host_mirror;
	out->size	  = size;
	out->host_ptr = host_data;
	out->offset	  = 0;
	out->owner	  = self;
	return OK;
}

static void gl_buffer_free(backend *self, buffer *buf) {
	gl_priv *p = self->priv;
	if (!buf || !buf->handle)
		return;
	gl_buf *b = as_glbuf(buf);

	if (b->usage == GL_DYNAMIC_COPY)
		gl_scratch_pool_release(p, b);
	else
		gl_buf_free(p, b);

	buf->handle	  = NULL;
	buf->size	  = 0;
	buf->host_ptr = NULL;
	buf->offset	  = 0;
}

static status_code gl_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n) {
	(void)self;
	gl_buf *b	  = as_glbuf(buf);
	size_t	bytes = (size_t)n * sizeof(float);
	gl_buf_sync_to_host(b);
	memcpy(host_dst, (char *)b->host_mirror + buf->offset, bytes);
	return OK;
}

static status_code gl_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n) {
	(void)self;
	gl_buf *b	  = as_glbuf(buf);
	size_t	bytes = (size_t)n * sizeof(float);
	memcpy((char *)b->host_mirror + buf->offset, host_src, bytes);
	b->host_dirty	= 1;
	b->device_dirty = 0;
	return OK;
}

static status_code gl_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	(void)self;
	gl_buf *sb	  = as_glbuf(src);
	gl_buf *db	  = as_glbuf(dst);
	size_t	bytes = (size_t)n * sizeof(float);
	gl_buf_sync_to_host(sb);
	memcpy((char *)db->host_mirror + dst->offset, (char *)sb->host_mirror + src->offset, bytes);
	db->host_dirty	 = 1;
	db->device_dirty = 0;
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

static void gl_synchronize(backend *self) {
	gl_priv *p = self->priv;
	if (!p->ctx.dpy)
		return;
	if (gl_ensure_context(p) != OK)
		return;
	glFinish();
	GLenum err = glGetError();
	if (err != GL_NO_ERROR && !p->device_lost) {
		WARN("gl: glFinish returned glErr=0x%04x; marking device as lost", (unsigned)err);
		p->device_lost = 1;
		return;
	}
	for (int i = 0; i < p->all_bufs_count; i++)
		gl_buf_sync_to_host(p->all_bufs[i]);
}

static void gl_begin_batch(backend *self) {
	(void)self;
}

static void gl_end_batch(backend *self) {
	(void)self;
}

static status_code gl_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	gl_priv *p = self->priv;

	const int ARGMAX_WG_CHUNK	= 8192;
	const int ARGMAX_MAX_GROUPS = 256;
	int		  groups			= (n + ARGMAX_WG_CHUNK - 1) / ARGMAX_WG_CHUNK;
	if (groups > ARGMAX_MAX_GROUPS)
		groups = ARGMAX_MAX_GROUPS;
	if (groups < 1)
		groups = 1;

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
	status_code		   s = gl_dispatch_all_dirty(p, &p->p_argmax_stage1, s1_bufs, 2, s1_unames,
												 s1_uniforms, 2, (GLuint)groups, 1, 1, 1);
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

static status_code gl_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
								  int token, int dim, buffer *x_out) {
	gl_priv		*p = self->priv;
	gl_pipeline *pipe;
	switch (tok_embd_type) {
	case GGML_TYPE_F32:
		pipe = &p->p_embd_lookup_f32;
		break;
	case GGML_TYPE_F16:
		pipe = &p->p_embd_lookup_f16;
		break;
	case GGML_TYPE_BF16:
		pipe = &p->p_embd_lookup_bf16;
		break;
	default: {
		gl_buf *eb = as_glbuf(tok_embd);
		gl_buf *ob = as_glbuf(x_out);
		if (!eb || !ob)
			return ERR_INVALID_ARG;
		gl_buf_sync_to_host(eb);
		gl_buf_sync_to_host(ob);
		size_t		   row_stride = ggml_row_size(tok_embd_type, (size_t)dim);
		const uint8_t *src =
			(const uint8_t *)eb->host_mirror + tok_embd->offset + (size_t)token * row_stride;
		float	*dst  = (float *)((uint8_t *)ob->host_mirror + x_out->offset);
		backend *host = backend_host();
		if (!host || !host->dequant_row)
			return ERR_UNSUPPORTED;
		status_code dq = host->dequant_row(host, tok_embd_type, src, dim, dst);
		if (dq != OK)
			return dq;
		ob->host_dirty	 = 1;
		ob->device_dirty = 0;
		return OK;
	}
	}

	size_t	row_stride = ggml_row_size(tok_embd_type, dim);
	gl_buf *bufs[2]	   = {as_glbuf(tok_embd), as_glbuf(x_out)};
	gl_buf_sync_to_device(bufs[0]);
	gl_buf_sync_to_device(bufs[1]);

	int				   uniforms[3] = {token, dim, (int)row_stride};
	static const char *unames[3]   = {"u_token", "u_dim", "u_row_stride"};
	GLuint			   groups	   = (GLuint)((dim + 63) / 64);
	status_code		   s = gl_dispatch(p, pipe, bufs, 2, unames, uniforms, 3, groups, 1, 1);
	if (s == OK)
		bufs[1]->device_dirty = 1;
	return s;
}

static status_code gl_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							  float eps) {
	gl_priv *p		 = self->priv;
	gl_buf	*bufs[3] = {as_glbuf(x), as_glbuf(w), as_glbuf(y)};
	gl_buf_sync_to_device(bufs[0]);
	gl_buf_sync_to_device(bufs[1]);
	gl_buf_sync_to_device(bufs[2]);

	if (n <= 0)
		return OK;

	glUseProgram(p->p_rmsnorm.program);
	gl_bind_ssbo(0, x);
	gl_bind_ssbo(1, w);
	gl_bind_ssbo(2, y);
	gl_pu(&p->p_rmsnorm, "u_n", n);
	gl_puf(&p->p_rmsnorm, "u_eps", eps);

	GLuint groups = 1;
	glDispatchCompute(groups, 1, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: rmsnorm dispatch failed (glErr=0x%04x)", (unsigned)err);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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
									  int rows) {
	if (n_heads <= 0 || head_dim <= 0 || rows <= 0)
		return OK;
	if (!p->p_rmsnorm_per_head.program)
		return ERR_UNSUPPORTED;
	if (has_weight && !w)
		return ERR_INVALID_ARG;

	gl_buf *xb = as_glbuf(x);
	gl_buf *yb = as_glbuf(y);
	gl_buf *wb = has_weight ? as_glbuf(w) : NULL;

	if (!has_weight) {
		status_code s = gl_ensure_dummy(p);
		if (s != OK)
			return s;
	}

	gl_buf_sync_to_device(xb);
	gl_buf_sync_to_device(yb);
	if (wb)
		gl_buf_sync_to_device(wb);

	glUseProgram(p->p_rmsnorm_per_head.program);
	gl_bind_ssbo(0, x);
	if (wb)
		gl_bind_ssbo(1, w);
	else
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, p->dummy->name);
	gl_bind_ssbo(2, y);
	gl_pu(&p->p_rmsnorm_per_head, "u_n_heads", n_heads);
	gl_pu(&p->p_rmsnorm_per_head, "u_head_dim", head_dim);
	gl_puf(&p->p_rmsnorm_per_head, "u_eps", eps);
	gl_pu(&p->p_rmsnorm_per_head, "u_has_weight", has_weight);
	gl_pu(&p->p_rmsnorm_per_head, "u_rows", rows);

	glDispatchCompute((GLuint)n_heads, (GLuint)rows, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: rmsnorm_per_head dispatch failed (glErr=0x%04x, heads=%d, hd=%d, rows=%d)",
			  (unsigned)err, n_heads, head_dim, rows);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	yb->device_dirty = 1;
	return OK;
}

static status_code gl_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
									   int n_heads, int head_dim, float eps) {
	return gl_rmsnorm_ph_impl(self->priv, x, w, y, n_heads, head_dim, eps, 1, 1);
}

static status_code gl_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n,
									   float eps) {
	return gl_rmsnorm_ph_impl(self->priv, x, NULL, y, 1, n, eps, 0, 1);
}

static status_code gl_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y,
												int n_heads, int head_dim, float eps) {
	return gl_rmsnorm_ph_impl(self->priv, x, NULL, y, n_heads, head_dim, eps, 0, 1);
}

static status_code gl_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
								  const buffer *residual, buffer *y, int n, float eps,
								  float out_scale) {
	status_code s = gl_rmsnorm(self, x, w, y, n, eps);
	if (s != OK)
		return s;
	s = gl_elementwise(self->priv, y, residual, NULL, n, 1, 0.0f, 0, 1);
	if (s != OK)
		return s;
	if (out_scale != 1.0f)
		return gl_scale_inplace(self, y, out_scale, n);
	return OK;
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

	glUseProgram(p->p_elementwise.program);
	gl_bind_ssbo(0, x);
	if (y)
		gl_bind_ssbo(1, y);
	else
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dummy->name);
	if (z)
		gl_bind_ssbo(2, z);
	else
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, dummy->name);
	gl_pu(&p->p_elementwise, "u_n", n);
	gl_pu(&p->p_elementwise, "u_mode", mode);
	gl_puf(&p->p_elementwise, "u_scale", scale);
	gl_pu(&p->p_elementwise, "u_aux", aux);
	gl_pu(&p->p_elementwise, "u_m", rows);

	GLuint groups = (GLuint)((n + 127) / 128);
	GLuint gy	  = (GLuint)MIN(rows, 65535);
	GLuint gz	  = (GLuint)((rows + 65534) / 65535);
	glDispatchCompute(groups, gy, gz);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: elementwise dispatch failed (glErr=0x%04x, n=%d, groups=%u, mode=%d)",
			  (unsigned)err, n, groups, mode);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	bufs[0]->device_dirty = 1;
	if (z)
		bufs[2]->device_dirty = 1;
	return OK;
}

static status_code gl_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	status_code s = gl_elementwise(self->priv, x, y, NULL, n, 1, 0.0f, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: add_inplace: routing to CPU due to bad buffer");
		backend *host = backend_host();
		if (host && host->add_inplace)
			return host->add_inplace(host, x, y, n);
	}
	return s;
}

static status_code gl_scale_inplace(backend *self, buffer *x, float scale, int n) {
	status_code s = gl_elementwise(self->priv, x, NULL, NULL, n, 2, scale, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: scale_inplace: routing to CPU due to bad buffer");
		backend *host = backend_host();
		if (host && host->scale_inplace)
			return host->scale_inplace(host, x, scale, n);
	}
	return s;
}

static status_code gl_softcap(backend *self, buffer *x, float cap, int n) {
	if (cap <= 0.0f || n <= 0)
		return OK;
	status_code s = gl_elementwise(self->priv, x, NULL, NULL, n, 4, cap, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: softcap: routing to CPU due to bad buffer");
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
	status_code s =
		gl_elementwise(self->priv, q, mixed, gate, n_heads * head_dim, 5, 0.0f, head_dim, n_rows);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: split_qgate: routing to CPU due to bad buffer");
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
	status_code s = gl_elementwise(self->priv, out, gate, NULL, n, 6, 0.0f, 0, n_rows);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: attn_output_gate: routing to CPU due to bad buffer");
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
	status_code s = gl_elementwise(self->priv, ple, proj, NULL, n, 7, combine_scale, 0, 1);
	if (s == ERR_INVALID_ARG) {
		WARN("gl: ple_combine: routing to CPU due to bad buffer");
		backend *host = backend_host();
		if (host && host->ple_combine)
			return host->ple_combine(host, ple, proj, n, combine_scale);
	}
	return s;
}

static status_code gl_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n) {
	gl_priv *p		 = self->priv;
	gl_buf	*bufs[3] = {as_glbuf(gate), as_glbuf(up), as_glbuf(out)};
	for (int i = 0; i < 3; i++)
		gl_buf_sync_to_device(bufs[i]);

	if (n <= 0)
		return OK;

	glUseProgram(p->p_ffn_activate.program);
	gl_bind_ssbo(0, gate);
	gl_bind_ssbo(1, up);
	gl_bind_ssbo(2, out);
	gl_pu(&p->p_ffn_activate, "u_n", n);
	gl_pu(&p->p_ffn_activate, "u_act", 0);
	GLuint groups = (GLuint)((n + 127) / 128);
	glDispatchCompute(groups, 1, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: ffn_activate dispatch failed (glErr=0x%04x)", (unsigned)err);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	bufs[2]->device_dirty = 1;
	return OK;
}

static status_code gl_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
									  buffer *out, int n, int activation) {
	gl_priv *p		 = self->priv;
	gl_buf	*bufs[3] = {as_glbuf(gate), as_glbuf(up), as_glbuf(out)};

	for (int i = 0; i < 3; i++) {
		if (!bufs[i] || bufs[i]->size_bytes == 0 || bufs[i]->size_bytes > (size_t)(1ull << 40)) {
			WARN("gl: ffn_activate_ex: bad buffer %d, routing to CPU", i);
			backend *host = backend_host();
			if (host && host->ffn_activate_ex)
				return host->ffn_activate_ex(host, gate, up, out, n, activation);
			return ERR_UNSUPPORTED;
		}
	}

	for (int i = 0; i < 3; i++)
		gl_buf_sync_to_device(bufs[i]);

	if (n <= 0)
		return OK;

	glUseProgram(p->p_ffn_activate.program);
	gl_bind_ssbo(0, gate);
	gl_bind_ssbo(1, up);
	gl_bind_ssbo(2, out);
	gl_pu(&p->p_ffn_activate, "u_n", n);
	gl_pu(&p->p_ffn_activate, "u_act", activation);
	GLuint groups = (GLuint)((n + 127) / 128);
	glDispatchCompute(groups, 1, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: ffn_activate_ex dispatch failed (glErr=0x%04x, act=%d)", (unsigned)err,
			  activation);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	bufs[2]->device_dirty = 1;
	return OK;
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

	glUseProgram(p->p_rope.program);
	gl_bind_ssbo(0, vec);
	gl_bind_ssbo(1, &cos_b);
	gl_bind_ssbo(2, &sin_b);
	gl_pu(&p->p_rope, "u_n_heads", n_heads);
	gl_pu(&p->p_rope, "u_head_dim", head_dim);
	gl_pu(&p->p_rope, "u_neox", self->rope_neox);

	GLuint groups = (GLuint)((total + 127) / 128);
	glDispatchCompute(groups, 1, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: rope dispatch failed (glErr=0x%04x)", (unsigned)err);
		p->device_lost = 1;
		self->buffer_free(self, &cos_b);
		self->buffer_free(self, &sin_b);
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	bufs[0]->device_dirty = 1;

	self->buffer_free(self, &cos_b);
	self->buffer_free(self, &sin_b);
	return OK;
}

static status_code gl_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						   const float *rope_cos_base, const float *rope_sin_base) {
	int half = head_dim / 2;
	if (n_heads <= 0 || half <= 0)
		return OK;
	return gl_rope_apply(self, vec, n_heads, head_dim, rope_cos_base + (size_t)pos * half,
						 rope_sin_base + (size_t)pos * half);
}

static status_code gl_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
							  int head_dim, int pos, const float *rope_cos_base,
							  const float *rope_sin_base) {
	status_code s = gl_rope(self, q, n_heads, head_dim, pos, rope_cos_base, rope_sin_base);
	if (s != OK)
		return s;
	return gl_rope(self, k, n_kv_heads, head_dim, pos, rope_cos_base, rope_sin_base);
}

static status_code gl_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
							   const float *rope_cos_base, const float *rope_sin_base,
							   const float *freq_factors) {
	if (!freq_factors)
		return gl_rope(self, vec, n_heads, head_dim, pos, rope_cos_base, rope_sin_base);

	int half = head_dim / 2;
	if (n_heads <= 0 || half <= 0)
		return OK;

	float *cs = xmalloc((size_t)half * 2 * sizeof(float));
	if (!cs)
		return ERR_OUT_OF_MEMORY;
	float *cos_row = cs;
	float *sin_row = cs + half;

	double theta_scale = pow((double)self->rope_theta, -2.0 / (double)head_dim);
	double base_freq   = 1.0;
	for (int j = 0; j < half; j++) {
		double ff;
		float  fv = freq_factors[j];
		ff		  = (fv >= 1e10f || fv == 0.0f) ? 0.0 : (double)fv;
		if (ff > 0.0) {
			double angle = base_freq * ff * (double)pos;
			cos_row[j]	 = (float)cos(angle);
			sin_row[j]	 = (float)sin(angle);
		} else {
			cos_row[j] = 1.0f;
			sin_row[j] = 0.0f;
		}
		base_freq *= theta_scale;
	}

	status_code s = gl_rope_apply(self, vec, n_heads, head_dim, cos_row, sin_row);
	free(cs);
	return s;
}

static status_code gl_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
									  int n_kv_heads, int head_dim, int rope_dim, int pos_start,
									  const float *rope_cos_base, const float *rope_sin_base,
									  int n_rows) {
	gl_priv *p = self->priv;
	if (n_rows <= 0 || rope_dim <= 0)
		return OK;

	int half = rope_dim / 2;
	if (half <= 0 || n_heads <= 0 || n_kv_heads <= 0)
		return OK;
	if (!p->p_partial_rope_qk.program)
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

	glUseProgram(p->p_partial_rope_qk.program);
	gl_bind_ssbo(0, q);
	gl_bind_ssbo(1, k);
	gl_bind_ssbo(2, &cos_b);
	gl_bind_ssbo(3, &sin_b);
	gl_pu(&p->p_partial_rope_qk, "u_n_heads", n_heads);
	gl_pu(&p->p_partial_rope_qk, "u_n_kv_heads", n_kv_heads);
	gl_pu(&p->p_partial_rope_qk, "u_head_dim", head_dim);
	gl_pu(&p->p_partial_rope_qk, "u_rope_dim", rope_dim);
	gl_pu(&p->p_partial_rope_qk, "u_pos0", pos_start);
	gl_pu(&p->p_partial_rope_qk, "u_n_rows", n_rows);

	int	   total_heads = n_heads + n_kv_heads;
	GLuint gx		   = (GLuint)((total_heads + 63) / 64);
	GLuint gy		   = (GLuint)MIN(n_rows, 65535);
	GLuint gz		   = (GLuint)((n_rows + 65534) / 65535);
	glDispatchCompute(gx, gy, gz);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: partial_rope_qk dispatch failed (glErr=0x%04x, rows=%d, heads=%d, "
			  "head_dim=%d, rope_dim=%d)",
			  (unsigned)err, n_rows, total_heads, head_dim, rope_dim);
		p->device_lost = 1;
		self->buffer_free(self, &cos_b);
		self->buffer_free(self, &sin_b);
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	qb->device_dirty = 1;
	kb->device_dirty = 1;

	self->buffer_free(self, &cos_b);
	self->buffer_free(self, &sin_b);
	return OK;
}

static int gl_matmul_type_native(backend *self, uint32_t w_type) {
	(void)self;
	return w_type == GGML_TYPE_F32 || w_type == GGML_TYPE_F16 || w_type == GGML_TYPE_BF16 ||
		   w_type == GGML_TYPE_IQ4_NL || w_type == GGML_TYPE_Q4_0 || w_type == GGML_TYPE_Q4_1 ||
		   w_type == GGML_TYPE_Q5_0 || w_type == GGML_TYPE_Q5_1 || w_type == GGML_TYPE_Q8_0 ||
		   w_type == GGML_TYPE_Q4_K || w_type == GGML_TYPE_Q5_K || w_type == GGML_TYPE_Q6_K;
}

static int gl_matmul_pipe_for_type(gl_priv *p, uint32_t w_type, gl_pipeline **out_pipe,
								   int *out_block_size) {
	gl_pipeline *pipe		= NULL;
	int			 block_size = 0;
	switch (w_type) {
	case GGML_TYPE_F32:
		pipe = &p->p_matmul_f32;
		break;
	case GGML_TYPE_F16:
		pipe = &p->p_matmul_f16;
		break;
	case GGML_TYPE_BF16:
		pipe = &p->p_matmul_bf16;
		break;
	case GGML_TYPE_IQ4_NL:
		pipe	   = &p->p_matmul_iq4_nl;
		block_size = 32;
		break;
	case GGML_TYPE_Q4_0:
		pipe	   = &p->p_matmul_q4_0;
		block_size = 32;
		break;
	case GGML_TYPE_Q4_1:
		pipe	   = &p->p_matmul_q4_1;
		block_size = 32;
		break;
	case GGML_TYPE_Q5_0:
		pipe	   = &p->p_matmul_q5_0;
		block_size = 32;
		break;
	case GGML_TYPE_Q5_1:
		pipe	   = &p->p_matmul_q5_1;
		block_size = 32;
		break;
	case GGML_TYPE_Q8_0:
		pipe	   = &p->p_matmul_q8_0;
		block_size = 32;
		break;
	case GGML_TYPE_Q4_K:
		pipe	   = &p->p_matmul_q4_k;
		block_size = 256;
		break;
	case GGML_TYPE_Q5_K:
		pipe	   = &p->p_matmul_q5_k;
		block_size = 256;
		break;
	case GGML_TYPE_Q6_K:
		pipe	   = &p->p_matmul_q6_k;
		block_size = 256;
		break;
	default:
		return 0;
	}
	if (!pipe->program)
		return 0;
	*out_pipe		= pipe;
	*out_block_size = block_size;
	return 1;
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

static status_code gl_matmul_dispatch(gl_priv *p, gl_pipeline *pipe, int block_size,
									  const buffer *w, const buffer *x, buffer *y,
									  const buffer *residual, int n, int k) {
	if (n <= 0 || k <= 0)
		return OK;

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

	glUseProgram(pipe->program);
	gl_bind_ssbo(0, w);
	gl_bind_ssbo(1, x);
	gl_bind_ssbo(2, y);
	if (mode == 1)
		gl_bind_ssbo(3, residual);
	else
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, p->dummy->name);
	gl_pu(pipe, "u_n", n);

	if (block_size == 0) {
		gl_pu(pipe, "u_k", k);
	} else {
		int k_blocks = k / block_size;
		gl_pu(pipe, "u_k_blocks", k_blocks);
	}
	gl_pu(pipe, "u_add_residual", mode);

	int	   rows_per_wg = (pipe == &p->p_matmul_f32 || pipe == &p->p_matmul_iq4_nl) ? 4 : 1;
	int	   total_wg	   = (n + rows_per_wg - 1) / rows_per_wg;
	GLuint gx, gy;
	if ((GLuint)total_wg <= GL_MAX_DISPATCH_PER_DIM) {
		gx = (GLuint)total_wg;
		gy = 1;
	} else {
		gx = GL_MAX_DISPATCH_PER_DIM;
		gy = ((GLuint)total_wg + GL_MAX_DISPATCH_PER_DIM - 1) / GL_MAX_DISPATCH_PER_DIM;
	}
	glDispatchCompute(gx, gy, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: matmul dispatch failed (glErr=0x%04x, n=%d, k=%d, mode=%d)", (unsigned)err, n, k,
			  mode);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	bufs[2]->device_dirty = 1;
	return OK;
}

static status_code gl_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							 buffer *y, int n, int k) {
	gl_priv *p = self->priv;
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
		status_code s = gl_matmul_dispatch(p, pipe, block_size, w, x, y, NULL, n, k);
		if (s == ERR_UNSUPPORTED)
			goto host_fallback;
		return s;
	}

host_fallback:
	gl_buf_sync_to_host(as_glbuf(w));
	gl_buf_sync_to_host(as_glbuf(x));
	gl_buf_sync_to_host(as_glbuf(y));
	{
		backend *host = backend_host();
		if (host && host->matmul)
			return host->matmul(host, w, w_type, x, y, n, k);
		return ERR_UNSUPPORTED;
	}
}

static status_code gl_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								   buffer *y, int n, int k, int m) {
	gl_priv *p = self->priv;
	if (n <= 0 || k <= 0 || m <= 0)
		return OK;
	if (m == 1)
		return gl_matmul(self, w, w_type, x, y, n, k);

	gl_pipeline *pipe		= NULL;
	int			 block_size = 0;
	switch (w_type) {
	case GGML_TYPE_F32:
		pipe = &p->p_matmul_f32;
		break;
	case GGML_TYPE_F16:
		pipe = &p->p_matmul_f16;
		break;
	case GGML_TYPE_BF16:
		pipe = &p->p_matmul_bf16;
		break;
	case GGML_TYPE_IQ4_NL:
		pipe	   = &p->p_matmul_iq4_nl;
		block_size = 32;
		break;
	case GGML_TYPE_Q4_0:
		pipe	   = &p->p_matmul_q4_0;
		block_size = 32;
		break;
	case GGML_TYPE_Q4_1:
		pipe	   = &p->p_matmul_q4_1;
		block_size = 32;
		break;
	case GGML_TYPE_Q5_0:
		pipe	   = &p->p_matmul_q5_0;
		block_size = 32;
		break;
	case GGML_TYPE_Q5_1:
		pipe	   = &p->p_matmul_q5_1;
		block_size = 32;
		break;
	case GGML_TYPE_Q8_0:
		pipe	   = &p->p_matmul_q8_0;
		block_size = 32;
		break;
	case GGML_TYPE_Q4_K:
		pipe	   = &p->p_matmul_q4_k;
		block_size = 256;
		break;
	case GGML_TYPE_Q5_K:
		pipe	   = &p->p_matmul_q5_k;
		block_size = 256;
		break;
	case GGML_TYPE_Q6_K:
		pipe	   = &p->p_matmul_q6_k;
		block_size = 256;
		break;
	default:
		goto host_fallback;
	}
	if (!pipe->program)
		goto host_fallback;
	if ((size_t)m * (size_t)k * sizeof(float) > x->size ||
		(size_t)m * (size_t)n * sizeof(float) > y->size)
		goto host_fallback;
	if (w_type == GGML_TYPE_F32 && (k & 3) != 0)
		goto host_fallback;
	{
		GLint ssbo_align = 0;
		glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &ssbo_align);
		size_t align = (size_t)(ssbo_align > 0 ? ssbo_align : 1);
		if ((x->offset % align) != 0 || (y->offset % align) != 0 ||
			((size_t)k * sizeof(float)) % align != 0 || ((size_t)n * sizeof(float)) % align != 0)
			goto host_fallback;
	}

	{
		gl_buf *wgb		= as_glbuf(w);
		gl_buf *xgb		= as_glbuf(x);
		gl_buf *ygb		= as_glbuf(y);
		gl_buf *bufs[3] = {wgb, xgb, ygb};

		for (int i = 0; i < 3; i++) {
			if (!bufs[i] || bufs[i]->size_bytes == 0 || bufs[i]->size_bytes > (size_t)(1ull << 40))
				goto host_fallback;
		}

		gl_buf_sync_to_device(bufs[0]);
		gl_buf_sync_to_device(bufs[1]);
		gl_buf_sync_to_device(bufs[2]);

		glUseProgram(pipe->program);
		gl_bind_ssbo(0, w);
		glUniform1i(0, n);

		if (block_size == 0) {
			glUniform1i(1, k);
		} else {
			int k_blocks = k / block_size;
			glUniform1i(1, k_blocks);
		}
		glUniform1i(2, 0);
		if (gl_ensure_dummy(p) != OK)
			goto host_fallback;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, p->dummy->name);

		int	   rows_per_wg = (w_type == GGML_TYPE_F32 || w_type == GGML_TYPE_IQ4_NL) ? 4 : 1;
		int	   total_wg	   = (n + rows_per_wg - 1) / rows_per_wg;
		GLuint gx, gy;
		if ((GLuint)total_wg <= GL_MAX_DISPATCH_PER_DIM) {
			gx = (GLuint)total_wg;
			gy = 1;
		} else {
			gx = GL_MAX_DISPATCH_PER_DIM;
			gy = ((GLuint)total_wg + GL_MAX_DISPATCH_PER_DIM - 1) / GL_MAX_DISPATCH_PER_DIM;
		}
		for (int i = 0; i < m; i++) {
			buffer xv =
				buffer_slice(x, (size_t)i * (size_t)k * sizeof(float), (size_t)k * sizeof(float));
			buffer yv =
				buffer_slice(y, (size_t)i * (size_t)n * sizeof(float), (size_t)n * sizeof(float));
			gl_bind_ssbo(1, &xv);
			gl_bind_ssbo(2, &yv);
			glDispatchCompute(gx, gy, 1);
		}
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			ERROR("gl: matmul_batch dispatch failed (glErr=0x%04x, type=%u, m=%d)", (unsigned)err,
				  w_type, m);
			p->device_lost = 1;
			return ERR_INTERNAL;
		}
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		ygb->device_dirty = 1;
		return OK;
	}

host_fallback:
	gl_buf_sync_to_host(as_glbuf(w));
	gl_buf_sync_to_host(as_glbuf(x));
	gl_buf_sync_to_host(as_glbuf(y));
	{
		backend *host = backend_host();
		if (host && host->matmul_batch)
			return host->matmul_batch(host, w, w_type, x, y, n, k, m);
		return ERR_UNSUPPORTED;
	}
}

static status_code gl_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *x, const buffer *residual, buffer *y, int n,
									  int k) {
	gl_priv *p = self->priv;
	if (n <= 0 || k <= 0)
		return OK;

	status_code	 s			= ERR_UNSUPPORTED;
	gl_pipeline *pipe		= NULL;
	int			 block_size = 0;
	if (gl_matmul_pipe_for_type(p, w_type, &pipe, &block_size) && gl_matmul_buf_ok(w) &&
		gl_matmul_buf_ok(x) && gl_matmul_buf_ok(y) && gl_matmul_buf_ok(residual) &&
		(w_type != GGML_TYPE_F32 || gl_matmul_f32_align_ok(x, y, residual, k)) &&
		(block_size == 0 || (k % block_size) == 0))
		s = gl_matmul_dispatch(p, pipe, block_size, w, x, y, residual, n, k);

	if (s != ERR_UNSUPPORTED)
		return s;

	gl_buf_sync_to_host(as_glbuf(w));
	gl_buf_sync_to_host(as_glbuf(x));
	gl_buf_sync_to_host(as_glbuf(residual));
	gl_buf_sync_to_host(as_glbuf(y));
	backend *host = backend_host();
	if (host && host->matmul_residual)
		return host->matmul_residual(host, w, w_type, x, residual, y, n, k);
	return ERR_UNSUPPORTED;
}

#define GL_MATMUL_MULTI_MAX 8

static status_code gl_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
								   int n_matmuls) {
	gl_priv *p = self->priv;
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

		for (int i = 0; i < n_matmuls; i++) {
			if (n_list[i] <= 0)
				continue;
			status_code s =
				gl_matmul_dispatch(p, pipes[i], block_sizes[i], w[i], x, y[i], NULL, n_list[i], k);
			if (s == ERR_UNSUPPORTED)
				goto host_fallback;
			if (s != OK)
				return s;
		}
		return OK;
	}

host_fallback:
	for (int i = 0; i < n_matmuls; i++) {
		gl_buf_sync_to_host(as_glbuf(w[i]));
		gl_buf_sync_to_host(as_glbuf(y[i]));
	}
	gl_buf_sync_to_host(as_glbuf(x));
	{
		backend *host = backend_host();
		if (host && host->matmul_multi)
			return host->matmul_multi(host, w, w_types, x, y, n_list, k, n_matmuls);
		return ERR_UNSUPPORTED;
	}
}

static status_code gl_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	gl_priv *p			 = self->priv;
	int		 n_kv_layers = desc->n_kv_layers > 0 ? desc->n_kv_layers : 1;
	int		 n_kv_heads	 = desc->n_kv_heads;
	int		 head_dim	 = desc->head_dim;
	int		 n_ctx		 = desc->n_ctx;

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

	gl_buf *kb = gl_buf_alloc(p, total_bytes, GL_DYNAMIC_COPY, NULL);
	if (!kb)
		return ERR_OUT_OF_MEMORY;
	gl_buf *vb = gl_buf_alloc(p, total_bytes, GL_DYNAMIC_COPY, NULL);
	if (!vb) {
		gl_buf_free(p, kb);
		return ERR_OUT_OF_MEMORY;
	}

	p->kv_head_dim	 = head_dim;
	p->kv_n_kv_heads = n_kv_heads;
	p->kv_n_ctx		 = n_ctx;
	p->kv_n_layers	 = n_kv_layers;

	k_out->handle	= kb->host_mirror;
	k_out->size		= total_bytes;
	k_out->host_ptr = NULL;
	k_out->offset	= 0;
	k_out->owner	= self;

	v_out->handle	= vb->host_mirror;
	v_out->size		= total_bytes;
	v_out->host_ptr = NULL;
	v_out->offset	= 0;
	v_out->owner	= self;
	return OK;
}

static void gl_kv_free(backend *self, buffer *k, buffer *v) {
	gl_priv *p = self->priv;
	if (k && k->handle) {
		gl_buf_free(p, as_glbuf(k));
		k->handle	= NULL;
		k->size		= 0;
		k->host_ptr = NULL;
		k->offset	= 0;
	}
	if (v && v->handle) {
		gl_buf_free(p, as_glbuf(v));
		v->handle	= NULL;
		v->size		= 0;
		v->host_ptr = NULL;
		v->offset	= 0;
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
								  int m) {
	gl_priv *p		  = self->priv;
	int		 n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	if (m <= 0 || n_active <= 0 || head_dim <= 0)
		return OK;
	if (!p->p_kv_put.program)
		return ERR_UNSUPPORTED;

	gl_buf *kin = as_glbuf(k_in);
	gl_buf *vin = as_glbuf(v_in);
	gl_buf *kb	= as_glbuf(k);
	gl_buf *vb	= as_glbuf(v);

	gl_buf_sync_to_device(kin);
	gl_buf_sync_to_device(vin);
	gl_buf_sync_to_device(kb);
	gl_buf_sync_to_device(vb);

	int kv_q8 = (p->kv_quant == KV_QUANT_Q8_0);

	size_t layer_stride = p->kv_layer_stride ? p->kv_layer_stride
											 : (kv_q8 ? (size_t)n_kv_heads * n_ctx * p->kv_row_bytes
													  : (size_t)n_kv_heads * n_ctx * head_dim);
	size_t kvh_stride = p->kv_kvh_stride
							? p->kv_kvh_stride
							: (kv_q8 ? (size_t)n_ctx * p->kv_row_bytes : (size_t)n_ctx * head_dim);
	int	   row_stride_u = kv_q8 ? (int)p->kv_row_bytes : head_dim;
	int	   layer_off	= (int)((size_t)layer * layer_stride);

	glUseProgram(p->p_kv_put.program);
	gl_bind_ssbo(0, k_in);
	gl_bind_ssbo(1, v_in);
	gl_bind_ssbo(2, k);
	gl_bind_ssbo(3, v);
	gl_pu(&p->p_kv_put, "u_layer_off", layer_off);
	gl_pu(&p->p_kv_put, "u_pos_start", pos_start);
	gl_pu(&p->p_kv_put, "u_kvh_stride", (int)kvh_stride);
	gl_pu(&p->p_kv_put, "u_row_stride", row_stride_u);
	gl_pu(&p->p_kv_put, "u_head_dim", head_dim);
	gl_pu(&p->p_kv_put, "u_kv_q8", kv_q8 ? 1 : 0);
	gl_pu(&p->p_kv_put, "u_in_row_stride", in_row_stride);

	glDispatchCompute((GLuint)m, (GLuint)n_active, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: kv_put dispatch failed (glErr=0x%04x, m=%d, n_active=%d, layer=%d)",
			  (unsigned)err, m, n_active, layer);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	kb->device_dirty = 1;
	vb->device_dirty = 1;
	kb->host_dirty	 = 0;
	vb->host_dirty	 = 0;
	return OK;
}

static status_code gl_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
							 const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
							 int n_ctx, int n_kv_heads_active) {
	return gl_kv_put_impl(self, k, v, layer, pos, k_in, v_in, 0, n_kv_heads, head_dim, n_ctx,
						  n_kv_heads_active, 1);
}

static status_code gl_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								   const buffer *k_in, const buffer *v_in, int in_row_stride,
								   int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								   int m) {
	return gl_kv_put_impl(self, k, v, layer, pos_start, k_in, v_in, in_row_stride, n_kv_heads,
						  head_dim, n_ctx, n_kv_heads_active, m);
}

static status_code gl_attention_host_fallback(backend *self, const buffer *q, const buffer *k_cache,
											  const buffer *v_cache, buffer *out, int layer,
											  int pos, int n_heads, int n_kv_heads, int n_active,
											  int head_dim, int n_ctx, float scale,
											  int attn_start) {
	gl_priv *p = self->priv;

	gl_buf_sync_to_host(as_glbuf(q));
	gl_buf_sync_to_host(as_glbuf(k_cache));
	gl_buf_sync_to_host(as_glbuf(v_cache));

	int n_pos = pos + 1 - attn_start;
	if (n_pos <= 0)
		return ERR_INVALID_ARG;

	size_t layer_stride =
		p->kv_layer_stride ? p->kv_layer_stride : (size_t)n_kv_heads * n_ctx * head_dim;
	size_t kvh_stride = p->kv_kvh_stride ? p->kv_kvh_stride : (size_t)n_ctx * head_dim;

	const uint16_t *kd_base = (const uint16_t *)as_glbuf(k_cache)->host_mirror +
							  k_cache->offset / sizeof(uint16_t) + (size_t)layer * layer_stride;
	const uint16_t *vd_base = (const uint16_t *)as_glbuf(v_cache)->host_mirror +
							  v_cache->offset / sizeof(uint16_t) + (size_t)layer * layer_stride;
	const float	   *qf		= (const float *)((char *)as_glbuf(q)->host_mirror + q->offset);
	float		   *outf	= (float *)((char *)as_glbuf(out)->host_mirror + out->offset);

	int n_groups = (n_heads + n_active - 1) / n_active;
	if (n_groups <= 0)
		return ERR_INVALID_ARG;

	if (p->fb_scores_cap < (size_t)n_pos) {
		free(p->fb_scores);
		p->fb_scores = xmalloc((size_t)n_pos * sizeof(float));
		if (!p->fb_scores) {
			p->fb_scores_cap = 0;
			return ERR_OUT_OF_MEMORY;
		}
		p->fb_scores_cap = (size_t)n_pos;
	}
	float *scores = p->fb_scores;

	if (p->kv_quant == KV_QUANT_Q8_0) {
		const uint8_t *kdb = (const uint8_t *)as_glbuf(k_cache)->host_mirror + k_cache->offset +
							 (size_t)layer * layer_stride;
		const uint8_t *vdb = (const uint8_t *)as_glbuf(v_cache)->host_mirror + v_cache->offset +
							 (size_t)layer * layer_stride;
		size_t		   row_bytes =
			p->kv_row_bytes
				? p->kv_row_bytes
				: (size_t)((head_dim + KV_Q8_0_BLOCK - 1) / KV_Q8_0_BLOCK * KV_Q8_0_BLOCK_BYTES);
		if (p->fb_rowbuf_cap < (size_t)head_dim) {
			free(p->fb_rowbuf);
			p->fb_rowbuf = xmalloc((size_t)head_dim * sizeof(float));
			if (!p->fb_rowbuf) {
				p->fb_rowbuf_cap = 0;
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

	int	   kv_q8 = (p->kv_quant == KV_QUANT_Q8_0);
	size_t layer_off;
	int	   row_stride_u;
	if (kv_q8) {
		size_t layer_stride_b =
			p->kv_layer_stride ? p->kv_layer_stride : (size_t)n_kv_heads * n_ctx * p->kv_row_bytes;
		layer_off	 = (size_t)layer * layer_stride_b;
		row_stride_u = (int)p->kv_row_bytes;
	} else {
		size_t layer_stride =
			p->kv_layer_stride ? p->kv_layer_stride : (size_t)n_kv_heads * n_ctx * head_dim;
		layer_off	 = (size_t)layer * layer_stride;
		row_stride_u = head_dim;
	}

	size_t scores_bytes = (size_t)n_heads * n_ctx * sizeof(float);
	if (!p->attn_scores || p->attn_scores_cap < scores_bytes) {
		if (p->attn_scores)
			gl_buf_free(p, p->attn_scores);
		p->attn_scores = gl_buf_alloc(p, scores_bytes, GL_DYNAMIC_COPY, NULL);
		if (!p->attn_scores)
			return ERR_OUT_OF_MEMORY;
		p->attn_scores_cap = scores_bytes;
	}

	gl_buf *qbuf = as_glbuf(q);
	gl_buf *kbuf = as_glbuf(k_cache);
	gl_buf *vbuf = as_glbuf(v_cache);
	gl_buf *obuf = as_glbuf(out);

	gl_buf_sync_to_device(qbuf);
	gl_buf_sync_to_device(kbuf);
	gl_buf_sync_to_device(vbuf);
	gl_buf_sync_to_device(obuf);
	p->attn_scores->host_dirty = 0;

	glUseProgram(p->p_attention.program);
	gl_bind_ssbo(0, q);
	gl_bind_ssbo(1, k_cache);
	gl_bind_ssbo(2, v_cache);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, p->attn_scores->name);
	gl_bind_ssbo(4, out);

	gl_pu(&p->p_attention, "u_layer_off", (int)layer_off);
	gl_pu(&p->p_attention, "u_pos", pos);
	gl_pu(&p->p_attention, "u_n_heads", n_heads);
	gl_pu(&p->p_attention, "u_n_kv_heads", n_kv_heads);
	gl_pu(&p->p_attention, "u_head_dim", head_dim);
	gl_pu(&p->p_attention, "u_n_ctx", n_ctx);
	gl_puf(&p->p_attention, "u_scale", scale);
	gl_pu(&p->p_attention, "u_stride_head_dim", row_stride_u);
	gl_pu(&p->p_attention, "u_attn_start", attn_start);
	gl_pu(&p->p_attention, "u_kv_q8", kv_q8 ? 1 : 0);

	GLuint groups = (GLuint)n_heads;
	glDispatchCompute(groups, 1, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: attention dispatch failed (glErr=0x%04x, layer=%d, pos=%d, "
			  "attn_start=%d, n_heads=%d, head_dim=%d)",
			  (unsigned)err, layer, pos, attn_start, n_heads, head_dim);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

	int	   kv_q8 = (p->kv_quant == KV_QUANT_Q8_0);
	size_t layer_off;
	int	   row_stride_u;
	if (kv_q8) {
		size_t layer_stride_b =
			p->kv_layer_stride ? p->kv_layer_stride : (size_t)n_kv_heads * n_ctx * p->kv_row_bytes;
		layer_off	 = (size_t)layer * layer_stride_b;
		row_stride_u = (int)p->kv_row_bytes;
	} else {
		size_t layer_stride =
			p->kv_layer_stride ? p->kv_layer_stride : (size_t)n_kv_heads * n_ctx * head_dim;
		layer_off	 = (size_t)layer * layer_stride;
		row_stride_u = head_dim;
	}

	gl_buf *qbuf = as_glbuf(q);
	gl_buf *kbuf = as_glbuf(k_cache);
	gl_buf *vbuf = as_glbuf(v_cache);
	gl_buf *obuf = as_glbuf(out);

	gl_buf_sync_to_device(qbuf);
	gl_buf_sync_to_device(kbuf);
	gl_buf_sync_to_device(vbuf);
	gl_buf_sync_to_device(obuf);

	glUseProgram(p->p_attention_batch.program);
	gl_bind_ssbo(0, q);
	gl_bind_ssbo(1, k_cache);
	gl_bind_ssbo(2, v_cache);
	gl_bind_ssbo(3, k_cache);
	gl_bind_ssbo(4, out);

	gl_pu(&p->p_attention_batch, "u_layer_off", (int)layer_off);
	gl_pu(&p->p_attention_batch, "u_pos_start", pos_start);
	gl_pu(&p->p_attention_batch, "u_n_heads", n_heads);
	gl_pu(&p->p_attention_batch, "u_n_kv_heads", n_kv_heads);
	gl_pu(&p->p_attention_batch, "u_head_dim", head_dim);
	gl_pu(&p->p_attention_batch, "u_n_ctx", n_ctx);
	gl_puf(&p->p_attention_batch, "u_scale", scale);
	gl_pu(&p->p_attention_batch, "u_stride_head_dim", row_stride_u);
	gl_pu(&p->p_attention_batch, "u_attn_start", attn_start);
	gl_pu(&p->p_attention_batch, "u_kv_q8", kv_q8 ? 1 : 0);
	gl_pu(&p->p_attention_batch, "u_m", m);
	gl_pu(&p->p_attention_batch, "u_window", window);

	GLuint groups_y = (GLuint)((m + GL_ATTENTION_Q_CHUNK - 1) / GL_ATTENTION_Q_CHUNK);
	glDispatchCompute((GLuint)n_heads, groups_y, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: attention_batch dispatch failed (glErr=0x%04x, layer=%d, pos_start=%d, "
			  "attn_start=%d, n_heads=%d, head_dim=%d, m=%d)",
			  (unsigned)err, layer, pos_start, attn_start, n_heads, head_dim, m);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	obuf->device_dirty = 1;
	return OK;
}

static status_code gl_attention(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
								int n_kv_heads, int head_dim, int n_ctx, int flash_attn,
								float scale, int n_kv_heads_active) {
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
	if (m <= 0 || n <= 0)
		return OK;
	if (m == 1)
		return gl_rmsnorm(self, x, w, y, n, eps);
	return gl_rmsnorm_ph_impl(self->priv, x, w, y, 1, n, eps, 1, m);
}

static status_code gl_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w,
											 buffer *y, int n_heads, int head_dim, float eps,
											 int m) {
	return gl_rmsnorm_ph_impl(self->priv, x, w, y, n_heads, head_dim, eps, 1, m);
}

static status_code gl_rmsnorm_noweight_batch(backend *self, const buffer *x, buffer *y, int n,
											 float eps, int m) {
	return gl_rmsnorm_ph_impl(self->priv, x, NULL, y, 1, n, eps, 0, m);
}

static status_code gl_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x, buffer *y,
													  int n_heads, int head_dim, float eps, int m) {
	return gl_rmsnorm_ph_impl(self->priv, x, NULL, y, n_heads, head_dim, eps, 0, m);
}

static status_code gl_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	return gl_elementwise(self->priv, x, y, NULL, n, 1, 0.0f, 0, m);
}

static status_code gl_rope_batch_dispatch(backend *self, buffer *vec, int n_heads, int head_dim,
										  int pos_start, const float *rope_cos_base,
										  const float *rope_sin_base, int m) {
	gl_priv *p	  = self->priv;
	int		 half = head_dim / 2;
	if (n_heads <= 0 || half <= 0 || m <= 0)
		return OK;

	size_t		tbl = (size_t)m * (size_t)half * sizeof(float);
	status_code s	= buffer_ensure_scratch(self, &p->rope_cos_tbl, tbl);
	if (s != OK)
		return s;
	s = buffer_ensure_scratch(self, &p->rope_sin_tbl, tbl);
	if (s != OK)
		return s;
	for (int i = 0; i < m; i++) {
		buffer c  = buffer_slice(&p->rope_cos_tbl, (size_t)i * (size_t)half * sizeof(float),
								 (size_t)half * sizeof(float));
		buffer si = buffer_slice(&p->rope_sin_tbl, (size_t)i * (size_t)half * sizeof(float),
								 (size_t)half * sizeof(float));
		s = self->buffer_write_f32(self, &c, rope_cos_base + (size_t)(pos_start + i) * half, half);
		if (s != OK)
			return s;
		s = self->buffer_write_f32(self, &si, rope_sin_base + (size_t)(pos_start + i) * half, half);
		if (s != OK)
			return s;
	}

	gl_buf *vb = as_glbuf(vec);
	if (!vb || vb->size_bytes == 0)
		return ERR_INVALID_ARG;
	gl_buf_sync_to_device(vb);
	gl_buf_sync_to_device(as_glbuf(&p->rope_cos_tbl));
	gl_buf_sync_to_device(as_glbuf(&p->rope_sin_tbl));

	glUseProgram(p->p_rope_batch.program);
	gl_bind_ssbo(0, vec);
	gl_bind_ssbo(1, &p->rope_cos_tbl);
	gl_bind_ssbo(2, &p->rope_sin_tbl);
	gl_pu(&p->p_rope_batch, "u_n_heads", n_heads);
	gl_pu(&p->p_rope_batch, "u_head_dim", head_dim);
	gl_pu(&p->p_rope_batch, "u_m", m);
	gl_pu(&p->p_rope_batch, "u_neox", self->rope_neox);
	gl_pu(&p->p_rope_batch, "u_row_stride", n_heads * head_dim);

	int total = m * n_heads * half;
	glDispatchCompute((GLuint)((total + 127) / 128), 1, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: rope_batch dispatch failed (glErr=0x%04x, heads=%d, hd=%d, m=%d)", (unsigned)err,
			  n_heads, head_dim, m);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	vb->device_dirty = 1;
	return OK;
}

static status_code gl_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim,
								 int pos_start, const float *rope_cos_base,
								 const float *rope_sin_base, int m) {
	if (m <= 0)
		return OK;
	if (m == 1)
		return gl_rope(self, vec, n_heads, head_dim, pos_start, rope_cos_base, rope_sin_base);
	return gl_rope_batch_dispatch(self, vec, n_heads, head_dim, pos_start, rope_cos_base,
								  rope_sin_base, m);
}

static status_code gl_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads,
									int n_kv_heads, int head_dim, int pos_start,
									const float *rope_cos_base, const float *rope_sin_base, int m) {
	if (m <= 0)
		return OK;
	if (m == 1)
		return gl_rope_qk(self, q, k, n_heads, n_kv_heads, head_dim, pos_start, rope_cos_base,
						  rope_sin_base);
	status_code s = gl_rope_batch_dispatch(self, q, n_heads, head_dim, pos_start, rope_cos_base,
										   rope_sin_base, m);
	if (s != OK)
		return s;
	return gl_rope_batch_dispatch(self, k, n_kv_heads, head_dim, pos_start, rope_cos_base,
								  rope_sin_base, m);
}

static status_code gl_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim,
									 int pos_start, const float *rope_cos_base,
									 const float *rope_sin_base, const float *freq_factors, int m) {
	if (m <= 0)
		return OK;
	if (!freq_factors)
		return gl_rope_batch(self, vec, n_heads, head_dim, pos_start, rope_cos_base, rope_sin_base,
							 m);
	if (m == 1)
		return gl_rope_ext(self, vec, n_heads, head_dim, pos_start, rope_cos_base, rope_sin_base,
						   freq_factors);

	int half = head_dim / 2;
	if (n_heads <= 0 || half <= 0)
		return OK;

	size_t tbl	= (size_t)m * (size_t)half * sizeof(float);
	float *tblc = xmalloc(tbl);
	float *tbls = xmalloc(tbl);
	if (!tblc || !tbls) {
		free(tblc);
		free(tbls);
		return ERR_OUT_OF_MEMORY;
	}
	double theta_scale = pow((double)self->rope_theta, -2.0 / (double)head_dim);
	for (int i = 0; i < m; i++) {
		double base_freq = 1.0;
		int	   pos		 = pos_start + i;
		for (int j = 0; j < half; j++) {
			float  fv = freq_factors[j];
			double ff = (fv >= 1e10f || fv == 0.0f) ? 0.0 : (double)fv;
			if (ff > 0.0) {
				double angle			   = base_freq * ff * (double)pos;
				tblc[(size_t)i * half + j] = (float)cos(angle);
				tbls[(size_t)i * half + j] = (float)sin(angle);
			} else {
				tblc[(size_t)i * half + j] = 1.0f;
				tbls[(size_t)i * half + j] = 0.0f;
			}
			base_freq *= theta_scale;
		}
	}
	status_code s = gl_rope_batch_dispatch(self, vec, n_heads, head_dim, 0, tblc, tbls, m);
	free(tblc);
	free(tbls);
	return s;
}

static status_code gl_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
									  const buffer *v_cache, buffer *out, int layer, int pos_start,
									  int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									  int flash_attn, float scale, int n_kv_heads_active, int m) {
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
	if (m <= 0 || n <= 0)
		return OK;
	if (m == 1)
		return gl_ffn_activate_ex(self, gate, up, out, n, activation);
	size_t total = (size_t)n * (size_t)m;
	if (total > (size_t)INT_MAX)
		goto rowwise;
	{
		gl_priv *p		 = self->priv;
		gl_buf	*bufs[3] = {as_glbuf(gate), as_glbuf(up), as_glbuf(out)};
		for (int i = 0; i < 3; i++)
			gl_buf_sync_to_device(bufs[i]);
		glUseProgram(p->p_ffn_activate.program);
		gl_bind_ssbo(0, gate);
		gl_bind_ssbo(1, up);
		gl_bind_ssbo(2, out);
		gl_pu(&p->p_ffn_activate, "u_n", (int)total);
		gl_pu(&p->p_ffn_activate, "u_act", activation);
		GLuint groups = (GLuint)((total + 127) / 128);
		glDispatchCompute(groups, 1, 1);
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			ERROR("gl: ffn_activate_batch dispatch failed (glErr=0x%04x)", (unsigned)err);
			p->device_lost = 1;
			return ERR_INTERNAL;
		}
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

static status_code gl_ffn_activate_fused_batch(backend *self, const buffer *fused, buffer *out,
											   int n, int activation, int m) {
	if (m <= 0 || n <= 0)
		return OK;
	gl_priv *p	= self->priv;
	gl_buf	*fb = as_glbuf(fused);
	gl_buf	*ob = as_glbuf(out);
	if (!fb || !ob || fb->size_bytes == 0 || ob->size_bytes == 0)
		return ERR_INVALID_ARG;
	gl_buf_sync_to_device(fb);
	gl_buf_sync_to_device(ob);

	glUseProgram(p->p_ffn_activate_fused_batch.program);
	gl_bind_ssbo(0, fused);
	gl_bind_ssbo(1, out);
	gl_pu(&p->p_ffn_activate_fused_batch, "u_n", n);
	gl_pu(&p->p_ffn_activate_fused_batch, "u_m", m);
	gl_pu(&p->p_ffn_activate_fused_batch, "u_act", activation);

	size_t total = (size_t)n * (size_t)m;
	GLuint gx	 = (GLuint)((total + 127) / 128);
	GLuint gy	 = 1;
	if (gx > GL_MAX_DISPATCH_PER_DIM) {
		gy = (GLuint)((total / 128 + GL_MAX_DISPATCH_PER_DIM - 1) / GL_MAX_DISPATCH_PER_DIM);
		gx = GL_MAX_DISPATCH_PER_DIM;
	}
	glDispatchCompute(gx, gy, 1);
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		ERROR("gl: ffn_activate_fused_batch dispatch failed (glErr=0x%04x, n=%d, m=%d)",
			  (unsigned)err, n, m);
		p->device_lost = 1;
		return ERR_INTERNAL;
	}
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	ob->device_dirty = 1;
	return OK;
}

static status_code gl_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
										 const buffer *x, buffer **y, const int *n_list, int k,
										 int n_matmuls, int m) {
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
	if (gate_scale == 1.0f && up_scale == 1.0f)
		return gl_ffn_activate_ex(self, gate, up, out, n, act);
	buffer		gs = {0}, us = {0};
	size_t		bytes = (size_t)n * sizeof(float);
	status_code s	  = self->buffer_alloc_scratch(self, bytes, &gs);
	if (s != OK)
		return s;
	s = self->buffer_alloc_scratch(self, bytes, &us);
	if (s != OK) {
		self->buffer_free(self, &gs);
		return s;
	}
	s = self->copy_buffer(self, gate, &gs, n);
	if (s == OK)
		s = self->copy_buffer(self, up, &us, n);
	if (s == OK && gate_scale != 1.0f)
		s = gl_scale_inplace(self, &gs, gate_scale, n);
	if (s == OK && up_scale != 1.0f)
		s = gl_scale_inplace(self, &us, up_scale, n);
	if (s == OK)
		s = gl_ffn_activate_ex(self, (const buffer *)&gs, (const buffer *)&us, out, n, act);
	self->buffer_free(self, &gs);
	self->buffer_free(self, &us);
	return s;
}

static status_code gl_dequant_row(backend *self, uint32_t type, const void *src, int n_elems,
								  float *dst) {
	(void)self;
	if (!src || !dst || n_elems <= 0)
		return ERR_INVALID_ARG;
	backend *host = backend_host();
	if (!host || !host->dequant_row)
		return ERR_UNSUPPORTED;
	return host->dequant_row(host, type, src, n_elems, dst);
}

static status_code gl_moe_expert_ffn(backend *self, const buffer *x, buffer *out,
									 const moe_resident_expert *e, int dim, int inter) {
	if (!e || !e->gate_w || !e->down_w)
		return ERR_INVALID_ARG;
	gl_buf_sync_to_host(as_glbuf(x));
	gl_buf_sync_to_host(as_glbuf(out));
	gl_buf_sync_to_host(as_glbuf(e->gate_w));
	if (e->up_w)
		gl_buf_sync_to_host(as_glbuf(e->up_w));
	gl_buf_sync_to_host(as_glbuf(e->down_w));
	backend *host = backend_host();
	if (host && host->moe_expert_ffn)
		return host->moe_expert_ffn(host, x, out, e, dim, inter);
	return ERR_UNSUPPORTED;
}

static status_code gl_moe_experts_batch(backend *self, const buffer *xb, buffer *out, int n_rows,
										int dim, int inter, int use_gelu, int n_experts,
										const moe_resident_expert *experts, const int *counts,
										const int *rows_packed, const float *weights_packed) {
	gl_buf_sync_to_host(as_glbuf(xb));
	gl_buf_sync_to_host(as_glbuf(out));
	for (int i = 0; i < n_experts; i++) {
		if (experts[i].gate_w)
			gl_buf_sync_to_host(as_glbuf(experts[i].gate_w));
		if (experts[i].up_w)
			gl_buf_sync_to_host(as_glbuf(experts[i].up_w));
		if (experts[i].down_w)
			gl_buf_sync_to_host(as_glbuf(experts[i].down_w));
	}
	backend *host = backend_host();
	if (host && host->moe_experts_batch)
		return host->moe_experts_batch(host, xb, out, n_rows, dim, inter, use_gelu, n_experts,
									   experts, counts, rows_packed, weights_packed);
	return ERR_UNSUPPORTED;
}

static status_code gl_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name	  = "opengl";
	out->priority = 80;
	out->caps	  = BCAP_HOST_VISIBLE_BUFFERS | BCAP_RMSNORM_ADD | BCAP_KV_QUANT_Q8_0 |
					BCAP_MATMUL_RESIDUAL | BCAP_MULTI_MATMUL;

	out->probe					= gl_probe;
	out->device_count			= gl_device_count;
	out->init					= gl_init;
	out->free					= gl_free;
	out->buffer_alloc_weight	= gl_buffer_alloc_weight;
	out->buffer_alloc_scratch	= gl_buffer_alloc_scratch;
	out->buffer_alloc_from_host = gl_buffer_alloc_from_host;
	out->buffer_free			= gl_buffer_free;
	out->buffer_read_f32		= gl_buffer_read_f32;
	out->buffer_write_f32		= gl_buffer_write_f32;
	out->copy_buffer			= gl_copy_buffer;
	out->ple_combine			= gl_ple_combine;
	out->mem_available			= gl_mem_available;
	out->mem_total				= gl_mem_total;
	out->synchronize			= gl_synchronize;
	out->begin_batch			= gl_begin_batch;
	out->end_batch				= gl_end_batch;

	out->argmax							 = gl_argmax;
	out->embd_lookup					 = gl_embd_lookup;
	out->rmsnorm						 = gl_rmsnorm;
	out->rmsnorm_per_head				 = gl_rmsnorm_per_head;
	out->rmsnorm_noweight				 = gl_rmsnorm_noweight;
	out->rmsnorm_noweight_per_head		 = gl_rmsnorm_noweight_per_head;
	out->rmsnorm_add					 = gl_rmsnorm_add;
	out->matmul							 = gl_matmul;
	out->matmul_type_native				 = gl_matmul_type_native;
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
	out->attention						 = gl_attention;
	out->attention_swa					 = gl_attention_swa;
	out->rmsnorm_batch					 = gl_rmsnorm_batch;
	out->rmsnorm_per_head_batch			 = gl_rmsnorm_per_head_batch;
	out->rmsnorm_noweight_batch			 = gl_rmsnorm_noweight_batch;
	out->rmsnorm_noweight_per_head_batch = gl_rmsnorm_noweight_per_head_batch;
	out->add_batch						 = gl_add_batch;
	out->rope_batch						 = gl_rope_batch;
	out->rope_qk_batch					 = gl_rope_qk_batch;
	out->rope_ext_batch					 = gl_rope_ext_batch;
	out->attention_batch				 = gl_attention_batch;
	out->attention_swa_batch			 = gl_attention_swa_batch;
	out->ffn_activate_batch				 = gl_ffn_activate_batch;
	out->ffn_activate_fused_batch		 = gl_ffn_activate_fused_batch;
	out->matmul_multi_batch				 = gl_matmul_multi_batch;
	out->moe_activate					 = gl_moe_activate;
	out->moe_expert_ffn					 = gl_moe_expert_ffn;
	out->moe_experts_batch				 = gl_moe_experts_batch;
	out->dequant_row					 = gl_dequant_row;

	return OK;
}

BACKEND_REGISTER("opengl", gl_ctor)
