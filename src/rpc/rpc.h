#ifndef RPC_H
#define RPC_H

#include "backend/backend.h"

#include <pthread.h>

#define RPC_DEFAULT_PORT 50052
#define RPC_DEFAULT_BIND "0.0.0.0"
#define RPC_MAGIC 0x4b505250u
#define RPC_VERSION 1u
#define RPC_MAX_ENDPOINTS 16
#define RPC_HOST_MAX 128
#define RPC_IDENT_MAX 192
#define RPC_MAX_FRAME ((size_t)1 << 30)
#define RPC_FLUSH_BYTES ((size_t)1 << 18)
#define RPC_UPLOAD_CHUNK ((size_t)1 << 24)
#define RPC_MAX_VALS 160
#define RPC_MAX_BUFS 2048
#define RPC_MAX_TABLES 32
#define RPC_TYPE_MAX 128
#define RPC_CONNECT_TIMEOUT_MS 3000
#define RPC_WR_INLINE 1024
#define RPC_KEEPALIVE_IDLE_S 10
#define RPC_KEEPALIVE_INTERVAL_S 5
#define RPC_KEEPALIVE_COUNT 3
#define RPC_MEM_CACHE_US 1000000
#define RPC_PLAN_CACHE 64

#define RPC_FIELD_OPS(X)                                                                           \
	X(kv_put)                                                                                      \
	X(kv_put_batch) X(embd_lookup) X(rmsnorm) X(matmul) X(matmul_residual) X(matmul_qonly)         \
		X(prequantize_x) X(add_inplace) X(scale_inplace) X(softcap) X(split_qgate)                 \
			X(attn_output_gate) X(copy_buffer) X(ple_combine) X(ffn_activate) X(ffn_activate_ex)   \
				X(rmsnorm_per_head) X(rmsnorm_noweight) X(rmsnorm_noweight_per_head)               \
					X(rmsnorm_add) X(rmsnorm_add_batch) X(rmsnorm_add_norm_batch)                  \
						X(rmsnorm_per_head_batch) X(rmsnorm_noweight_batch)                        \
							X(rmsnorm_noweight_per_head_batch) X(matmul_ffn_down)                  \
								X(matmul_ffn_down_batch) X(rmsnorm_batch) X(matmul_batch)          \
									X(add_batch) X(ffn_activate_batch) X(ffn_activate_fused_batch) \
										X(attention) X(attention_swa) X(attention_batch)           \
											X(attention_swa_batch) X(attention_full_batch)         \
												X(matmul_gate_act_batch) X(matmul_softcap_batch)   \
													X(layernorm_modulate) X(gated_add) X(add_bias) \
														X(activate) X(rope_table) X(kv_put_mla)    \
															X(moe_activate) X(moe_router_topk)

#define RPC_SPECIAL_OPS(X)                                                                         \
	X(rope)                                                                                        \
	X(rope_qk) X(rope_ext) X(rope_ext_batch) X(rope_batch) X(qk_norm_rope) X(rope_qk_batch)        \
		X(partial_rope_qk) X(attention_mla) X(matmul_multi) X(matmul_multi_batch) X(argmax)        \
			X(argmax_to_reg) X(read_token_reg) X(sample) X(sample_to_reg) X(sample_history_set)    \
				X(buffer_read_f32) X(buffer_read_f32_x2) X(buffer_write_f32)                       \
					X(buffer_alloc_weight) X(buffer_alloc_scratch) X(buffer_free)                  \
						X(buffer_alloc_from_host) X(dequant_weight) X(kv_alloc) X(kv_free)         \
							X(kv_alloc_mla) X(gated_delta_net) X(hc_init) X(hc_pre) X(hc_post)     \
								X(hc_final) X(kda) X(mla_absorbed) X(attention_dense) X(im2col3x3) \
									X(upsample_shortcut_add) X(moe_expert_ffn)                     \
										X(moe_experts_batch) X(synchronize) X(submit)              \
											X(begin_batch) X(end_batch) X(set_gpu_profile)         \
												X(matmul_type_native) X(dequant_type_native)       \
													X(repack_plan)                                    \
													X(mem_available) X(mem_total)

typedef enum {
	RPC_EXT_DEVICE = 1,
	RPC_EXT_MEM_TOTAL,
} rpc_hello_ext;

#define RPC_ALL_OPS(X) RPC_FIELD_OPS(X) RPC_SPECIAL_OPS(X)

#define RPC_ENUM_ENTRY(name) RPC_OP_##name,

typedef enum {
	RPC_CTL_INFO = 1,
	RPC_CTL_HELLO,
	RPC_CTL_PARAMS,
	RPC_CTL_TABLE,
	RPC_CTL_UPLOAD_BEGIN,
	RPC_CTL_UPLOAD_DATA,
	RPC_ALL_OPS(RPC_ENUM_ENTRY) RPC_OP_END,
} rpc_op;

#define RPC_OP_COUNT ((int)RPC_OP_END)

typedef enum {
	RPC_VAL_BUF = 1,
	RPC_VAL_I32,
	RPC_VAL_U32,
	RPC_VAL_F32,
} rpc_val_kind;

typedef enum {
	RPC_F_I32,
	RPC_F_F32,
	RPC_F_CBUF,
	RPC_F_BUF,
	RPC_F_WEIGHT,
} rpc_field_kind;

typedef enum {
	RPC_DESC_GDN,
	RPC_DESC_HC_INIT,
	RPC_DESC_HC_FINAL,
	RPC_DESC_HC_PRE,
	RPC_DESC_HC_POST,
	RPC_DESC_KDA,
	RPC_DESC_MLA_ABSORBED,
	RPC_DESC_DENSE_ATTENTION,
	RPC_DESC_IM2COL,
	RPC_DESC_UPSAMPLE_SHORTCUT,
} rpc_desc_kind;

typedef struct {
	int				fd;
	pthread_mutex_t mtx;
	uint8_t		   *out;
	size_t			out_len;
	size_t			out_cap;
	bool			dead;
} rpc_conn;

typedef struct {
	char host[RPC_HOST_MAX];
	int	 port;
} rpc_endpoint;

typedef struct {
	char device[BACKEND_DEVICE_NAME_MAX];
	char identity[RPC_IDENT_MAX];
} rpc_info;

typedef struct {
	int32_t	 status;
	uint8_t *data;
	size_t	 len;
} rpc_reply;

typedef struct {
	uint8_t *p;
	size_t	 len;
	size_t	 cap;
	bool	 bad;
	uint8_t	 inline_buf[RPC_WR_INLINE];
} rpc_wr;

typedef struct {
	const uint8_t *p;
	size_t		   len;
	size_t		   pos;
	bool		   bad;
} rpc_rd;

typedef struct {
	uint8_t kind;
	union {
		const buffer *b;
		int32_t		  i;
		uint32_t	  u;
		float		  f;
	} v;
} rpc_val;

typedef bool (*rpc_resolve_fn)(void *ctx, uint32_t id, uint64_t off, uint64_t size, buffer *out);

typedef struct {
	rpc_rd		   rd;
	rpc_resolve_fn resolve;
	void		  *ctx;
	buffer		   bufs[RPC_MAX_BUFS];
	rpc_val		   vals[RPC_MAX_VALS];
	int			   n_bufs;
	int			   n_vals;
} rpc_msg;

typedef struct {
	size_t		   offset;
	rpc_field_kind kind;
} rpc_field;

typedef struct {
	const rpc_field *fields;
	int				 n_fields;
} rpc_layout;

void rpc_wr_init(rpc_wr *w);
void rpc_wr_free(rpc_wr *w);
void rpc_wr_u8(rpc_wr *w, uint8_t v);
void rpc_wr_u16(rpc_wr *w, uint16_t v);
void rpc_wr_u32(rpc_wr *w, uint32_t v);
void rpc_wr_i32(rpc_wr *w, int32_t v);
void rpc_wr_u64(rpc_wr *w, uint64_t v);
void rpc_wr_f32(rpc_wr *w, float v);
void rpc_wr_bytes(rpc_wr *w, const void *data, size_t n);
void rpc_wr_str(rpc_wr *w, const char *s);
void rpc_wr_buf(rpc_wr *w, const backend *owner, const buffer *b);
void rpc_wr_val(rpc_wr *w, const backend *owner, rpc_val v);
void rpc_wr_desc(rpc_wr *w, const backend *owner, const void *desc, rpc_desc_kind kind);

void		rpc_rd_init(rpc_rd *r, const void *p, size_t len);
uint8_t		rpc_rd_u8(rpc_rd *r);
uint16_t	rpc_rd_u16(rpc_rd *r);
uint32_t	rpc_rd_u32(rpc_rd *r);
int32_t		rpc_rd_i32(rpc_rd *r);
uint64_t	rpc_rd_u64(rpc_rd *r);
float		rpc_rd_f32(rpc_rd *r);
const void *rpc_rd_bytes(rpc_rd *r, size_t n);
void		rpc_rd_str(rpc_rd *r, char *out, size_t cap);

void rpc_msg_init(rpc_msg *m, rpc_resolve_fn resolve, void *ctx, const void *p, size_t len);
const buffer *rpc_msg_buf(rpc_msg *m);
bool		  rpc_msg_vals(rpc_msg *m);
bool		  rpc_msg_desc(rpc_msg *m, void *desc, rpc_desc_kind kind);
void		  rpc_wr_expert(rpc_wr *w, const backend *owner, const moe_resident_expert *e);
bool		  rpc_msg_expert(rpc_msg *m, moe_resident_expert *e);

rpc_val rpc_val_buf(const buffer *b);
rpc_val rpc_val_i32(int32_t v);
rpc_val rpc_val_u32(uint32_t v);
rpc_val rpc_val_f32(float v);

#define RPC_V(x)                                                                                   \
	_Generic((x),                                                                                  \
		const buffer *: rpc_val_buf,                                                               \
		buffer *: rpc_val_buf,                                                                     \
		int: rpc_val_i32,                                                                          \
		unsigned int: rpc_val_u32,                                                                 \
		float: rpc_val_f32)(x)

status_code rpc_connect(rpc_conn *c, const rpc_endpoint *ep, int timeout_ms);
void		rpc_conn_close(rpc_conn *c);
status_code rpc_send(rpc_conn *c, uint32_t op, const void *payload, size_t len);
status_code rpc_flush(rpc_conn *c);
status_code rpc_call(rpc_conn *c, uint32_t op, const void *payload, size_t len, rpc_reply *out);
void		rpc_reply_free(rpc_reply *r);

int			rpc_listen(const char *host, int port, char *err, size_t errlen);
status_code rpc_read_frame(int fd, uint32_t *op, uint8_t **payload, size_t *len);
status_code rpc_write_reply(int fd, int32_t status, const void *payload, size_t len);

int			rpc_endpoint_parse(const char *text, rpc_endpoint *out, char *err, size_t errlen);
void		rpc_endpoint_format(const rpc_endpoint *ep, char *out, size_t cap);
int			rpc_endpoints_load(rpc_endpoint *out, int max);
int			rpc_endpoints_save(const rpc_endpoint *eps, int n, char *err, size_t errlen);
status_code rpc_probe(const rpc_endpoint *ep, rpc_info *out);

size_t rpc_tensor_bytes(uint32_t type, int n_dims, const uint64_t *dims);

#endif
