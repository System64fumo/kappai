#ifndef RPC_SESSION_H
#define RPC_SESSION_H

#include "rpc/rpc.h"

typedef struct {
	buffer buf;
	void  *owned;
	bool   used;
} srv_buf;

typedef struct {
	float *data;
	size_t cap_floats;
} srv_table;

typedef struct {
	int			fd;
	char		device[BACKEND_DEVICE_NAME_MAX];
	backend	   *be;
	bool		emulate_moe;
	srv_buf	   *bufs;
	uint32_t	n_bufs;
	srv_table	tables[RPC_MAX_TABLES * 4];
	uint8_t	   *upload;
	size_t		upload_len;
	size_t		upload_total;
	status_code sticky;
	float_buf	ws;
	float_buf	moe_scratch;
	rpc_msg	   *msg;
} srv_session;

void session_run(int fd, const char *device);

status_code resident_expert_ffn(srv_session *s, const buffer *x, buffer *out,
								const moe_resident_expert *e, int dim, int inter);
status_code resident_experts_batch(srv_session *s, const buffer *xb, buffer *out, int n_rows,
								   int dim, int inter, int use_gelu, int n_experts,
								   const moe_resident_expert *experts, const int *counts,
								   const int *rows_packed, const float *weights_packed);

#endif
