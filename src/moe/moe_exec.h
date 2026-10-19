#ifndef MOE_EXEC_H
#define MOE_EXEC_H

#include "backend/backend.h"
#include "moe/moe_stream.h"

struct model;
struct compute_scratch;

#define MOE_EXEC_MAX_TARGETS 4

typedef struct {
	struct model		   *m;
	struct compute_scratch *s;
	const float			   *weights;
	const float			   *x;
	const buffer		   *xb_dev;
	buffer				   *out_dev;
	int						layer;
	int						dim;
	int						K;
	int						I;
	int						use_gelu;
} moe_exec_request;

typedef struct moe_exec_target moe_exec_target;

typedef struct {
	status_code (*submit)(moe_exec_target *t, const moe_exec_request *rq, const int *ids);
	status_code (*collect)(moe_exec_target *t, const moe_exec_request *rq, float *out);
} moe_exec_ops;

struct moe_exec_target {
	const moe_exec_ops *ops;
	backend			   *be;
	moe_expert_slot	   *slots;
	int					direct;
};

typedef struct {
	moe_exec_target targets[MOE_EXEC_MAX_TARGETS];
	int				n;
	int				n_direct;
} moe_exec_pending;

int moe_exec_dispatch(moe_exec_request *rq, backend *local, const buffer *xb, const int *expert_ids,
					  int *host_ids, moe_exec_pending *pend);

status_code moe_exec_gather(moe_exec_pending *pend, const moe_exec_request *rq, int discard);

int moe_exec_direct_capable(const backend *a, const buffer *xb, const buffer *out);

int moe_exec_staged_capable(const backend *a, const buffer *out);

status_code moe_exec_upload_host(struct compute_scratch *s, backend *a, buffer *out,
								 const float *host, int dim, int accumulate);

#endif
