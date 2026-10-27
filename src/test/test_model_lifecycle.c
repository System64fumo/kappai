#include "config.h"
#include "log.h"
#include "test_core.h"

typedef struct {
	void *handles[128];
	bool  live[128];
	int	  n;
	int	  released;
	int	  duplicate_frees;
	void (*buffer_free)(backend *, buffer *);
} lifecycle_allocations;

static lifecycle_allocations allocations;

static status_code lifecycle_copy_weight(backend *b, const tensor_desc *desc, buffer *out) {
	if (allocations.n == (int)ARRAY_LEN(allocations.handles))
		return ERR_INTERNAL;
	size_t bytes = ggml_row_size(desc->type, desc->dims[0]);
	for (int i = 1; i < desc->n_dims; i++)
		bytes *= desc->dims[i];
	void *p = xmalloc(bytes);
	memcpy(p, desc->host_data, bytes);
	*out							   = (buffer){.handle = p, .size = bytes, .owner = b};
	allocations.handles[allocations.n] = p;
	allocations.live[allocations.n++]  = true;
	return OK;
}

static void lifecycle_free_buffer(backend *b, buffer *buf) {
	if (buf && buf->handle) {
		for (int i = 0; i < allocations.n; i++) {
			if (allocations.handles[i] != buf->handle)
				continue;
			if (!allocations.live[i]) {
				allocations.duplicate_frees++;
				memset(buf, 0, sizeof(*buf));
				return;
			}
			allocations.live[i] = false;
			allocations.released++;
			break;
		}
	}
	allocations.buffer_free(b, buf);
}

void run_model_lifecycle_tests(void) {
	config	  saved		= *config_get();
	log_level saved_log = log_get_level();
	log_set_level((log_level)(LOG_ERROR + 1));
	((config *)config_get())->n_threads = 2;
	for (int use_mmap = 0; use_mmap <= 1; use_mmap++) {
		for (int owned = 0; owned <= 1; owned++) {
			backend	   *b  = NULL;
			status_code st = backend_create("cpu", 0, &b);
			if (st != OK) {
				record_resultf(OPFAM_EDGE_CASE, "model.qkv_lifecycle", 0, "backend=%d", st);
				continue;
			}
			allocations = (lifecycle_allocations){.buffer_free = b->buffer_free};
			if (owned) {
				b->buffer_alloc_weight = lifecycle_copy_weight;
				b->buffer_free		   = lifecycle_free_buffer;
			}
			model m;
			st = model_load_parse(&m, synth_chat_model_path, b, use_mmap, "none");
			if (st == OK) {
				m.fuse_config = "qkv";
				st			  = model_upload_weights(&m);
			}
			int ok = st == OK && m.qkv_fused_layers == m.n_layers;
			for (int i = 0; ok && i < m.n_layers; i++) {
				const layer_weights *l = &m.layers[i];
				ok = l->qkv_fused && l->qkv_w.buf.handle &&
					 l->qkv_w.buf.handle == l->wq.buf.handle &&
					 l->qkv_w.buf.handle == l->wk.buf.handle &&
					 l->qkv_w.buf.handle == l->wv.buf.handle &&
					 (owned ? l->qkv_fused_host == NULL : l->qkv_fused_host == l->qkv_w.buf.handle);
			}
			record_resultf(OPFAM_EDGE_CASE, "model.qkv_ownership", ok, "mmap=%d owned=%d status=%d",
						   use_mmap, owned, st);
			model_free(&m);
			ok = allocations.duplicate_frees == 0 && allocations.released == allocations.n;
			record_resultf(OPFAM_EDGE_CASE, "model.qkv_shutdown", ok,
						   "mmap=%d owned=%d released=%d/%d duplicate_frees=%d", use_mmap, owned,
						   allocations.released, allocations.n, allocations.duplicate_frees);
			backend_destroy(b);
		}
	}
	*((config *)config_get()) = saved;
	log_set_level(saved_log);
}
