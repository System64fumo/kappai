#ifndef GL_CONTEXT_H
#define GL_CONTEXT_H

#include "common.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <gbm.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
	EGLDisplay		   dpy;
	EGLContext		   ctx;
	EGLConfig		   cfg;
	struct gbm_device *gbm;
	int				   drm_fd;

	int has_compute;
	int max_wg_size;
	int max_wg_size_arr[3];
	int max_wg_invocations;
	int max_shared_bytes;
	int max_ssbo_bindings;
	int max_dispatch[3];

	char renderer[128];
} gl_context;

status_code gl_context_init(gl_context *out);

void gl_context_free(gl_context *ctx);

status_code gl_context_probe(void);

#endif
