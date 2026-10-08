#include "gl_context.h"
#include "backend/backend.h"
#include "common.h"
#include "log.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int open_drm_render_node(int *out_index) {
	static const int MAX_NODES = 8;
	for (int i = 0; i < MAX_NODES; i++) {
		char path[64];
		snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
		int fd = open(path, O_RDWR | O_CLOEXEC);
		if (fd >= 0) {
			*out_index = i;
			return fd;
		}
	}
	return -1;
}

static int g_probed = -1;

status_code gl_context_probe(void) {
	if (g_probed >= 0)
		return g_probed == 1 ? OK : ERR_UNSUPPORTED;

	EGLDisplay prev_dpy	 = eglGetCurrentDisplay();
	EGLContext prev_ctx	 = eglGetCurrentContext();
	EGLSurface prev_draw = eglGetCurrentSurface(EGL_DRAW);
	EGLSurface prev_read = eglGetCurrentSurface(EGL_READ);

	gl_context tmp;
	memset(&tmp, 0, sizeof(tmp));
	status_code s = gl_context_init(&tmp);
	if (s == OK) {
		gl_context_free(&tmp);
		g_probed = 1;
	} else {
		g_probed = 0;
	}

	if (prev_ctx != EGL_NO_CONTEXT && prev_dpy != EGL_NO_DISPLAY)
		eglMakeCurrent(prev_dpy, prev_draw, prev_read, prev_ctx);

	return s == OK ? OK : ERR_UNSUPPORTED;
}

status_code gl_context_init(gl_context *out) {
	memset(out, 0, sizeof(*out));

	int drm_index = 0;
	int fd		  = open_drm_render_node(&drm_index);
	if (fd < 0) {
		DEBUG("gl: no /dev/dri/renderD128+ node available");
		return ERR_UNSUPPORTED;
	}

	struct gbm_device *gbm = gbm_create_device(fd);
	if (!gbm) {
		DEBUG("gl: gbm_create_device failed on renderD%d", drm_index);
		close(fd);
		return ERR_UNSUPPORTED;
	}

	EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, gbm, NULL);
	if (dpy == EGL_NO_DISPLAY) {
		DEBUG("gl: eglGetPlatformDisplay failed");
		gbm_device_destroy(gbm);
		close(fd);
		return ERR_UNSUPPORTED;
	}

	EGLint major = 0, minor = 0;
	if (!eglInitialize(dpy, &major, &minor)) {
		DEBUG("gl: eglInitialize failed (eglErr=0x%04x)", (unsigned)eglGetError());
		gbm_device_destroy(gbm);
		close(fd);
		return ERR_UNSUPPORTED;
	}

	if (!eglBindAPI(EGL_OPENGL_ES_API)) {
		ERROR("gl: eglBindAPI(ES) failed");
		eglTerminate(dpy);
		gbm_device_destroy(gbm);
		close(fd);
		return ERR_UNSUPPORTED;
	}

	static const EGLint config_attribs[] = {
		EGL_RENDERABLE_TYPE,
		EGL_OPENGL_ES3_BIT_KHR,
		EGL_NONE,
	};
	EGLConfig cfg;
	EGLint	  cfg_count = 0;
	if (!eglChooseConfig(dpy, config_attribs, &cfg, 1, &cfg_count) || cfg_count < 1) {
		ERROR("gl: eglChooseConfig failed (eglErr=0x%04x)", (unsigned)eglGetError());
		eglTerminate(dpy);
		gbm_device_destroy(gbm);
		close(fd);
		return ERR_UNSUPPORTED;
	}

	static const EGLint ctx31_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 3, EGL_CONTEXT_MINOR_VERSION_KHR, 1, EGL_NONE,
	};
	EGLint ctx30_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION,
		3,
		EGL_NONE,
	};

	EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx31_attribs);
	if (ctx == EGL_NO_CONTEXT) {
		EGLint e = eglGetError();
		DEBUG("gl: GLES 3.1 context creation failed (eglErr=0x%04x); trying 3.0", (unsigned)e);
		ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx30_attribs);
	}
	if (ctx == EGL_NO_CONTEXT) {
		ERROR("gl: eglCreateContext failed (eglErr=0x%04x)", (unsigned)eglGetError());
		eglTerminate(dpy);
		gbm_device_destroy(gbm);
		close(fd);
		return ERR_UNSUPPORTED;
	}

	if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
		ERROR("gl: eglMakeCurrent(surfaceless) failed (eglErr=0x%04x)", (unsigned)eglGetError());
		eglDestroyContext(dpy, ctx);
		eglTerminate(dpy);
		gbm_device_destroy(gbm);
		close(fd);
		return ERR_UNSUPPORTED;
	}

	GLint has_compute = 0;
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &has_compute);
	int usable = (has_compute > 0) && (glGetError() == GL_NO_ERROR);

	out->dpy				= dpy;
	out->ctx				= ctx;
	out->cfg				= cfg;
	out->gbm				= gbm;
	out->drm_fd				= fd;
	out->has_compute		= usable ? 1 : 0;
	out->max_wg_invocations = has_compute > 0 ? has_compute : 0;

	if (usable) {
		for (int i = 0; i < 3; i++)
			glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, (GLuint)i, &out->max_wg_size_arr[i]);
		out->max_wg_size = out->max_wg_size_arr[0];
		for (int i = 0; i < 3; i++)
			glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, (GLuint)i, &out->max_dispatch[i]);
		GLint shared = 0;
		glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &shared);
		out->max_shared_bytes = shared;
		GLint bindings		  = 0;
		glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &bindings);
		out->max_ssbo_bindings = bindings;

		GLenum drain;
		while ((drain = glGetError()) != GL_NO_ERROR) {
			DEBUG("gl: drained GL error 0x%04x during context init queries", (unsigned)drain);
		}
	}

	const GLubyte *renderer = glGetString(GL_RENDERER);
	snprintf(out->renderer, sizeof(out->renderer), "%s",
			 renderer ? (const char *)renderer : "(unknown)");

	if (!usable) {
		WARN("gl: GLES 3.1 context created but compute appears unavailable; "
			 "backend will be unusable until a real GPU driver is installed");
	}

	DEBUG("gl: EGL/GLES 3.1 context up on %s (compute=%d, max_wg=%d, shared=%dKB, bindings=%d)",
		  out->renderer, out->has_compute, out->max_wg_size, out->max_shared_bytes / 1024,
		  out->max_ssbo_bindings);
	return OK;
}

void gl_context_free(gl_context *ctx) {
	if (!ctx)
		return;
	if (ctx->ctx) {
		eglMakeCurrent(ctx->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		eglDestroyContext(ctx->dpy, ctx->ctx);
	}
	if (ctx->dpy)
		eglTerminate(ctx->dpy);
	if (ctx->gbm)
		gbm_device_destroy(ctx->gbm);
	if (ctx->drm_fd >= 0)
		close(ctx->drm_fd);
	memset(ctx, 0, sizeof(*ctx));
}
