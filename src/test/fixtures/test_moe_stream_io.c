#define _GNU_SOURCE
#include "model.h"
#include "config.h"
#include "moe/moe_stream.h"
#include "moe/moe_common.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Standalone fault-injection regression; build/run as documented in the numeric report.
 * Injection is restricted to this fixture inode. Run cases sequentially. */
static const char *path;
static dev_t dev;
static ino_t ino;
static int reject_direct, active, calls, direct_calls, buffered_calls, errors;
static const char *fault;
int open(const char *p, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap);
    }
    if (reject_direct && (flags & O_DIRECT) && !strcmp(p, path)) {
        errno = EINVAL; return -1;
    }
    return syscall(SYS_openat, AT_FDCWD, p, flags, mode);
}
ssize_t pread(int fd, void *buf, size_t len, off_t off) {
    struct stat sb;
    if (active && !fstat(fd, &sb) && sb.st_dev == dev && sb.st_ino == ino) {
        int direct = !!(fcntl(fd, F_GETFL) & O_DIRECT);
        calls++; if (direct) direct_calls++; else buffered_calls++;
        if (!strcmp(fault, "eio") || (!strcmp(fault, "eio-once") && calls == 1) ||
            (!strcmp(fault, "direct-eio") && direct) ||
            (!strcmp(fault, "partial-eio") && calls > 1)) {
            errors++; errno = EIO; return -1;
        }
        if (!strcmp(fault, "eintr") && calls == 1) {
            errors++; errno = EINTR; return -1;
        }
        if ((!strcmp(fault, "short") || !strcmp(fault, "partial-eio")) && len > 64)
            len = 64;
    }
    return syscall(SYS_pread64, fd, buf, len, off);
}
static status_code resolve(model *m, int eid, moe_expert_slot *s, int prep) {
    if (!prep) return moe_stream_resolve(m, 0, &eid, 1, s);
    moe_stream_op *op = moe_stream_resolve_prep(m, 0, &eid, 1, s);
    if (!op) return OK;
    for (int i = 0; i < moe_stream_op_n_items(op); i++)
        if (moe_stream_op_compute_k(op, i) < 0) moe_stream_op_fill_run(op, i, 0);
    status_code rc = moe_stream_op_finish(op);
    moe_stream_op_free(op); return rc;
}
static size_t bad_cells(const moe_expert_slot *s, size_t cells) {
    assert(s->eid == 1 && s->gate_w && s->up_w && s->down_w);
    const float *w[] = {s->gate_w, s->up_w, s->down_w};
    size_t bad = 0;
    for (int t = 0; t < 3; t++) for (size_t i = 0; i < cells; i++) bad += w[t][i] != 2.0f;
    return bad;
}
static double exec(model *m, moe_expert_slot *s) {
    int d = m->dim;
    float *x = calloc(d, sizeof(float)), *out = calloc(d, sizeof(float));
    float *scratch = calloc(4*d, sizeof(float));
    assert(x && out && scratch);
    for (int i = 0; i < d; i++) x[i] = 1;
    moe_expert_ctx cx = {.a=backend_host(), .inter=d, .dim=d, .scratch=scratch, .out=out};
    assert(moe_expert_exec(&cx, s, x, 1.0f, 0) == OK);
    double result = out[0];
    for (int i = 1; i < d; i++) assert(out[i] == out[0] || (isnan(out[i]) && isnan(out[0])));
    free(x); free(out); free(scratch); return result;
}
int main(int argc, char **argv) {
    assert(argc == 5);
    path = getenv("KAPPAI_IO_FIXTURE");
    if (!path) path = "/tmp/kappai-moe-stream-io-fixture.bin";
    const char *mode = argv[1]; int prep = !strcmp(argv[2], "prep"); fault = argv[3];
    int mapped = !strcmp(argv[4], "mmap");
    reject_direct = !strcmp(mode, "buffered");
    int d = !strcmp(mode, "aligned") ? 32 : 8;
    assert(!mapped || ((!strcmp(mode, "bounce") || !strcmp(mode, "aligned")) && strcmp(fault, "eof")));
    assert((strcmp(fault, "partial-eio") && strcmp(fault, "short")) || reject_direct);
    size_t cells = (size_t)d*d, bytes = cells*sizeof(float);
    size_t stride = (3*bytes+4095)&~(size_t)4095, size = 2*stride+4096;
    int fd = open(path, O_CREAT|O_TRUNC|O_RDWR, 0600); assert(fd >= 0);
    float *data = calloc(1, size); assert(data);
    for (size_t i = 0; i < 3*cells; i++) data[i] = 1;
    for (size_t i = stride/4; i < stride/4+3*cells; i++) data[i] = 2;
    assert(write(fd, data, size) == (ssize_t)size);
    struct stat sb; assert(!fstat(fd, &sb)); dev = sb.st_dev; ino = sb.st_ino;
    void *map = NULL;
    if (mapped) { map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0); assert(map != MAP_FAILED); }
    config cfg = config_defaults(); cfg.n_threads=1; cfg.moe_stream=1; cfg.moe_pin=0; cfg.moe_cache_cap=1;
    config_init(&cfg);
    arch_info arch = {.is_moe=1, .gguf_name="fixture"};
    struct expert_desc experts[2] = {0}; struct layer_weights layer = {.experts=experts};
    model m = {0}; m.arch_info=&arch; m.n_layers=1; m.layers=&layer; m.dim=d;
    m.moe.n_experts=2; m.moe.n_experts_used=1; m.moe.moe_intermediate=d;
    m.model_path=(char *)path; m.use_mmap=mapped; m.gctx.fd=mapped?fd:-1;
    m.gctx.map=map; m.gctx.map_size=mapped?size:0;
    for (int i = 0; i < 2; i++) {
        experts[i].gate_type=experts[i].up_type=experts[i].down_type=GGML_TYPE_F32;
        experts[i].gate_scale=experts[i].up_scale=experts[i].down_scale=1;
        experts[i].gate_off=i*stride; experts[i].up_off=i*stride+bytes; experts[i].down_off=i*stride+2*bytes;
        if (mapped) {
            experts[i].gate_w=(char *)map+experts[i].gate_off;
            experts[i].up_w=(char *)map+experts[i].up_off;
            experts[i].down_w=(char *)map+experts[i].down_off;
        }
    }
    assert(moe_stream_cache_init(&m)==OK);
    moe_expert_slot s={0}; assert(resolve(&m,0,&s,prep)==OK);
    assert(s.gate_w && *(float *)s.gate_w==1 && *(float *)s.up_w==1 && *(float *)s.down_w==1);
    moe_stream_release_slot(&m,0,&s);
    if (!strcmp(fault,"eof")) assert(!ftruncate(fd,stride));
    active=1; status_code rc=resolve(&m,1,&s,prep); active=0;
    int should_fail = !mapped && (!strcmp(fault,"eof") ||
        !strcmp(fault,"eio") || !strcmp(fault,"partial-eio") ||
        (reject_direct && !strcmp(fault,"eio-once")));
    size_t bad=0; double value=0;
    if (should_fail) { assert(rc==ERR_IO && !s.gate_w); }
    else {
        assert(rc==OK); bad=bad_cells(&s,cells); value=exec(&m,&s);
        assert(bad==0);
    }
    if (strcmp(fault,"healthy") && strcmp(fault,"short") && strcmp(fault,"eof")) assert(errors>0);
    if (!strcmp(fault,"direct-eio") && !mapped) assert(buffered_calls>0);
    /* Independent real-valued reference: sum_j 2 * SiLU(2*d) * (2*d). */
    double u=2.0*d, reference=d*2.0*(u/(1.0+exp(-u)))*u;
    printf("%s/%s/%s/%s rc=%d calls=%d direct=%d buffered=%d errors=%d bad=%zu/%zu out=%.12g reference=%.12g\n",
           mode,argv[2],fault,argv[4],rc,calls,direct_calls,buffered_calls,errors,bad,3*cells,value,reference);
    if (!should_fail) assert(fabs(value-reference)<1e-5*reference);
    moe_stream_release_slot(&m,0,&s);
    /* Restore the EOF fixture, then distinguish failed-fill recovery from a poisoned cache hit. */
    if (!strcmp(fault,"eof")) {
        assert(!ftruncate(fd,size));
        assert(pwrite(fd,data,size,0)==(ssize_t)size);
    }
    int previous_calls=calls;
    fault="healthy"; active=1; memset(&s,0,sizeof(s)); rc=resolve(&m,1,&s,prep); active=0;
    assert(rc==OK); size_t retry_bad=bad_cells(&s,cells);
    printf("retry rc=%d new_reads=%d bad=%zu/%zu out=%.12g\n",rc,calls-previous_calls,retry_bad,3*cells,exec(&m,&s));
    assert(retry_bad==0);
    if (should_fail) assert(calls>previous_calls);
    else assert(calls==previous_calls);
    moe_stream_release_slot(&m,0,&s); moe_stream_cache_free(m.moe_cache); moe_stream_thread_cleanup();
    if (mapped) assert(!munmap(map,size));
    assert(!close(fd)); assert(!unlink(path)); free(data); return 0;
}
