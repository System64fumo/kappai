#if defined(__APPLE__) && KAPPAI_TASK_UCONTEXT
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#endif

#include "task.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>

#if KAPPAI_TASK_UCONTEXT

#include <assert.h>
#include <stdint.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

struct server_task {
	ucontext_t	   context;
	ucontext_t	   caller;
	pthread_t	   owner;
	server_task_fn fn;
	void		  *ud;
	void		  *mapping;
	size_t		   mapping_size;
	bool		   done;
};

static _Thread_local server_task *task_entering;

static void server_task_main(void) {
	server_task *task = task_entering;
	task->fn(task, task->ud);
	task->done = true;
}

int server_task_create(server_task **out, size_t stack_size, server_task_fn fn, void *ud) {
	if (!out)
		return EINVAL;
	*out = NULL;
	if (!fn)
		return EINVAL;
	long page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0)
		return EINVAL;
	size_t page = (size_t)page_size;
	if (stack_size == 0)
		stack_size = (size_t)16 << 20;
	if (stack_size < page)
		return EINVAL;
	if (stack_size > SIZE_MAX - 3 * page)
		return ENOMEM;
	stack_size		  = (stack_size + page - 1) / page * page;
	server_task *task = calloc(1, sizeof(*task));
	if (!task)
		return ENOMEM;
	task->fn		   = fn;
	task->ud		   = ud;
	task->owner		   = pthread_self();
	task->mapping_size = stack_size + 2 * page;
	int flags		   = MAP_PRIVATE;
#ifdef MAP_ANONYMOUS
	flags |= MAP_ANONYMOUS;
#else
	flags |= MAP_ANON;
#endif
	task->mapping = mmap(NULL, task->mapping_size, PROT_NONE, flags, -1, 0);
	if (task->mapping == MAP_FAILED) {
		int rc = errno;
		free(task);
		return rc;
	}
	void *stack = (char *)task->mapping + page;
	if (mprotect(stack, stack_size, PROT_READ | PROT_WRITE) != 0 ||
		getcontext(&task->context) != 0) {
		int rc = errno;
		munmap(task->mapping, task->mapping_size);
		free(task);
		return rc;
	}
	task->context.uc_stack.ss_sp	= stack;
	task->context.uc_stack.ss_size	= stack_size;
	task->context.uc_stack.ss_flags = 0;
	task->context.uc_link			= &task->caller;
	makecontext(&task->context, server_task_main, 0);
	*out = task;
	return 0;
}

bool server_task_resume(server_task *task) {
	assert(pthread_equal(task->owner, pthread_self()));
	if (!task->done) {
		server_task *previous = task_entering;
		task_entering		  = task;
		if (swapcontext(&task->caller, &task->context) != 0)
			abort();
		task_entering = previous;
	}
	return task->done;
}

void server_task_yield(server_task *task) {
	assert(pthread_equal(task->owner, pthread_self()));
	assert(task_entering == task && !task->done);
	if (swapcontext(&task->context, &task->caller) != 0)
		abort();
}

void server_task_join(server_task *task) {
	if (!task)
		return;
	assert(task->done && pthread_equal(task->owner, pthread_self()));
	munmap(task->mapping, task->mapping_size);
	free(task);
}

#else

struct server_task {
	pthread_t		thread;
	pthread_mutex_t mtx;
	pthread_cond_t	cv;
	server_task_fn	fn;
	void		   *ud;
	bool			running;
	bool			done;
};

static void *server_task_main(void *ud) {
	server_task *task = ud;
	pthread_mutex_lock(&task->mtx);
	while (!task->running)
		pthread_cond_wait(&task->cv, &task->mtx);
	pthread_mutex_unlock(&task->mtx);

	task->fn(task, task->ud);

	pthread_mutex_lock(&task->mtx);
	task->running = false;
	task->done	  = true;
	pthread_cond_broadcast(&task->cv);
	pthread_mutex_unlock(&task->mtx);
	return NULL;
}

int server_task_create(server_task **out, size_t stack_size, server_task_fn fn, void *ud) {
	if (!out)
		return EINVAL;
	*out = NULL;
	if (!fn)
		return EINVAL;
	server_task *task = calloc(1, sizeof(*task));
	if (!task)
		return ENOMEM;
	task->fn = fn;
	task->ud = ud;
	int rc	 = pthread_mutex_init(&task->mtx, NULL);
	if (rc != 0)
		goto fail_alloc;
	rc = pthread_cond_init(&task->cv, NULL);
	if (rc != 0)
		goto fail_mutex;
	pthread_attr_t attr;
	rc = pthread_attr_init(&attr);
	if (rc != 0)
		goto fail_cond;
	if (stack_size > 0)
		rc = pthread_attr_setstacksize(&attr, stack_size);
	if (rc == 0)
		rc = pthread_create(&task->thread, &attr, server_task_main, task);
	pthread_attr_destroy(&attr);
	if (rc != 0)
		goto fail_cond;
	*out = task;
	return 0;

fail_cond:
	pthread_cond_destroy(&task->cv);
fail_mutex:
	pthread_mutex_destroy(&task->mtx);
fail_alloc:
	free(task);
	return rc;
}

bool server_task_resume(server_task *task) {
	pthread_mutex_lock(&task->mtx);
	if (!task->done) {
		task->running = true;
		pthread_cond_broadcast(&task->cv);
		while (task->running)
			pthread_cond_wait(&task->cv, &task->mtx);
	}
	bool done = task->done;
	pthread_mutex_unlock(&task->mtx);
	return done;
}

void server_task_yield(server_task *task) {
	pthread_mutex_lock(&task->mtx);
	task->running = false;
	pthread_cond_broadcast(&task->cv);
	while (!task->running)
		pthread_cond_wait(&task->cv, &task->mtx);
	pthread_mutex_unlock(&task->mtx);
}

void server_task_join(server_task *task) {
	if (!task)
		return;
	pthread_join(task->thread, NULL);
	pthread_cond_destroy(&task->cv);
	pthread_mutex_destroy(&task->mtx);
	free(task);
}

#endif
