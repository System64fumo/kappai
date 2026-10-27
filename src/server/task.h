#ifndef SERVER_TASK_H
#define SERVER_TASK_H

#include <stdbool.h>
#include <stddef.h>

#ifndef KAPPAI_TASK_UCONTEXT
#define KAPPAI_TASK_UCONTEXT 0
#endif

#if KAPPAI_TASK_UCONTEXT
#define SERVER_TASK_THREADED 0
#define SERVER_TASK_BACKEND_NAME "ucontext"
#else
#define SERVER_TASK_THREADED 1
#define SERVER_TASK_BACKEND_NAME "pthread"
#endif

typedef struct server_task server_task;
typedef void (*server_task_fn)(server_task *task, void *ud);

int	 server_task_create(server_task **out, size_t stack_size, server_task_fn fn, void *ud);
bool server_task_resume(server_task *task);
void server_task_yield(server_task *task);
void server_task_join(server_task *task);

#endif
