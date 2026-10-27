#include "server/task.h"
#include "test_core.h"

#include <errno.h>
#include <pthread.h>

typedef struct {
	atomic_int	  active;
	atomic_int	  errors;
	atomic_int	  cleaned;
	pthread_key_t key;
	pthread_t	  scheduler;
} task_test_shared;

typedef struct {
	task_test_shared *shared;
	int				  id;
	int				  steps;
	int				  request;
	int				  response;
	int				  visits;
	bool			  stop;
	bool			  completed;
} task_test_data;

static void task_test_cleanup(void *ud) {
	atomic_int *cleaned = ud;
	atomic_fetch_add(cleaned, 1);
}

static void task_test_run(server_task *task, void *ud) {
	task_test_data	 *data	 = ud;
	task_test_shared *shared = data->shared;
	if (!!pthread_equal(pthread_self(), shared->scheduler) != !SERVER_TASK_THREADED)
		atomic_fetch_add(&shared->errors, 1);
#if SERVER_TASK_THREADED
	static _Thread_local int tls_visits;
	if (pthread_setspecific(shared->key, &shared->cleaned) != 0)
		atomic_fetch_add(&shared->errors, 1);
#endif
	int local = data->id;
	for (int i = 0; i < data->steps && !data->stop; i++) {
		if (atomic_fetch_add(&shared->active, 1) != 0)
			atomic_fetch_add(&shared->errors, 1);
		if (i > 0 && data->response != local * 2)
			atomic_fetch_add(&shared->errors, 1);
#if SERVER_TASK_THREADED
		if (++tls_visits != i + 1)
			atomic_fetch_add(&shared->errors, 1);
#endif
		local += i + 1;
		data->request = local;
		data->visits++;
		atomic_fetch_sub(&shared->active, 1);
		server_task_yield(task);
	}
	data->completed = true;
}

static void task_test_immediate(server_task *task, void *ud) {
	(void)task;
	int *result = ud;
	*result		= 123;
}

static void task_test_child(server_task *task, void *ud) {
	int *value = ud;
	*value	   = 1;
	server_task_yield(task);
	*value = 2;
}

static void task_test_nested(server_task *task, void *ud) {
	int			*result = ud;
	int			 value	= 0;
	server_task *child	= NULL;
	if (server_task_create(&child, 0, task_test_child, &value) != 0)
		return;
	bool first		 = server_task_resume(child);
	int	 first_value = value;
	server_task_yield(task);
	bool second = server_task_resume(child);
	if (!second)
		while (!server_task_resume(child)) {
		}
	server_task_join(child);
	*result = !first && first_value == 1 && second && value == 2;
}

void run_server_task_tests(void) {
	server_task *task	= NULL;
	int			 result = 0;
	int			 rc		= server_task_create(&task, 0, NULL, &result);
	record_resultf(OPFAM_EDGE_CASE, "server.task_invalid_entry", rc == EINVAL && task == NULL,
				   "rc=%d", rc);
	rc = server_task_create(&task, 1, task_test_immediate, &result);
	record_resultf(OPFAM_EDGE_CASE, "server.task_invalid_stack", rc == EINVAL && task == NULL,
				   "rc=%d", rc);
	rc = server_task_create(&task, (size_t)16 << 20, task_test_immediate, &result);
	record_resultf(OPFAM_EDGE_CASE, "server.task_create", rc == 0, "rc=%d", rc);
	if (rc == 0) {
		record_resultf(OPFAM_EDGE_CASE, "server.task_initially_parked", result == 0, "result=%d",
					   result);
		bool done = server_task_resume(task);
		record_resultf(OPFAM_EDGE_CASE, "server.task_immediate", done && result == 123,
					   "done=%d result=%d", done, result);
		record_resultf(OPFAM_EDGE_CASE, "server.task_completed_resume", server_task_resume(task),
					   "completed task stays completed");
		server_task_join(task);
	}
	result = 0;
	rc	   = server_task_create(&task, 0, task_test_nested, &result);
	record_resultf(OPFAM_EDGE_CASE, "server.task_nested_create", rc == 0, "rc=%d", rc);
	if (rc == 0) {
		bool yielded = !server_task_resume(task);
		bool done	 = server_task_resume(task);
		while (!done)
			done = server_task_resume(task);
		server_task_join(task);
		record_resultf(OPFAM_EDGE_CASE, "server.task_nested_resume", yielded && result == 1,
					   "yielded=%d result=%d", yielded, result);
	}

	task_test_shared shared = {.scheduler = pthread_self()};
	rc						= pthread_key_create(&shared.key, task_test_cleanup);
	record_resultf(OPFAM_EDGE_CASE, "server.task_cleanup_key", rc == 0, "rc=%d", rc);
	if (rc != 0)
		return;
	server_task	  *tasks[4] = {0};
	task_test_data data[4]	= {0};
	const int	   steps[]	= {1, 17, 97, 64};
	int			   created	= 0;
	for (int i = 0; i < 4; i++) {
		data[i] = (task_test_data){.shared = &shared, .id = i + 1, .steps = steps[i]};
		rc		= server_task_create(&tasks[i], 0, task_test_run, &data[i]);
		if (rc != 0)
			break;
		created++;
	}
	record_resultf(OPFAM_EDGE_CASE, "server.task_multiple_create", created == 4, "created=%d rc=%d",
				   created, rc);
	for (int round = 0; round <= 97; round++) {
		for (int i = 0; i < created; i++) {
			if (round == 7 && i == 2)
				data[i].stop = true;
			bool done = server_task_resume(tasks[i]);
			if (done != data[i].completed || atomic_load(&shared.active) != 0)
				atomic_fetch_add(&shared.errors, 1);
			atomic_fetch_add(&shared.active, 1);
			data[i].response = data[i].request * 2;
			atomic_fetch_sub(&shared.active, 1);
		}
	}
	for (int i = 0; i < created; i++) {
		server_task_join(tasks[i]);
		int expected = i == 2 ? 7 : steps[i];
		record_resultf(OPFAM_EDGE_CASE, "server.task_round_robin",
					   data[i].completed && data[i].visits == expected,
					   "task=%d visits=%d expected=%d", i, data[i].visits, expected);
	}
	record_resultf(OPFAM_EDGE_CASE, "server.task_handoff", atomic_load(&shared.errors) == 0,
				   "errors=%d", atomic_load(&shared.errors));
	record_resultf(OPFAM_EDGE_CASE, "server.task_thread_cleanup",
				   atomic_load(&shared.cleaned) == (SERVER_TASK_THREADED ? created : 0),
				   "cleaned=%d", atomic_load(&shared.cleaned));
	pthread_key_delete(shared.key);
}
