#include "memconfig.h"
#include "monitor.h"
#include "test_core.h"
#include "threadpool.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>

typedef struct {
	atomic_int completed;
} portability_job;

static void portability_chunk(int begin, int end, int tid, void *ud) {
	(void)tid;
	portability_job *job = ud;
	atomic_fetch_add(&job->completed, end - begin);
}

static void test_posix_file_io(void) {
	char path[512];
	snprintf(path, sizeof(path), "%s/posix-io-XXXXXX", synth_fixture_dir);
	int fd = mkstemp(path);
	if (fd < 0) {
		record_resultf(OPFAM_EDGE_CASE, "posix.file_fixture", 0, "mkstemp: %s", strerror(errno));
		return;
	}
	const char data[] = "0123456789";
	int		   ready  = write(fd, data, sizeof(data)) == (ssize_t)sizeof(data);
	record_resultf(OPFAM_EDGE_CASE, "posix.file_fixture", ready, "write fixture");
	if (ready) {
		file_advise(fd, 0, 0, FILE_ADVICE_RANDOM);
		file_advise(fd, 0, sizeof(data), FILE_ADVICE_RELEASE);
		char buf[32] = {0};
		lseek(fd, 3, SEEK_SET);
		ssize_t n  = gguf_pread_full(fd, buf, 5, 2);
		int		ok = n == 5 && memcmp(buf, "23456", 5) == 0 && lseek(fd, 0, SEEK_CUR) == 3;
		record_resultf(OPFAM_EDGE_CASE, "posix.positioned_read", ok, "read=%zd", n);
		n = gguf_pread_full(fd, buf, sizeof(buf), 8);
		record_resultf(OPFAM_EDGE_CASE, "posix.read_eof", n == 3 && memcmp(buf, "89", 3) == 0,
					   "read=%zd", n);
		n = gguf_pread_full(-1, buf, 1, 0);
		record_resultf(OPFAM_EDGE_CASE, "posix.read_error", n == -1 && errno == EBADF,
					   "read=%zd errno=%d", n, errno);
		n = gguf_pread_full(fd, buf, 1, UINT64_MAX);
		record_resultf(OPFAM_EDGE_CASE, "posix.read_offset_overflow", n == -1 && errno == EOVERFLOW,
					   "read=%zd errno=%d", n, errno);

		int direct_fd = open(path, O_RDONLY);
		if (direct_fd >= 0) {
			size_t		align = 0;
			status_code s	  = direct_io_probe_fd(direct_fd, "test", 0, 1, &align);
			ok				  = s == OK ? align >= sizeof(void *) && (align & (align - 1)) == 0
										: fcntl(direct_fd, F_GETFD) < 0 && errno == EBADF;
			record_resultf(OPFAM_EDGE_CASE, "posix.optional_direct_io", ok, "status=%d align=%zu",
						   s, align);
			if (s == OK)
				close(direct_fd);
		} else {
			record_resultf(OPFAM_EDGE_CASE, "posix.optional_direct_io", 0, "open failed");
		}
	}
	close(fd);
	unlink(path);
}

static void test_posix_monitor(void) {
	char path[512];
	snprintf(path, sizeof(path), "%s/mon-XXXXXX", synth_fixture_dir);
	int fd = mkstemp(path);
	if (fd < 0) {
		record_resultf(OPFAM_EDGE_CASE, "posix.monitor_fixture", 0, "mkstemp failed");
		return;
	}
	close(fd);
	unlink(path);
	monitor		mon;
	status_code s = monitor_init(&mon, path);
	record_resultf(OPFAM_EDGE_CASE, "posix.monitor_init", s == OK, "status=%d", s);
	if (s != OK) {
		monitor_free(&mon);
		return;
	}
	int flags	= fcntl(mon.fd, F_GETFL);
	int fdflags = fcntl(mon.fd, F_GETFD);
	int ok		= flags >= 0 && fdflags >= 0 && (flags & O_NONBLOCK) && (fdflags & FD_CLOEXEC);
	record_resultf(OPFAM_EDGE_CASE, "posix.monitor_listener_flags", ok, "nonblock and cloexec");
	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0) {
		record_resultf(OPFAM_EDGE_CASE, "posix.monitor_connect", 0, "socket failed");
		monitor_free(&mon);
		return;
	}
	struct sockaddr_un addr = {0};
	addr.sun_family			= AF_UNIX;
	size_t len				= strlen(path);
	if (len >= sizeof(addr.sun_path)) {
		record_resultf(OPFAM_EDGE_CASE, "posix.monitor_connect", 0, "socket path too long");
		close(fd);
		monitor_free(&mon);
		return;
	}
	memcpy(addr.sun_path, path, len + 1);
	socklen_t addr_len = (socklen_t)(offsetof(struct sockaddr_un, sun_path) + len + 1);
	ok				   = connect(fd, (struct sockaddr *)&addr, addr_len) == 0;
	record_resultf(OPFAM_EDGE_CASE, "posix.monitor_connect", ok, "connect");
	if (ok) {
		monitor_poll(&mon);
		int client = mon.clients[0];
		flags	   = fcntl(client, F_GETFL);
		fdflags	   = fcntl(client, F_GETFD);
		ok		   = client >= 0 && flags >= 0 && fdflags >= 0 && (flags & O_NONBLOCK) &&
					 (fdflags & FD_CLOEXEC);
		record_resultf(OPFAM_EDGE_CASE, "posix.monitor_client_flags", ok, "nonblock and cloexec");
		monitor_send(&mon, "%s", "{\"test\":true}");
		struct pollfd pfd		 = {.fd = fd, .events = POLLIN};
		char		  buf[64]	 = {0};
		ssize_t		  n			 = poll(&pfd, 1, 1000) > 0 ? read(fd, buf, sizeof(buf)) : -1;
		const char	  expected[] = "{\"test\":true}\n";
		ok						 = n == (ssize_t)(sizeof(expected) - 1) &&
								   memcmp(buf, expected, sizeof(expected) - 1) == 0;
		record_resultf(OPFAM_EDGE_CASE, "posix.monitor_send", ok, "read=%zd", n);
		close(fd);
		fd = -1;
		monitor_send(&mon, "%s", "disconnected");
		record_resultf(OPFAM_EDGE_CASE, "posix.monitor_disconnect",
					   atomic_load(&mon.n_clients) == 0, "clients=%d", atomic_load(&mon.n_clients));
	}
	if (fd >= 0)
		close(fd);
	monitor_free(&mon);
	ok = access(path, F_OK) < 0 && errno == ENOENT;
	record_resultf(OPFAM_EDGE_CASE, "posix.monitor_cleanup", ok, "socket removed");
}

void run_portability_tests(void) {
#if defined(__APPLE__)
	run_host_memory_tests();
#endif
	test_posix_file_io();
	test_posix_monitor();
	void *local = xmalloc(16);
	tlocal_register(&local);
	tlocal_free_all();
	record_resultf(OPFAM_EDGE_CASE, "posix.thread_local_cleanup", local == NULL,
				   "registered allocation freed");
	const char data[] = {'a', '\0', 'b', 'a', 'b', 'a', 'b'};
	static const struct {
		const char *needle;
		size_t		len;
		size_t		needle_len;
		int			offset;
	} cases[] = {
		{"", 0, 0, 0},
		{"a", 0, 1, -1},
		{"ab", sizeof(data), 2, 3},
		{"ab", 4, 2, -1},
		{"ab", 2, 2, -1},
		{"\0b", sizeof(data), 2, 1},
		{"bab", sizeof(data), 3, 2},
		{"abab", sizeof(data), 4, 3},
		{"abababab", sizeof(data), 8, -1},
	};
	for (size_t i = 0; i < ARRAY_LEN(cases); i++) {
		const char *hit	   = str_find_n(data, cases[i].len, cases[i].needle, cases[i].needle_len);
		int			offset = hit ? (int)(hit - data) : -1;
		record_resultf(OPFAM_EDGE_CASE, "common.bounded_search", offset == cases[i].offset,
					   "case=%zu offset=%d", i, offset);
	}
	int n = available_cpu_count();
	record_resultf(OPFAM_EDGE_CASE, "common.cpu_count", n >= 1, "cpus=%d", n);
#if defined(__linux__) || defined(__APPLE__)
	size_t total	 = get_total_memory();
	size_t available = get_available_memory();
	record_resultf(OPFAM_EDGE_CASE, "memory.total", total > 0, "bytes=%zu", total);
	record_resultf(OPFAM_EDGE_CASE, "memory.available", available > 0 && available <= total,
				   "available=%zu total=%zu", available, total);
#endif

	portability_job job	 = {0};
	tpool		   *pool = tpool_create(2);
	for (int i = 0; i < 32; i++)
		tpool_parallel_for(pool, 127, 1, portability_chunk, &job);
	tpool_destroy(pool);
	record_resultf(OPFAM_EDGE_CASE, "threadpool.repeated_dispatch",
				   atomic_load(&job.completed) == 32 * 127, "completed=%d",
				   atomic_load(&job.completed));
}
