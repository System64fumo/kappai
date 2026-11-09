#include "backend/backend.h"
#include "log.h"
#include "rpc/rpc.h"
#include "rpc/server/session.h"

#include <getopt.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <sys/socket.h>

typedef struct {
	int	 fd;
	char device[BACKEND_DEVICE_NAME_MAX];
} conn_args;

static void usage(const char *prog) {
	printf("usage: %s [--host <addr>] [--port <port>] [--device <device>]\n"
		   "  --host    address to listen on (default %s)\n"
		   "  --port    port to listen on (default %d)\n"
		   "  --device  device to share with the cluster (default: best available)\n",
		   prog, RPC_DEFAULT_BIND, RPC_DEFAULT_PORT);
}

static void set_keepalive(int fd) {
	int on = 1;
	setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
#ifdef TCP_KEEPIDLE
	int idle = RPC_KEEPALIVE_IDLE_S, interval = RPC_KEEPALIVE_INTERVAL_S, count = RPC_KEEPALIVE_COUNT;
	setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
	setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
	setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
#endif
}

static void *connection_thread(void *arg) {
	conn_args *c = arg;
	session_run(c->fd, c->device);
	free(c);
	return NULL;
}

static bool device_valid(const char *device) {
	char name[BACKEND_DEVICE_NAME_MAX];
	int	 index = 0;
	if (backend_parse_device(device, name, sizeof(name), &index) != 0 || strcmp(name, "rpc") == 0)
		return false;
	char avail[BACKEND_MAX * 8][BACKEND_DEVICE_NAME_MAX];
	int	 n = backend_enum_devices(avail, BACKEND_MAX * 8);
	for (int i = 0; i < n; i++)
		if (strcmp(avail[i], device) == 0)
			return true;
	return strcmp(device, "cpu0") == 0;
}

int main(int argc, char **argv) {
	log_init(log_default_config());
	signal(SIGPIPE, SIG_IGN);

	const char *host							= RPC_DEFAULT_BIND;
	int			port							= RPC_DEFAULT_PORT;
	char		device[BACKEND_DEVICE_NAME_MAX] = "";

	static const struct option opts[] = {{"host", required_argument, NULL, 'h'},
										 {"port", required_argument, NULL, 'p'},
										 {"device", required_argument, NULL, 'd'},
										 {"help", no_argument, NULL, '?'},
										 {0}};
	int						   c;
	while ((c = getopt_long(argc, argv, "", opts, NULL)) != -1) {
		switch (c) {
		case 'h':
			host = optarg;
			break;
		case 'p':
			port = atoi(optarg);
			break;
		case 'd':
			snprintf(device, sizeof(device), "%s", optarg);
			break;
		default:
			usage(argv[0]);
			return c == '?' && optopt == 0 ? 0 : 1;
		}
	}
	if (port < 1 || port > 65535) {
		ERROR("invalid port %d", port);
		return 1;
	}
	if (!device[0])
		backend_best_device(device, sizeof(device));
	if (!device_valid(device)) {
		ERROR("device '%s' is not available on this machine (see kappai-cli --list-devices)",
			  device);
		return 1;
	}

	char err[256];
	int	 listen_fd = rpc_listen(host, port, err, sizeof(err));
	if (listen_fd < 0) {
		ERROR("%s", err);
		return 1;
	}
	INFO("rpc server sharing '%s' on %s:%d", device, host, port);

	for (;;) {
		int fd = accept(listen_fd, NULL, NULL);
		if (fd < 0)
			continue;
		conn_args *args = xcalloc(1, sizeof(*args));
		args->fd		= fd;
		snprintf(args->device, sizeof(args->device), "%s", device);
		int one = 1;
		setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
		set_keepalive(fd);
		pthread_t tid;
		if (pthread_create(&tid, NULL, connection_thread, args) != 0) {
			close(fd);
			free(args);
			continue;
		}
		pthread_detach(tid);
	}
}
