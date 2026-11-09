#include "rpc_page.h"

#include "dialog.h"
#include "rpc/rpc.h"
#include "schema.h"
#include "ui.h"

#include <string.h>

#define STATUS_MAX 160

typedef struct {
	rpc_endpoint eps[RPC_MAX_ENDPOINTS];
	char		 status[RPC_MAX_ENDPOINTS][STATUS_MAX];
	int			 n;
	int			 row;
	char		 message[SC_ERR_MAX];
	int			 message_kind;
} page_state;

static void set_message(page_state *st, int kind, const char *msg) {
	snprintf(st->message, sizeof(st->message), "%s", msg);
	st->message_kind = kind;
}

static void probe_one(page_state *st, int i) {
	rpc_info	info;
	status_code s = rpc_probe(&st->eps[i], &info);
	if (s != OK) {
		snprintf(st->status[i], STATUS_MAX, "unreachable");
		return;
	}
	snprintf(st->status[i], STATUS_MAX, "%s (%s)", info.identity[0] ? info.identity : "device",
			 info.device);
}

static void probe_all(page_state *st) {
	for (int i = 0; i < st->n; i++)
		probe_one(st, i);
}

static void draw(const page_state *st) {
	erase();
	ui_putn(0, 2, " rpc workers ", 13, A_REVERSE | A_BOLD);
	move(1, 1);
	attron(ui_attr(UI_DIM));
	for (int i = 1; i < COLS - 1; i++)
		addch(ACS_HLINE);
	attroff(ui_attr(UI_DIM));

	if (st->n == 0)
		ui_putn(3, 3, "no workers added; press A to add one", COLS - 6, ui_attr(UI_DIM));
	for (int i = 0; i < st->n; i++) {
		char name[16], addr[RPC_HOST_MAX + 16];
		snprintf(name, sizeof(name), "rpc%d", i);
		rpc_endpoint_format(&st->eps[i], addr, sizeof(addr));
		bool   selected = i == st->row;
		chtype attr		= selected ? (A_REVERSE | A_BOLD) : 0;
		int	   y		= 3 + i;
		if (selected) {
			ui_putn(y, 1, ">", 1, A_BOLD);
			ui_fill(y, 3, 1, COLS - 5, A_REVERSE);
		}
		ui_putn(y, 3, name, 8, attr);
		ui_putn(y, 12, addr, 32, attr);
		ui_putn(y, 46, st->status[i], COLS - 48, attr);
	}
	if (st->message[0])
		ui_putn(LINES - 2, 2, st->message, COLS - 4, ui_attr(st->message_kind));
	ui_putn(LINES - 1, 2, "Up/Down: select   A: add   X: remove   T: test   Esc: back", COLS - 4,
			ui_attr(UI_DIM));
	refresh();
}

static int validate_endpoint(const char *value, char *err, size_t errlen, void *ud) {
	(void)ud;
	rpc_endpoint ep;
	return rpc_endpoint_parse(value, &ep, err, errlen);
}

static void persist(page_state *st) {
	char err[SC_ERR_MAX];
	if (rpc_endpoints_save(st->eps, st->n, err, sizeof(err)) < 0) {
		set_message(st, UI_ERR, err);
		beep();
		return;
	}
	set_message(st, UI_OK, "saved");
}

static void add_worker(page_state *st) {
	if (st->n >= RPC_MAX_ENDPOINTS) {
		set_message(st, UI_ERR, "worker limit reached");
		return;
	}
	char buf[RPC_HOST_MAX + 16];
	if (ui_input("Worker address (host or host:port)", "", buf, sizeof(buf), validate_endpoint,
				 NULL, NULL) <= 0)
		return;
	char err[SC_ERR_MAX];
	if (rpc_endpoint_parse(buf, &st->eps[st->n], err, sizeof(err)) < 0) {
		set_message(st, UI_ERR, err);
		return;
	}
	probe_one(st, st->n);
	st->row = st->n++;
	persist(st);
}

static void remove_worker(page_state *st) {
	if (st->n == 0)
		return;
	for (int i = st->row; i + 1 < st->n; i++) {
		st->eps[i] = st->eps[i + 1];
		memcpy(st->status[i], st->status[i + 1], STATUS_MAX);
	}
	st->n--;
	st->row = st->row >= st->n ? MAX(st->n - 1, 0) : st->row;
	persist(st);
}

void rpc_page_run(void) {
	page_state st;
	memset(&st, 0, sizeof(st));
	st.n = rpc_endpoints_load(st.eps, RPC_MAX_ENDPOINTS);
	probe_all(&st);

	for (;;) {
		draw(&st);
		int ch = ui_getch();
		if (ch == KBD_INTERRUPT || ch == 27)
			return;
		switch (ch) {
		case KEY_UP:
			st.row = st.n ? (st.row + st.n - 1) % st.n : 0;
			break;
		case KEY_DOWN:
			st.row = st.n ? (st.row + 1) % st.n : 0;
			break;
		case 'a':
		case 'A':
			add_worker(&st);
			break;
		case 'x':
		case 'X':
			remove_worker(&st);
			break;
		case 't':
		case 'T':
			probe_all(&st);
			set_message(&st, UI_OK, "workers tested");
			break;
		default:
			break;
		}
	}
}
