#include "test_core.h"
#include "test_synth_gguf.h"

#include <unistd.h>

void run_gguf_tests(void) {
	static const struct {
		uint32_t type;
		size_t	 width;
	} cases[] = {
		{GGUF_TYPE_U8, 1},	{GGUF_TYPE_I8, 1},	{GGUF_TYPE_BOOL, 1}, {GGUF_TYPE_U16, 2},
		{GGUF_TYPE_I16, 2}, {GGUF_TYPE_U32, 4}, {GGUF_TYPE_I32, 4},	 {GGUF_TYPE_F32, 4},
		{GGUF_TYPE_U64, 8}, {GGUF_TYPE_I64, 8}, {GGUF_TYPE_F64, 8},
	};
	char path[512];
	snprintf(path, sizeof(path), "%s/gguf-array-XXXXXX", synth_fixture_dir);
	int fd = mkstemp(path);
	if (fd < 0) {
		record_resultf(OPFAM_EDGE_CASE, "gguf.array_fixture", 0, "mkstemp failed");
		return;
	}
	close(fd);
	uint8_t		   data[24];
	const uint16_t endian_probe = 1;
	int			   big_endian	= *(const uint8_t *)&endian_probe == 0;
	for (size_t i = 0; i < sizeof(data); i++)
		data[i] = (uint8_t)(0x80 + i * 7);
	for (size_t ci = 0; ci < ARRAY_LEN(cases); ci++) {
		size_t	width = cases[ci].width;
		uint8_t expected[sizeof(data)];
		for (size_t i = 0; i < 3; i++)
			for (size_t j = 0; j < width; j++)
				expected[i * width + j] = data[i * width + (big_endian ? width - 1 - j : j)];
		for (int empty = 0; empty <= 1; empty++) {
			size_t count = empty ? 0 : 3;
			tsg_build_array_file(path, cases[ci].type, count, data, count * width);
			gguf_ctx	ctx;
			status_code st = gguf_load(&ctx, path);
			int			ok = st == OK;
			if (ok)
				ok = ctx.n_kv == 1 && ctx.kv_arr_type[0] == cases[ci].type &&
					 ctx.kv_arr_len[0] == count &&
					 memcmp(ctx.kv_arr_data[0], expected, count * width) == 0;
			record_resultf(OPFAM_EDGE_CASE, "gguf.numeric_array", ok, "type=%u count=%zu",
						   cases[ci].type, count);
			gguf_free(&ctx);
		}
	}
	static const struct {
		uint32_t type;
		uint64_t count;
	} invalid[] = {
		{GGUF_TYPE_U8, 1000},		 {GGUF_TYPE_U16, UINT64_MAX}, {GGUF_TYPE_U32, UINT64_MAX},
		{GGUF_TYPE_U64, UINT64_MAX}, {GGUF_TYPE_ARRAY, 0},		  {UINT32_MAX, 0},
	};
	for (size_t i = 0; i < ARRAY_LEN(invalid); i++) {
		tsg_build_array_file(path, invalid[i].type, invalid[i].count, data, sizeof(data));
		gguf_ctx	ctx;
		status_code st = gguf_load(&ctx, path);
		record_resultf(OPFAM_EDGE_CASE, "gguf.invalid_array", st == ERR_FORMAT, "type=%u",
					   invalid[i].type);
		gguf_free(&ctx);
	}
	static const struct {
		uint64_t	tensors;
		uint64_t	metadata;
		status_code expected;
	} headers[] = {
		{0, 0, OK},
		{UINT64_MAX, 0, ERR_FORMAT},
		{0, UINT64_MAX, ERR_FORMAT},
		{UINT64_C(1) << 32, 0, ERR_FORMAT},
		{0, UINT64_C(1) << 32, ERR_FORMAT},
		{1, 0, ERR_FORMAT},
		{0, 1, ERR_FORMAT},
	};
	for (size_t i = 0; i < ARRAY_LEN(headers); i++) {
		uint8_t header[32] = {'G', 'G', 'U', 'F', 3};
		for (size_t byte = 0; byte < 8; byte++) {
			header[8 + byte]  = (uint8_t)(headers[i].tensors >> (8 * byte));
			header[16 + byte] = (uint8_t)(headers[i].metadata >> (8 * byte));
		}
		FILE *f		= fopen(path, "wb");
		int	  ready = 0;
		if (f) {
			ready = fwrite(header, 1, sizeof(header), f) == sizeof(header);
			if (fclose(f) != 0)
				ready = 0;
		}
		if (!ready) {
			record_resultf(OPFAM_EDGE_CASE, "gguf.header_counts", 0, "fixture write failed");
			continue;
		}
		gguf_ctx	ctx;
		status_code st = gguf_load(&ctx, path);
		int			ok = st == headers[i].expected;
		if (st == OK)
			ok = ok && ctx.n_tensors == headers[i].tensors && ctx.n_kv == headers[i].metadata;
		record_resultf(OPFAM_EDGE_CASE, "gguf.header_counts", ok, "case=%zu status=%d", i, st);
		gguf_free(&ctx);
	}
	unlink(path);
}
