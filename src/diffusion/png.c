#include "diffusion/diffusion.h"
#include "log.h"

#include <stdio.h>

#define DEFLATE_STORED_MAX 65535

static uint32_t crc_table[256];

static void crc_init(void) {
	for (uint32_t n = 0; n < 256; n++) {
		uint32_t c = n;
		for (int k = 0; k < 8; k++)
			c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
		crc_table[n] = c;
	}
}

static uint32_t crc_update(uint32_t crc, const uint8_t *buf, size_t len) {
	for (size_t i = 0; i < len; i++)
		crc = crc_table[(crc ^ buf[i]) & 0xff] ^ (crc >> 8);
	return crc;
}

static void put_be32(uint8_t *p, uint32_t v) {
	p[0] = (uint8_t)(v >> 24);
	p[1] = (uint8_t)(v >> 16);
	p[2] = (uint8_t)(v >> 8);
	p[3] = (uint8_t)v;
}

static bool write_chunk(FILE *fp, const char type[4], const uint8_t *data, size_t len) {
	uint8_t head[8];
	put_be32(head, (uint32_t)len);
	memcpy(head + 4, type, 4);
	uint32_t crc = crc_update(0xffffffffu, (const uint8_t *)type, 4);
	crc			 = crc_update(crc, data, len) ^ 0xffffffffu;
	uint8_t tail[4];
	put_be32(tail, crc);
	return fwrite(head, 1, 8, fp) == 8 && (len == 0 || fwrite(data, 1, len, fp) == len) &&
		   fwrite(tail, 1, 4, fp) == 4;
}

static uint32_t adler32(const uint8_t *buf, size_t len) {
	uint32_t a = 1, b = 0;
	for (size_t i = 0; i < len; i++) {
		a = (a + buf[i]) % 65521;
		b = (b + a) % 65521;
	}
	return (b << 16) | a;
}

static uint8_t *zlib_stored(const uint8_t *raw, size_t raw_len, size_t *out_len) {
	size_t	 n_blocks = (raw_len + DEFLATE_STORED_MAX - 1) / DEFLATE_STORED_MAX;
	size_t	 cap	  = 2 + raw_len + n_blocks * 5 + 4;
	uint8_t *out	  = xmalloc(cap);
	size_t	 pos	  = 0;
	out[pos++]		  = 0x78;
	out[pos++]		  = 0x01;
	for (size_t off = 0; off < raw_len || off == 0; off += DEFLATE_STORED_MAX) {
		size_t len = MIN(raw_len - off, (size_t)DEFLATE_STORED_MAX);
		bool   fin = off + len >= raw_len;
		out[pos++] = fin ? 1 : 0;
		out[pos++] = (uint8_t)(len & 0xff);
		out[pos++] = (uint8_t)(len >> 8);
		out[pos++] = (uint8_t)(~len & 0xff);
		out[pos++] = (uint8_t)((~len >> 8) & 0xff);
		memcpy(out + pos, raw + off, len);
		pos += len;
		if (fin)
			break;
	}
	put_be32(out + pos, adler32(raw, raw_len));
	*out_len = pos + 4;
	return out;
}

static bool write_png_file(FILE *fp, const diffusion_image *img) {
	static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};

	size_t	 row_bytes = (size_t)img->width * img->channels;
	size_t	 raw_len   = (row_bytes + 1) * img->height;
	uint8_t *raw	   = xmalloc(raw_len);
	for (int y = 0; y < img->height; y++) {
		raw[(size_t)y * (row_bytes + 1)] = 0;
		memcpy(raw + (size_t)y * (row_bytes + 1) + 1, img->pixels + (size_t)y * row_bytes,
			   row_bytes);
	}
	size_t	 z_len;
	uint8_t *z = zlib_stored(raw, raw_len, &z_len);
	free(raw);

	uint8_t ihdr[13] = {0};
	put_be32(ihdr, (uint32_t)img->width);
	put_be32(ihdr + 4, (uint32_t)img->height);
	ihdr[8] = 8;
	ihdr[9] = img->channels == 4 ? 6 : 2;
	bool ok = fwrite(sig, 1, sizeof(sig), fp) == sizeof(sig) && write_chunk(fp, "IHDR", ihdr, 13) &&
			  write_chunk(fp, "IDAT", z, z_len) && write_chunk(fp, "IEND", NULL, 0);
	free(z);
	return ok;
}

status_code diffusion_image_write_png(const char *path, const diffusion_image *img) {
	if (!crc_table[1])
		crc_init();

	char tmp[4096];
	snprintf(tmp, sizeof(tmp), "%s.tmp", path);
	FILE *fp = fopen(tmp, "wb");
	if (!fp) {
		ERROR("cannot open '%s' for writing", tmp);
		return ERR_IO;
	}
	bool ok = write_png_file(fp, img);
	ok &= fclose(fp) == 0;
	if (!ok || rename(tmp, path) != 0) {
		remove(tmp);
		ERROR("failed to write '%s'", path);
		return ERR_IO;
	}
	return OK;
}

void diffusion_image_free(diffusion_image *img) {
	free(img->pixels);
	memset(img, 0, sizeof(*img));
}
