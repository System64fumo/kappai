#ifndef TUI_HF_META_H
#define TUI_HF_META_H

#include <stdbool.h>
#include <stddef.h>

#define HF_META_ARCH_MAX 48
#define HF_META_TYPE_SLOTS 48
#define HF_META_BAD_TYPES 8

typedef struct {
	char	  arch[HF_META_ARCH_MAX];
	long long n_layer;
	long long n_embd;
	long long n_head;
	long long head_kv;
	long long kv_sum;
	long long key_len;
	long long val_len;
	long long n_ctx;
	long long n_expert;
	long long n_nextn;
	bool	  kv_ok;
	bool	  tensors_ok;
	long long n_tensors;
	double	  type_bytes[HF_META_TYPE_SLOTS];
	unsigned  bad_types[HF_META_BAD_TYPES];
	int		  n_bad_types;
	int		  n_bad_tensors;
	int		  n_bad_mtp_tensors;
} hf_meta;

void   hf_meta_init(hf_meta *m);
int	   hf_meta_scan(const unsigned char *buf, size_t len, hf_meta *m);
double hf_meta_kv_per_token(const hf_meta *m, double elem_bytes);
bool   hf_meta_blocked(const hf_meta *m);
void   hf_meta_bad_list(const hf_meta *m, char *out, size_t cap);
void   hf_meta_mix(const hf_meta *m, char *out, size_t cap);

#endif
