#ifndef HCS_H
#define HCS_H

#include "config.h"

#include <stdbool.h>
#include <stddef.h>

#define HCS_MAX_DEVICES 8
#define HCS_NAME_MAX 64
#define HCS_ERR_MAX 4096

#define HCS_TASK_PREFILL 1u
#define HCS_TASK_DECODE 2u
#define HCS_TASKS_ALL (HCS_TASK_PREFILL | HCS_TASK_DECODE)

#define HCS_COUNT_ALL (-1)
#define HCS_COUNT_UNSET (-2)

typedef struct {
	char	 name[HCS_NAME_MAX];
	unsigned tasks;
	int		 layers;
	int		 experts;
} hcs_device;

typedef struct {
	bool	   enabled;
	int		   n_devices;
	hcs_device devices[HCS_MAX_DEVICES];
} hcs_config;

typedef struct {
	char	   name[HCS_NAME_MAX];
	hcs_device dev;
	unsigned   seen;
} hcs_section;

typedef struct {
	char		devices[HCS_MAX_DEVICES * (HCS_NAME_MAX + 2)];
	bool		has_devices;
	bool		has_enabled;
	bool		enabled;
	bool		devices_seen;
	hcs_section sections[HCS_MAX_DEVICES];
	int			n_sections;
} hcs_loader;

typedef struct {
	bool has_device;
	bool has_ngl;
	bool has_cache;
	char device[HCS_NAME_MAX];
	int	 ngl;
	int	 cache;
} hcs_legacy;

void hcs_defaults(hcs_config *h);
void hcs_set_simple(hcs_config *h, const char *device);

int			hcs_parse_count(const char *text, int *out);
const char *hcs_format_count(int value, char *buf, size_t cap);
int			hcs_parse_tasks(const char *text, unsigned *out);
void		hcs_format_tasks(unsigned tasks, char *buf, size_t cap);
void		hcs_format_devices(const hcs_config *h, char *buf, size_t cap);

void hcs_loader_init(hcs_loader *l);
int	 hcs_loader_open_section(hcs_loader *l, const char *device, char *err, size_t errlen);
int	 hcs_loader_set(hcs_loader *l, int section, const char *key, const char *value, char *err,
					size_t errlen);
int	 hcs_loader_finish(const hcs_loader *l, hcs_config *out, char *err, size_t errlen);

int hcs_migrate_legacy(hcs_config *h, const hcs_legacy *legacy, char *err, size_t errlen);

int hcs_validate(const hcs_config *h, const char *model_path, char *err, size_t errlen);

void hcs_materialize(const hcs_config *h, config *cfg);

#endif
