#include "backend/backend.h"
#include "common.h"
#include "log.h"
#include "profile.h"
#include "recipe.h"
#include "shaders_embedded.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(expr)                                                                             \
	do {                                                                                           \
		VkResult _r = (expr);                                                                      \
		if (_r != VK_SUCCESS)                                                                      \
			return vk_check_fail(p, _r, #expr, __FILE__, __LINE__);                                \
	} while (0)

#define VK_DESC_CACHE_CAP 256
#define VK_MAX_BINDINGS 8
#define VK_RING_DEPTH 4
#define VK_PROF_SLOT_CAP 4096
#define VK_QMM_MIN_M 16
#define VK_READ_SET_CAP 64
#define VK_QMV_ROWS 2
#define VK_QMV_COLS 4
#define VK_QMM_TILE 64
#define VK_XQ_BLOCK_BYTES 48
#define VK_ATTN_DEC_MAX_M 4
#define VK_ATTN_DEC_MAX_HEAD_DIM 512
#define VK_ATTN_DEC_TARGET_WG 256
#define VK_ATTN_DEC_MAX_SPLITS 64
#define VK_ATTN_DEC_MIN_CHUNK 16
#define VK_PROF_NAMES 128
#define VK_DEAD_BUF_CAP 64
#define VK_DIRTY_MAX 128
#define VK_DIRTY_TABLE_SIZE 256
#define VK_MOE_META_SLOTS 2
#define VK_SCRATCH_POOL_CAP 32

typedef enum {
	VK_NN_LAYERNORM_MODULATE,
	VK_NN_ROPE_TABLE,
	VK_NN_ATTENTION_DENSE,
	VK_NN_IM2COL,
	VK_NN_UPSAMPLE_SHORTCUT,
	VK_NN_COUNT,
} vk_nn_pipe;

enum elem_mode {
	ELEM_MODE_COPY			   = 0,
	ELEM_MODE_ADD_INPLACE	   = 1,
	ELEM_MODE_SCALE_INPLACE	   = 2,
	ELEM_MODE_PLE_COMBINE	   = 3,
	ELEM_MODE_SOFTCAP		   = 4,
	ELEM_MODE_SPLIT_QGATE	   = 5,
	ELEM_MODE_ATTN_OUTPUT_GATE = 6,
	ELEM_MODE_ACT_SILU		   = 7,
	ELEM_MODE_ACT_GELU		   = 8,
	ELEM_MODE_ADD_BIAS		   = 9,
	ELEM_MODE_GATED_ADD		   = 10,
};

typedef struct {
	VkDescriptorSet set;
	VkBuffer		bufs[VK_MAX_BINDINGS];
	VkDeviceSize	offs[VK_MAX_BINDINGS];
	int				valid;
	uint64_t		last_used;
	uint64_t		hash;
} vk_desc_slot;

typedef struct {
	VkPipeline			  pipeline;
	VkPipelineLayout	  layout;
	VkDescriptorSetLayout set_layout;
	int					  n_bindings;
	vk_desc_slot		  desc_cache[VK_DESC_CACHE_CAP];
	const char			 *name;
} vk_pipeline_set;

enum {
	VQ_Q4_0,
	VQ_Q4_1,
	VQ_Q5_0,
	VQ_Q5_1,
	VQ_Q8_0,
	VQ_IQ4_NL,
	VQ_MM_COUNT,
	VQ_Q4_K = VQ_MM_COUNT,
	VQ_Q5_K,
	VQ_Q6_K,
	VQ_DENSE,
	VQ_F32 = VQ_DENSE,
	VQ_F16,
	VQ_BF16,
	VQ_COUNT
};

#define VK_ATTN_CACHE_CAP 6
#define VK_ATTN_KIND_MAX 4

#define VK_ATTN_FLASH_BATCH_ENABLED 1

enum {
	VK_ATTN_PLAIN = 0,
	VK_ATTN_FLASH,
	VK_ATTN_PLAIN_BATCH,
	VK_ATTN_FLASH_BATCH,
};

typedef struct {
	vk_pipeline_set set[VK_ATTN_CACHE_CAP];
	int				head_dim[VK_ATTN_CACHE_CAP];
	int				n_groups[VK_ATTN_CACHE_CAP];
	int				unsupported[VK_ATTN_CACHE_CAP];
	int				count;
} vk_attn_cache;

typedef struct {
	VkBuffer				  buf;
	VkDeviceMemory			  mem;
	size_t					  size;
	void					 *mapped;
	struct vk_suballoc_block *sub_block;
	VkDeviceSize			  sub_off;
	VkDeviceSize			  sub_size;
} vk_buf;

typedef struct {
	vk_buf		*cos_b;
	vk_buf		*sin_b;
	VkDeviceSize cos_off;
	VkDeviceSize sin_off;
} vk_rope_tables;

typedef struct vk_suballoc_range {
	VkDeviceSize off;
	VkDeviceSize size;
} vk_suballoc_range;

typedef struct vk_suballoc_block {
	VkDeviceMemory			  mem;
	VkDeviceSize			  size;
	uint32_t				  memory_type;
	vk_suballoc_range		 *ranges;
	int						  n_ranges;
	int						  cap_ranges;
	struct vk_suballoc_block *next;
} vk_suballoc_block;

typedef struct {
	int		chunked;
	int		n_chunks;
	vk_buf	single;
	vk_buf *chunks;
	size_t	per_layer_size;
	int		n_kv_layers;
	size_t *layer_off_elems;
	size_t *layer_bytes;
} vk_kv_store;

typedef struct {
	vk_kv_store store;
} vk_kv_handle;

typedef struct {
	VkCommandBuffer cmd;
	int				recording;
	int				has_work;
	uint64_t		signal_value;
	int				submitted;
} vk_ring_slot;

typedef struct {
	VkInstance				 instance;
	VkDebugUtilsMessengerEXT debug_messenger;
	VkPhysicalDevice		 phys;
	VkDevice				 dev;
	VkQueue					 queue;
	VkCommandBuffer			 cmd;
	VkDescriptorPool		 desc_pool;
	VkFence					 fence;
	VkSemaphore				 timeline;
	uint64_t				 timeline_value;
	uint64_t				 timeline_completed;
	vk_suballoc_block		*sub_blocks;
	size_t					 device_local_allocated;
	uint64_t				 desc_tick;
	uint64_t				 total_desc_allocs;
	VkPipeline				 last_pipeline;
	const char				*last_pipeline_name;
	VkDescriptorSet			 last_desc_set;
	uint64_t				 diag_desc_allocs;
	uint64_t				 diag_buf_allocs;
	uint64_t				 stream_seq;
	size_t					 attn_part_cap;
	const float				*rope_cos_base_alt;
	const float				*rope_sin_base_alt;
	const float				*rope_cos_base;
	const float				*rope_sin_base;
	const float				*rope_ff_base;
	buffer					*rope_cos_buf_active;
	buffer					*rope_sin_buf_active;
	size_t					 staging_cap;
	size_t					 wbatch_off;
	uint64_t				 staging_pending_value;
	uint64_t				 last_signal_value;
	uint64_t				 pending_free_after;
	size_t					 argmax_partial_cap;
	size_t					 moe_batch_arena_cap;
	size_t					 moe_meta_cap;
	void				   **kv_handles;
	size_t					 xq_cap[2];
	uint64_t				 tok_signal[BACKEND_TOKEN_REGS];
	uint64_t				 moe_meta_signal[VK_MOE_META_SLOTS];
	VkCommandPool			 cmd_pools[VK_RING_DEPTH];
	struct {
		const vk_buf *buf;
		VkDeviceSize  off;
		int			  k, m;
		uint64_t	  seq;
	} xq_fused;
	buffer attn_part_buf;
	buffer rope_cos_buf;
	buffer rope_sin_buf;
	buffer rope_ff_buf;
	buffer rope_cos_buf_alt;
	buffer rope_sin_buf_alt;
	buffer attn_scores_buf;
	buffer ffn_act_buf;
	vk_buf staging_buf;
	vk_buf tok_reg_buf;
	vk_buf tok_rb_buf;
	vk_buf argmax_partial_buf;
	vk_buf moe_arena;
	vk_buf moe_batch_arena;
	vk_buf dummy_buf;
	vk_buf iq3s_grid_buf;
	vk_buf iqx_grid_buf;
	struct {
		uint64_t dispatches;
		uint64_t barriers;
		uint64_t submits;
		uint64_t waits;
		uint64_t uploads;
		uint64_t downloads;
		uint64_t wait_us;
	} stats;
	buffer xq_buf[2];
	struct {
		uint32_t	 max_shared_memory;
		uint32_t	 max_workgroup_size[3];
		uint32_t	 max_workgroup_invocations;
		VkDeviceSize storage_buffer_offset_alignment;
		VkDeviceSize max_storage_buffer_range;
		uint32_t	 subgroup_size;
		int			 unified_memory;
		int			 is_mali;
		int			 is_power_vr;
		int			 is_adreno;
		int			 is_amd;
		int			 supports_subgroup_basic;
		int			 supports_subgroup_arithmetic;
		int			 supports_integer_dot_product;
		int			 supports_16bit_storage;
		uint32_t	 max_workgroup_count[2];
	} caps;
	vk_buf							 moe_meta_slots[VK_MOE_META_SLOTS];
	vk_ring_slot					 ring[VK_RING_DEPTH];
	vk_buf							*scratch_pool[VK_SCRATCH_POOL_CAP];
	size_t							 scratch_pool_sizes[VK_SCRATCH_POOL_CAP];
	VkBuffer						 dead_bufs[VK_DEAD_BUF_CAP];
	VkBuffer						 read_set[VK_READ_SET_CAP];
	VkPhysicalDeviceMemoryProperties mem_props;
	VkBuffer						 dirty_table[VK_DIRTY_TABLE_SIZE];
	struct {
		int			 enabled;
		VkQueryPool	 pool;
		double		 ns_per_tick;
		uint64_t	 ts_mask;
		const char **rec;
		int			 n_rec[VK_RING_DEPTH];
		int			 reset_done[VK_RING_DEPTH];
		const char	*names[VK_PROF_NAMES];
		uint64_t	 ns[VK_PROF_NAMES];
		uint64_t	 calls[VK_PROF_NAMES];
		int			 n_names;
	} prof;
	VkDescriptorSet pending_free_sets[VK_DESC_CACHE_CAP * 4];
	vk_pipeline_set p_kv_put;
	vk_pipeline_set p_kv_shift_gather;
	vk_pipeline_set p_embd_lookup;
	vk_pipeline_set p_argmax;
	vk_pipeline_set p_argmax_reduce;
	vk_pipeline_set p_moe_gather;
	vk_pipeline_set p_moe_combine;
	vk_pipeline_set p_matmul_q4_0_batch;
	vk_pipeline_set p_matmul_q4_0_res_batch;
	vk_pipeline_set p_matmul_q4_0_dual_batch;
	vk_pipeline_set p_matmul_q4_0_triple_batch;
	vk_pipeline_set p_matmul_q4_1_batch;
	vk_pipeline_set p_matmul_q4_1_res_batch;
	vk_pipeline_set p_matmul_q5_0_batch;
	vk_pipeline_set p_matmul_q5_0_res_batch;
	vk_pipeline_set p_matmul_q5_1_batch;
	vk_pipeline_set p_matmul_q5_1_res_batch;
	vk_pipeline_set p_matmul_q8_0_batch;
	vk_pipeline_set p_matmul_q8_0_res_batch;
	vk_pipeline_set p_matmul_q4_k_batch;
	vk_pipeline_set p_matmul_q4_k_res_batch;
	vk_pipeline_set p_matmul_q4_k_dual_batch;
	vk_pipeline_set p_matmul_q5_k_batch;
	vk_pipeline_set p_matmul_q5_k_res_batch;
	vk_pipeline_set p_matmul_q6_k_batch;
	vk_pipeline_set p_matmul_q6_k_res_batch;
	vk_pipeline_set p_matmul_q6_k_dual_batch;
	vk_pipeline_set p_matmul_iq3_s_batch;
	vk_pipeline_set p_matmul_iq3_s_res_batch;
	vk_pipeline_set p_matmul_iq2_s_batch;
	vk_pipeline_set p_matmul_iq2_s_res_batch;
	vk_pipeline_set p_matmul_iq2_xxs_batch;
	vk_pipeline_set p_matmul_iq2_xxs_res_batch;
	vk_pipeline_set p_matmul_iq3_xxs_batch;
	vk_pipeline_set p_matmul_iq3_xxs_res_batch;
	vk_pipeline_set p_matmul_iq4_xs_batch;
	vk_pipeline_set p_matmul_iq4_xs_res_batch;
	vk_pipeline_set p_matmul_f32_batch;
	vk_pipeline_set p_matmul_f32_res_batch;
	vk_pipeline_set p_matmul_f16_batch;
	vk_pipeline_set p_matmul_f16_res_batch;
	vk_pipeline_set p_matmul_bf16_batch;
	vk_pipeline_set p_matmul_bf16_res_batch;
	vk_pipeline_set p_matmul_f32_wide_batch;
	vk_pipeline_set p_matmul_f32_wide_res_batch;
	vk_pipeline_set p_matmul_f16_wide_batch;
	vk_pipeline_set p_matmul_f16_wide_res_batch;
	vk_pipeline_set p_matmul_bf16_wide_batch;
	vk_pipeline_set p_matmul_bf16_wide_res_batch;
	vk_pipeline_set p_matmul_iq4_nl_batch;
	vk_pipeline_set p_attn_decode;
	vk_pipeline_set p_attn_merge;
	vk_pipeline_set p_rmsnorm_batch;
	vk_pipeline_set p_rmsnorm_sg_batch;
	vk_pipeline_set p_rmsnorm_noweight_batch;
	vk_pipeline_set p_rmsnorm_noweight_sg_batch;
	vk_pipeline_set p_rmsnorm_per_head_batch;
	vk_pipeline_set p_rmsnorm_per_head_sg_batch;
	vk_pipeline_set p_rmsnorm_noweight_per_head_batch;
	vk_pipeline_set p_rmsnorm_noweight_per_head_sg_batch;
	vk_pipeline_set p_rmsnorm_add_batch;
	vk_pipeline_set p_rmsnorm_sg_quant;
	vk_pipeline_set p_rmsnorm_add_norm;
	vk_pipeline_set p_rope_batch;
	vk_pipeline_set p_rope_ext_batch;
	vk_pipeline_set p_qk_norm_rope;
	vk_pipeline_set p_rope_qk_batch;
	vk_pipeline_set p_dequant;
	vk_pipeline_set p_ffn_activate_batch;
	vk_pipeline_set p_ffn_activate_fused_batch;
	vk_pipeline_set p_elementwise_batch;
	vk_pipeline_set p_q8_quant[2];
	vk_pipeline_set p_nn[VK_NN_COUNT];
	vk_pipeline_set p_qmm[VQ_MM_COUNT];
	vk_pipeline_set p_qmv[VQ_COUNT][2];
	vk_attn_cache	attn_cache[VK_ATTN_KIND_MAX];
	uint32_t		queue_family;
	int				timeline_supported;
	int				ring_cur;
	int				sub_block_count;
	int				has_memory_budget_ext;
	int				pending_dispatches;
	int				cmd_recording;
	int				last_desc_pipeline_match;
	int				dead_buf_count;
	int				pending_free_count;
	int				dirty_count;
	int				matmul_wg_size;
	int				matmul_rows_per_thread;
	int				matmul_lanes_per_row;
	int				matmul_tile_k;
	int				matmul_m_per_wg;
	int				qcore_ok;
	int				rope_buf_cap;
	int				rope_ff_cap;
	int				rope_buf_cap_alt;
	int				rope_half_stored_alt;
	int				n_ctx_stored;
	int				kv_head_dim_max;
	int				rope_ff_head_dim;
	int				rope_ff_n_ctx;
	float			rope_ff_theta;
	int				rope_half_stored;
	int				attn_scores_cap;
	int				ffn_act_cap;
	int				wbatch_count;
	int				staging_read_recorded;
	int				tok_pipe;
	int				moe_arena_inter;
	int				moe_arena_dim;
	int				moe_meta_cur;
	int				kv_handle_count;
	int				kv_handle_cap;
	int				device_lost;
	int				device_lost_warned;
	int				scratch_pool_count;
	int				batch_active;
	int				n_read;
	int				read_overflow;
	struct {
		int diag;
		int abort_on_error;
		int validate;
	} debug;
} vk_priv;

static inline uint32_t vk_f32_to_bits(float f) {
	union {
		float	 v;
		uint32_t b;
	} u = {.v = f};
	return u.b;
}

static inline float vk_f32_from_bits(uint32_t b) {
	union {
		uint32_t b;
		float	 v;
	} u = {.b = b};
	return u.v;
}

static void vk_invalidate_desc_cache_for_buf(vk_priv *p, VkBuffer freed);

static status_code vk_ensure_staging_cap(vk_priv *p, size_t need);
static void		   vk_kv_store_free(vk_priv *p, vk_kv_store *store);
static void		   vk_dirty_clear(vk_priv *p);

static void vk_reads_clear(vk_priv *p) {
	p->n_read		 = 0;
	p->read_overflow = 0;
}

static int vk_reads_has(const vk_priv *p, VkBuffer buf) {
	if (p->read_overflow)
		return 1;
	for (int i = 0; i < p->n_read; i++)
		if (p->read_set[i] == buf)
			return 1;
	return 0;
}

static void vk_reads_add(vk_priv *p, VkBuffer buf) {
	if (vk_reads_has(p, buf))
		return;
	if (p->n_read == VK_READ_SET_CAP)
		p->read_overflow = 1;
	else
		p->read_set[p->n_read++] = buf;
}

static status_code vk_timeline_wait(vk_priv *p, uint64_t value);
static void		   vk_timeline_poll(vk_priv *p);
static int		   vk_timeline_done(vk_priv *p, uint64_t value);

static int g_probed = -1;

#define VK_SUBALLOC_BLOCK_SIZE (64ull << 20)
#define VK_SUBALLOC_MAX_REQ (4ull << 20)
#define VK_SUBALLOC_MAX_BLOCKS 8

#define VK_QCORE_PIPES (4 + VQ_MM_COUNT + 2 * VQ_COUNT)

#define VQ_MV(t, label)                                                                            \
	{                                                                                              \
		shader_##t##_spv, sizeof(shader_##t##_spv), {                                              \
			"q_matvec_" label, "q_matvec_m4_" label                                                \
		}                                                                                          \
	}
#define VQ_MM(t, label) {shader_##t##_spv, sizeof(shader_##t##_spv), "q_matmul_" label}

static const uint32_t VQ_MV_SPEC[2][2] = {{1, 4}, {VK_QMV_COLS, 2}};

#define VK_WEIGHT_TAIL_PAD 32

#define VK_WEIGHT_STAGE_CHUNK_BYTES ((size_t)(32 << 20))
#define VK_WEIGHT_STAGE_BATCH_BYTES ((size_t)(64 << 20))

#define FLASH_MAX_DPL 16
#define FLASH_MAX_SPLIT 64
#define FLASH_K_PAD 4

#define VK_DENSE_ATTN_MAX_HEAD_DIM 2048

static const char *vk_result_str(int r) {
	switch (r) {
	case 0:
		return "VK_SUCCESS";
	case 1:
		return "VK_NOT_READY";
	case -1:
		return "VK_TIMEOUT";
	case -2:
		return "VK_EVENT_SET";
	case -3:
		return "VK_EVENT_RESET";
	case -4:
		return "VK_ERROR_DEVICE_LOST";
	case -5:
		return "VK_ERROR_MEMORY_MAP_FAILED";
	case -6:
		return "VK_ERROR_LAYER_NOT_PRESENT";
	case -7:
		return "VK_ERROR_EXTENSION_NOT_PRESENT";
	case -8:
		return "VK_ERROR_FEATURE_NOT_PRESENT";
	case -9:
		return "VK_ERROR_INCOMPATIBLE_DRIVER";
	case -10:
		return "VK_ERROR_TOO_MANY_OBJECTS";
	case -11:
		return "VK_ERROR_FORMAT_NOT_SUPPORTED";
	case -12:
		return "VK_ERROR_FRAGMENTED_POOL";
	case -13:
		return "VK_ERROR_UNKNOWN";
	case -1000069000:
		return "VK_ERROR_OUT_OF_HOST_MEMORY";
	case -1000069001:
		return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
	case -1000069002:
		return "VK_ERROR_INITIALIZATION_FAILED";
	case -1000069003:
		return "VK_ERROR_DEVICE_REMOVED";
	case -1000001004:
		return "VK_ERROR_VALIDATION_FAILED_EXT";
	case -1000002002:
		return "VK_ERROR_INVALID_SHADER_NV";
	default:
		return "VK_ERROR_UNKNOWN_RESULT";
	}
}

static status_code vk_probe(void) {
	if (g_probed >= 0)
		return g_probed == 1 ? OK : ERR_UNSUPPORTED;

	VkApplicationInfo	 app = {.sType		= VK_STRUCTURE_TYPE_APPLICATION_INFO,
								.apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType			  = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
								.pApplicationInfo = &app};
	VkInstance			 inst;
	if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
		g_probed = 0;
		return ERR_UNSUPPORTED;
	}

	uint32_t count = 0;
	vkEnumeratePhysicalDevices(inst, &count, NULL);
	vkDestroyInstance(inst, NULL);

	g_probed = (count > 0) ? 1 : 0;
	return g_probed == 1 ? OK : ERR_UNSUPPORTED;
}

static int vk_device_count(void) {
	VkApplicationInfo	 app = {.sType		= VK_STRUCTURE_TYPE_APPLICATION_INFO,
								.apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType			  = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
								.pApplicationInfo = &app};
	VkInstance			 inst;
	if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
		return 0;
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(inst, &count, NULL);
	vkDestroyInstance(inst, NULL);
	return (int)count;
}

static status_code vk_device_identity_query(VkPhysicalDevice phys, char *out, size_t cap);

static status_code vk_device_identity(int device_index, char *out, size_t cap) {
	VkApplicationInfo	 app = {.sType		= VK_STRUCTURE_TYPE_APPLICATION_INFO,
								.apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType			  = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
								.pApplicationInfo = &app};
	VkInstance			 inst;
	if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
		return ERR_UNSUPPORTED;

	uint32_t count = 0;
	vkEnumeratePhysicalDevices(inst, &count, NULL);
	if (count == 0) {
		vkDestroyInstance(inst, NULL);
		return ERR_UNSUPPORTED;
	}
	if (device_index < 0 || (uint32_t)device_index >= count) {
		vkDestroyInstance(inst, NULL);
		return ERR_INVALID_ARG;
	}
	VkPhysicalDevice *devs = xmalloc(count * sizeof(VkPhysicalDevice));
	vkEnumeratePhysicalDevices(inst, &count, devs);
	VkPhysicalDevice phys = devs[device_index];
	free(devs);

	status_code rc = vk_device_identity_query(phys, out, cap);
	vkDestroyInstance(inst, NULL);
	return rc;
}

static status_code vk_device_identity_query(VkPhysicalDevice phys, char *out, size_t cap) {
	int has_pci_ext = 0;
	{
		uint32_t ext_count = 0;
		vkEnumerateDeviceExtensionProperties(phys, NULL, &ext_count, NULL);
		if (ext_count > 0) {
			VkExtensionProperties *exts = xmalloc(ext_count * sizeof(VkExtensionProperties));
			vkEnumerateDeviceExtensionProperties(phys, NULL, &ext_count, exts);
			for (uint32_t i = 0; i < ext_count; i++)
				has_pci_ext |=
					strcmp(exts[i].extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0;
			free(exts);
		}
	}

	VkPhysicalDeviceIDProperties id_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
	};
	VkPhysicalDevicePCIBusInfoPropertiesEXT pci_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
	};
	VkPhysicalDeviceProperties2 props2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &id_props,
	};
	VkPhysicalDeviceProperties2 *pci_tail = NULL;
	if (has_pci_ext) {
		VkPhysicalDeviceProperties2 *node = xmalloc(sizeof(*node));
		*node							  = (VkPhysicalDeviceProperties2){
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &pci_props,
		};
		id_props.pNext = node;
		pci_tail	   = node;
	}
	vkGetPhysicalDeviceProperties2(phys, &props2);
	if (pci_tail)
		free(pci_tail);

	if (has_pci_ext && (pci_props.pciDomain != 0 || pci_props.pciBus != 0 ||
						pci_props.pciDevice != 0 || pci_props.pciFunction != 0)) {
		snprintf(out, cap, "PCI:%04x:%02x:%02x.%x", pci_props.pciDomain, pci_props.pciBus,
				 pci_props.pciDevice, pci_props.pciFunction);
		return OK;
	}

	uint64_t lo = 0, hi = 0;
	memcpy(&lo, id_props.deviceUUID, sizeof(lo));
	memcpy(&hi, id_props.deviceUUID + sizeof(lo), sizeof(hi));
	if (lo || hi) {
		snprintf(out, cap, "UUID:%016llx%016llx", (unsigned long long)lo, (unsigned long long)hi);
		return OK;
	}
	return ERR_UNSUPPORTED;
}

static status_code vk_check_fail(vk_priv *p, VkResult r, const char *expr, const char *file,
								 int line) {
	ERROR("vk: %s returned %d at %s:%d", expr, (int)r, file, line);
	if (p && r == VK_ERROR_DEVICE_LOST)
		p->device_lost = 1;
	if (p && p->debug.abort_on_error)
		abort();
	return ERR_INTERNAL;
}

static uint32_t find_memory_type(vk_priv *p, uint32_t type_bits, VkMemoryPropertyFlags want) {
	for (uint32_t i = 0; i < p->mem_props.memoryTypeCount; i++) {
		if (!(type_bits & (1u << i)))
			continue;
		if ((p->mem_props.memoryTypes[i].propertyFlags & want) == want)
			return i;
	}
	return UINT32_MAX;
}

static void vk_suballoc_release(vk_priv *p, vk_buf *b) {
	(void)p;
	vk_suballoc_block *blk = b->sub_block;
	VkDeviceSize	   lo  = b->sub_off;
	int				   pos = 0;
	for (int i = 0; i < blk->n_ranges; i++) {
		if (blk->ranges[i].off < lo)
			pos = i + 1;
		else
			break;
	}
	arr_reserve((void **)&(blk->ranges), (blk->n_ranges), &(blk->cap_ranges),
				sizeof(*(blk->ranges)));
	memmove(blk->ranges + pos + 1, blk->ranges + pos,
			(size_t)(blk->n_ranges - pos) * sizeof(*blk->ranges));
	blk->ranges[pos].off  = lo;
	blk->ranges[pos].size = b->sub_size;
	blk->n_ranges++;
	if (pos > 0 && blk->ranges[pos - 1].off + blk->ranges[pos - 1].size == lo) {
		blk->ranges[pos - 1].size += blk->ranges[pos].size;
		memmove(blk->ranges + pos, blk->ranges + pos + 1,
				(size_t)(blk->n_ranges - pos - 1) * sizeof(*blk->ranges));
		blk->n_ranges--;
		pos--;
	}
	if (pos + 1 < blk->n_ranges &&
		blk->ranges[pos].off + blk->ranges[pos].size == blk->ranges[pos + 1].off) {
		blk->ranges[pos].size += blk->ranges[pos + 1].size;
		memmove(blk->ranges + pos + 1, blk->ranges + pos + 2,
				(size_t)(blk->n_ranges - pos - 2) * sizeof(*blk->ranges));
		blk->n_ranges--;
	}
}

static status_code vk_suballoc_place(vk_priv *p, VkBuffer buf, const VkMemoryRequirements *req,
									 vk_buf *out) {
	vk_suballoc_block *blk = p->sub_blocks;
	while (blk) {
		if ((req->memoryTypeBits & (1u << blk->memory_type)))
			break;
		blk = blk->next;
	}

	if (!blk && p->sub_block_count < VK_SUBALLOC_MAX_BLOCKS) {
		VkMemoryRequirements block_req;
		vkGetBufferMemoryRequirements(p->dev, buf, &block_req);
		VkDeviceSize block_size = VK_SUBALLOC_BLOCK_SIZE;
		if (block_req.size > block_size)
			block_size = block_req.size;

		uint32_t mt =
			find_memory_type(p, block_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (mt == UINT32_MAX)
			return ERR_OUT_OF_MEMORY;
		VkMemoryAllocateInfo mai = {
			.sType			 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize	 = block_size,
			.memoryTypeIndex = mt,
		};
		VkDeviceMemory mem;
		if (vkAllocateMemory(p->dev, &mai, NULL, &mem) != VK_SUCCESS)
			return ERR_OUT_OF_MEMORY;
		p->device_local_allocated += block_size;
		blk					= xcalloc(1, sizeof(*blk));
		blk->mem			= mem;
		blk->size			= block_size;
		blk->memory_type	= mt;
		blk->cap_ranges		= 1;
		blk->ranges			= xmalloc(sizeof(*blk->ranges));
		blk->ranges[0].off	= 0;
		blk->ranges[0].size = block_size;
		blk->n_ranges		= 1;
		blk->next			= p->sub_blocks;
		p->sub_blocks		= blk;
		p->sub_block_count++;
	}

	if (!blk)
		return ERR_OUT_OF_MEMORY;

	for (int i = 0; i < blk->n_ranges; i++) {
		VkDeviceSize aligned = ALIGN_UP(blk->ranges[i].off, (VkDeviceSize)req->alignment);
		VkDeviceSize slack	 = aligned - blk->ranges[i].off;
		if (slack >= blk->ranges[i].size)
			continue;
		if (blk->ranges[i].size - slack < req->size)
			continue;

		VkDeviceSize used_off  = aligned;
		VkDeviceSize used_size = req->size;
		VkDeviceSize tail_size = blk->ranges[i].size - slack - used_size;
		if (tail_size > 0) {
			blk->ranges[i].off	= blk->ranges[i].off + slack + used_size;
			blk->ranges[i].size = tail_size;
		} else {
			memmove(blk->ranges + i, blk->ranges + i + 1,
					(size_t)(blk->n_ranges - i - 1) * sizeof(*blk->ranges));
			blk->n_ranges--;
		}
		VK_CHECK(vkBindBufferMemory(p->dev, buf, blk->mem, used_off));
		out->mem	   = blk->mem;
		out->sub_block = blk;
		out->sub_off   = used_off;
		out->sub_size  = used_size;
		return OK;
	}
	return ERR_OUT_OF_MEMORY;
}

static status_code vk_alloc_buffer(vk_priv *p, size_t size, VkBufferUsageFlags usage,
								   VkMemoryPropertyFlags mem_flags, vk_buf *out) {
	if (size == 0)
		size = 1;
	if (p->debug.diag) {
		p->diag_buf_allocs++;
		fprintf(stderr, "[VK_DIAG] buf_alloc #%llu size=%zu\n",
				(unsigned long long)p->diag_buf_allocs, size);
	}
	if ((usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && p->caps.max_storage_buffer_range > 0 &&
		size > p->caps.max_storage_buffer_range) {
		ERROR("vk: buffer alloc %zu bytes exceeds maxStorageBufferRange (%llu)", size,
			  (unsigned long long)p->caps.max_storage_buffer_range);
		return ERR_OUT_OF_MEMORY;
	}

	VkBufferCreateInfo bci = {
		.sType		 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size		 = size,
		.usage		 = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VK_CHECK(vkCreateBuffer(p->dev, &bci, NULL, &out->buf));

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(p->dev, out->buf, &req);

	out->mapped	   = NULL;
	out->sub_block = NULL;
	out->sub_off   = 0;
	out->sub_size  = 0;

	if (size <= VK_SUBALLOC_MAX_REQ && !(mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
		if (vk_suballoc_place(p, out->buf, &req, out) == OK) {
			out->size = size;
			return OK;
		}
		out->sub_block = NULL;
	}

	uint32_t mt = find_memory_type(p, req.memoryTypeBits, mem_flags);
	if (mt == UINT32_MAX) {
		vkDestroyBuffer(p->dev, out->buf, NULL);
		return ERR_UNSUPPORTED;
	}

	VkMemoryAllocateInfo mai = {
		.sType			 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize	 = req.size,
		.memoryTypeIndex = mt,
	};
	if (vkAllocateMemory(p->dev, &mai, NULL, &out->mem) != VK_SUCCESS) {
		vkDestroyBuffer(p->dev, out->buf, NULL);
		return ERR_OUT_OF_MEMORY;
	}
	vkBindBufferMemory(p->dev, out->buf, out->mem, 0);
	out->size = size;

	VkMemoryPropertyFlags actual_flags = p->mem_props.memoryTypes[mt].propertyFlags;
	if (actual_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		p->device_local_allocated += req.size;
	if (actual_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
		if (vkMapMemory(p->dev, out->mem, 0, VK_WHOLE_SIZE, 0, &out->mapped) != VK_SUCCESS) {
			out->mapped = NULL;
		}
	}
	return OK;
}

static void vk_free_buffer(vk_priv *p, vk_buf *b) {
	if (!b->buf)
		return;
	vk_invalidate_desc_cache_for_buf(p, b->buf);
	if (b->mapped) {
		vkUnmapMemory(p->dev, b->mem);
		b->mapped = NULL;
	}
	vkDestroyBuffer(p->dev, b->buf, NULL);
	if (b->sub_block) {
		vk_suballoc_release(p, b);
	} else {
		if (b->size > 0 && b->size <= p->device_local_allocated)
			p->device_local_allocated -= b->size;
		vkFreeMemory(p->dev, b->mem, NULL);
	}
	b->buf		 = VK_NULL_HANDLE;
	b->mem		 = VK_NULL_HANDLE;
	b->size		 = 0;
	b->sub_block = NULL;
	b->sub_off	 = 0;
	b->sub_size	 = 0;
}

static status_code vk_flush(vk_priv *p);
static status_code vk_ensure_token_regs(vk_priv *p);

static void vk_queue_desc_free(vk_priv *p, VkDescriptorSet set) {
	if (set == VK_NULL_HANDLE)
		return;
	if (p->last_signal_value > p->pending_free_after)
		p->pending_free_after = p->last_signal_value;
	if (p->pending_free_count >=
		(int)(sizeof(p->pending_free_sets) / sizeof(p->pending_free_sets[0]))) {
		status_code fs = vk_flush(p);
		if (fs != OK) {
			ERROR("vk: queue_desc_free flush failed (%s), dropping %d frees", status_str(fs),
				  p->pending_free_count);
			p->pending_free_count = 0;
		} else if (p->pending_free_count >=
				   (int)(sizeof(p->pending_free_sets) / sizeof(p->pending_free_sets[0]))) {
			vkFreeDescriptorSets(p->dev, p->desc_pool, (uint32_t)p->pending_free_count,
								 p->pending_free_sets);
			p->pending_free_count = 0;
		}
	}
	p->pending_free_sets[p->pending_free_count++] = set;
}

static void vk_flush_pending_desc_frees(vk_priv *p) {
	if (p->pending_free_count == 0)
		return;
	if (!vk_timeline_done(p, p->pending_free_after))
		return;
	vkFreeDescriptorSets(p->dev, p->desc_pool, (uint32_t)p->pending_free_count,
						 p->pending_free_sets);
	p->pending_free_count = 0;
}

static vk_pipeline_set *vk_qcore_pipe(vk_priv *p, int i) {
	if (i < 2)
		return &p->p_q8_quant[i];
	if (i == 2)
		return &p->p_attn_decode;
	if (i == 3)
		return &p->p_attn_merge;
	i -= 4;
	if (i < VQ_MM_COUNT)
		return &p->p_qmm[i];
	i -= VQ_MM_COUNT;
	return &p->p_qmv[i / 2][i % 2];
}

static void vk_desc_cache_drop_buf(vk_priv *p, vk_pipeline_set *ps, VkBuffer freed) {
	if (!ps->pipeline)
		return;
	for (int j = 0; j < VK_DESC_CACHE_CAP; j++) {
		vk_desc_slot *s = &ps->desc_cache[j];
		if (!s->valid)
			continue;
		for (int k = 0; k < ps->n_bindings; k++) {
			if (s->bufs[k] == freed) {
				vk_queue_desc_free(p, s->set);
				s->valid = 0;
				s->set	 = VK_NULL_HANDLE;
				break;
			}
		}
	}
}

static void vk_invalidate_desc_cache_for_buf(vk_priv *p, VkBuffer freed) {
	if (p->total_desc_allocs == 0)
		return;

	if (p->dead_buf_count < VK_DEAD_BUF_CAP) {
		for (int i = 0; i < p->dead_buf_count; i++) {
			if (p->dead_bufs[i] == freed)
				return;
		}
		p->dead_bufs[p->dead_buf_count++] = freed;
	} else {
		vk_pipeline_set *pipelines[] = {

			&p->p_matmul_iq4_nl_batch,

			&p->p_kv_put,
			&p->p_kv_shift_gather,
			&p->p_embd_lookup,
			&p->p_argmax,
			&p->p_argmax_reduce,
			&p->p_moe_gather,
			&p->p_moe_combine,
			&p->p_matmul_q4_0_batch,
			&p->p_matmul_q4_0_res_batch,
			&p->p_matmul_q4_0_dual_batch,
			&p->p_matmul_q4_0_triple_batch,
			&p->p_matmul_q4_1_batch,
			&p->p_matmul_q4_1_res_batch,
			&p->p_matmul_q5_0_batch,
			&p->p_matmul_q5_0_res_batch,
			&p->p_matmul_q5_1_batch,
			&p->p_matmul_q5_1_res_batch,
			&p->p_matmul_q8_0_batch,
			&p->p_matmul_q8_0_res_batch,
			&p->p_matmul_q4_k_batch,
			&p->p_matmul_q4_k_res_batch,
			&p->p_matmul_q4_k_dual_batch,
			&p->p_matmul_q5_k_batch,
			&p->p_matmul_q5_k_res_batch,
			&p->p_matmul_q6_k_batch,
			&p->p_matmul_q6_k_res_batch,
			&p->p_matmul_q6_k_dual_batch,
			&p->p_matmul_iq3_s_batch,
			&p->p_matmul_iq3_s_res_batch,
			&p->p_matmul_iq2_s_batch,
			&p->p_matmul_iq2_s_res_batch,
			&p->p_matmul_iq2_xxs_batch,
			&p->p_matmul_iq2_xxs_res_batch,
			&p->p_matmul_iq3_xxs_batch,
			&p->p_matmul_iq3_xxs_res_batch,
			&p->p_matmul_iq4_xs_batch,
			&p->p_matmul_iq4_xs_res_batch,
			&p->p_matmul_f32_batch,
			&p->p_matmul_f32_res_batch,
			&p->p_matmul_f16_batch,
			&p->p_matmul_f16_res_batch,
			&p->p_matmul_bf16_batch,
			&p->p_matmul_bf16_res_batch,
			&p->p_matmul_iq4_nl_batch,
			&p->p_rmsnorm_batch,
			&p->p_rmsnorm_sg_batch,
			&p->p_rmsnorm_noweight_batch,
			&p->p_rmsnorm_noweight_sg_batch,
			&p->p_rmsnorm_per_head_batch,
			&p->p_rmsnorm_per_head_sg_batch,
			&p->p_rmsnorm_noweight_per_head_batch,
			&p->p_rmsnorm_noweight_per_head_sg_batch,
			&p->p_rmsnorm_add_batch,
			&p->p_rmsnorm_sg_quant,
			&p->p_rmsnorm_add_norm,
			&p->p_rope_batch,
			&p->p_rope_ext_batch,
			&p->p_qk_norm_rope,
			&p->p_rope_qk_batch,
			&p->p_ffn_activate_batch,
			&p->p_elementwise_batch,
			&p->p_dequant,
		};
		for (int i = 0; i < VK_NN_COUNT; i++)
			vk_desc_cache_drop_buf(p, &p->p_nn[i], freed);
		const int n_pipelines = (int)(sizeof(pipelines) / sizeof(pipelines[0]));
		for (int i = 0; i < n_pipelines; i++)
			vk_desc_cache_drop_buf(p, pipelines[i], freed);
		for (int i = 0; i < VK_QCORE_PIPES; i++)
			vk_desc_cache_drop_buf(p, vk_qcore_pipe(p, i), freed);
		for (int ci = 0; ci < VK_ATTN_KIND_MAX; ci++) {
			vk_attn_cache *c = &p->attn_cache[ci];
			for (int i = 0; i < c->count; i++)
				vk_desc_cache_drop_buf(p, &c->set[i], freed);
		}
		p->dead_buf_count = 0;
	}
}

static int vk_buf_is_dead(vk_priv *p, VkBuffer buf) {
	for (int i = 0; i < p->dead_buf_count; i++) {
		if (p->dead_bufs[i] == buf)
			return 1;
	}
	return 0;
}

static void vk_prof_resolve(vk_priv *p, int idx) {
	int n = p->prof.n_rec[idx];
	if (!p->prof.pool || n == 0)
		return;
	p->prof.n_rec[idx] = 0;
	uint64_t *ts	   = xmalloc((size_t)n * 2 * sizeof(uint64_t));
	VkResult  r =
		vkGetQueryPoolResults(p->dev, p->prof.pool, (uint32_t)(idx * VK_PROF_SLOT_CAP * 2),
							  (uint32_t)(n * 2), (size_t)n * 2 * sizeof(uint64_t), ts,
							  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	for (int i = 0; r == VK_SUCCESS && i < n; i++) {
		const char *name = p->prof.rec[idx * VK_PROF_SLOT_CAP + i];
		int			k	 = 0;
		while (k < p->prof.n_names && p->prof.names[k] != name)
			k++;
		if (k == p->prof.n_names) {
			if (k == VK_PROF_NAMES)
				continue;
			p->prof.names[p->prof.n_names++] = name;
		}
		uint64_t ticks = (ts[2 * i + 1] - ts[2 * i]) & p->prof.ts_mask;
		p->prof.ns[k] += (uint64_t)((double)ticks * p->prof.ns_per_tick);
		p->prof.calls[k]++;
	}
	free(ts);
}

static int vk_prof_begin(vk_priv *p, const char *name) {
	int idx = p->ring_cur;
	if (!p->prof.enabled || p->prof.n_rec[idx] >= VK_PROF_SLOT_CAP)
		return -1;
	if (!p->prof.reset_done[idx]) {
		vkCmdResetQueryPool(p->cmd, p->prof.pool, (uint32_t)(idx * VK_PROF_SLOT_CAP * 2),
							VK_PROF_SLOT_CAP * 2);
		p->prof.reset_done[idx] = 1;
	}
	int q									= p->prof.n_rec[idx]++;
	p->prof.rec[idx * VK_PROF_SLOT_CAP + q] = name ? name : "?";
	vkCmdWriteTimestamp(p->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->prof.pool,
						(uint32_t)((idx * VK_PROF_SLOT_CAP + q) * 2));
	return q;
}

static void vk_prof_end(vk_priv *p, int q) {
	if (q < 0)
		return;
	vkCmdWriteTimestamp(p->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->prof.pool,
						(uint32_t)((p->ring_cur * VK_PROF_SLOT_CAP + q) * 2 + 1));
}

static void vk_prof_report(vk_priv *p) {
	if (!p->prof.enabled)
		return;
	for (int i = 0; i < VK_RING_DEPTH; i++)
		if (p->ring[i].submitted || !p->ring[i].recording)
			vk_prof_resolve(p, i);
	uint64_t total = 0;
	int		 order[VK_PROF_NAMES];
	for (int i = 0; i < p->prof.n_names; i++) {
		total += p->prof.ns[i];
		int j = i;
		while (j > 0 && p->prof.ns[order[j - 1]] < p->prof.ns[i]) {
			order[j] = order[j - 1];
			j--;
		}
		order[j] = i;
	}
	INFO("vk: GPU kernel time %.1f ms total (host waited %.1f ms)", (double)total / 1e6,
		 (double)p->stats.wait_us / 1000.0);
	for (int i = 0; i < p->prof.n_names && i < 24 && total; i++) {
		int k = order[i];
		INFO("vk:   %-30s %8.1f ms %5.1f%% %8llu calls %7.1f us/call", p->prof.names[k],
			 (double)p->prof.ns[k] / 1e6, 100.0 * (double)p->prof.ns[k] / (double)total,
			 (unsigned long long)p->prof.calls[k],
			 (double)p->prof.ns[k] / 1e3 / (double)p->prof.calls[k]);
	}
	vkDestroyQueryPool(p->dev, p->prof.pool, NULL);
	free(p->prof.rec);
	p->prof.pool	= VK_NULL_HANDLE;
	p->prof.enabled = 0;
}

static void vk_set_gpu_profile(backend *self, int enable) {
	vk_priv *p = self->priv;
	if (!enable || p->prof.enabled || !p->dev)
		return;
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(p->phys, &props);
	uint32_t nq = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(p->phys, &nq, NULL);
	VkQueueFamilyProperties *qf = xcalloc(nq ? nq : 1, sizeof(*qf));
	vkGetPhysicalDeviceQueueFamilyProperties(p->phys, &nq, qf);
	uint32_t bits = p->queue_family < nq ? qf[p->queue_family].timestampValidBits : 0;
	free(qf);
	if (bits == 0 || props.limits.timestampPeriod <= 0.0f) {
		WARN("vk: queue has no timestamp support; per-kernel GPU timing disabled");
		return;
	}
	VkQueryPoolCreateInfo qpci = {
		.sType		= VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType	= VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = VK_RING_DEPTH * VK_PROF_SLOT_CAP * 2,
	};
	if (vkCreateQueryPool(p->dev, &qpci, NULL, &p->prof.pool) != VK_SUCCESS) {
		WARN("vk: vkCreateQueryPool failed; per-kernel GPU timing disabled");
		return;
	}
	p->prof.rec			= xcalloc(VK_RING_DEPTH * VK_PROF_SLOT_CAP, sizeof(const char *));
	p->prof.ns_per_tick = (double)props.limits.timestampPeriod;
	p->prof.ts_mask		= bits >= 64 ? UINT64_MAX : ((1ull << bits) - 1);
	for (int i = 0; i < VK_RING_DEPTH; i++)
		p->prof.reset_done[i] = 0;
	p->prof.enabled = 1;
}

static status_code vk_ring_begin(vk_priv *p, int idx) {
	vk_ring_slot *r = &p->ring[idx];
	vk_prof_resolve(p, idx);
	p->prof.reset_done[idx] = 0;
	vkResetCommandPool(p->dev, p->cmd_pools[idx], 0);
	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	VK_CHECK(vkBeginCommandBuffer(r->cmd, &bi));
	vk_reads_clear(p);
	r->recording				= 1;
	r->has_work					= 0;
	r->submitted				= 0;
	p->cmd						= r->cmd;
	p->cmd_recording			= 1;
	p->pending_dispatches		= 0;
	p->last_pipeline			= VK_NULL_HANDLE;
	p->last_desc_set			= VK_NULL_HANDLE;
	p->last_desc_pipeline_match = 0;
	return OK;
}

static status_code vk_timeline_wait(vk_priv *p, uint64_t value) {
	if (value <= p->timeline_completed)
		return OK;
	VkSemaphoreWaitInfo wi = {
		.sType			= VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = 1,
		.pSemaphores	= &p->timeline,
		.pValues		= &value,
	};
	uint64_t t0 = time_us();
	VK_CHECK(vkWaitSemaphores(p->dev, &wi, UINT64_MAX));
	p->stats.waits++;
	p->stats.wait_us += time_us() - t0;
	p->timeline_completed = value;
	return OK;
}

static void vk_timeline_poll(vk_priv *p) {
	if (!p->timeline_supported || p->timeline == VK_NULL_HANDLE)
		return;
	uint64_t v = 0;
	if (vkGetSemaphoreCounterValue(p->dev, p->timeline, &v) == VK_SUCCESS &&
		v > p->timeline_completed)
		p->timeline_completed = v;
}

static int vk_timeline_done(vk_priv *p, uint64_t value) {
	if (value == 0 || value <= p->timeline_completed)
		return 1;
	vk_timeline_poll(p);
	return value <= p->timeline_completed;
}

static void vk_ring_invalidate(vk_priv *p, int idx) {
	vk_ring_slot *r = &p->ring[idx];
	(void)vkResetCommandPool(p->dev, p->cmd_pools[idx], 0);
	r->recording			= 0;
	r->has_work				= 0;
	r->submitted			= 0;
	p->prof.n_rec[idx]		= 0;
	p->prof.reset_done[idx] = 0;
	if (p->cmd == r->cmd) {
		p->cmd						= VK_NULL_HANDLE;
		p->cmd_recording			= 0;
		p->last_pipeline			= VK_NULL_HANDLE;
		p->last_desc_set			= VK_NULL_HANDLE;
		p->last_desc_pipeline_match = 0;
	}
	p->pending_dispatches = 0;
	vk_dirty_clear(p);
}

static status_code vk_queue_submit_checked(vk_priv *p, int idx, const VkSubmitInfo *si,
										   VkFence fence) {
	p->stats.submits++;
	VkResult res = vkQueueSubmit(p->queue, 1, si, fence);
	if (res == VK_SUCCESS)
		return OK;
	ERROR("vk: vkQueueSubmit failed: %d (%s) at %s:%d%s%s", (int)res, vk_result_str((int)res),
		  __FILE__, __LINE__, p->last_pipeline_name ? "; last pipeline: " : "",
		  p->last_pipeline_name ? p->last_pipeline_name : "");
	if (res == VK_ERROR_DEVICE_LOST)
		p->device_lost = 1;
	if (p->debug.abort_on_error)
		abort();
	vk_ring_invalidate(p, idx);
	return ERR_INTERNAL;
}

static status_code vk_ring_submit(vk_priv *p, int idx, int wait_now) {
	vk_ring_slot *r = &p->ring[idx];
	if (!r->recording || r->submitted)
		return OK;

	VkResult end_res = vkEndCommandBuffer(r->cmd);
	if (end_res != VK_SUCCESS) {
		if (p->pending_dispatches <= 4) {
			ERROR("vk: vkEndCommandBuffer failed (%d) with only %d dispatch(es) "
				  "recorded -- a shader in this batch is too heavy for this driver. "
				  "Check for large const arrays or excessive register pressure in "
				  "the most recently dispatched shader.",
				  (int)end_res, p->pending_dispatches);
		} else {
			ERROR("vk: vkEndCommandBuffer failed (%d), dispatches=%d batch_active=%d"
				  " -- command buffer was too large for this driver;"
				  " check for large const arrays or excessive register pressure",
				  (int)end_res, p->pending_dispatches, p->batch_active);
		}
		if (p && p->debug.abort_on_error)
			abort();
		vk_ring_invalidate(p, idx);
		vk_flush_pending_desc_frees(p);
		return ERR_INTERNAL;
	}

	if (p->timeline_supported) {
		p->timeline_value++;
		r->signal_value = p->timeline_value;

		uint64_t					  wait_value = p->last_signal_value;
		uint32_t					  wait_count = wait_value > 0 ? 1u : 0u;
		VkPipelineStageFlags		  wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		VkTimelineSemaphoreSubmitInfo tsi		 = {
			.sType					   = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues	   = &r->signal_value,
			.waitSemaphoreValueCount   = wait_count,
			.pWaitSemaphoreValues	   = wait_count ? &wait_value : NULL,
		};
		VkSubmitInfo si = {
			.sType				  = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext				  = &tsi,
			.commandBufferCount	  = 1,
			.pCommandBuffers	  = &r->cmd,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores	  = &p->timeline,
			.waitSemaphoreCount	  = wait_count,
			.pWaitSemaphores	  = wait_count ? &p->timeline : NULL,
			.pWaitDstStageMask	  = wait_count ? &wait_stage : NULL,
		};
		status_code submit_st = vk_queue_submit_checked(p, idx, &si, VK_NULL_HANDLE);
		if (submit_st != OK)
			return submit_st;
		r->submitted		 = 1;
		r->recording		 = 0;
		p->last_signal_value = r->signal_value;

		if (wait_now) {
			status_code ws = vk_timeline_wait(p, r->signal_value);
			if (ws != OK) {
				vk_ring_invalidate(p, idx);
				return ws;
			}

			vk_flush_pending_desc_frees(p);
		}
		return OK;
	}

	VkSubmitInfo si = {
		.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers	= &r->cmd,
	};
	vkResetFences(p->dev, 1, &p->fence);
	status_code submit_st = vk_queue_submit_checked(p, idx, &si, p->fence);
	if (submit_st != OK)
		return submit_st;
	r->submitted = 1;
	r->recording = 0;
	{
		uint64_t t0		  = time_us();
		VkResult wait_res = vkWaitForFences(p->dev, 1, &p->fence, VK_TRUE, UINT64_MAX);
		p->stats.waits++;
		p->stats.wait_us += time_us() - t0;
		if (wait_res != VK_SUCCESS) {
			if (wait_res == VK_ERROR_DEVICE_LOST)
				p->device_lost = 1;
			ERROR("vk: vkWaitForFences returned %d", (int)wait_res);
			vk_ring_invalidate(p, idx);
			return ERR_INTERNAL;
		}
	}
	vk_flush_pending_desc_frees(p);
	return OK;
}

static status_code vk_run_cmd(vk_priv *p) {
	int			idx		 = p->ring_cur;
	int			wait_now = p->timeline_supported ? 0 : 1;
	status_code s		 = vk_ring_submit(p, idx, wait_now);
	if (s != OK)
		return s;

	vk_ring_slot *cur = &p->ring[idx];
	if (cur->submitted) {
		if (p->pending_free_count && cur->signal_value > p->pending_free_after)
			p->pending_free_after = cur->signal_value;
		if (p->wbatch_count || p->staging_read_recorded) {
			if (cur->signal_value > p->staging_pending_value)
				p->staging_pending_value = cur->signal_value;
			p->wbatch_count			 = 0;
			p->wbatch_off			 = 0;
			p->staging_read_recorded = 0;
		}
	}

	p->ring_cur		 = (idx + 1) % VK_RING_DEPTH;
	vk_ring_slot *nr = &p->ring[p->ring_cur];
	if (nr->recording && nr->has_work) {
		status_code ss = vk_ring_submit(p, p->ring_cur, wait_now);
		if (ss != OK)
			return ss;
	}
	nr = &p->ring[p->ring_cur];
	if (wait_now == 0 && nr->submitted && nr->signal_value > p->timeline_completed) {
		status_code ws = vk_timeline_wait(p, nr->signal_value);
		if (ws != OK)
			return ws;
	}
	if (!nr->recording) {
		status_code bs = vk_ring_begin(p, p->ring_cur);
		if (bs != OK)
			return bs;
	}
	vk_dirty_clear(p);
	if (wait_now == 1) {
		p->wbatch_count			 = 0;
		p->wbatch_off			 = 0;
		p->staging_pending_value = 0;
	}
	return OK;
}

static status_code vk_run_cmd_sync(vk_priv *p) {
	status_code s = vk_run_cmd(p);
	if (s != OK)
		return s;
	if (p->timeline_supported && p->last_signal_value > p->timeline_completed) {
		status_code ws = vk_timeline_wait(p, p->last_signal_value);
		if (ws != OK)
			return ws;
	}
	p->wbatch_count			 = 0;
	p->wbatch_off			 = 0;
	p->staging_pending_value = 0;
	p->staging_read_recorded = 0;
	return OK;
}

static status_code vk_flush(vk_priv *p) {
	if (p->device_lost)
		return ERR_INTERNAL;

	vk_ring_slot *r = &p->ring[p->ring_cur];

	if (r->has_work) {
		status_code s = vk_run_cmd(p);
		if (s != OK)
			return s;
	}
	if (!p->timeline_supported)
		return OK;

	if (p->last_signal_value > p->timeline_completed) {
		status_code ws = vk_timeline_wait(p, p->last_signal_value);
		if (ws != OK)
			return ws;
	}

	p->wbatch_count			 = 0;
	p->wbatch_off			 = 0;
	p->staging_pending_value = 0;
	p->staging_read_recorded = 0;
	vk_flush_pending_desc_frees(p);

	r = &p->ring[p->ring_cur];
	if (!r->recording) {
		status_code bs = vk_ring_begin(p, p->ring_cur);
		if (bs != OK)
			return bs;
	}
	return OK;
}

static status_code vk_wbatch_flush(vk_priv *p) {
	if (p->staging_pending_value && !vk_timeline_done(p, p->staging_pending_value)) {
		status_code ws = vk_timeline_wait(p, p->staging_pending_value);
		if (ws != OK)
			return ws;
	}
	if (!p->wbatch_count)
		return OK;
	status_code s	= vk_flush(p);
	p->wbatch_count = 0;
	p->wbatch_off	= 0;
	return s;
}

static status_code vk_create_pipeline_spec(vk_priv *p, const uint32_t *spv, size_t spv_len,
										   int n_bindings, uint32_t push_size,
										   const uint32_t *spec_data, uint32_t spec_size,
										   vk_pipeline_set *out);

static status_code vk_create_pipeline(vk_priv *p, const uint32_t *spv, size_t spv_len,
									  int n_bindings, uint32_t push_size, vk_pipeline_set *out) {
	return vk_create_pipeline_spec(p, spv, spv_len, n_bindings, push_size, NULL, 0, out);
}

static status_code vk_create_pipeline_spec(vk_priv *p, const uint32_t *spv, size_t spv_len,
										   int n_bindings, uint32_t push_size,
										   const uint32_t *spec_data, uint32_t spec_size,
										   vk_pipeline_set *out) {
	memset(out->desc_cache, 0, sizeof(out->desc_cache));
	out->n_bindings = n_bindings;

	VkShaderModuleCreateInfo smci = {
		.sType	  = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = spv_len,
		.pCode	  = spv,
	};
	VkShaderModule mod;
	VK_CHECK(vkCreateShaderModule(p->dev, &smci, NULL, &mod));

	VkDescriptorSetLayoutBinding bindings[8];
	for (int i = 0; i < n_bindings; i++) {
		bindings[i] = (VkDescriptorSetLayoutBinding){
			.binding		 = (uint32_t)i,
			.descriptorType	 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags		 = VK_SHADER_STAGE_COMPUTE_BIT,
		};
	}
	VkDescriptorSetLayoutCreateInfo dslci = {
		.sType		  = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = (uint32_t)n_bindings,
		.pBindings	  = bindings,
	};
	if (vkCreateDescriptorSetLayout(p->dev, &dslci, NULL, &out->set_layout) != VK_SUCCESS) {
		vkDestroyShaderModule(p->dev, mod, NULL);
		out->set_layout = VK_NULL_HANDLE;
		out->layout		= VK_NULL_HANDLE;
		out->pipeline	= VK_NULL_HANDLE;
		return ERR_INTERNAL;
	}

	VkPushConstantRange pcr = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size};
	VkPipelineLayoutCreateInfo plci = {
		.sType					= VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount			= 1,
		.pSetLayouts			= &out->set_layout,
		.pushConstantRangeCount = push_size > 0 ? 1u : 0u,
		.pPushConstantRanges	= push_size > 0 ? &pcr : NULL,
	};
	if (vkCreatePipelineLayout(p->dev, &plci, NULL, &out->layout) != VK_SUCCESS) {
		vkDestroyDescriptorSetLayout(p->dev, out->set_layout, NULL);
		vkDestroyShaderModule(p->dev, mod, NULL);
		out->set_layout = VK_NULL_HANDLE;
		out->layout		= VK_NULL_HANDLE;
		out->pipeline	= VK_NULL_HANDLE;
		return ERR_INTERNAL;
	}

	VkSpecializationMapEntry map_entries[8];
	int						 n_spec = 0;
	if (spec_data && spec_size > 0) {
		n_spec = (int)(spec_size / 4);
		if (n_spec > 8)
			n_spec = 8;
		for (int i = 0; i < n_spec; i++) {
			map_entries[i] = (VkSpecializationMapEntry){
				.constantID = (uint32_t)i,
				.offset		= (uint32_t)(i * 4),
				.size		= 4,
			};
		}
	}

	VkSpecializationInfo spec_info = {0};
	if (spec_data && spec_size > 0) {
		spec_info.mapEntryCount = (uint32_t)n_spec;
		spec_info.pMapEntries	= map_entries;
		spec_info.dataSize		= spec_size;
		spec_info.pData			= spec_data;
	}

	VkComputePipelineCreateInfo cpci = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage =
			{
				.sType				 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage				 = VK_SHADER_STAGE_COMPUTE_BIT,
				.module				 = mod,
				.pName				 = "main",
				.pSpecializationInfo = (spec_data && spec_size > 0) ? &spec_info : NULL,
			},
		.layout = out->layout,
	};
	VkResult pr = vkCreateComputePipelines(p->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &out->pipeline);
	vkDestroyShaderModule(p->dev, mod, NULL);
	if (pr != VK_SUCCESS) {
		ERROR("vk: vkCreateComputePipelines failed (result=%d).", (int)pr);
		ERROR("vk:       Device maxComputeSharedMemorySize=%u bytes.", p->caps.max_shared_memory);
		ERROR("vk:       If the shader uses > %u bytes of shared memory, reduce the "
			  "tile size.",
			  p->caps.max_shared_memory);
		vkDestroyPipelineLayout(p->dev, out->layout, NULL);
		vkDestroyDescriptorSetLayout(p->dev, out->set_layout, NULL);
		out->set_layout = VK_NULL_HANDLE;
		out->layout		= VK_NULL_HANDLE;
		out->pipeline	= VK_NULL_HANDLE;
		return ERR_INTERNAL;
	}
	return OK;
}

static void vk_destroy_pipeline(vk_priv *p, vk_pipeline_set *ps) {
	if (!ps->pipeline)
		return;
	vk_ring_slot *r = &p->ring[p->ring_cur];
	if (r->has_work && !p->device_lost) {
		status_code fs = vk_flush(p);
		if (fs != OK) {
			ERROR("vk: vk_destroy_pipeline: flush failed (%s)", status_str(fs));
		}
	}
	if (r->has_work) {
		r->has_work = 0;
		vk_dirty_clear(p);
	}
	vk_flush_pending_desc_frees(p);
	vkDestroyPipeline(p->dev, ps->pipeline, NULL);
	vkDestroyPipelineLayout(p->dev, ps->layout, NULL);
	vkDestroyDescriptorSetLayout(p->dev, ps->set_layout, NULL);
	memset(ps, 0, sizeof(*ps));
}

static inline uint32_t vk_dirty_index(VkBuffer buf) {
	return (uint32_t)(((uintptr_t)buf >> 4) * 0x9E3779B9u);
}

static void vk_dirty_clear(vk_priv *p) {
	memset(p->dirty_table, 0, sizeof(p->dirty_table));
	p->dirty_count = 0;
	vk_reads_clear(p);
}

static int vk_dirty_lookup(vk_priv *p, VkBuffer buf) {
	const uint32_t mask = VK_DIRTY_TABLE_SIZE - 1;
	uint32_t	   idx	= vk_dirty_index(buf) & mask;
	for (uint32_t i = 0; i < VK_DIRTY_TABLE_SIZE; i++) {
		VkBuffer s = p->dirty_table[(idx + i) & mask];
		if (s == VK_NULL_HANDLE)
			return 0;
		if (s == buf)
			return 1;
	}
	return 0;
}

static void vk_dirty_insert(vk_priv *p, VkBuffer buf) {
	const uint32_t mask	 = VK_DIRTY_TABLE_SIZE - 1;
	uint32_t	   idx	 = vk_dirty_index(buf) & mask;
	uint32_t	   reuse = VK_DIRTY_TABLE_SIZE;
	for (uint32_t i = 0; i < VK_DIRTY_TABLE_SIZE; i++) {
		VkBuffer *s = &p->dirty_table[(idx + i) & mask];
		if (*s == buf)
			return;
		if (*s == VK_NULL_HANDLE) {
			if (reuse < VK_DIRTY_TABLE_SIZE)
				s = &p->dirty_table[reuse];
			*s = buf;
			p->dirty_count++;
			return;
		}
		if (*s == (VkBuffer)0x1 && reuse == VK_DIRTY_TABLE_SIZE)
			reuse = (idx + i) & mask;
	}
}

static void vk_dirty_remove(vk_priv *p, VkBuffer buf) {
	const uint32_t mask = VK_DIRTY_TABLE_SIZE - 1;
	uint32_t	   idx	= vk_dirty_index(buf) & mask;
	for (uint32_t i = 0; i < VK_DIRTY_TABLE_SIZE; i++) {
		VkBuffer *s = &p->dirty_table[(idx + i) & mask];
		if (*s == VK_NULL_HANDLE)
			return;
		if (*s == buf) {
			*s = (VkBuffer)0x1;
			p->dirty_count--;
			return;
		}
	}
}

static inline uint64_t vk_desc_hash(const VkBuffer *bufs, const VkDeviceSize *offs, int n_bufs) {
	uint64_t h = FNV1A_OFFSET_BASIS;
	for (int i = 0; i < n_bufs; i++) {
		h = fnv1a_update(h, (const char *)&bufs[i], sizeof(bufs[i]));
		h = fnv1a_update(h, (const char *)&offs[i], sizeof(offs[i]));
	}
	return h;
}

static void vk_dirty_add(vk_priv *p, const VkBuffer *bufs, int n_bufs, uint32_t write_mask) {
	for (int i = 0; i < n_bufs; i++) {
		if (!(write_mask & (1u << i)))
			continue;
		if (p->dirty_count >= VK_DIRTY_MAX) {
			VkMemoryBarrier barrier = {
				.sType		   = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
			};
			p->stats.barriers++;
			vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
								 NULL);
			vk_dirty_clear(p);
		}
		if (!vk_dirty_lookup(p, bufs[i]))
			vk_dirty_insert(p, bufs[i]);
	}
}

static status_code vk_dispatch_ex(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs,
								  const VkDeviceSize *offs, const VkDeviceSize *ranges, int n_bufs,
								  const void *push, uint32_t push_size, uint32_t groups_x,
								  uint32_t groups_y, uint32_t write_mask) {
	if (p->device_lost)
		return ERR_INTERNAL;

	if (groups_x == 0)
		groups_x = 1;
	if (groups_y == 0)
		groups_y = 1;
	if (groups_x > p->caps.max_workgroup_count[0] || groups_y > p->caps.max_workgroup_count[1]) {
		ERROR("vk: dispatch %s (%u, %u) exceeds device limit (%u, %u)", ps->name, groups_x,
			  groups_y, p->caps.max_workgroup_count[0], p->caps.max_workgroup_count[1]);
		return ERR_INVALID_ARG;
	}

	p->stream_seq++;
	if (!p->ring[p->ring_cur].recording) {
		status_code bs = vk_ring_begin(p, p->ring_cur);
		if (bs != OK)
			return bs;
	}
	int flush_limit;
	if (p->caps.is_mali || p->caps.is_power_vr || p->caps.is_adreno) {
		flush_limit = 256;
	} else {
		flush_limit = p->batch_active ? 4096 : 256;
	}
	if (p->pending_dispatches >= flush_limit) {
		status_code fs = vk_run_cmd(p);
		if (fs != OK)
			return fs;
		if (!p->ring[p->ring_cur].recording) {
			status_code bs = vk_ring_begin(p, p->ring_cur);
			if (bs != OK)
				return bs;
		}
	}

	if (n_bufs > VK_MAX_BINDINGS)
		return ERR_INVALID_ARG;

	for (int i = 0; i < n_bufs; i++) {
		if (!bufs[i]) {
			ERROR("vk: vk_dispatch_ex: NULL vk_buf at index %d (pipeline=%s)", i,
				  ps->name ? ps->name : "?");
			return ERR_INVALID_ARG;
		}
		if (bufs[i]->buf == VK_NULL_HANDLE) {
			ERROR("vk: vk_dispatch_ex: VK_NULL_HANDLE VkBuffer at index %d "
				  "(pipeline=%s) — buffer was destroyed or never allocated",
				  i, ps->name ? ps->name : "?");
			return ERR_INVALID_ARG;
		}
	}

	VkBuffer	 key[VK_MAX_BINDINGS];
	VkDeviceSize key_offs[VK_MAX_BINDINGS];
	for (int i = 0; i < n_bufs; i++) {
		key[i]		= bufs[i]->buf;
		key_offs[i] = offs ? offs[i] : 0;
	}

	int war = 0;
	for (int i = 0; i < n_bufs && !war; i++)
		war = ((write_mask >> i) & 1u) && vk_reads_has(p, key[i]);
	if (war) {
		VkMemoryBarrier barrier = {
			.sType		   = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		};
		p->stats.barriers++;
		vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
							 NULL);
		vk_dirty_clear(p);
	}

	if (p->dirty_count > 0) {
		VkBufferMemoryBarrier bars[VK_MAX_BINDINGS];
		int					  nbar = 0;
		for (int i = 0; i < n_bufs; i++) {
			if (!vk_dirty_lookup(p, key[i]))
				continue;
			bars[nbar++] = (VkBufferMemoryBarrier){
				.sType				 = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				.srcAccessMask		 = VK_ACCESS_SHADER_WRITE_BIT,
				.dstAccessMask		 = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.buffer				 = key[i],
				.offset				 = 0,
				.size				 = VK_WHOLE_SIZE,
			};
			vk_dirty_remove(p, key[i]);
		}
		if (nbar > 0) {
			vk_reads_clear(p);
			p->stats.barriers++;
			vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, (uint32_t)nbar,
								 bars, 0, NULL);
		}
		if (p->dirty_count == 0)
			vk_dirty_clear(p);
	}

	const size_t key_bytes = sizeof(VkBuffer) * (size_t)n_bufs;
	const size_t off_bytes = sizeof(VkDeviceSize) * (size_t)n_bufs;
	uint64_t	 key_hash  = vk_desc_hash(key, key_offs, n_bufs);

	VkDescriptorSet set		  = VK_NULL_HANDLE;
	int				miss_slot = -1;
	int				lru_slot  = 0;
	uint64_t		lru_tick  = UINT64_MAX;

	int has_dead = 0;
	if (p->dead_buf_count > 0) {
		for (int i = 0; i < n_bufs; i++) {
			if (vk_buf_is_dead(p, key[i])) {
				has_dead = 1;
				break;
			}
		}
	}

	if (!has_dead) {
		const uint32_t VK_DESC_PROBE_MAX = 32;
		uint32_t	   probe			 = (uint32_t)(key_hash % VK_DESC_CACHE_CAP);
		for (uint32_t i = 0; i < VK_DESC_PROBE_MAX; i++) {
			uint32_t	  idx = (probe + i) % VK_DESC_CACHE_CAP;
			vk_desc_slot *s	  = &ps->desc_cache[idx];
			if (!s->valid) {
				if (miss_slot < 0)
					miss_slot = (int)idx;
				continue;
			}
			if (s->hash != key_hash) {
				if (s->last_used < lru_tick) {
					lru_tick = s->last_used;
					lru_slot = (int)idx;
				}
				continue;
			}
			if (memcmp(s->bufs, key, key_bytes) == 0 && memcmp(s->offs, key_offs, off_bytes) == 0) {
				set = s->set;
				p->desc_tick++;
				s->last_used = p->desc_tick;
				goto have_set;
			}
			if (s->last_used < lru_tick) {
				lru_tick = s->last_used;
				lru_slot = (int)idx;
			}
		}
	}

	if (has_dead) {
		for (int i = 0; i < VK_DESC_CACHE_CAP; i++) {
			vk_desc_slot *s = &ps->desc_cache[i];
			if (!s->valid) {
				if (miss_slot < 0)
					miss_slot = i;
				continue;
			}
			int dead = 0;
			for (int b = 0; b < n_bufs; b++) {
				if (vk_buf_is_dead(p, s->bufs[b])) {
					dead = 1;
					break;
				}
			}
			if (dead) {
				vk_queue_desc_free(p, s->set);
				s->valid = 0;
				s->set	 = VK_NULL_HANDLE;
				if (miss_slot < 0)
					miss_slot = i;
			}
		}
	}

	{
		if (miss_slot < 0)
			miss_slot = lru_slot;
		vk_desc_slot *s = &ps->desc_cache[miss_slot];

		if (s->valid && s->set != VK_NULL_HANDLE) {
			vk_queue_desc_free(p, s->set);
			s->set	 = VK_NULL_HANDLE;
			s->valid = 0;
		}

		VkDescriptorSetAllocateInfo dsai = {
			.sType				= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool		= p->desc_pool,
			.descriptorSetCount = 1,
			.pSetLayouts		= &ps->set_layout,
		};
		VkDescriptorSet new_set;
		VK_CHECK(vkAllocateDescriptorSets(p->dev, &dsai, &new_set));
		p->total_desc_allocs++;
		if (p->debug.diag) {
			p->diag_desc_allocs++;
			fprintf(stderr, "[VK_DIAG] desc_alloc #%llu pipeline=%s\n",
					(unsigned long long)p->diag_desc_allocs, ps->name ? ps->name : "?");
		}

		VkDescriptorBufferInfo binfo[VK_MAX_BINDINGS];
		VkWriteDescriptorSet   writes[VK_MAX_BINDINGS];
		for (int i = 0; i < n_bufs; i++) {
			binfo[i] = (VkDescriptorBufferInfo){
				.buffer = key[i],
				.offset = key_offs[i],
				.range	= ranges ? ranges[i] : VK_WHOLE_SIZE,
			};
			writes[i] = (VkWriteDescriptorSet){
				.sType			 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet			 = new_set,
				.dstBinding		 = (uint32_t)i,
				.descriptorCount = 1,
				.descriptorType	 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo	 = &binfo[i],
			};
		}
		vkUpdateDescriptorSets(p->dev, (uint32_t)n_bufs, writes, 0, NULL);

		s->set = new_set;
		memcpy(s->bufs, key, key_bytes);
		memcpy(s->offs, key_offs, off_bytes);
		s->hash	 = key_hash;
		s->valid = 1;
		p->desc_tick++;
		s->last_used = p->desc_tick;
		set			 = new_set;
	}

have_set:;

	if (p->last_pipeline != ps->pipeline) {
		vkCmdBindPipeline(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipeline);
		p->last_pipeline			= ps->pipeline;
		p->last_desc_pipeline_match = 0;
	}
	if (!p->last_desc_pipeline_match || p->last_desc_set != set) {
		vkCmdBindDescriptorSets(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, 1, &set, 0,
								NULL);
		p->last_desc_set			= set;
		p->last_desc_pipeline_match = 1;
	}
	if (push_size > 0)
		vkCmdPushConstants(p->cmd, ps->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);

	int pq = vk_prof_begin(p, ps->name);
	vkCmdDispatch(p->cmd, groups_x, groups_y, 1);
	vk_prof_end(p, pq);
	p->stats.dispatches++;

	vk_dirty_add(p, key, n_bufs, write_mask);
	for (int i = 0; i < n_bufs; i++)
		if (!((write_mask >> i) & 1u))
			vk_reads_add(p, key[i]);

	p->pending_dispatches++;
	p->ring[p->ring_cur].has_work = 1;
	return OK;
}

static status_code vk_dispatch(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs, int n_bufs,
							   const void *push, uint32_t push_size, uint32_t groups_x) {
	uint32_t wmask = (n_bufs > 0) ? (1u << (n_bufs - 1)) : 0;
	return vk_dispatch_ex(p, ps, bufs, NULL, NULL, n_bufs, push, push_size, groups_x, 1, wmask);
}

static status_code vk_dispatch_2d_ex(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs,
									 const VkDeviceSize *offs, const VkDeviceSize *ranges,
									 int n_bufs, const void *push, uint32_t push_size,
									 uint32_t groups_x, uint32_t groups_y, uint32_t write_mask) {
	return vk_dispatch_ex(p, ps, bufs, offs, ranges, n_bufs, push, push_size, groups_x, groups_y,
						  write_mask);
}

static VkBool32 vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT	  severity,
								  VkDebugUtilsMessageTypeFlagsEXT			  type,
								  const VkDebugUtilsMessengerCallbackDataEXT *data,
								  void										 *user_data) {
	(void)type;
	(void)user_data;
	const char *level = "INFO";
	if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
		level = "ERROR";
	else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
		level = "WARN";
	fprintf(stderr, "[VK_VALIDATE][%s] %s\n", level, data->pMessage ? data->pMessage : "?");
	return VK_FALSE;
}

static void vk_qcore_init(vk_priv *p) {
	static const struct {
		const uint32_t *spv;
		size_t			len;
		const char	   *name[2];
	} MV[VQ_COUNT] = {
		[VQ_Q4_0] = VQ_MV(q_matvec_q4_0, "q4_0"),  [VQ_Q4_1] = VQ_MV(q_matvec_q4_1, "q4_1"),
		[VQ_Q5_0] = VQ_MV(q_matvec_q5_0, "q5_0"),  [VQ_Q5_1] = VQ_MV(q_matvec_q5_1, "q5_1"),
		[VQ_Q8_0] = VQ_MV(q_matvec_q8_0, "q8_0"),  [VQ_IQ4_NL] = VQ_MV(q_matvec_iq4_nl, "iq4_nl"),
		[VQ_Q4_K] = VQ_MV(kq_matvec_q4_k, "q4_k"), [VQ_Q5_K] = VQ_MV(kq_matvec_q5_k, "q5_k"),
		[VQ_Q6_K] = VQ_MV(kq_matvec_q6_k, "q6_k"), [VQ_F32] = VQ_MV(dense_matvec_f32, "f32"),
		[VQ_F16] = VQ_MV(dense_matvec_f16, "f16"), [VQ_BF16] = VQ_MV(dense_matvec_bf16, "bf16"),
	};
	static const struct {
		const uint32_t *spv;
		size_t			len;
		const char	   *name;
	} MM[VQ_MM_COUNT] = {
		[VQ_Q4_0] = VQ_MM(q_matmul_q4_0, "q4_0"), [VQ_Q4_1] = VQ_MM(q_matmul_q4_1, "q4_1"),
		[VQ_Q5_0] = VQ_MM(q_matmul_q5_0, "q5_0"), [VQ_Q5_1] = VQ_MM(q_matmul_q5_1, "q5_1"),
		[VQ_Q8_0] = VQ_MM(q_matmul_q8_0, "q8_0"), [VQ_IQ4_NL] = VQ_MM(q_matmul_iq4_nl, "iq4_nl"),
	};
	if (vk_create_pipeline(p, shader_attn_decode_spv, sizeof(shader_attn_decode_spv), 5, 40,
						   &p->p_attn_decode) == OK)
		p->p_attn_decode.name = "attn_decode";
	if (vk_create_pipeline(p, shader_attn_merge_spv, sizeof(shader_attn_merge_spv), 2, 8,
						   &p->p_attn_merge) == OK)
		p->p_attn_merge.name = "attn_merge";
	p->qcore_ok = 0;
	if (!p->caps.supports_integer_dot_product || !p->caps.supports_16bit_storage ||
		!p->caps.supports_subgroup_arithmetic || p->caps.max_workgroup_invocations < 256 ||
		p->caps.max_shared_memory < 16384)
		return;
	if (vk_create_pipeline(p, shader_q8_quant_spv, sizeof(shader_q8_quant_spv), 3, 16,
						   &p->p_q8_quant[0]) != OK ||
		vk_create_pipeline(p, shader_q8k_quant_spv, sizeof(shader_q8k_quant_spv), 3, 16,
						   &p->p_q8_quant[1]) != OK)
		return;
	p->p_q8_quant[0].name = "q8_quant";
	p->p_q8_quant[1].name = "q8k_quant";
	for (int t = 0; t < VQ_COUNT; t++) {
		for (int c = 0; c < 2; c++) {
			if (vk_create_pipeline_spec(p, MV[t].spv, MV[t].len, 4, 16, VQ_MV_SPEC[c],
										sizeof(VQ_MV_SPEC[c]), &p->p_qmv[t][c]) != OK)
				return;
			p->p_qmv[t][c].name = MV[t].name[c];
		}
	}
	for (int t = 0; t < VQ_MM_COUNT; t++) {
		if (vk_create_pipeline(p, MM[t].spv, MM[t].len, 4, 16, &p->p_qmm[t]) != OK)
			return;
		p->p_qmm[t].name = MM[t].name;
	}
	p->qcore_ok = 1;
}

static status_code vk_init(backend *self, int device_index) {
	vk_priv *p = xcalloc(1, sizeof(vk_priv));
	self->priv = p;

	p->debug.diag			= getenv("VK_DIAG") != NULL;
	p->debug.validate		= getenv("VK_VALIDATE") != NULL;
	p->debug.abort_on_error = getenv("VK_ABORT_ON_ERROR") != NULL;

	VkApplicationInfo	 app			  = {.sType		 = VK_STRUCTURE_TYPE_APPLICATION_INFO,
											 .apiVersion = VK_API_VERSION_1_3};
	VkInstanceCreateInfo ici			  = {.sType			   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
											 .pApplicationInfo = &app};
	const char			*validation_layer = "VK_LAYER_KHRONOS_validation";
	const char			*debug_ext		  = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
	if (p->debug.validate) {
		uint32_t layer_count = 0;
		vkEnumerateInstanceLayerProperties(&layer_count, NULL);
		VkLayerProperties *layers =
			layer_count ? xmalloc(layer_count * sizeof(VkLayerProperties)) : NULL;
		if (layers)
			vkEnumerateInstanceLayerProperties(&layer_count, layers);
		int layer_found = 0;
		for (uint32_t i = 0; i < layer_count; i++) {
			if (strcmp(layers[i].layerName, validation_layer) == 0) {
				layer_found = 1;
				break;
			}
		}
		free(layers);
		if (!layer_found) {
			WARN("vk: VK_VALIDATE=1 requested but %s not installed", validation_layer);
			p->debug.validate = 0;
		}
	}
	if (p->debug.validate) {
		ici.enabledLayerCount		= 1;
		ici.ppEnabledLayerNames		= &validation_layer;
		ici.enabledExtensionCount	= 1;
		ici.ppEnabledExtensionNames = &debug_ext;
	}
	if (vkCreateInstance(&ici, NULL, &p->instance) != VK_SUCCESS) {
		app.apiVersion = VK_API_VERSION_1_2;
		if (vkCreateInstance(&ici, NULL, &p->instance) != VK_SUCCESS) {
			app.apiVersion = VK_API_VERSION_1_1;
			if (vkCreateInstance(&ici, NULL, &p->instance) != VK_SUCCESS)
				return ERR_INTERNAL;
		}
	}

	if (p->debug.validate) {
		PFN_vkCreateDebugUtilsMessengerEXT create_fn =
			(PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
				p->instance, "vkCreateDebugUtilsMessengerEXT");
		if (create_fn) {
			VkDebugUtilsMessengerCreateInfoEXT dbci = {
				.sType			 = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
				.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
								   VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
								   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
								   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
				.messageType	 = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
								   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
								   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
				.pfnUserCallback = vk_debug_callback,
			};
			if (create_fn(p->instance, &dbci, NULL, &p->debug_messenger) != VK_SUCCESS) {
				WARN("vk: failed to create debug messenger; validation output will not "
					 "be printed");
			}
		} else {
			WARN("vk: VK_EXT_debug_utils not available, no validation output");
		}
	}

	uint32_t count = 0;
	vkEnumeratePhysicalDevices(p->instance, &count, NULL);
	if (count == 0)
		return ERR_UNSUPPORTED;
	if (device_index < 0 || (uint32_t)device_index >= count)
		return ERR_NOT_FOUND;

	VkPhysicalDevice *devs = xmalloc(count * sizeof(VkPhysicalDevice));
	vkEnumeratePhysicalDevices(p->instance, &count, devs);
	p->phys = devs[device_index];
	free(devs);

	vkGetPhysicalDeviceMemoryProperties(p->phys, &p->mem_props);

	uint32_t qf_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(p->phys, &qf_count, NULL);
	VkQueueFamilyProperties *qfs = xmalloc(qf_count * sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(p->phys, &qf_count, qfs);

	p->queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < qf_count; i++) {
		if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			p->queue_family = i;
			break;
		}
	}
	free(qfs);
	if (p->queue_family == UINT32_MAX)
		return ERR_UNSUPPORTED;

	int has_timeline_ext	  = 0;
	int has_dot_product_ext	  = 0;
	int has_memory_budget_ext = 0;
	{
		uint32_t ext_count = 0;
		vkEnumerateDeviceExtensionProperties(p->phys, NULL, &ext_count, NULL);
		if (ext_count > 0) {
			VkExtensionProperties *exts = xmalloc(ext_count * sizeof(VkExtensionProperties));
			vkEnumerateDeviceExtensionProperties(p->phys, NULL, &ext_count, exts);
			for (uint32_t i = 0; i < ext_count; i++) {
				if (strcmp(exts[i].extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) {
					has_timeline_ext = 1;
				} else if (strcmp(exts[i].extensionName,
								  VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) {
					has_dot_product_ext = 1;
#ifdef VK_EXT_memory_budget
				} else if (strcmp(exts[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) ==
						   0) {
					has_memory_budget_ext = 1;
#endif
				}
			}
			free(exts);
		}
	}

	VkPhysicalDeviceProperties dev_props_for_version;
	vkGetPhysicalDeviceProperties(p->phys, &dev_props_for_version);
	uint32_t device_api_version			= dev_props_for_version.apiVersion;
	int		 device_supports_v2_queries = (device_api_version >= VK_API_VERSION_1_1);

	VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dot_product_feat = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
	};
	VkPhysicalDeviceVulkan11Features vk11_feat = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext = &dot_product_feat,
	};
	VkPhysicalDeviceVulkan12Features vk12_feat = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &vk11_feat,
	};
	VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feat = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
		.pNext =
			(device_api_version >= VK_API_VERSION_1_2) ? &vk12_feat : (void *)&dot_product_feat,
	};
	VkPhysicalDeviceFeatures2 feat2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &timeline_feat,
	};
	if (device_supports_v2_queries &&
		(has_timeline_ext || device_api_version >= VK_API_VERSION_1_2)) {
		vkGetPhysicalDeviceFeatures2(p->phys, &feat2);
	}
	int timeline_available = timeline_feat.timelineSemaphore ? 1 : 0;

	int int8_available		  = (device_api_version >= VK_API_VERSION_1_2) && vk12_feat.shaderInt8;
	int int16_available		  = (device_api_version >= VK_API_VERSION_1_2) &&
								vk11_feat.storageBuffer16BitAccess && feat2.features.shaderInt16;
	int dot_product_available = (device_api_version >= VK_API_VERSION_1_3 || has_dot_product_ext) &&
								dot_product_feat.shaderIntegerDotProduct;

	uint64_t max_timeline_diff = 0;
	if (timeline_available) {
		VkPhysicalDeviceTimelineSemaphoreProperties timeline_props = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES,
		};
		VkPhysicalDeviceProperties2 props2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &timeline_props,
		};
		if (device_supports_v2_queries &&
			(has_timeline_ext || device_api_version >= VK_API_VERSION_1_2)) {
			vkGetPhysicalDeviceProperties2(p->phys, &props2);
			max_timeline_diff = timeline_props.maxTimelineSemaphoreValueDifference;
		}

		const uint64_t MIN_USABLE_TIMELINE_DIFF = 4096;
		if (max_timeline_diff < MIN_USABLE_TIMELINE_DIFF) {
			WARN("vulkan: timeline semaphore reported but "
				 "maxTimelineSemaphoreValueDifference=%llu is too small to "
				 "trust -- falling back to fence-based sync",
				 (unsigned long long)max_timeline_diff);
			timeline_available = 0;
		}
	}

	const char *device_exts[3];
	uint32_t	n_device_exts = 0;
	if (timeline_available && has_timeline_ext && device_api_version < VK_API_VERSION_1_2) {
		device_exts[n_device_exts++] = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
	}
	if (dot_product_available && has_dot_product_ext && device_api_version < VK_API_VERSION_1_3) {
		device_exts[n_device_exts++] = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
	}
#ifdef VK_EXT_memory_budget
	if (has_memory_budget_ext) {
		device_exts[n_device_exts++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
	}
#endif

	float					prio = 1.0f;
	VkDeviceQueueCreateInfo dqci = {
		.sType			  = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = p->queue_family,
		.queueCount		  = 1,
		.pQueuePriorities = &prio,
	};
	VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR enable_dot_product = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
		.shaderIntegerDotProduct = dot_product_available ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceVulkan11Features enable_vk11 = {
		.sType					  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext					  = &enable_dot_product,
		.storageBuffer16BitAccess = int16_available ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceVulkan12Features enable_vk12 = {
		.sType			   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext			   = &enable_vk11,
		.shaderInt8		   = int8_available ? VK_TRUE : VK_FALSE,
		.timelineSemaphore = timeline_available ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceTimelineSemaphoreFeatures enable_timeline = {
		.sType			   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
		.pNext			   = &enable_dot_product,
		.timelineSemaphore = timeline_available ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceFeatures2 enable_feat2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = (device_api_version >= VK_API_VERSION_1_2) ? (void *)&enable_vk12
															: (void *)&enable_timeline,
		.features.shaderInt16 = int16_available ? VK_TRUE : VK_FALSE,
	};
	VkDeviceCreateInfo dci = {
		.sType					 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext					 = &enable_feat2,
		.queueCreateInfoCount	 = 1,
		.pQueueCreateInfos		 = &dqci,
		.enabledExtensionCount	 = n_device_exts,
		.ppEnabledExtensionNames = n_device_exts ? device_exts : NULL,
	};
	if (vkCreateDevice(p->phys, &dci, NULL, &p->dev) != VK_SUCCESS) {
		timeline_available			= 0;
		int16_available				= 0;
		dot_product_available		= 0;
		has_memory_budget_ext		= 0;
		dci.pNext					= NULL;
		dci.enabledExtensionCount	= 0;
		dci.ppEnabledExtensionNames = NULL;
		if (vkCreateDevice(p->phys, &dci, NULL, &p->dev) != VK_SUCCESS)
			return ERR_INTERNAL;
	}
	p->timeline_supported				 = timeline_available;
	p->caps.supports_integer_dot_product = dot_product_available;
	p->caps.supports_16bit_storage		 = int16_available;
	p->has_memory_budget_ext			 = has_memory_budget_ext;

	vkGetDeviceQueue(p->dev, p->queue_family, 0, &p->queue);

	VkPhysicalDeviceProperties props		= dev_props_for_version;
	p->caps.max_shared_memory				= props.limits.maxComputeSharedMemorySize;
	p->caps.max_workgroup_size[0]			= props.limits.maxComputeWorkGroupSize[0];
	p->caps.max_workgroup_size[1]			= props.limits.maxComputeWorkGroupSize[1];
	p->caps.max_workgroup_size[2]			= props.limits.maxComputeWorkGroupSize[2];
	p->caps.max_workgroup_invocations		= props.limits.maxComputeWorkGroupInvocations;
	p->caps.storage_buffer_offset_alignment = props.limits.minStorageBufferOffsetAlignment;
	p->caps.max_workgroup_count[0]			= props.limits.maxComputeWorkGroupCount[0];
	p->caps.max_workgroup_count[1]			= props.limits.maxComputeWorkGroupCount[1];
	p->caps.max_storage_buffer_range		= props.limits.maxStorageBufferRange;

	p->caps.is_mali		= (props.vendorID == 0x13B5);
	p->caps.is_power_vr = (props.vendorID == 0x1010);
	p->caps.is_adreno	= (props.vendorID == 0x5143);
	p->caps.is_amd		= (props.vendorID == 0x1002);

	p->caps.subgroup_size = 1;
#ifdef VK_VERSION_1_1
	VkPhysicalDeviceSubgroupProperties subgroup_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
	};
	VkPhysicalDeviceProperties2 pdp2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &subgroup_props,
	};
	vkGetPhysicalDeviceProperties2(p->phys, &pdp2);
	p->caps.subgroup_size				 = subgroup_props.subgroupSize;
	VkSubgroupFeatureFlags sf			 = subgroup_props.supportedOperations;
	p->caps.supports_subgroup_basic		 = (sf & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
	p->caps.supports_subgroup_arithmetic = (sf & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
#endif

	p->caps.unified_memory = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
							 props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
	if (p->caps.unified_memory)
		self->caps |= BCAP_UNIFIED_MEMORY;

	log_tag("VLK", "%.*s", (int)backend_name_brief_len(props.deviceName), props.deviceName);
	DEBUG("vulkan: device: %s", props.deviceName);

	if (p->caps.is_mali || p->caps.is_power_vr || p->caps.is_adreno) {
		p->matmul_wg_size		  = 32;
		p->matmul_rows_per_thread = 2;
		p->matmul_tile_k		  = 1024;
		p->matmul_m_per_wg		  = 4;
	} else {
		p->matmul_wg_size		  = 64;
		p->matmul_rows_per_thread = 4;
		p->matmul_tile_k		  = 2048;
		p->matmul_m_per_wg		  = 4;
	}
	if (p->caps.is_mali) {
		if (p->caps.subgroup_size <= 8) {
			p->matmul_wg_size		  = 32;
			p->matmul_rows_per_thread = 1;
			p->matmul_tile_k		  = 512;
			p->matmul_m_per_wg		  = 2;
		} else {
			p->matmul_wg_size		  = 96;
			p->matmul_rows_per_thread = 1;
			p->matmul_tile_k		  = 1152;
			p->matmul_m_per_wg		  = 2;
		}
	}
	p->matmul_lanes_per_row = 1;
	if (p->caps.is_mali || p->caps.is_adreno || p->caps.is_amd)
		p->matmul_lanes_per_row = 8;
	if (p->caps.is_adreno) {
		p->matmul_wg_size		  = 128;
		p->matmul_rows_per_thread = 1;
		p->matmul_tile_k		  = 256;
		p->matmul_m_per_wg		  = 2;
	}
	if (p->caps.is_amd) {
		p->matmul_wg_size		  = 128;
		p->matmul_rows_per_thread = 1;
		p->matmul_tile_k		  = 2048;
		p->matmul_m_per_wg		  = 4;
	}

	VkCommandPoolCreateInfo cpci = {
		.sType			  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.queueFamilyIndex = p->queue_family,
	};
	for (int i = 0; i < VK_RING_DEPTH; i++) {
		if (vkCreateCommandPool(p->dev, &cpci, NULL, &p->cmd_pools[i]) != VK_SUCCESS)
			return ERR_INTERNAL;
	}

	{
		VkCommandBufferAllocateInfo cbai = {
			.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		for (int i = 0; i < VK_RING_DEPTH; i++) {
			cbai.commandPool = p->cmd_pools[i];
			if (vkAllocateCommandBuffers(p->dev, &cbai, &p->ring[i].cmd) != VK_SUCCESS)
				return ERR_INTERNAL;
		}
	}

	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	if (vkCreateFence(p->dev, &fci, NULL, &p->fence) != VK_SUCCESS)
		return ERR_INTERNAL;

	if (p->timeline_supported) {
		VkSemaphoreTypeCreateInfo type_ci = {
			.sType		   = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
			.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
			.initialValue  = 0,
		};
		VkSemaphoreCreateInfo sem_ci = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = &type_ci,
		};
		if (vkCreateSemaphore(p->dev, &sem_ci, NULL, &p->timeline) != VK_SUCCESS) {
			WARN("timeline semaphore creation failed, falling back to fence sync");
			p->timeline_supported = 0;
		}
	}
	p->timeline_value	  = 0;
	p->timeline_completed = 0;

	VkDescriptorPoolSize	   dps	= {.type			= VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
									   .descriptorCount = 65536};
	VkDescriptorPoolCreateInfo dpci = {
		.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags		   = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets	   = 8192,
		.poolSizeCount = 1,
		.pPoolSizes	   = &dps,
	};
	if (vkCreateDescriptorPool(p->dev, &dpci, NULL, &p->desc_pool) != VK_SUCCESS)
		return ERR_INTERNAL;

	p->ring_cur = 0;
	{
		status_code bs = vk_ring_begin(p, 0);
		if (bs != OK)
			return bs;
	}

	status_code s;
	uint32_t	spec_matmul[3] = {
		(uint32_t)p->matmul_wg_size,
		(uint32_t)p->matmul_rows_per_thread,
		(uint32_t)p->matmul_tile_k,
	};

	uint32_t spec_matmul_dense_narrow[6] = {
		(uint32_t)p->matmul_wg_size,	   (uint32_t)p->matmul_rows_per_thread,
		(uint32_t)p->matmul_tile_k,		   1u,
		(uint32_t)p->matmul_lanes_per_row, (uint32_t)p->matmul_wg_size,
	};

	uint32_t dense_tile_k	= (uint32_t)p->matmul_tile_k;
	uint32_t dense_m_per_wg = (uint32_t)p->matmul_m_per_wg;
	if (p->caps.max_shared_memory > 0) {
		double	 budget		  = (double)p->caps.max_shared_memory * 0.75;
		uint32_t max_m_per_wg = (uint32_t)(budget / (4.0 * (double)dense_tile_k));
		if (max_m_per_wg < 1)
			max_m_per_wg = 1;
		if (dense_m_per_wg > max_m_per_wg)
			dense_m_per_wg = max_m_per_wg;
	}
	if (dense_m_per_wg < 1)
		dense_m_per_wg = 1;
	p->matmul_m_per_wg			 = (int)dense_m_per_wg;
	uint32_t spec_matmul_wide[6] = {
		(uint32_t)p->matmul_wg_size,
		(uint32_t)p->matmul_rows_per_thread,
		dense_tile_k,
		dense_m_per_wg,
		(uint32_t)p->matmul_lanes_per_row,
		(uint32_t)p->matmul_wg_size,
	};

	uint32_t kquant_tile_k = (uint32_t)p->matmul_tile_k;
	if (kquant_tile_k < 256)
		kquant_tile_k = 256;
	if (kquant_tile_k % 256 != 0)
		kquant_tile_k += 256 - (kquant_tile_k % 256);
	if (p->caps.max_shared_memory > 0) {
		double	 budget			= (double)p->caps.max_shared_memory * 0.75;
		double	 bytes_per_elem = 4.0 + 4.0 + (4.0 / 256.0) + (4.0 * 16.0 / 256.0);
		uint32_t max_tile		= (uint32_t)(budget / bytes_per_elem);
		max_tile -= max_tile % 256;
		if (max_tile < 256)
			max_tile = 256;
		if (max_tile < kquant_tile_k)
			kquant_tile_k = max_tile;
	}
	uint32_t spec_matmul_kquant[3] = {
		(uint32_t)p->matmul_wg_size,
		(uint32_t)p->matmul_rows_per_thread,
		kquant_tile_k,
	};

	uint32_t spec_matmul_iq3s[3] = {
		(uint32_t)p->matmul_wg_size,
		(uint32_t)p->matmul_rows_per_thread,
		kquant_tile_k,
	};

	s = vk_create_pipeline(p, shader_kv_put_spv, shader_kv_put_spv_len, 4, 36, &p->p_kv_put);
	if (s != OK)
		return s;
	p->p_kv_put.name = "kv_put";
	s = vk_create_pipeline(p, shader_kv_shift_gather_spv, shader_kv_shift_gather_spv_len, 6, 36,
						   &p->p_kv_shift_gather);
	if (s != OK)
		return s;
	p->p_kv_shift_gather.name = "kv_shift_gather";
	s = vk_create_pipeline(p, shader_embd_lookup_spv, shader_embd_lookup_spv_len, 4, 16,
						   &p->p_embd_lookup);
	if (s != OK)
		return s;
	p->p_embd_lookup.name = "embd_lookup";
	s = vk_create_pipeline(p, shader_argmax_spv, shader_argmax_spv_len, 2, 8, &p->p_argmax);
	if (s != OK)
		return s;
	p->p_argmax.name = "argmax";
	s = vk_create_pipeline(p, shader_argmax_reduce_spv, shader_argmax_reduce_spv_len, 3, 8,
						   &p->p_argmax_reduce);
	if (s != OK)
		return s;
	p->p_argmax_reduce.name = "argmax_reduce";
	s = vk_create_pipeline(p, shader_moe_gather_spv, shader_moe_gather_spv_len, 3, 8,
						   &p->p_moe_gather);
	if (s != OK)
		return s;
	p->p_moe_gather.name = "moe_gather";
	s = vk_create_pipeline(p, shader_moe_combine_spv, shader_moe_combine_spv_len, 4, 12,
						   &p->p_moe_combine);
	if (s != OK)
		return s;
	p->p_moe_combine.name = "moe_combine";
	{
		VkBufferUsageFlags	  grid_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VkMemoryPropertyFlags grid_flags =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		if (p->caps.unified_memory)
			grid_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		s = vk_alloc_buffer(p, sizeof(ggml_iq3s_grid), grid_usage, grid_flags, &p->iq3s_grid_buf);
		if (s != OK) {
			WARN("vulkan: failed to allocate IQ3_S grid buffer -- "
				 "IQ3_S grid path will be disabled");
			p->iq3s_grid_buf.buf = VK_NULL_HANDLE;
		} else {
			memcpy(p->iq3s_grid_buf.mapped, ggml_iq3s_grid, sizeof(ggml_iq3s_grid));
		}
	}
	{
		uint8_t				 *iqx		= NULL;
		size_t				  iqx_bytes = sizeof(ggml_iq2s_grid) + sizeof(ggml_iq2xxs_grid) +
										  sizeof(ggml_iq3xxs_grid) + sizeof(ggml_ksigns_iq2xs);
		VkBufferUsageFlags	  iqx_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VkMemoryPropertyFlags iqx_flags =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		if (p->caps.unified_memory)
			iqx_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		s = vk_alloc_buffer(p, iqx_bytes, iqx_usage, iqx_flags, &p->iqx_grid_buf);
		if (s != OK) {
			WARN("vulkan: failed to allocate IQX grid buffer -- "
				 "IQ2_S/IQ2_XXS/IQ3_XXS grid paths will be disabled");
			p->iqx_grid_buf.buf = VK_NULL_HANDLE;
		} else {
			iqx = (uint8_t *)p->iqx_grid_buf.mapped;
			memcpy(iqx, ggml_iq2s_grid, sizeof(ggml_iq2s_grid));
			iqx += sizeof(ggml_iq2s_grid);
			memcpy(iqx, ggml_iq2xxs_grid, sizeof(ggml_iq2xxs_grid));
			iqx += sizeof(ggml_iq2xxs_grid);
			memcpy(iqx, ggml_iq3xxs_grid, sizeof(ggml_iq3xxs_grid));
			iqx += sizeof(ggml_iq3xxs_grid);
			memcpy(iqx, ggml_ksigns_iq2xs, sizeof(ggml_ksigns_iq2xs));
		}
	}

	s = vk_create_pipeline_spec(p, shader_matmul_f32_batch_spv, shader_matmul_f32_batch_spv_len, 3,
								12, spec_matmul_dense_narrow, sizeof(spec_matmul_dense_narrow),
								&p->p_matmul_f32_batch);
	if (s != OK)
		WARN("vulkan: failed to create matmul_f32_batch pipeline -- "
			 "batched prefill will fall back to per-token dispatch");
	else
		p->p_matmul_f32_batch.name = "matmul_f32_batch";

	s = vk_create_pipeline_spec(
		p, shader_matmul_f32_residual_batch_spv, shader_matmul_f32_residual_batch_spv_len, 4, 12,
		spec_matmul_dense_narrow, sizeof(spec_matmul_dense_narrow), &p->p_matmul_f32_res_batch);
	if (s != OK)
		WARN("vulkan: failed to create matmul_f32_res_batch pipeline");
	else
		p->p_matmul_f32_res_batch.name = "matmul_f32_res_batch";

#define VK_CREATE_BATCH_MATMUL_FIELD(field, spv_prefix, n_bind, push_sz, spec_arr)                 \
	do {                                                                                           \
		s = vk_create_pipeline_spec(p, shader_##spv_prefix##_batch_spv,                            \
									shader_##spv_prefix##_batch_spv_len, n_bind, push_sz,          \
									spec_arr, sizeof(spec_arr), &p->p_matmul_##field##_batch);     \
		if (s != OK)                                                                               \
			WARN("vulkan: failed to create " #spv_prefix "_batch pipeline");                       \
		else                                                                                       \
			p->p_matmul_##field##_batch.name = #spv_prefix "_batch";                               \
	} while (0)

#define VK_CREATE_BATCH_MATMUL_RES_FIELD(field, spv_prefix, n_bind, push_sz, spec_arr)             \
	do {                                                                                           \
		s = vk_create_pipeline_spec(p, shader_##spv_prefix##_residual_batch_spv,                   \
									shader_##spv_prefix##_residual_batch_spv_len, n_bind, push_sz, \
									spec_arr, sizeof(spec_arr), &p->p_matmul_##field##_res_batch); \
		if (s != OK)                                                                               \
			WARN("vulkan: failed to create " #spv_prefix "_res_batch pipeline");                   \
		else                                                                                       \
			p->p_matmul_##field##_res_batch.name = #spv_prefix "_res_batch";                       \
	} while (0)

#define VK_CREATE_BATCH_MATMUL_NMAT(qname, spv_prefix, variant, n_bind, push_sz, spec_arr)         \
	do {                                                                                           \
		s = vk_create_pipeline_spec(p, shader_##spv_prefix##_##variant##_batch_spv,                \
									shader_##spv_prefix##_##variant##_batch_spv_len, n_bind,       \
									push_sz, spec_arr, sizeof(spec_arr),                           \
									&p->p_matmul_##qname##_##variant##_batch);                     \
		if (s != OK)                                                                               \
			WARN("vulkan: failed to create " #spv_prefix "_" #variant "_batch pipeline");          \
		else                                                                                       \
			p->p_matmul_##qname##_##variant##_batch.name = #spv_prefix "_" #variant "_batch";      \
	} while (0)

#define VK_CREATE_BATCH_MATMUL_DUAL(qname, spv_prefix, n_bind, push_sz, spec_arr)                  \
	VK_CREATE_BATCH_MATMUL_NMAT(qname, spv_prefix, dual, n_bind, push_sz, spec_arr)

	VK_CREATE_BATCH_MATMUL_FIELD(q4_0, matmul_q4_0, 3, 16, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q4_0, matmul_q4_0, 4, 16, spec_matmul);
	VK_CREATE_BATCH_MATMUL_DUAL(q4_0, matmul_q4_0, 5, 16, spec_matmul);
	VK_CREATE_BATCH_MATMUL_NMAT(q4_0, matmul_q4_0, triple, 7, 20, spec_matmul);

	VK_CREATE_BATCH_MATMUL_FIELD(q4_1, matmul_q4_1, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q4_1, matmul_q4_1, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL_FIELD(q5_0, matmul_q5_0, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q5_0, matmul_q5_0, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL_FIELD(q5_1, matmul_q5_1, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q5_1, matmul_q5_1, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL_FIELD(q8_0, matmul_q8_0, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q8_0, matmul_q8_0, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL_FIELD(q4_k, matmul_q4_k, 3, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q4_k, matmul_q4_k, 4, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_DUAL(q4_k, matmul_q4_k, 5, 16, spec_matmul_kquant);

	VK_CREATE_BATCH_MATMUL_FIELD(q5_k, matmul_q5_k, 3, 12, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q5_k, matmul_q5_k, 4, 12, spec_matmul_kquant);

	VK_CREATE_BATCH_MATMUL_FIELD(q6_k, matmul_q6_k, 3, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(q6_k, matmul_q6_k, 4, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_DUAL(q6_k, matmul_q6_k, 5, 16, spec_matmul_kquant);

	VK_CREATE_BATCH_MATMUL_FIELD(iq3_s, matmul_iq3_s, 4, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(iq3_s, matmul_iq3_s, 5, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_FIELD(iq2_s, matmul_iq2_s, 4, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(iq2_s, matmul_iq2_s, 5, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_FIELD(iq2_xxs, matmul_iq2_xxs, 4, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(iq2_xxs, matmul_iq2_xxs, 5, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_FIELD(iq3_xxs, matmul_iq3_xxs, 4, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(iq3_xxs, matmul_iq3_xxs, 5, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_FIELD(iq4_xs, matmul_iq4_xs, 3, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(iq4_xs, matmul_iq4_xs, 4, 16, spec_matmul_iq3s);

	VK_CREATE_BATCH_MATMUL_FIELD(f16, matmul_f16, 3, 12, spec_matmul_dense_narrow);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(f16, matmul_f16, 4, 12, spec_matmul_dense_narrow);
	VK_CREATE_BATCH_MATMUL_FIELD(bf16, matmul_bf16, 3, 12, spec_matmul_dense_narrow);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(bf16, matmul_bf16, 4, 12, spec_matmul_dense_narrow);

	VK_CREATE_BATCH_MATMUL_FIELD(f32_wide, matmul_f32, 3, 12, spec_matmul_wide);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(f32_wide, matmul_f32, 4, 12, spec_matmul_wide);
	VK_CREATE_BATCH_MATMUL_FIELD(f16_wide, matmul_f16, 3, 12, spec_matmul_wide);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(f16_wide, matmul_f16, 4, 12, spec_matmul_wide);
	VK_CREATE_BATCH_MATMUL_FIELD(bf16_wide, matmul_bf16, 3, 12, spec_matmul_wide);
	VK_CREATE_BATCH_MATMUL_RES_FIELD(bf16_wide, matmul_bf16, 4, 12, spec_matmul_wide);

#undef VK_CREATE_BATCH_MATMUL_DUAL
#undef VK_CREATE_BATCH_MATMUL_FIELD
#undef VK_CREATE_BATCH_MATMUL_RES_FIELD

	s = vk_create_pipeline_spec(p, shader_matmul_iq4_nl_batch_spv,
								shader_matmul_iq4_nl_batch_spv_len, 7, 32, spec_matmul,
								sizeof(spec_matmul), &p->p_matmul_iq4_nl_batch);
	if (s != OK)
		WARN("vulkan: failed to create matmul_iq4_nl_batch pipeline");
	else
		p->p_matmul_iq4_nl_batch.name = "matmul_iq4_nl_batch";

	vk_qcore_init(p);

	s = vk_create_pipeline(p, shader_rmsnorm_batch_spv, shader_rmsnorm_batch_spv_len, 3, 12,
						   &p->p_rmsnorm_batch);
	if (s == OK)
		p->p_rmsnorm_batch.name = "rmsnorm_batch";
	s = vk_create_pipeline(p, shader_rmsnorm_noweight_batch_spv,
						   shader_rmsnorm_noweight_batch_spv_len, 2, 12,
						   &p->p_rmsnorm_noweight_batch);
	if (s == OK)
		p->p_rmsnorm_noweight_batch.name = "rmsnorm_noweight_batch";
	s = vk_create_pipeline(p, shader_rmsnorm_per_head_batch_spv,
						   shader_rmsnorm_per_head_batch_spv_len, 3, 16,
						   &p->p_rmsnorm_per_head_batch);
	if (s == OK)
		p->p_rmsnorm_per_head_batch.name = "rmsnorm_per_head_batch";
	s = vk_create_pipeline(p, shader_rmsnorm_noweight_per_head_batch_spv,
						   shader_rmsnorm_noweight_per_head_batch_spv_len, 2, 16,
						   &p->p_rmsnorm_noweight_per_head_batch);
	if (s == OK)
		p->p_rmsnorm_noweight_per_head_batch.name = "rmsnorm_noweight_per_head_batch";

	if (p->caps.supports_subgroup_basic && p->caps.supports_subgroup_arithmetic &&
		p->caps.subgroup_size >= 16) {
		s = vk_create_pipeline(p, shader_rmsnorm_sg_batch_spv, shader_rmsnorm_sg_batch_spv_len, 3,
							   12, &p->p_rmsnorm_sg_batch);
		if (s == OK)
			p->p_rmsnorm_sg_batch.name = "rmsnorm_sg_batch";
		s = vk_create_pipeline(p, shader_rmsnorm_noweight_sg_batch_spv,
							   shader_rmsnorm_noweight_sg_batch_spv_len, 2, 12,
							   &p->p_rmsnorm_noweight_sg_batch);
		if (s == OK)
			p->p_rmsnorm_noweight_sg_batch.name = "rmsnorm_noweight_sg_batch";
		s = vk_create_pipeline(p, shader_rmsnorm_per_head_sg_batch_spv,
							   shader_rmsnorm_per_head_sg_batch_spv_len, 3, 16,
							   &p->p_rmsnorm_per_head_sg_batch);
		if (s == OK)
			p->p_rmsnorm_per_head_sg_batch.name = "rmsnorm_per_head_sg_batch";
		s = vk_create_pipeline(p, shader_rmsnorm_noweight_per_head_sg_batch_spv,
							   shader_rmsnorm_noweight_per_head_sg_batch_spv_len, 2, 16,
							   &p->p_rmsnorm_noweight_per_head_sg_batch);
		if (s == OK)
			p->p_rmsnorm_noweight_per_head_sg_batch.name = "rmsnorm_noweight_per_head_sg_batch";
		s = vk_create_pipeline(p, shader_rmsnorm_add_batch_spv, shader_rmsnorm_add_batch_spv_len, 4,
							   16, &p->p_rmsnorm_add_batch);
		if (s == OK)
			p->p_rmsnorm_add_batch.name = "rmsnorm_add_batch";
		s = vk_create_pipeline(p, shader_rmsnorm_sg_quant_batch_spv,
							   shader_rmsnorm_sg_quant_batch_spv_len, 4, 12,
							   &p->p_rmsnorm_sg_quant);
		if (s == OK)
			p->p_rmsnorm_sg_quant.name = "rmsnorm_sg_quant";
		s = vk_create_pipeline(p, shader_rmsnorm_add_norm_batch_spv,
							   shader_rmsnorm_add_norm_batch_spv_len, 7, 16,
							   &p->p_rmsnorm_add_norm);
		if (s == OK)
			p->p_rmsnorm_add_norm.name = "rmsnorm_add_norm";
	}

	memset(p->attn_cache, 0, sizeof(p->attn_cache));

	s = vk_create_pipeline(p, shader_rope_batch_spv, shader_rope_batch_spv_len, 3, 20,
						   &p->p_rope_batch);
	if (s == OK)
		p->p_rope_batch.name = "rope_batch";
	s = vk_create_pipeline(p, shader_rope_ext_batch_spv, shader_rope_ext_batch_spv_len, 4, 28,
						   &p->p_rope_ext_batch);
	if (s == OK)
		p->p_rope_ext_batch.name = "rope_ext_batch";
	s = vk_create_pipeline(p, shader_qk_norm_rope_spv, shader_qk_norm_rope_spv_len, 6, 24,
						   &p->p_qk_norm_rope);
	if (s == OK)
		p->p_qk_norm_rope.name = "qk_norm_rope";
	s = vk_create_pipeline(p, shader_rope_qk_batch_spv, shader_rope_qk_batch_spv_len, 4, 28,
						   &p->p_rope_qk_batch);
	if (s == OK)
		p->p_rope_qk_batch.name = "rope_qk_batch";

	s = vk_create_pipeline(p, shader_dequant_spv, shader_dequant_spv_len, 3, 16, &p->p_dequant);
	if (s == OK)
		p->p_dequant.name = "dequant";

	s = vk_create_pipeline(p, shader_ffn_activate_batch_spv, shader_ffn_activate_batch_spv_len, 3,
						   20, &p->p_ffn_activate_batch);
	if (s == OK)
		p->p_ffn_activate_batch.name = "ffn_activate_batch";

	s = vk_create_pipeline(p, shader_ffn_activate_fused_batch_spv,
						   shader_ffn_activate_fused_batch_spv_len, 2, 12,
						   &p->p_ffn_activate_fused_batch);
	if (s == OK)
		p->p_ffn_activate_fused_batch.name = "ffn_activate_fused_batch";

	s = vk_create_pipeline(p, shader_elementwise_batch_spv, shader_elementwise_batch_spv_len, 3, 20,
						   &p->p_elementwise_batch);
	if (s == OK)
		p->p_elementwise_batch.name = "elementwise_batch";

	static const struct {
		const uint32_t *spv;
		size_t			len;
		int				n_bufs;
		uint32_t		push_size;
		const char	   *name;
	} nn_pipes[VK_NN_COUNT] = {
		[VK_NN_LAYERNORM_MODULATE] = {shader_layernorm_modulate_spv,
									  shader_layernorm_modulate_spv_len, 3, 12,
									  "layernorm_modulate"},
		[VK_NN_ROPE_TABLE]		   = {shader_rope_table_spv, shader_rope_table_spv_len, 2, 12,
									  "rope_table"},
		[VK_NN_ATTENTION_DENSE]	   = {shader_attention_dense_spv, shader_attention_dense_spv_len, 6,
									  20, "attention_dense"},
		[VK_NN_IM2COL]			   = {shader_im2col_spv, shader_im2col_spv_len, 2, 28, "im2col"},
		[VK_NN_UPSAMPLE_SHORTCUT] = {shader_upsample_shortcut_spv, shader_upsample_shortcut_spv_len,
									 2, 28, "upsample_shortcut"},
	};
	for (int i = 0; i < VK_NN_COUNT; i++) {
		s = vk_create_pipeline(p, nn_pipes[i].spv, nn_pipes[i].len, nn_pipes[i].n_bufs,
							   nn_pipes[i].push_size, &p->p_nn[i]);
		if (s == OK)
			p->p_nn[i].name = nn_pipes[i].name;
		else
			WARN("vulkan: failed to create %s pipeline", nn_pipes[i].name);
	}

	return OK;
}

static void vk_free(backend *self) {
	vk_priv *p = self->priv;
	if (!p)
		return;
	if (p->dev) {
		vkDeviceWaitIdle(p->dev);
		INFO("vk: totals: %llu dispatches, %llu barriers, %llu submits, %llu waits, %llu uploads, "
			 "%llu downloads, %.1f ms waiting",
			 (unsigned long long)p->stats.dispatches, (unsigned long long)p->stats.barriers,
			 (unsigned long long)p->stats.submits, (unsigned long long)p->stats.waits,
			 (unsigned long long)p->stats.uploads, (unsigned long long)p->stats.downloads,
			 (double)p->stats.wait_us / 1000.0);
		vk_prof_report(p);

		for (int ci = 0; ci < VK_ATTN_KIND_MAX; ci++)
			for (int fi = 0; fi < VK_ATTN_CACHE_CAP; fi++)
				vk_destroy_pipeline(p, &p->attn_cache[ci].set[fi]);
		vk_destroy_pipeline(p, &p->p_kv_put);
		vk_destroy_pipeline(p, &p->p_kv_shift_gather);
		vk_destroy_pipeline(p, &p->p_embd_lookup);
		vk_destroy_pipeline(p, &p->p_argmax);
		vk_destroy_pipeline(p, &p->p_argmax_reduce);
		vk_destroy_pipeline(p, &p->p_moe_gather);
		vk_destroy_pipeline(p, &p->p_moe_combine);

		vk_destroy_pipeline(p, &p->p_matmul_q4_0_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_0_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_0_dual_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_0_triple_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_1_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_1_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q5_0_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q5_0_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q5_1_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q5_1_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q8_0_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q8_0_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_k_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_k_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_k_dual_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q5_k_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q5_k_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q6_k_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q6_k_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q6_k_dual_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq3_s_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq3_s_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq2_s_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq2_s_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq2_xxs_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq2_xxs_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq3_xxs_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq3_xxs_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq4_xs_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq4_xs_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f32_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f32_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f16_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f16_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_bf16_batch);
		vk_destroy_pipeline(p, &p->p_matmul_bf16_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f32_wide_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f32_wide_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f16_wide_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f16_wide_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_bf16_wide_batch);
		vk_destroy_pipeline(p, &p->p_matmul_bf16_wide_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_iq4_nl_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_sg_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_noweight_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_noweight_sg_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_per_head_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_per_head_sg_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_noweight_per_head_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_noweight_per_head_sg_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_add_batch);
		vk_destroy_pipeline(p, &p->p_rmsnorm_sg_quant);
		vk_destroy_pipeline(p, &p->p_rmsnorm_add_norm);
		vk_destroy_pipeline(p, &p->p_rope_batch);
		vk_destroy_pipeline(p, &p->p_rope_ext_batch);
		vk_destroy_pipeline(p, &p->p_qk_norm_rope);
		vk_destroy_pipeline(p, &p->p_rope_qk_batch);
		vk_destroy_pipeline(p, &p->p_dequant);
		vk_destroy_pipeline(p, &p->p_ffn_activate_batch);
		vk_destroy_pipeline(p, &p->p_ffn_activate_fused_batch);
		vk_destroy_pipeline(p, &p->p_elementwise_batch);
		for (int i = 0; i < VK_NN_COUNT; i++)
			vk_destroy_pipeline(p, &p->p_nn[i]);
		for (int i = 0; i < VK_QCORE_PIPES; i++)
			vk_destroy_pipeline(p, vk_qcore_pipe(p, i));
		for (int i = 0; i < 2; i++) {
			if (p->xq_buf[i].handle) {
				vk_free_buffer(p, (vk_buf *)p->xq_buf[i].handle);
				free(p->xq_buf[i].handle);
			}
		}
		if (p->attn_part_buf.handle) {
			vk_free_buffer(p, (vk_buf *)p->attn_part_buf.handle);
			free(p->attn_part_buf.handle);
		}

		if (p->rope_cos_buf.handle) {
			vk_free_buffer(p, (vk_buf *)p->rope_cos_buf.handle);
			free(p->rope_cos_buf.handle);
		}
		if (p->rope_sin_buf.handle) {
			vk_free_buffer(p, (vk_buf *)p->rope_sin_buf.handle);
			free(p->rope_sin_buf.handle);
		}
		if (p->rope_cos_buf_alt.handle) {
			vk_free_buffer(p, (vk_buf *)p->rope_cos_buf_alt.handle);
			free(p->rope_cos_buf_alt.handle);
		}
		if (p->rope_sin_buf_alt.handle) {
			vk_free_buffer(p, (vk_buf *)p->rope_sin_buf_alt.handle);
			free(p->rope_sin_buf_alt.handle);
		}
		if (p->rope_ff_buf.handle) {
			vk_free_buffer(p, (vk_buf *)p->rope_ff_buf.handle);
			free(p->rope_ff_buf.handle);
		}
		if (p->attn_scores_buf.handle) {
			vk_free_buffer(p, (vk_buf *)p->attn_scores_buf.handle);
			free(p->attn_scores_buf.handle);
		}
		if (p->ffn_act_buf.handle) {
			vk_free_buffer(p, (vk_buf *)p->ffn_act_buf.handle);
			free(p->ffn_act_buf.handle);
		}
		if (p->staging_buf.buf) {
			p->wbatch_count = 0;
			p->wbatch_off	= 0;
			vk_free_buffer(p, &p->staging_buf);
			p->staging_cap = 0;
		}
		if (p->tok_reg_buf.buf) {
			vk_free_buffer(p, &p->tok_reg_buf);
		}
		if (p->tok_rb_buf.buf) {
			vk_free_buffer(p, &p->tok_rb_buf);
		}
		if (p->argmax_partial_buf.buf) {
			vk_free_buffer(p, &p->argmax_partial_buf);
		}
		if (p->moe_arena.buf) {
			vk_free_buffer(p, &p->moe_arena);
		}
		if (p->moe_batch_arena.buf) {
			vk_free_buffer(p, &p->moe_batch_arena);
		}
		for (int i = 0; i < VK_MOE_META_SLOTS; i++) {
			if (p->moe_meta_slots[i].buf) {
				vk_free_buffer(p, &p->moe_meta_slots[i]);
			}
		}
		if (p->dummy_buf.buf) {
			vk_free_buffer(p, &p->dummy_buf);
		}
		if (p->iq3s_grid_buf.buf) {
			vk_free_buffer(p, &p->iq3s_grid_buf);
		}
		if (p->iqx_grid_buf.buf) {
			vk_free_buffer(p, &p->iqx_grid_buf);
		}

		for (int i = 0; i < p->scratch_pool_count; i++) {
			vk_free_buffer(p, p->scratch_pool[i]);
			free(p->scratch_pool[i]);
		}
		p->scratch_pool_count = 0;

		free(p->kv_handles);
		vk_suballoc_block *blk = p->sub_blocks;
		while (blk) {
			vk_suballoc_block *next = blk->next;
			if (blk->mem)
				vkFreeMemory(p->dev, blk->mem, NULL);
			free(blk->ranges);
			free(blk);
			blk = next;
		}
		p->sub_blocks	   = NULL;
		p->sub_block_count = 0;
		if (p->desc_pool)
			vkDestroyDescriptorPool(p->dev, p->desc_pool, NULL);
		if (p->fence)
			vkDestroyFence(p->dev, p->fence, NULL);
		if (p->timeline)
			vkDestroySemaphore(p->dev, p->timeline, NULL);
		if (p->cmd_pools[0])
			for (int i = 0; i < VK_RING_DEPTH; i++)
				if (p->cmd_pools[i])
					vkDestroyCommandPool(p->dev, p->cmd_pools[i], NULL);
		vkDestroyDevice(p->dev, NULL);
	}
	if (p->debug_messenger) {
		PFN_vkDestroyDebugUtilsMessengerEXT destroy_fn =
			(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
				p->instance, "vkDestroyDebugUtilsMessengerEXT");
		if (destroy_fn)
			destroy_fn(p->instance, p->debug_messenger, NULL);
	}
	if (p->instance)
		vkDestroyInstance(p->instance, NULL);
	free(p);
	self->priv = NULL;
}

static vk_buf *as_vkbuf(const buffer *b) {
	return (vk_buf *)b->handle;
}

static vk_buf *vk_dummy_buf(vk_priv *p) {
	if (!p->dummy_buf.buf) {
		VkBufferUsageFlags	  usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VkMemoryPropertyFlags flags =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		vk_alloc_buffer(p, 16, usage, flags, &p->dummy_buf);
	}
	return &p->dummy_buf;
}

static void vk_kv_registry_add(vk_priv *p, void *h) {
	arr_reserve((void **)&(p->kv_handles), (p->kv_handle_count), &(p->kv_handle_cap),
				sizeof(*(p->kv_handles)));
	p->kv_handles[p->kv_handle_count++] = h;
}

static int vk_kv_registry_remove(vk_priv *p, void *h) {
	for (int i = 0; i < p->kv_handle_count; i++) {
		if (p->kv_handles[i] == h) {
			p->kv_handles[i] = p->kv_handles[--p->kv_handle_count];
			return 1;
		}
	}
	return 0;
}

static vk_buf *vk_scratch_pool_take(vk_priv *p, size_t size, int *slot_out) {
	int best = -1;
	for (int i = 0; i < p->scratch_pool_count; i++) {
		if (p->scratch_pool_sizes[i] < size)
			continue;
		if (best < 0 || p->scratch_pool_sizes[i] < p->scratch_pool_sizes[best])
			best = i;
	}
	if (best < 0)
		return NULL;
	vk_buf *b = p->scratch_pool[best];
	if (slot_out)
		*slot_out = best;
	p->scratch_pool[best]		= p->scratch_pool[p->scratch_pool_count - 1];
	p->scratch_pool_sizes[best] = p->scratch_pool_sizes[p->scratch_pool_count - 1];
	p->scratch_pool_count--;
	return b;
}

static status_code vk_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	vk_priv *p = self->priv;

	vk_buf *b = vk_scratch_pool_take(p, size, NULL);
	if (b) {
		out->handle	  = b;
		out->size	  = size;
		out->host_ptr = b->mapped;
		out->owner	  = self;
		return OK;
	}

	b = xcalloc(1, sizeof(vk_buf));

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
							   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	VkMemoryPropertyFlags preferred;
	if (p->caps.unified_memory) {
		preferred = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	} else {
		preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	}

	status_code s = vk_alloc_buffer(p, size, usage, preferred, b);
	if (s != OK) {
		VkMemoryPropertyFlags fallback =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		s = vk_alloc_buffer(p, size, usage, fallback, b);
	}
	if (s != OK) {
		free(b);
		return s;
	}
	out->handle	  = b;
	out->size	  = size;
	out->host_ptr = b->mapped;
	out->owner	  = self;

	return OK;
}

static status_code vk_stage_upload(vk_priv *p, vk_buf *dst, const tensor_desc *desc, size_t size) {
	for (size_t off = 0; off < size; off += VK_WEIGHT_STAGE_CHUNK_BYTES) {
		size_t len		  = MIN(size - off, VK_WEIGHT_STAGE_CHUNK_BYTES);
		bool   batch_full = p->wbatch_off + len > p->staging_cap;
		if (!p->staging_buf.buf || (p->wbatch_count && batch_full) ||
			!vk_timeline_done(p, p->staging_pending_value)) {
			status_code fs = vk_wbatch_flush(p);
			if (fs != OK)
				return fs;
		}
		if (!p->wbatch_count) {
			status_code s = vk_ensure_staging_cap(p, MAX(len, VK_WEIGHT_STAGE_BATCH_BYTES));
			if (s != OK)
				return s;
		}

		status_code rs = tensor_desc_read(desc, off, p->staging_buf.mapped + p->wbatch_off, len);
		if (rs != OK)
			return rs;
		VkBufferCopy copy = {.srcOffset = p->wbatch_off, .dstOffset = off, .size = len};
		vkCmdCopyBuffer(p->cmd, p->staging_buf.buf, dst->buf, 1, &copy);
		VkMemoryBarrier wbar = {
			.sType		   = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
		vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &wbar, 0, NULL, 0, NULL);
		p->wbatch_off += len;
		p->wbatch_count++;
		p->ring[p->ring_cur].has_work = 1;
	}
	return OK;
}

static void vk_drain_gpu(vk_priv *p);

static status_code vk_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	vk_priv *p	  = self->priv;
	size_t	 size = (desc->n_dims == 1) ? ggml_row_size(desc->type, desc->dims[0])
										: ggml_row_size(desc->type, desc->dims[0]) * desc->dims[1];
	size_t	 alloc_size = size + VK_WEIGHT_TAIL_PAD;

	vk_buf *b = vk_scratch_pool_take(p, alloc_size, NULL);
	if (b) {
		vk_drain_gpu(p);
		if (b->mapped) {
			status_code s = tensor_desc_read(desc, 0, b->mapped, size);
			if (s != OK)
				return s;
		} else {
			status_code s = vk_stage_upload(p, b, desc, size);
			if (s != OK)
				return s;
		}

		out->handle	  = b;
		out->size	  = size;
		out->host_ptr = desc->host_data;
		out->owner	  = self;
		return OK;
	}

	vk_buf *bnew = xcalloc(1, sizeof(vk_buf));

	VkBufferUsageFlags weight_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
									  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
									  VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	if (p->caps.unified_memory) {
		VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
									  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
									  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		status_code			  s		= vk_alloc_buffer(p, alloc_size, weight_usage, flags, bnew);
		if (s != OK) {
			s = vk_alloc_buffer(
				p, size, weight_usage,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bnew);
		}
		if (s != OK) {
			ERROR("vk: weight alloc failed for %zu bytes on UMA device -- "
				  "per-BO size limit may be exceeded. Model may be too "
				  "large for this device.",
				  size);
			free(bnew);
			return s;
		}

		status_code rs = tensor_desc_read(desc, 0, bnew->mapped, size);
		if (rs != OK) {
			vk_free_buffer(p, bnew);
			free(bnew);
			return rs;
		}
	} else {
		status_code s =
			vk_alloc_buffer(p, alloc_size, weight_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bnew);
		if (s != OK) {
			s = vk_alloc_buffer(
				p, size, weight_usage,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bnew);
		}
		if (s != OK) {
			free(bnew);
			return s;
		}

		s = vk_stage_upload(p, bnew, desc, size);
		if (s != OK) {
			vk_free_buffer(p, bnew);
			free(bnew);
			return s;
		}
	}

	out->handle	  = bnew;
	out->size	  = size;
	out->host_ptr = desc->host_data;
	out->owner	  = self;
	return OK;
}

static void vk_buffer_free(backend *self, buffer *buf) {
	vk_priv *p = self->priv;
	if (!buf || !buf->handle)
		return;
	if (vk_kv_registry_remove(p, buf->handle)) {
		status_code fs = vk_flush(p);
		if (fs != OK && !p->device_lost) {
			ERROR("vk: vk_buffer_free: flush failed (%s)", status_str(fs));
		}
		vk_kv_handle *kh = (vk_kv_handle *)buf->handle;
		vk_kv_store_free(p, &kh->store);
		free(kh);
		buf->handle = NULL;
		buf->size	= 0;
		return;
	}
	vk_buf *b = as_vkbuf(buf);

	if (p->scratch_pool_count < VK_SCRATCH_POOL_CAP) {
		p->scratch_pool[p->scratch_pool_count]		 = b;
		p->scratch_pool_sizes[p->scratch_pool_count] = b->size;
		p->scratch_pool_count++;
		buf->handle = NULL;
		buf->size	= 0;
		return;
	}

	if (p->last_signal_value > p->timeline_completed || p->ring[p->ring_cur].has_work) {
		status_code fs = vk_flush(p);
		if (fs != OK && !p->device_lost) {
			ERROR("vk: vk_buffer_free: flush failed (%s)", status_str(fs));
		}
	}
	vk_free_buffer(p, b);
	free(b);
	buf->handle = NULL;
	buf->size	= 0;
}

static void vk_kv_free(backend *self, buffer *k, buffer *v) {
	vk_buffer_free(self, k);
	vk_buffer_free(self, v);
}

static status_code vk_ensure_staging_cap(vk_priv *p, size_t need) {
	if (need <= p->staging_cap && p->staging_buf.buf)
		return OK;

	if (p->staging_buf.buf) {
		if (!vk_timeline_done(p, p->staging_pending_value))
			vk_timeline_wait(p, p->staging_pending_value);
		vk_free_buffer(p, &p->staging_buf);
		p->staging_cap = 0;
	}

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags flags =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	status_code s = vk_alloc_buffer(p, need, usage, flags, &p->staging_buf);
	if (s != OK)
		return s;
	p->staging_cap = need;
	return OK;
}

static status_code vk_buffer_alloc_from_host(backend *self, const void *host_data, size_t size,
											 buffer *out) {
	vk_priv *p = self->priv;
	if (p->device_lost)
		return ERR_INTERNAL;

	size_t alloc_size = size + VK_WEIGHT_TAIL_PAD;

	vk_buf			  *b	 = xcalloc(1, sizeof(vk_buf));
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
							   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	status_code s = vk_alloc_buffer(p, alloc_size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b);
	if (s != OK) {
		s = vk_alloc_buffer(
			p, alloc_size, usage,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b);
	}
	if (s != OK) {
		free(b);
		return s;
	}

	if (b->mapped) {
		memcpy(b->mapped, host_data, size);
	} else {
		if (p->wbatch_count || !p->staging_buf.buf || p->wbatch_off + size > p->staging_cap ||
			!vk_timeline_done(p, p->staging_pending_value)) {
			status_code fs = vk_wbatch_flush(p);
			if (fs != OK) {
				vk_free_buffer(p, b);
				free(b);
				return fs;
			}
		}
		s = vk_ensure_staging_cap(p, size);
		if (s != OK) {
			vk_free_buffer(p, b);
			free(b);
			return s;
		}
		memcpy(p->staging_buf.mapped + p->wbatch_off, host_data, size);
		VkBufferCopy copy = {.srcOffset = p->wbatch_off, .dstOffset = 0, .size = size};
		vkCmdCopyBuffer(p->cmd, p->staging_buf.buf, b->buf, 1, &copy);
		VkMemoryBarrier wbar = {
			.sType		   = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
		vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &wbar, 0, NULL, 0, NULL);
		p->wbatch_off += size;
		p->wbatch_count++;
		p->ring[p->ring_cur].has_work = 1;
	}

	out->handle	  = b;
	out->size	  = size;
	out->host_ptr = NULL;
	out->owner	  = self;
	return OK;
}

static status_code vk_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								   buffer *y, int n, int k, int m);
static status_code vk_scale_inplace(backend *self, buffer *x, float scale, int n);
static status_code vk_ffn_activate_batch_scales(backend *self, const buffer *gate, const buffer *up,
												buffer *out, int n, int activation, int m, float gs,
												float us);
static status_code vk_add_inplace(backend *self, buffer *x, const buffer *y, int n);
static status_code vk_moe_experts_batch(backend *self, const buffer *xb, buffer *out, int n_rows,
										int dim, int inter, int use_gelu, int n_experts,
										const moe_resident_expert *experts, const int *counts,
										const int *rows_packed, const float *weights_packed);

static status_code vk_moe_expert_body(backend *self, const moe_resident_expert *e, const buffer *x,
									  buffer *gate_v, buffer *up_v, buffer *act_v, buffer *y_v,
									  int dim, int inter, int m) {
	int act = e->use_gelu ? ACTIVATION_GELU : ACTIVATION_SILU;

	if (e->gate_up_fused) {
		status_code s =
			vk_matmul_batch(self, e->gate_w, e->gate_type, x, gate_v, inter * 2, dim, m);
		if (s != OK)
			return s;
	} else {
		status_code s = vk_matmul_batch(self, e->gate_w, e->gate_type, x, gate_v, inter, dim, m);
		if (s != OK)
			return s;
		s = vk_matmul_batch(self, e->up_w, e->up_type, x, up_v, inter, dim, m);
		if (s != OK)
			return s;
	}
	status_code s = vk_ffn_activate_batch_scales(self, gate_v, up_v, act_v, inter, act, m,
												 e->gate_scale, e->up_scale);
	if (s != OK)
		return s;

	return vk_matmul_batch(self, e->down_w, e->down_type, act_v, y_v, dim, inter, m);
}

static status_code vk_moe_expert_ffn(backend *self, const buffer *x, buffer *out,
									 const moe_resident_expert *e, int dim, int inter) {
	vk_priv *p = self->priv;
	if (p->device_lost)
		return ERR_INTERNAL;
	if (!x || !out || !e || !e->gate_w || !e->down_w || (!e->up_w && !e->gate_up_fused))
		return ERR_INVALID_ARG;
	if ((e->gate_w->owner && e->gate_w->owner != self) ||
		(e->down_w->owner && e->down_w->owner != self) ||
		(e->up_w && e->up_w->owner && e->up_w->owner != self))
		return ERR_UNSUPPORTED;
	if (p->moe_arena.buf && (p->moe_arena_inter != inter || p->moe_arena_dim != dim)) {
		vk_free_buffer(p, &p->moe_arena);
		p->moe_arena.buf = VK_NULL_HANDLE;
	}
	if (!p->moe_arena.buf) {
		size_t		bytes = ((size_t)3 * inter + dim) * sizeof(float);
		status_code s	  = vk_alloc_buffer(p, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
											VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &p->moe_arena);
		if (s != OK)
			return s;
		p->moe_arena_inter = inter;
		p->moe_arena_dim   = dim;
	}

	buffer arena = {0};
	arena.handle = &p->moe_arena;
	arena.owner	 = self;

	buffer gate_v = buffer_slice(&arena, 0, (size_t)2 * inter * sizeof(float));
	buffer up_v =
		buffer_slice(&arena, (size_t)inter * sizeof(float), (size_t)inter * sizeof(float));
	buffer act_v =
		buffer_slice(&arena, (size_t)2 * inter * sizeof(float), (size_t)inter * sizeof(float));
	buffer y_v =
		buffer_slice(&arena, (size_t)3 * inter * sizeof(float), (size_t)dim * sizeof(float));

	status_code s = vk_moe_expert_body(self, e, x, &gate_v, &up_v, &act_v, &y_v, dim, inter, 1);
	if (s != OK)
		return s;

	float ws = e->weight * e->down_scale;
	if (ws != 1.0f)
		vk_scale_inplace(self, &y_v, ws, dim);
	return vk_add_inplace(self, out, &y_v, dim);
}

static status_code vk_moe_experts_batch(backend *self, const buffer *xb, buffer *out, int n_rows,
										int dim, int inter, int use_gelu, int n_experts,
										const moe_resident_expert *experts, const int *counts,
										const int *rows_packed, const float *weights_packed) {
	(void)n_rows;
	(void)use_gelu;
	vk_priv *p = self->priv;
	if (p->device_lost)
		return ERR_INTERNAL;
	if (!p->p_moe_gather.pipeline || !p->p_moe_combine.pipeline)
		return ERR_UNSUPPORTED;
	for (int i = 0; i < n_experts; i++) {
		const moe_resident_expert *ex = &experts[i];
		if (counts[i] <= 0 || !ex->gate_w)
			continue;
		if ((ex->gate_w->owner && ex->gate_w->owner != self) ||
			(ex->down_w && ex->down_w->owner && ex->down_w->owner != self) ||
			(ex->up_w && ex->up_w->owner && ex->up_w->owner != self))
			return ERR_UNSUPPORTED;
	}

	size_t align = p->caps.storage_buffer_offset_alignment > 0
					   ? (size_t)p->caps.storage_buffer_offset_alignment
					   : 256;
	if (align < 256)
		align = 256;

	size_t meta_need = 0;
	for (int i = 0; i < n_experts; i++)
		meta_need += ALIGN_UP((size_t)counts[i] * 4, align) * 2;
	if (meta_need == 0)
		return ERR_INVALID_ARG;
	if (!p->moe_meta_slots[0].buf || p->moe_meta_cap < meta_need) {
		for (int i = 0; i < VK_MOE_META_SLOTS; i++) {
			if (p->moe_meta_slots[i].buf)
				vk_free_buffer(p, &p->moe_meta_slots[i]);
			status_code s = vk_alloc_buffer(p, meta_need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
											VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
												VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
											&p->moe_meta_slots[i]);
			if (s != OK)
				return s;
			p->moe_meta_signal[i] = 0;
		}
		p->moe_meta_cap = meta_need;
		p->moe_meta_cur = 0;
	}

	int		slot	 = p->moe_meta_cur;
	vk_buf *moe_meta = &p->moe_meta_slots[slot];
	p->moe_meta_cur	 = (slot + 1) % VK_MOE_META_SLOTS;

	if (!moe_meta->mapped)
		return ERR_INTERNAL;

	if (p->timeline_supported) {
		if (!vk_timeline_done(p, p->moe_meta_signal[slot])) {
			status_code ws = vk_timeline_wait(p, p->moe_meta_signal[slot]);
			if (ws != OK)
				return ws;
		}
	} else if (p->last_signal_value > p->timeline_completed || p->ring[p->ring_cur].has_work) {
		status_code fs = vk_flush(p);
		if (fs != OK)
			return fs;
	}

	size_t batch_bytes = 0;
	for (int i = 0; i < n_experts; i++) {
		size_t cnt = (size_t)counts[i];
		batch_bytes += ALIGN_UP(cnt * dim * 4, align);
		batch_bytes += ALIGN_UP(cnt * 2 * inter * 4, align);
		batch_bytes += ALIGN_UP(cnt * inter * 4, align);
		batch_bytes += ALIGN_UP(cnt * dim * 4, align);
	}
	if (!p->moe_batch_arena.buf || p->moe_batch_arena_cap < batch_bytes) {
		if (p->moe_batch_arena.buf)
			vk_free_buffer(p, &p->moe_batch_arena);
		status_code s = vk_alloc_buffer(p, batch_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
										VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &p->moe_batch_arena);
		if (s != OK)
			return s;
		p->moe_batch_arena_cap = batch_bytes;
	}

	int *r_off	 = xmalloc((size_t)n_experts * sizeof(int));
	int *w_off	 = xmalloc((size_t)n_experts * sizeof(int));
	int *x_off	 = xmalloc((size_t)n_experts * sizeof(int));
	int *gu_off	 = xmalloc((size_t)n_experts * sizeof(int));
	int *act_off = xmalloc((size_t)n_experts * sizeof(int));
	int *y_off	 = xmalloc((size_t)n_experts * sizeof(int));

	size_t mo = 0, ao = 0, packed = 0;
	for (int i = 0; i < n_experts; i++) {
		size_t cb = (size_t)counts[i] * 4;
		r_off[i]  = (int)mo;
		memcpy((uint8_t *)moe_meta->mapped + mo, rows_packed + packed, cb);
		mo += ALIGN_UP(cb, align);
		w_off[i] = (int)mo;
		memcpy((uint8_t *)moe_meta->mapped + mo, weights_packed + packed, cb);
		mo += ALIGN_UP(cb, align);
		x_off[i] = (int)ao;
		ao += ALIGN_UP((size_t)counts[i] * dim * 4, align);
		gu_off[i] = (int)ao;
		ao += ALIGN_UP((size_t)counts[i] * 2 * inter * 4, align);
		act_off[i] = (int)ao;
		ao += ALIGN_UP((size_t)counts[i] * inter * 4, align);
		y_off[i] = (int)ao;
		ao += ALIGN_UP((size_t)counts[i] * dim * 4, align);
		packed += (size_t)counts[i];
	}

	buffer arena = {0};
	arena.handle = &p->moe_batch_arena;
	arena.owner	 = self;

	status_code st = OK;
	for (int i = 0; i < n_experts && st == OK; i++) {
		int cnt = counts[i];
		if (cnt <= 0 || !experts[i].gate_w)
			continue;
		const moe_resident_expert *e = &experts[i];

		buffer x_v	 = buffer_slice(&arena, x_off[i], (size_t)cnt * dim * 4);
		buffer gu_v	 = buffer_slice(&arena, gu_off[i], (size_t)cnt * 2 * inter * 4);
		buffer g_v	 = buffer_slice(&gu_v, 0, (size_t)cnt * inter * 4);
		buffer u_v	 = buffer_slice(&gu_v, (size_t)cnt * inter * 4, (size_t)cnt * inter * 4);
		buffer act_v = buffer_slice(&arena, act_off[i], (size_t)cnt * inter * 4);
		buffer y_v	 = buffer_slice(&arena, y_off[i], (size_t)cnt * dim * 4);

		struct {
			int32_t cnt, dim;
		} gather_push		  = {cnt, dim};
		vk_buf		*gbufs[3] = {as_vkbuf(xb), moe_meta, &p->moe_batch_arena};
		VkDeviceSize goffs[3] = {xb->offset, r_off[i], x_off[i]};
		uint32_t	 groups	  = (uint32_t)(((size_t)cnt * dim + 255) / 256);
		st = vk_dispatch_ex(p, &p->p_moe_gather, gbufs, goffs, NULL, 3, &gather_push,
							sizeof(gather_push), groups, 1, 1u << 2);
		if (st != OK)
			break;

		int act = e->use_gelu ? ACTIVATION_GELU : ACTIVATION_SILU;
		if (e->gate_up_fused) {
			st = vk_matmul_batch(self, e->gate_w, e->gate_type, &x_v, &gu_v, inter * 2, dim, cnt);
			if (st != OK)
				break;
			for (int c = 0; c < cnt && st == OK; c++) {
				buffer g_row = buffer_slice(&gu_v, (size_t)c * 2 * inter * 4, (size_t)inter * 4);
				buffer u_row =
					buffer_slice(&gu_v, ((size_t)c * 2 + 1) * inter * 4, (size_t)inter * 4);
				buffer a_row = buffer_slice(&act_v, (size_t)c * inter * 4, (size_t)inter * 4);
				st = vk_ffn_activate_batch_scales(self, &g_row, &u_row, &a_row, inter, act, 1,
												  e->gate_scale, e->up_scale);
			}
			if (st != OK)
				break;
			st = vk_matmul_batch(self, e->down_w, e->down_type, &act_v, &y_v, dim, inter, cnt);
			if (st != OK)
				break;
		} else {
			st = vk_moe_expert_body(self, e, &x_v, &g_v, &u_v, &act_v, &y_v, dim, inter, cnt);
			if (st != OK)
				break;
		}

		struct {
			int32_t cnt, dim;
			float	ds;
		} cpush				  = {cnt, dim, e->down_scale};
		vk_buf		*cbufs[4] = {&p->moe_batch_arena, moe_meta, moe_meta, as_vkbuf(out)};
		VkDeviceSize coffs[4] = {y_off[i], w_off[i], r_off[i], out->offset};
		st = vk_dispatch_ex(p, &p->p_moe_combine, cbufs, coffs, NULL, 4, &cpush, sizeof(cpush),
							groups, 1, 1u << 3);
		if (st != OK)
			break;
	}

	free(r_off);
	free(w_off);
	free(x_off);
	free(gu_off);
	free(act_off);
	free(y_off);

	if (st == OK && p->timeline_supported && p->ring[p->ring_cur].has_work) {
		status_code ss = vk_run_cmd(p);
		if (ss != OK)
			return ss;
		p->moe_meta_signal[slot] = p->last_signal_value;
	}

	return st;
}

static status_code vk_buffer_upload(backend *self, buffer *buf, const void *host_src, size_t size) {
	vk_priv *p = self->priv;
	p->stream_seq++;
	vk_buf *vb	= as_vkbuf(buf);
	size_t	off = buf->offset;

	if (vb->mapped) {
		vk_ring_slot *r = &p->ring[p->ring_cur];
		if (!p->device_lost && (r->has_work || p->last_signal_value > p->timeline_completed)) {
			status_code fs = vk_flush(p);
			if (fs != OK)
				return fs;
		}
		if (r->has_work) {
			r->has_work = 0;
			vk_dirty_clear(p);
		}
		memcpy((uint8_t *)vb->mapped + off, host_src, size);
		return OK;
	}

	if (p->device_lost)
		return ERR_INTERNAL;

	status_code fs = vk_flush(p);
	if (fs != OK)
		return fs;

	status_code s = vk_ensure_staging_cap(p, size);
	if (s != OK)
		return s;

	memcpy(p->staging_buf.mapped, host_src, size);
	p->stats.uploads++;

	VkBufferCopy copy = {.dstOffset = off, .size = size};
	vkCmdCopyBuffer(p->cmd, p->staging_buf.buf, vb->buf, 1, &copy);
	p->staging_read_recorded = 1;
	return vk_run_cmd(p);
}

static status_code vk_buf_download_raw(backend *self, vk_buf *vb, size_t off, void *host_dst,
									   size_t size) {
	vk_priv *p = self->priv;

	if (vb->mapped) {
		vk_ring_slot *r = &p->ring[p->ring_cur];
		if (!p->device_lost && (r->has_work || p->last_signal_value > p->timeline_completed)) {
			status_code fs = vk_flush(p);
			if (fs != OK)
				return fs;
		}
		if (r->has_work) {
			r->has_work = 0;
			vk_dirty_clear(p);
		}
		memcpy(host_dst, (uint8_t *)vb->mapped + off, size);
		return OK;
	}

	if (p->device_lost)
		return ERR_INTERNAL;

	if (p->dirty_count > 0) {
		VkMemoryBarrier barrier = {
			.sType		   = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		};
		vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
		vk_dirty_clear(p);
	}

	status_code s = vk_ensure_staging_cap(p, size);
	if (s != OK)
		return s;

	VkBufferCopy copy = {.srcOffset = off, .size = size};
	vkCmdCopyBuffer(p->cmd, vb->buf, p->staging_buf.buf, 1, &copy);

	status_code rs = vk_run_cmd_sync(p);
	if (rs != OK)
		return rs;

	memcpy(host_dst, p->staging_buf.mapped, size);
	p->stats.downloads++;
	return OK;
}

static status_code vk_buffer_download(backend *self, const buffer *buf, void *host_dst,
									  size_t size) {
	vk_buf *vb = as_vkbuf(buf);
	return vk_buf_download_raw(self, vb, buf->offset, host_dst, size);
}

static status_code vk_buffer_read_f32(backend *self, const buffer *buf, float *host_dst, int n) {
	return vk_buffer_download(self, buf, host_dst, (size_t)n * sizeof(float));
}

static status_code vk_buffer_write_f32(backend *self, buffer *buf, const float *host_src, int n) {
	return vk_buffer_upload(self, buf, host_src, (size_t)n * sizeof(float));
}

static status_code vk_zero_buffer(vk_priv *p, vk_buf *b) {
	if (!b->buf || !b->size)
		return OK;
	vk_ring_slot *r = &p->ring[p->ring_cur];
	if (!r->recording) {
		status_code bs = vk_ring_begin(p, p->ring_cur);
		if (bs != OK)
			return bs;
	}
	p->stream_seq++;
	vkCmdFillBuffer(p->cmd, b->buf, 0, VK_WHOLE_SIZE, 0);
	r->has_work = 1;
	return vk_run_cmd_sync(p);
}

static status_code vk_kv_store_alloc(vk_priv *p, size_t total, size_t per_layer_size,
									 int n_kv_layers, const size_t *layer_bytes,
									 size_t *layer_off_elems, vk_kv_store *out) {
	memset(out, 0, sizeof(*out));
	out->n_kv_layers	 = n_kv_layers > 0 ? n_kv_layers : 1;
	out->layer_off_elems = NULL;
	out->layer_bytes	 = NULL;
	if (layer_bytes && n_kv_layers > 0) {
		out->layer_bytes = xmalloc((size_t)n_kv_layers * sizeof(size_t));
		memcpy(out->layer_bytes, layer_bytes, (size_t)n_kv_layers * sizeof(size_t));
		out->per_layer_size = 0;
		for (int i = 0; i < n_kv_layers; i++)
			if (layer_bytes[i] > out->per_layer_size)
				out->per_layer_size = layer_bytes[i];
	} else {
		out->per_layer_size = per_layer_size;
	}
	if (layer_off_elems && n_kv_layers > 0) {
		out->layer_off_elems = xmalloc((size_t)(n_kv_layers + 1) * sizeof(size_t));
		memcpy(out->layer_off_elems, layer_off_elems, (size_t)(n_kv_layers + 1) * sizeof(size_t));
	}

	size_t			   per_layer_for_checks = layer_bytes ? out->per_layer_size : per_layer_size;
	VkBufferUsageFlags kv_usage				= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
											  VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
											  VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	VkMemoryPropertyFlags kv_preferred;
	VkMemoryPropertyFlags kv_fallback;
	if (p->caps.unified_memory) {
		kv_preferred = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
					   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		kv_fallback	 = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	} else {
		kv_preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		kv_fallback	 = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	}

	int needs_chunking =
		p->caps.max_storage_buffer_range > 0 && total > p->caps.max_storage_buffer_range;

	if (!needs_chunking && p->caps.unified_memory && n_kv_layers > 1) {
		const size_t UMA_CHUNK_THRESHOLD = 128u * 1024u * 1024u;
		if (total > UMA_CHUNK_THRESHOLD)
			needs_chunking = 1;
	}

	if (!needs_chunking) {
		status_code s = vk_alloc_buffer(p, total, kv_usage, kv_preferred, &out->single);
		if (s != OK)
			s = vk_alloc_buffer(p, total, kv_usage, kv_fallback, &out->single);
		if (s == OK) {
			out->chunked = 0;
			s			 = vk_zero_buffer(p, &out->single);
			if (s != OK)
				return s;
			return OK;
		}
		if (n_kv_layers > 1) {
			WARN("vulkan: single KV allocation of %zu bytes failed -- "
				 "retrying as %d per-layer chunks of %zu bytes each "
				 "(common on UMA drivers with per-BO size limits)",
				 total, n_kv_layers, per_layer_for_checks);
		} else {
			ERROR("kv_alloc: failed to allocate %zu bytes for KV cache "
				  "(single layer, cannot chunk further). The device may not "
				  "have enough memory for this context length. Try a "
				  "smaller --ctx-size or --device cpu.",
				  total);
			return ERR_OUT_OF_MEMORY;
		}
	}

	if (p->caps.max_storage_buffer_range > 0 &&
		per_layer_for_checks > p->caps.max_storage_buffer_range) {
		ERROR("kv_alloc: a single KV layer needs %zu bytes, which still "
			  "exceeds this device's maxStorageBufferRange (%llu bytes); "
			  "this model's context/head configuration cannot fit on this "
			  "device. Try a smaller --ctx-size or run with --device cpu.",
			  per_layer_for_checks, (unsigned long long)p->caps.max_storage_buffer_range);
		return ERR_OUT_OF_MEMORY;
	}

	int		n	   = out->n_kv_layers;
	vk_buf *chunks = xcalloc((size_t)n, sizeof(vk_buf));
	for (int i = 0; i < n; i++) {
		size_t		layer_sz = layer_bytes ? layer_bytes[i] : per_layer_size;
		status_code s		 = vk_alloc_buffer(p, layer_sz, kv_usage, kv_preferred, &chunks[i]);
		if (s != OK)
			s = vk_alloc_buffer(p, layer_sz, kv_usage, kv_fallback, &chunks[i]);
		if (s != OK) {
			for (int j = 0; j < i; j++)
				vk_free_buffer(p, &chunks[j]);
			free(chunks);
			ERROR("kv_alloc: failed to allocate KV layer %d/%d (%zu bytes). "
				  "Available device memory may be insufficient for this model "
				  "and context length. Try a smaller --ctx-size or --device cpu.",
				  i, n, layer_sz);
			return s;
		}
		s = vk_zero_buffer(p, &chunks[i]);
		if (s != OK) {
			for (int j = 0; j <= i; j++)
				vk_free_buffer(p, &chunks[j]);
			free(chunks);
			return s;
		}
	}
	out->chunked  = 1;
	out->n_chunks = n;
	out->chunks	  = chunks;
	return OK;
}

static void vk_kv_store_free(vk_priv *p, vk_kv_store *store) {
	if (store->chunked) {
		for (int i = 0; i < store->n_chunks; i++)
			vk_free_buffer(p, &store->chunks[i]);
		free(store->chunks);
	} else {
		vk_free_buffer(p, &store->single);
	}
	free(store->layer_bytes);
	store->layer_bytes = NULL;
	free(store->layer_off_elems);
	store->layer_off_elems = NULL;
}

static vk_buf *vk_kv_layer_buf(const vk_kv_handle *h, int layer, int *shader_layer_out) {
	if (h->store.chunked) {
		int idx = layer;
		if (idx < 0)
			idx = 0;
		if (idx >= h->store.n_chunks)
			idx = h->store.n_chunks - 1;
		if (shader_layer_out)
			*shader_layer_out = 0;
		return &h->store.chunks[idx];
	}
	if (shader_layer_out)
		*shader_layer_out = layer;
	return (vk_buf *)&h->store.single;
}

static size_t vk_kv_layer_off_elems(const vk_kv_handle *h, int layer) {
	if (layer < 0)
		layer = 0;
	vk_kv_store *s = (vk_kv_store *)&h->store;
	if (s->chunked || s->n_kv_layers < 1)
		return 0;
	if (s->layer_off_elems) {
		if (layer > s->n_kv_layers)
			layer = s->n_kv_layers;
		return s->layer_off_elems[layer];
	}
	size_t per_layer_elems = s->per_layer_size / sizeof(uint16_t);
	return (size_t)layer * per_layer_elems;
}

static size_t vk_device_local_heap_total(vk_priv *p) {
	size_t total = 0;
	for (uint32_t i = 0; i < p->mem_props.memoryHeapCount; i++) {
		if (p->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
			total += p->mem_props.memoryHeaps[i].size;
	}
	return total;
}

static size_t vk_mem_total(backend *self) {
	vk_priv *p = self->priv;
	return vk_device_local_heap_total(p);
}

static size_t vk_mem_available(backend *self) {
	vk_priv *p = self->priv;

#ifdef VK_EXT_memory_budget
	if (p->has_memory_budget_ext) {
		VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
		};
		VkPhysicalDeviceMemoryProperties2 props2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
			.pNext = &budget,
		};
		vkGetPhysicalDeviceMemoryProperties2(p->phys, &props2);

		size_t avail = 0;
		for (uint32_t i = 0; i < p->mem_props.memoryHeapCount; i++) {
			if (!(p->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
				continue;
			VkDeviceSize heap_budget = budget.heapBudget[i];
			VkDeviceSize heap_usage	 = budget.heapUsage[i];
			if (heap_budget == 0)
				heap_budget = p->mem_props.memoryHeaps[i].size;
			avail += heap_usage < heap_budget ? (size_t)(heap_budget - heap_usage) : 0;
		}
		return avail;
	}
#endif

	size_t heap_total = vk_device_local_heap_total(p);
	if (heap_total <= p->device_local_allocated)
		return 0;
	return heap_total - p->device_local_allocated;
}

static status_code vk_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	vk_priv *p		   = self->priv;
	p->n_ctx_stored	   = desc->n_ctx;
	p->kv_head_dim_max = desc->head_dim;

	int n_kv_layers	   = desc->n_kv_layers > 0 ? desc->n_kv_layers : 1;
	int has_layer_dims = desc->layer_head_dim && desc->layer_n_kv_heads && n_kv_layers > 0;

	size_t *layer_bytes		  = NULL;
	size_t *layer_off_elems	  = NULL;
	size_t	per_layer_uniform = 0;
	size_t	total;
	if (has_layer_dims) {
		layer_bytes		= xcalloc((size_t)n_kv_layers, sizeof(size_t));
		layer_off_elems = xcalloc((size_t)n_kv_layers + 1, sizeof(size_t));
		size_t acc		= 0;
		for (int i = 0; i < n_kv_layers; i++) {
			size_t elems =
				(size_t)desc->layer_n_kv_heads[i] * desc->n_ctx * (size_t)desc->layer_head_dim[i];
			layer_bytes[i]	   = elems * sizeof(uint16_t);
			layer_off_elems[i] = acc;
			if (elems > per_layer_uniform)
				per_layer_uniform = elems;
			acc += elems;
		}
		layer_off_elems[n_kv_layers] = acc;
		per_layer_uniform *= sizeof(uint16_t);
		total = acc * sizeof(uint16_t);
	} else {
		per_layer_uniform =
			(size_t)desc->n_ctx * desc->n_kv_heads * desc->head_dim * sizeof(uint16_t);
		total = per_layer_uniform * (size_t)n_kv_layers;
	}

	vk_kv_handle *kh = xcalloc(1, sizeof(vk_kv_handle));
	vk_kv_handle *vh = xcalloc(1, sizeof(vk_kv_handle));

	status_code s = vk_kv_store_alloc(p, total, per_layer_uniform, n_kv_layers, layer_bytes,
									  layer_off_elems, &kh->store);
	if (s != OK) {
		free(layer_bytes);
		free(layer_off_elems);
		free(kh);
		free(vh);
		return s;
	}

	s = vk_kv_store_alloc(p, total, 0, n_kv_layers, layer_bytes, layer_off_elems, &vh->store);
	free(layer_bytes);
	free(layer_off_elems);
	if (s != OK) {
		vk_kv_store_free(p, &kh->store);
		free(kh);
		free(vh);
		return s;
	}

	if (kh->store.chunked) {
		INFO("vulkan: KV cache (%zu bytes total) exceeds maxStorageBufferRange; "
			 "split into %d per-layer buffers",
			 total, n_kv_layers);
	}

	k_out->handle	= kh;
	k_out->size		= total;
	k_out->host_ptr = NULL;
	k_out->owner	= self;
	v_out->handle	= vh;
	v_out->size		= total;
	v_out->host_ptr = NULL;
	v_out->owner	= self;
	vk_kv_registry_add(p, kh);
	vk_kv_registry_add(p, vh);
	return OK;
}

static status_code vk_kv_put_impl(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								  const buffer *k_in, const buffer *v_in, int in_row_stride,
								  int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								  int m, int in_row0) {
	vk_priv *p		  = self->priv;
	int		 n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	const vk_kv_handle *kh = (const vk_kv_handle *)k->handle;
	const vk_kv_handle *vh = (const vk_kv_handle *)v->handle;
	vk_buf			   *kb = vk_kv_layer_buf(kh, layer, NULL);
	vk_buf			   *vb = vk_kv_layer_buf(vh, layer, NULL);

	struct {
		uint32_t layer_off;
		int32_t	 pos_start, n_kv_heads, head_dim, n_ctx, stride_head_dim;
		int32_t	 in_row_stride, m_tokens, in_row0;
	} push					 = {(uint32_t)vk_kv_layer_off_elems(kh, layer),
								pos_start,
								n_active,
								head_dim,
								n_ctx,
								head_dim,
								in_row_stride > 0 ? in_row_stride : n_active * head_dim,
								m,
								in_row0};
	vk_buf		*bufs[4]	 = {kb, vb, as_vkbuf(k_in), as_vkbuf(v_in)};
	VkDeviceSize offs[4]	 = {0, 0, k_in->offset, v_in->offset};
	int			 total_pairs = n_active * (head_dim / 2);
	uint64_t	 total		 = (uint64_t)total_pairs * (uint64_t)m;
	uint32_t	 groups		 = (uint32_t)((total + 127) / 128);
	return vk_dispatch_ex(p, &p->p_kv_put, bufs, offs, NULL, 4, &push, sizeof(push), groups, 1,
						  0x3);
}

static status_code vk_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
							 const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
							 int n_ctx, int n_kv_heads_active) {
	return vk_kv_put_impl(self, k, v, layer, pos, k_in, v_in, 0, n_kv_heads, head_dim, n_ctx,
						  n_kv_heads_active, 1, 0);
}

static status_code vk_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								   const buffer *k_in, const buffer *v_in, int in_row_stride,
								   int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								   int m) {
	return vk_kv_put_impl(self, k, v, layer, pos_start, k_in, v_in, in_row_stride, n_kv_heads,
						  head_dim, n_ctx, n_kv_heads_active, m, 0);
}

static status_code vk_kv_shift(backend *self, buffer *k, buffer *v, int layer,
							   const kv_shift_chunk *chunks, int n_chunks, const float *cos_tbl,
							   const float *sin_tbl, int n_kv_heads, int head_dim, int n_ctx,
							   int n_kv_heads_active) {
	vk_priv *p		  = self->priv;
	int		 n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int		 half	  = head_dim / 2;
	if (n_chunks <= 0 || n_active <= 0 || half <= 0)
		return OK;

	int total_rows = 0;
	for (int i = 0; i < n_chunks; i++)
		total_rows += chunks[i].n_rows;
	size_t tmp_bytes  = (size_t)total_rows * (size_t)n_active * (size_t)head_dim * sizeof(float);
	size_t tbl_floats = (size_t)n_chunks * (size_t)half;

	buffer		k_tmp = {0}, v_tmp = {0}, cos_b = {0}, sin_b = {0};
	status_code s = buffer_ensure_scratch(self, &k_tmp, tmp_bytes);
	if (s == OK)
		s = buffer_ensure_scratch(self, &v_tmp, tmp_bytes);
	if (s == OK)
		s = buffer_ensure_scratch(self, &cos_b, tbl_floats * sizeof(float));
	if (s == OK)
		s = buffer_ensure_scratch(self, &sin_b, tbl_floats * sizeof(float));
	if (s != OK)
		goto done;
	vk_buffer_write_f32(self, &cos_b, cos_tbl, (int)tbl_floats);
	vk_buffer_write_f32(self, &sin_b, sin_tbl, (int)tbl_floats);

	const vk_kv_handle *kh		= (const vk_kv_handle *)k->handle;
	const vk_kv_handle *vh		= (const vk_kv_handle *)v->handle;
	vk_buf			   *bufs[6] = {vk_kv_layer_buf(kh, layer, NULL),
								   vk_kv_layer_buf(vh, layer, NULL),
								   as_vkbuf(&k_tmp),
								   as_vkbuf(&v_tmp),
								   as_vkbuf(&cos_b),
								   as_vkbuf(&sin_b)};
	int					row0	= 0;
	for (int i = 0; i < n_chunks && s == OK; i++) {
		struct {
			uint32_t layer_off;
			int32_t	 src_pos, n_kv_heads, head_dim, n_ctx, neox, tmp_row0, tbl_off, m_rows;
		} push			= {(uint32_t)vk_kv_layer_off_elems(kh, layer),
						   chunks[i].src_pos,
						   n_active,
						   head_dim,
						   n_ctx,
						   self->rope_neox ? 1 : 0,
						   row0,
						   i * half,
						   chunks[i].n_rows};
		uint64_t total	= (uint64_t)n_active * (uint64_t)half * (uint64_t)chunks[i].n_rows;
		uint32_t groups = (uint32_t)((total + 127) / 128);
		s = vk_dispatch_ex(p, &p->p_kv_shift_gather, bufs, NULL, NULL, 6, &push, sizeof(push),
						   groups, 1, 0xC);
		row0 += chunks[i].n_rows;
	}

	row0 = 0;
	for (int i = 0; i < n_chunks && s == OK; i++) {
		s = vk_kv_put_impl(self, k, v, layer, chunks[i].dst_pos, &k_tmp, &v_tmp,
						   n_active * head_dim, n_kv_heads, head_dim, n_ctx, n_kv_heads_active,
						   chunks[i].n_rows, row0);
		row0 += chunks[i].n_rows;
	}

done:
	if (k_tmp.owner)
		self->buffer_free(self, &k_tmp);
	if (v_tmp.owner)
		self->buffer_free(self, &v_tmp);
	if (cos_b.owner)
		self->buffer_free(self, &cos_b);
	if (sin_b.owner)
		self->buffer_free(self, &sin_b);
	return s;
}

static status_code vk_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
								  int token, int dim, buffer *x_out) {
	vk_priv *p = self->priv;

	if (!tok_embd->handle && tok_embd->host_ptr) {
		if (TOKEN_IS_REG(token))
			return ERR_UNSUPPORTED;
		backend_report_host_fallback(
			self, "embd_lookup", HFB_BUF_HOST_RESIDENT,
			"token embedding buffer is host-resident; embedding lookup executed "
			"on host (cpu)");
		backend *host = backend_host();
		if (host && host->embd_lookup) {
			float *cpu_out	   = xmalloc((size_t)dim * sizeof(float));
			buffer cpu_out_buf = {
				.handle	  = cpu_out,
				.host_ptr = cpu_out,
				.size	  = (size_t)dim * sizeof(float),
				.offset	  = 0,
				.owner	  = NULL,
			};
			buffer embd_cpu_view = *tok_embd;
			embd_cpu_view.handle = (void *)tok_embd->host_ptr;
			embd_cpu_view.owner	 = NULL;
			status_code st =
				host->embd_lookup(host, &embd_cpu_view, tok_embd_type, token, dim, &cpu_out_buf);
			if (st == OK)
				st = vk_buffer_write_f32(self, x_out, cpu_out, dim);
			free(cpu_out);
			return st;
		}
		return ERR_UNSUPPORTED;
	}

	int shader_type = -1;
	switch (tok_embd_type) {
	case GGML_TYPE_F32:
		shader_type = 0;
		break;
	case GGML_TYPE_F16:
		shader_type = 1;
		break;
	case GGML_TYPE_Q4_0:
		shader_type = 2;
		break;
	case GGML_TYPE_Q4_1:
		shader_type = 3;
		break;
	case GGML_TYPE_Q5_0:
		shader_type = 6;
		break;
	case GGML_TYPE_Q5_1:
		shader_type = 7;
		break;
	case GGML_TYPE_Q8_0:
		shader_type = 8;
		break;
	case GGML_TYPE_Q4_K:
		shader_type = 12;
		break;
	case GGML_TYPE_Q5_K:
		shader_type = 13;
		break;
	case GGML_TYPE_Q6_K:
		shader_type = 14;
		break;
	case GGML_TYPE_IQ4_NL:
		shader_type = 20;
		break;
	case GGML_TYPE_IQ3_S:
		shader_type = 21;
		break;
	case GGML_TYPE_IQ2_XXS:
		shader_type = 16;
		break;
	case GGML_TYPE_IQ3_XXS:
		shader_type = 18;
		break;
	case GGML_TYPE_IQ2_S:
		shader_type = 22;
		break;
	case GGML_TYPE_IQ4_XS:
		shader_type = 23;
		break;
	case GGML_TYPE_BF16:
		shader_type = 30;
		break;
	default:
		break;
	}

	if (shader_type >= 0) {

		size_t row_stride = ggml_row_size(tok_embd_type, dim);
		struct {
			int32_t token, dim, type, row_stride;
		} push		  = {token, dim, shader_type, (int)row_stride};
		vk_buf *dummy = vk_dummy_buf(p);
		int		embd_needs_iqx =
			(tok_embd_type == GGML_TYPE_IQ2_S || tok_embd_type == GGML_TYPE_IQ2_XXS ||
			 tok_embd_type == GGML_TYPE_IQ3_XXS);
		vk_buf *grid_buf = (tok_embd_type == GGML_TYPE_IQ3_S && p->iq3s_grid_buf.buf)
							   ? &p->iq3s_grid_buf
						   : (embd_needs_iqx && p->iqx_grid_buf.buf) ? &p->iqx_grid_buf
																	 : dummy;
		if (TOKEN_IS_REG(token)) {
			status_code rs = vk_ensure_token_regs(p);
			if (rs != OK)
				return rs;
		}
		vk_buf		*bufs[4] = {as_vkbuf(tok_embd), as_vkbuf(x_out), grid_buf,
								TOKEN_IS_REG(token) ? &p->tok_reg_buf : dummy};
		VkDeviceSize offs[4] = {tok_embd->offset, x_out->offset, 0, 0};
		uint32_t	 groups	 = (uint32_t)((dim + 63) / 64);
		return vk_dispatch_ex(p, &p->p_embd_lookup, bufs, offs, NULL, 4, &push, sizeof(push),
							  groups, 1, 0x2);
	}

	size_t		   row_stride = ggml_row_size(tok_embd_type, dim);
	const uint8_t *embd_host;
	if (TOKEN_IS_REG(token))
		return ERR_UNSUPPORTED;
	if (tok_embd->host_ptr) {
		embd_host = (const uint8_t *)tok_embd->host_ptr + ((size_t)token * row_stride);
	} else {
		return ERR_UNSUPPORTED;
	}

	float *dst_host = xmalloc((size_t)dim * sizeof(float));
	backend_report_host_fallback(
		self, "embd_lookup", HFB_WEIGHT_TYPE,
		"embedding type '%s' (type=%u) has no vulkan shader; row dequantized "
		"on host (cpu)",
		ggml_type_name(tok_embd_type), tok_embd_type);
	backend *host = backend_host();
	if ((!host || !host->dequant_row) ||
		(tok_embd_type != GGML_TYPE_F32 && tok_embd_type != GGML_TYPE_F16 &&
		 tok_embd_type != GGML_TYPE_IQ3_S && tok_embd_type != GGML_TYPE_IQ2_S &&
		 tok_embd_type != GGML_TYPE_IQ2_XXS && tok_embd_type != GGML_TYPE_IQ3_XXS &&
		 tok_embd_type != GGML_TYPE_IQ4_XS)) {
		free(dst_host);
		return ERR_UNSUPPORTED;
	}
	status_code dq = host->dequant_row(host, tok_embd_type, embd_host, dim, dst_host);
	if (dq != OK) {
		free(dst_host);
		return dq;
	}

	status_code s = vk_buffer_write_f32(self, x_out, dst_host, dim);
	free(dst_host);
	return s;
}

static vk_pipeline_set *vk_matmul_batch_pipeline(vk_priv *p, uint32_t w_type, int residual,
												 int use_wide);
static status_code vk_dispatch_iq4_nl_unified(vk_priv *p, int mode, const buffer *w0,
											  const buffer *b1, const buffer *b2, buffer *y0,
											  buffer *y1_opt, int n, int n1, int k, int activation,
											  int has_residual);

static status_code vk_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									int n, float eps, int m);
static status_code vk_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w,
											 buffer *y, int n_heads, int head_dim, float eps,
											 int m);
static status_code vk_rmsnorm_noweight_batch(backend *self, const buffer *x, buffer *y, int n,
											 float eps, int m);
static status_code vk_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x, buffer *y,
													  int n_heads, int head_dim, float eps, int m);
static status_code vk_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								   buffer *y, int n, int k, int m);
static status_code vk_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
										 const buffer *x, buffer **y, const int *n_list, int k,
										 int n_matmuls, int m);
static status_code vk_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim,
								 int pos_start, const float *rope_cos_base,
								 const float *rope_sin_base, int m);
static status_code vk_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads,
									int n_kv_heads, int head_dim, int pos_start,
									const float *rope_cos_base, const float *rope_sin_base, int m);
static status_code vk_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim,
									 int pos_start, const float *rope_cos_base,
									 const float *rope_sin_base, const float *freq_factors, int m);
static status_code vk_ffn_activate_batch_scales(backend *self, const buffer *gate, const buffer *up,
												buffer *out, int n, int activation, int m, float gs,
												float us);
static status_code vk_add_batch(backend *self, buffer *x, const buffer *y, int n, int m);

static status_code vk_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							  float eps) {
	return vk_rmsnorm_batch(self, x, w, y, n, eps, 1);
}

static vk_pipeline_set *vk_matmul_pipeline(vk_priv *p, uint32_t w_type, int residual) {
	return vk_matmul_batch_pipeline(p, w_type, residual, 0);
}

static int vk_matmul_type_native(backend *self, uint32_t w_type) {
	vk_priv *p = self->priv;
	if (w_type == GGML_TYPE_IQ4_NL)
		return p->p_matmul_iq4_nl_batch.pipeline != NULL;
	vk_pipeline_set *ps = vk_matmul_pipeline(p, w_type, 0);
	if (!ps || !ps->pipeline)
		return 0;
	return 1;
}

static status_code vk_matmul_host_fallback(backend *self, const buffer *w, uint32_t w_type,
										   const buffer *x, const buffer *residual, buffer *y,
										   int n, int k) {
	int			has_residual = (residual != NULL);
	float	   *x_host		 = xmalloc((size_t)k * sizeof(float));
	status_code s			 = vk_buffer_read_f32(self, x, x_host, k);
	if (s != OK) {
		free(x_host);
		return s;
	}
	float *r_host = NULL;
	if (has_residual) {
		r_host = xmalloc((size_t)n * sizeof(float));
		s	   = vk_buffer_read_f32(self, residual, r_host, n);
		if (s != OK) {
			free(x_host);
			free(r_host);
			return s;
		}
	}
	float *y_host = xmalloc((size_t)n * sizeof(float));
	host_matmul_generic(w->host_ptr, w_type, x_host, y_host, n, k);
	if (has_residual)
		for (int i = 0; i < n; i++)
			y_host[i] += r_host[i];
	s = vk_buffer_write_f32(self, y, y_host, n);
	free(x_host);
	free(r_host);
	free(y_host);
	return s;
}

static int vk_matmul_groups_x(vk_priv *p, int n, uint32_t w_type) {
	int is_dense = (w_type == GGML_TYPE_F32 || w_type == GGML_TYPE_F16 || w_type == GGML_TYPE_BF16);
	int lpr		 = (is_dense && p->matmul_lanes_per_row > 0) ? p->matmul_lanes_per_row : 1;
	int rows	 = p->matmul_rows_per_thread;
	int per_wg	 = (p->matmul_wg_size / lpr) * rows;
	if (per_wg < 1)
		per_wg = 1;
	return (n + per_wg - 1) / per_wg;
}

static status_code vk_ensure_scratch(backend *self, buffer *buf, size_t *cap, size_t need) {
	vk_priv *p = self->priv;
	if (need <= *cap)
		return OK;
	if (buf->handle) {
		vk_drain_gpu(p);
		vk_free_buffer(p, as_vkbuf(buf));
		free(buf->handle);
		buf->handle = NULL;
		*cap		= 0;
	}
	size_t		c = need < ((size_t)1 << 20) ? ((size_t)1 << 20) : need;
	status_code s = vk_buffer_alloc_scratch(self, c, buf);
	if (s == OK)
		*cap = c;
	return s;
}

static int vk_qfamily(int t) {
	return t >= VQ_DENSE ? 2 : t >= VQ_MM_COUNT ? 1 : 0;
}

static int vk_qcore_type(vk_priv *p, uint32_t w_type, int k, int m) {
	static const struct {
		uint32_t type;
		int		 vq;
	} MAP[] = {
		{GGML_TYPE_Q4_0, VQ_Q4_0}, {GGML_TYPE_Q4_1, VQ_Q4_1}, {GGML_TYPE_Q5_0, VQ_Q5_0},
		{GGML_TYPE_Q5_1, VQ_Q5_1}, {GGML_TYPE_Q8_0, VQ_Q8_0}, {GGML_TYPE_IQ4_NL, VQ_IQ4_NL},
		{GGML_TYPE_Q4_K, VQ_Q4_K}, {GGML_TYPE_Q5_K, VQ_Q5_K}, {GGML_TYPE_Q6_K, VQ_Q6_K},
		{GGML_TYPE_F32, VQ_F32},   {GGML_TYPE_F16, VQ_F16},	  {GGML_TYPE_BF16, VQ_BF16},
	};
	static const int BLOCK[3] = {32, 256, 4};
	if (!p->qcore_ok || k <= 0)
		return -1;
	for (size_t i = 0; i < ARRAY_LEN(MAP); i++) {
		if (MAP[i].type != w_type)
			continue;
		int fam = vk_qfamily(MAP[i].vq);
		if (k % BLOCK[fam] != 0 || (fam != 0 && m >= VK_QMM_MIN_M))
			return -1;
		return MAP[i].vq;
	}
	return -1;
}

static status_code vk_qcore_quantize(backend *self, int family, const buffer *x, const buffer *up,
									 int act, int k, int m) {
	static const int QB[2]	   = {32, 256};
	static const int WORDS[2]  = {12, 76};
	static const int PER_WG[2] = {8, 1};
	vk_priv			*p		   = self->priv;
	if (family == 0 && !up && p->xq_fused.buf == as_vkbuf(x) && p->xq_fused.off == x->offset &&
		p->xq_fused.k == k && p->xq_fused.m == m && p->xq_fused.seq == p->stream_seq)
		return OK;
	int			kb	 = k / QB[family];
	size_t		need = (size_t)m * (size_t)kb * (size_t)WORDS[family] * 4;
	status_code s	 = vk_ensure_scratch(self, &p->xq_buf[family], &p->xq_cap[family], need);
	if (s != OK)
		return s;
	const buffer *u = up ? up : x;
	struct {
		int32_t k, m, x_stride, act;
	} push				 = {k, m, k, up ? act : -1};
	vk_buf		*bufs[3] = {as_vkbuf(x), as_vkbuf(u), as_vkbuf(&p->xq_buf[family])};
	VkDeviceSize offs[3] = {x->offset, u->offset, p->xq_buf[family].offset};
	return vk_dispatch_ex(p, &p->p_q8_quant[family], bufs, offs, NULL, 3, &push, sizeof(push),
						  (uint32_t)((kb + PER_WG[family] - 1) / PER_WG[family]), (uint32_t)m,
						  1u << 2);
}

static status_code vk_qcore_matmul(vk_priv *p, int t, const buffer *w, const buffer *x, buffer *y,
								   const buffer *res, int n, int k, int m) {
	static const int BLOCK[3] = {32, 256, 4};
	int				 res_mode = 0;
	if (res)
		res_mode = (as_vkbuf(res) == as_vkbuf(y) && res->offset == y->offset) ? 2 : 1;
	const buffer *xs = vk_qfamily(t) == 2 ? x : &p->xq_buf[vk_qfamily(t)];
	struct {
		int32_t n, kb, m, res_mode;
	} push					 = {n, k / BLOCK[vk_qfamily(t)], m, res_mode};
	vk_buf			*bufs[4] = {as_vkbuf(w), as_vkbuf(xs), as_vkbuf(y),
								res ? as_vkbuf(res) : vk_dummy_buf(p)};
	VkDeviceSize	 offs[4] = {w->offset, xs->offset, y->offset, res ? res->offset : 0};
	vk_pipeline_set *ps;
	uint32_t		 gx, gy;
	if (m >= VK_QMM_MIN_M) {
		ps = &p->p_qmm[t];
		gx = (uint32_t)((m + VK_QMM_TILE - 1) / VK_QMM_TILE);
		gy = (uint32_t)((n + VK_QMM_TILE - 1) / VK_QMM_TILE);
	} else {
		int c = m == 1 ? 0 : 1;
		ps	  = &p->p_qmv[t][c];
		gx	  = (uint32_t)((n + (int)VQ_MV_SPEC[c][1] - 1) / (int)VQ_MV_SPEC[c][1]);
		gy	  = (uint32_t)((m + (int)VQ_MV_SPEC[c][0] - 1) / (int)VQ_MV_SPEC[c][0]);
	}
	if (gx > p->caps.max_workgroup_count[0] || gy > p->caps.max_workgroup_count[1])
		return ERR_UNSUPPORTED;
	int			keeps_fused = p->xq_fused.seq == p->stream_seq && as_vkbuf(y) != p->xq_fused.buf;
	status_code s =
		vk_dispatch_ex(p, ps, bufs, offs, NULL, 4, &push, sizeof(push), gx, gy, 1u << 2);
	if (s == OK && keeps_fused)
		p->xq_fused.seq = p->stream_seq;
	return s;
}

static status_code vk_qcore_run(backend *self, int t, const buffer *w, const buffer *x,
								const buffer *up, int act, buffer *y, const buffer *res, int n,
								int k, int m) {
	if (vk_qfamily(t) == 2) {
		if (up)
			return ERR_UNSUPPORTED;
		return vk_qcore_matmul(self->priv, t, w, x, y, res, n, k, m);
	}
	status_code s = vk_qcore_quantize(self, vk_qfamily(t), x, up, act, k, m);
	if (s != OK)
		return s;
	return vk_qcore_matmul(self->priv, t, w, x, y, res, n, k, m);
}

static status_code vk_matmul_impl(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								  const buffer *residual, buffer *y, int n, int k) {
	vk_priv *p			  = self->priv;
	int		 has_residual = (residual != NULL);

	if (!w->handle && w->host_ptr) {
		backend_report_host_fallback(
			self, "matmul", HFB_BUF_HOST_RESIDENT,
			"weight buffer is host-resident (never uploaded to the device); "
			"matmul executed on host (cpu)");
		return vk_matmul_host_fallback(self, w, w_type, x, residual, y, n, k);
	}

	int qt = vk_qcore_type(p, w_type, k, 1);
	if (qt >= 0) {
		status_code qs = vk_qcore_run(self, qt, w, x, NULL, -1, y, residual, n, k, 1);
		if (qs != ERR_UNSUPPORTED)
			return qs;
	}

	vk_pipeline_set *ps = vk_matmul_pipeline(p, w_type, has_residual);

	if (w_type == GGML_TYPE_IQ4_NL && p->p_matmul_iq4_nl_batch.pipeline && k > 0 && (k % 32) == 0) {
		vk_buf		 *dummy = vk_dummy_buf(p);
		const buffer *b2	= has_residual
								  ? residual
								  : (const buffer *)&(buffer){
										.handle = dummy, .size = 16, .offset = 0, .owner = self};
		return vk_dispatch_iq4_nl_unified(p, 0, w, x, b2, y, NULL, n, 0, k, 0,
										  has_residual ? 1 : 0);
	}

	if (!ps) {
		if (!w->host_ptr) {
			ERROR("vk: matmul CPU fallback: no host_ptr for weight type=%u", w_type);
			return ERR_UNSUPPORTED;
		}
		backend_report_host_fallback(
			self, "matmul", HFB_WEIGHT_TYPE,
			"no vulkan shader for weight type '%s' (type=%u); matmul executed "
			"on host (cpu)",
			ggml_type_name(w_type), w_type);
		return vk_matmul_host_fallback(self, w, w_type, x, residual, y, n, k);
	}

	int is_iq3_s	 = (w_type == GGML_TYPE_IQ3_S);
	int is_iqx		 = (w_type == GGML_TYPE_IQ2_S || w_type == GGML_TYPE_IQ2_XXS ||
						w_type == GGML_TYPE_IQ3_XXS || w_type == GGML_TYPE_IQ4_XS);
	int is_kquant	 = (w_type == GGML_TYPE_Q4_K || w_type == GGML_TYPE_Q5_K ||
						w_type == GGML_TYPE_Q6_K || w_type == GGML_TYPE_IQ3_S || is_iqx);
	int unified_push = (w_type == GGML_TYPE_Q4_0 || w_type == GGML_TYPE_Q4_K ||
						w_type == GGML_TYPE_Q6_K || w_type == GGML_TYPE_IQ3_S || is_iqx);

	vk_buf		*bufs[5];
	VkDeviceSize offs[5];
	int			 n_bufs;
	if (has_residual) {
		bufs[0] = as_vkbuf(w);
		offs[0] = w->offset;
		bufs[1] = as_vkbuf(x);
		offs[1] = x->offset;
		bufs[2] = as_vkbuf(residual);
		offs[2] = residual->offset;
		bufs[3] = as_vkbuf(y);
		offs[3] = y->offset;
		n_bufs	= 4;
	} else {
		bufs[0] = as_vkbuf(w);
		offs[0] = w->offset;
		bufs[1] = as_vkbuf(x);
		offs[1] = x->offset;
		bufs[2] = as_vkbuf(y);
		offs[2] = y->offset;
		n_bufs	= 3;
	}
	if (is_iq3_s && p->iq3s_grid_buf.buf) {
		bufs[n_bufs] = &p->iq3s_grid_buf;
		offs[n_bufs] = 0;
		n_bufs++;
	}
	if (is_iqx && w_type != GGML_TYPE_IQ4_XS && p->iqx_grid_buf.buf) {
		bufs[n_bufs] = &p->iqx_grid_buf;
		offs[n_bufs] = 0;
		n_bufs++;
	}
	uint32_t groups;
	if (is_kquant) {
		groups = (uint32_t)((n + (int)p->matmul_wg_size - 1) / (int)p->matmul_wg_size);
	} else {
		groups = (uint32_t)vk_matmul_groups_x(p, n, w_type);
	}
	uint32_t wmask = has_residual ? 0x8 : (1u << 2);
	if (unified_push) {
		struct {
			int32_t n0, n1, k, m;
		} push = {n, 0, k, 1};
		return vk_dispatch_ex(p, ps, bufs, offs, NULL, n_bufs, &push, sizeof(push), groups, 1,
							  wmask);
	} else {
		struct {
			int32_t n, k, m;
		} push = {n, k, 1};
		return vk_dispatch_ex(p, ps, bufs, offs, NULL, n_bufs, &push, sizeof(push), groups, 1,
							  wmask);
	}
}

static status_code vk_matmul(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
							 buffer *y, int n, int k) {
	return vk_matmul_batch(self, w, w_type, x, y, n, k, 1);
}

static status_code vk_dispatch_iq4_nl_unified(vk_priv *p, int mode, const buffer *w0,
											  const buffer *b1, const buffer *b2, buffer *y0,
											  buffer *y1_opt, int n, int n1, int k, int activation,
											  int has_residual) {
	if (!p->p_matmul_iq4_nl_batch.pipeline)
		return ERR_UNSUPPORTED;
	vk_buf *dummy = vk_dummy_buf(p);

	struct {
		int32_t mode, n, n1, k, activation, has_residual, n2, m;
	} push = {mode, n, n1, k, activation, has_residual, 0, 1};

	vk_buf *bufs[7] = {
		as_vkbuf(w0), as_vkbuf(b1), as_vkbuf(b2), as_vkbuf(y0), y1_opt ? as_vkbuf(y1_opt) : dummy,
		dummy,		  dummy,
	};
	VkDeviceSize offs[7] = {
		w0->offset, b1->offset, b2->offset, y0->offset, y1_opt ? y1_opt->offset : 0, 0, 0,
	};

	int		 rows	= p->matmul_rows_per_thread;
	int		 wg		= p->matmul_wg_size;
	int		 n_max	= (mode == 1) ? ((n > n1) ? n : n1) : n;
	uint32_t groups = (uint32_t)((n_max + (wg * rows) - 1) / (wg * rows));
	uint32_t wmask	= (mode == 1) ? ((1u << 3) | (1u << 4)) : (1u << 3);
	return vk_dispatch_ex(p, &p->p_matmul_iq4_nl_batch, bufs, offs, NULL, 7, &push, sizeof(push),
						  groups, 1, wmask);
}

static status_code vk_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
								   int n_matmuls) {
	return vk_matmul_multi_batch(self, w, w_types, x, y, n_list, k, n_matmuls, 1);
}

static void vk_drain_gpu(vk_priv *p) {
	if (p->device_lost)
		return;
	if (vk_flush(p) != OK)
		ERROR("vk: draining the GPU failed; a buffer may be reused while still in use");
}

static status_code vk_ensure_ffn_act_buf(backend *self, int k, buffer *out) {
	vk_priv *p = self->priv;
	if (k <= p->ffn_act_cap) {
		*out	  = p->ffn_act_buf;
		out->size = (size_t)k * sizeof(float);
		return OK;
	}

	if (p->ffn_act_buf.handle) {
		vk_drain_gpu(p);
		vk_free_buffer(p, (vk_buf *)p->ffn_act_buf.handle);
		free(p->ffn_act_buf.handle);
		p->ffn_act_buf.handle = NULL;
		p->ffn_act_cap		  = 0;
	}

	status_code s = vk_buffer_alloc_scratch(self, (size_t)k * sizeof(float), &p->ffn_act_buf);
	if (s != OK)
		return s;
	p->ffn_act_cap = k;
	*out		   = p->ffn_act_buf;
	out->size	   = (size_t)k * sizeof(float);
	return OK;
}

static status_code vk_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *gate, const buffer *up, buffer *y, int n, int k,
									  int activation);

static status_code vk_matmul_ffn_down_batch(backend *self, const buffer *w, uint32_t w_type,
											const buffer *gate, const buffer *up, buffer *y, int n,
											int k, int activation, int m) {
	if (m <= 0)
		return ERR_INVALID_ARG;
	if (m == 1)
		return vk_matmul_ffn_down(self, w, w_type, gate, up, y, n, k, activation);

	if (n <= 0 || k <= 0)
		return ERR_INVALID_ARG;

	int qt = vk_qcore_type(self->priv, w_type, k, m);
	if (qt >= 0) {
		status_code qs = vk_qcore_run(self, qt, w, gate, up, activation, y, NULL, n, k, m);
		if (qs != ERR_UNSUPPORTED)
			return qs;
	}

	if (!vk_matmul_type_native(self, w_type)) {
		backend_report_host_fallback(
			self, "ffn_down", HFB_WEIGHT_TYPE,
			"no vulkan matmul for weight type '%s'; activation + matmul executed on "
			"host (cpu)",
			ggml_type_name(w_type));
		return ERR_UNSUPPORTED;
	}

	buffer		act = {0};
	status_code s	= vk_ensure_ffn_act_buf(self, m * k, &act);
	if (s != OK)
		return s;

	s = vk_ffn_activate_batch_scales(self, gate, up, &act, k, activation, m, 1.0f, 1.0f);
	if (s != OK)
		return s;

	return vk_matmul_batch(self, w, w_type, &act, y, n, k, m);
}

static status_code vk_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *gate, const buffer *up, buffer *y, int n, int k,
									  int activation) {
	vk_priv *p = self->priv;

	int qt = vk_qcore_type(p, w_type, k, 1);
	if (qt >= 0) {
		status_code qs = vk_qcore_run(self, qt, w, gate, up, activation, y, NULL, n, k, 1);
		if (qs != ERR_UNSUPPORTED)
			return qs;
	}

	if (w_type == GGML_TYPE_IQ4_NL && p->p_matmul_iq4_nl_batch.pipeline && k > 0 && (k % 32) == 0) {
		return vk_dispatch_iq4_nl_unified(p, 2, w, gate, up, y, NULL, n, 0, k, activation, 0);
	}

	if (n <= 0 || k <= 0)
		return ERR_INVALID_ARG;

	if (!vk_matmul_type_native(self, w_type)) {
		backend_report_host_fallback(
			self, "ffn_down", HFB_WEIGHT_TYPE,
			"no vulkan matmul for weight type '%s'; activation + matmul executed on "
			"host (cpu)",
			ggml_type_name(w_type));
		return ERR_UNSUPPORTED;
	}

	buffer		act = {0};
	status_code s	= vk_ensure_ffn_act_buf(self, k, &act);
	if (s != OK)
		return s;

	s = vk_ffn_activate_batch_scales(self, gate, up, &act, k, activation, 1, 1.0f, 1.0f);
	if (s != OK)
		return s;

	return vk_matmul_batch(self, w, w_type, &act, y, n, k, 1);
}

static status_code vk_alloc_rope_buf(vk_priv *p, size_t size, buffer *out) {
	vk_buf			  *b = xcalloc(1, sizeof(vk_buf));
	VkBufferUsageFlags usage =
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	VkMemoryPropertyFlags preferred =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	if (p->caps.unified_memory)
		preferred |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	status_code s = vk_alloc_buffer(p, size, usage, preferred, b);
	if (s != OK) {
		VkMemoryPropertyFlags fallback =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		s = vk_alloc_buffer(p, size, usage, fallback, b);
	}
	if (s != OK) {
		free(b);
		return s;
	}

	out->handle	  = b;
	out->size	  = size;
	out->host_ptr = NULL;
	return OK;
}

static status_code vk_ensure_rope_bufs(backend *self, int half, const float *cos_base,
									   const float *sin_base) {
	vk_priv *p	   = self->priv;
	int		 n_ctx = p->n_ctx_stored;
	if (n_ctx <= 0 || half <= 0)
		return ERR_INVALID_ARG;

	int need = n_ctx * half;

	buffer		 *cos_buf;
	buffer		 *sin_buf;
	int			 *cap;
	int			 *half_stored;
	const float **cos_base_p;
	const float **sin_base_p;

	if (p->rope_cos_base == cos_base && p->rope_sin_base == sin_base) {
		cos_buf		= &p->rope_cos_buf;
		sin_buf		= &p->rope_sin_buf;
		cap			= &p->rope_buf_cap;
		half_stored = &p->rope_half_stored;
		cos_base_p	= &p->rope_cos_base;
		sin_base_p	= &p->rope_sin_base;
	} else if (p->rope_cos_base_alt == cos_base && p->rope_sin_base_alt == sin_base) {
		cos_buf		= &p->rope_cos_buf_alt;
		sin_buf		= &p->rope_sin_buf_alt;
		cap			= &p->rope_buf_cap_alt;
		half_stored = &p->rope_half_stored_alt;
		cos_base_p	= &p->rope_cos_base_alt;
		sin_base_p	= &p->rope_sin_base_alt;
	} else if (p->rope_cos_base == NULL) {
		cos_buf		= &p->rope_cos_buf;
		sin_buf		= &p->rope_sin_buf;
		cap			= &p->rope_buf_cap;
		half_stored = &p->rope_half_stored;
		cos_base_p	= &p->rope_cos_base;
		sin_base_p	= &p->rope_sin_base;
	} else {
		if (p->rope_cos_base_alt == NULL) {
			cos_buf		= &p->rope_cos_buf_alt;
			sin_buf		= &p->rope_sin_buf_alt;
			cap			= &p->rope_buf_cap_alt;
			half_stored = &p->rope_half_stored_alt;
			cos_base_p	= &p->rope_cos_base_alt;
			sin_base_p	= &p->rope_sin_base_alt;
		} else {
			cos_buf		= &p->rope_cos_buf;
			sin_buf		= &p->rope_sin_buf;
			cap			= &p->rope_buf_cap;
			half_stored = &p->rope_half_stored;
			cos_base_p	= &p->rope_cos_base;
			sin_base_p	= &p->rope_sin_base;
		}
	}

	int need_upload =
		(need > *cap || *half_stored != half || *cos_base_p != cos_base || *sin_base_p != sin_base);

	if (need_upload) {
		if (need > *cap) {
			if (cos_buf->handle)
				vk_buffer_free(self, cos_buf);
			if (sin_buf->handle)
				vk_buffer_free(self, sin_buf);

			status_code s = vk_alloc_rope_buf(p, (size_t)need * sizeof(float), cos_buf);
			if (s != OK)
				return s;
			s = vk_alloc_rope_buf(p, (size_t)need * sizeof(float), sin_buf);
			if (s != OK)
				return s;
			*cap = need;
		}

		status_code s = vk_buffer_write_f32(self, cos_buf, cos_base, need);
		if (s != OK)
			return s;
		s = vk_buffer_write_f32(self, sin_buf, sin_base, need);
		if (s != OK)
			return s;
		*cos_base_p	 = cos_base;
		*sin_base_p	 = sin_base;
		*half_stored = half;
	}

	p->rope_cos_buf_active = cos_buf;
	p->rope_sin_buf_active = sin_buf;

	return OK;
}

static status_code vk_rope(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
						   const float *rope_cos_base, const float *rope_sin_base) {
	return vk_rope_batch(self, vec, n_heads, head_dim, pos, rope_cos_base, rope_sin_base, 1);
}

static status_code vk_rope_qk(backend *self, buffer *q, buffer *k, int n_heads, int n_kv_heads,
							  int head_dim, int pos, const float *rope_cos_base,
							  const float *rope_sin_base) {
	return vk_rope_qk_batch(self, q, k, n_heads, n_kv_heads, head_dim, pos, rope_cos_base,
							rope_sin_base, 1);
}

static status_code vk_ensure_scores_buf_batch(backend *self, int n_heads, int n_ctx, int m) {
	vk_priv *p		= self->priv;
	int		 needed = m * n_heads * n_ctx;
	if (needed <= p->attn_scores_cap)
		return OK;

	if (p->attn_scores_buf.handle) {
		vk_drain_gpu(p);
		vk_free_buffer(p, (vk_buf *)p->attn_scores_buf.handle);
		free(p->attn_scores_buf.handle);
		p->attn_scores_buf.handle = NULL;
	}

	status_code s =
		vk_buffer_alloc_scratch(self, (size_t)needed * sizeof(float), &p->attn_scores_buf);
	if (s != OK)
		return s;
	p->attn_scores_cap = needed;
	return OK;
}

static status_code vk_ensure_scores_buf(backend *self, int n_heads, int n_ctx) {
	return vk_ensure_scores_buf_batch(self, n_heads, n_ctx, 1);
}

static int vk_flash_layout(vk_priv *p, int head_dim, int n_groups, uint32_t spec[6]) {
	const int sg = (int)p->caps.subgroup_size;
	if (sg < 1 || head_dim < 1 || head_dim > 256 || n_groups < 1)
		return 0;

	int lanes = 0;
	for (int l = 1; l <= 16; l <<= 1) {
		if (l > sg || sg % l != 0 || head_dim % l != 0 || head_dim / l > FLASH_MAX_DPL)
			continue;
		lanes = l;
		break;
	}
	if (lanes == 0)
		return 0;

	int heads = n_groups;
	while (heads > 1 && (n_groups % heads != 0 || heads * lanes > sg))
		heads--;

	const int base	= heads * lanes;
	int		  split = 1;
	while (split * base < 128 && split < FLASH_MAX_SPLIT && (split * 2) * base <= 1024)
		split *= 2;

	int tile_t = 2048 / head_dim;
	while (tile_t > 4 && (tile_t & (tile_t - 1)))
		tile_t--;
	if (tile_t < 4)
		tile_t = 4;

	size_t kv_bytes = ((size_t)tile_t * ((size_t)head_dim + FLASH_K_PAD) +
					   (size_t)tile_t * (size_t)head_dim + (size_t)heads * (size_t)tile_t) *
					  sizeof(float);
	size_t pt_bytes =
		(size_t)heads * (size_t)split * (size_t)lanes * (FLASH_MAX_DPL + 2) * sizeof(float);
	if (p->caps.max_shared_memory > 0 && kv_bytes + pt_bytes > p->caps.max_shared_memory)
		return 0;

	spec[0] = (uint32_t)head_dim;
	spec[1] = (uint32_t)heads;
	spec[2] = (uint32_t)lanes;
	spec[3] = (uint32_t)(base * split);
	spec[4] = (uint32_t)tile_t;
	spec[5] = (uint32_t)split;
	return 1;
}

static int vk_flash_groups(vk_priv *p, int head_dim, int n_heads, int n_kv_heads) {
	uint32_t spec[6];
	int		 n_groups = n_kv_heads > 0 ? n_heads / n_kv_heads : 1;
	if (!vk_flash_layout(p, head_dim, n_groups, spec))
		return n_kv_heads;
	return n_kv_heads * n_groups / (int)spec[1];
}

static status_code vk_ensure_attn_pipeline(vk_priv *p, int kind, int head_dim, int n_groups,
										   const uint32_t *spec_data, uint32_t spec_size,
										   const uint32_t *spv, size_t spv_len, int n_bindings,
										   uint32_t push_size, const char *name,
										   vk_pipeline_set **out) {
	vk_attn_cache *c = &p->attn_cache[kind];
	for (int i = 0; i < c->count; i++) {
		if (c->head_dim[i] != head_dim || c->n_groups[i] != n_groups)
			continue;
		if (c->unsupported[i])
			return ERR_UNSUPPORTED;
		*out = &c->set[i];
		return OK;
	}
	if (c->count >= VK_ATTN_CACHE_CAP || !spec_data) {
		if (spec_data)
			WARN("attention pipeline cache full, %s head_dim=%d n_groups=%d", name, head_dim,
				 n_groups);
		if (c->count < VK_ATTN_CACHE_CAP) {
			c->head_dim[c->count]	 = head_dim;
			c->n_groups[c->count]	 = n_groups;
			c->unsupported[c->count] = 1;
			c->count++;
		}
		return ERR_UNSUPPORTED;
	}

	int slot			 = c->count;
	c->head_dim[slot]	 = head_dim;
	c->n_groups[slot]	 = n_groups;
	c->unsupported[slot] = 1;
	c->count++;

	status_code s = vk_create_pipeline_spec(p, spv, spv_len, n_bindings, push_size, spec_data,
											spec_size, &c->set[slot]);
	if (s != OK) {
		WARN("%s pipeline creation failed for head_dim=%d n_groups=%d", name, head_dim, n_groups);
		return s;
	}
	c->set[slot].name	 = name;
	c->unsupported[slot] = 0;
	*out				 = &c->set[slot];
	return OK;
}

static int vk_attn_tile_t(int head_dim) {
	return head_dim > 256 ? 8 : 16;
}

static status_code vk_attn_check_shared(vk_priv *p, int head_dim) {
	size_t q_bytes	= (size_t)head_dim * sizeof(float);
	size_t kv_bytes = (size_t)vk_attn_tile_t(head_dim) * (size_t)head_dim * sizeof(float);
	size_t shared	= q_bytes + kv_bytes;
	if (p->caps.max_shared_memory > 0 && shared > p->caps.max_shared_memory) {
		WARN("attention pipeline needs %zu bytes shared (device has %u), head_dim=%d", shared,
			 p->caps.max_shared_memory, head_dim);
		return ERR_UNSUPPORTED;
	}
	return OK;
}

static status_code vk_ensure_attn_spec(vk_priv *p, int kind, int head_dim, int n_groups,
									   const uint32_t *spv, size_t spv_len, int n_bindings,
									   uint32_t push_size, const char *name,
									   vk_pipeline_set **out) {
	if (vk_attn_check_shared(p, head_dim) != OK)
		return ERR_UNSUPPORTED;
	uint32_t spec_data[2] = {(uint32_t)head_dim, (uint32_t)vk_attn_tile_t(head_dim)};
	return vk_ensure_attn_pipeline(p, kind, head_dim, n_groups, spec_data, sizeof(spec_data), spv,
								   spv_len, n_bindings, push_size, name, out);
}

static status_code vk_ensure_attn_flash(vk_priv *p, int kind, int head_dim, int n_groups,
										const uint32_t *spv, size_t spv_len, int n_bindings,
										uint32_t push_size, const char *name,
										vk_pipeline_set **out) {
	uint32_t spec_data[6];
	if (!vk_flash_layout(p, head_dim, n_groups, spec_data))
		return vk_ensure_attn_pipeline(p, kind, head_dim, n_groups, NULL, 0, spv, spv_len,
									   n_bindings, push_size, name, out);
	return vk_ensure_attn_pipeline(p, kind, head_dim, n_groups, spec_data, sizeof(spec_data), spv,
								   spv_len, n_bindings, push_size, name, out);
}

static status_code vk_ensure_flash_pipeline(vk_priv *p, int head_dim, int n_groups,
											vk_pipeline_set **out) {
	return vk_ensure_attn_flash(p, VK_ATTN_FLASH, head_dim, n_groups, shader_attention_flash_spv,
								shader_attention_flash_spv_len, 4, 36, "attention_flash", out);
}

static status_code vk_ensure_attention_pipeline(vk_priv *p, int head_dim, vk_pipeline_set **out) {
	return vk_ensure_attn_spec(p, VK_ATTN_PLAIN, head_dim, 1, shader_attention_spv,
							   shader_attention_spv_len, 5, 36, "attention", out);
}

static status_code vk_attention_decode(backend *self, const buffer *q, vk_buf *kb, vk_buf *vb,
									   size_t layer_off, buffer *out, int pos_start, int n_heads,
									   int n_kv_heads, int head_dim, int n_ctx, float scale,
									   int window, int m) {
	vk_priv *p		  = self->priv;
	int		 start	  = window > 0 ? MAX(0, pos_start + 1 - window) : 0;
	int		 n_pos	  = MAX(pos_start + m - start, 1);
	int		 n_wg	  = n_heads * m;
	int		 n_splits = MIN((VK_ATTN_DEC_TARGET_WG + n_wg - 1) / n_wg, VK_ATTN_DEC_MAX_SPLITS);
	n_splits  = MAX(MIN(n_splits, (n_pos + VK_ATTN_DEC_MIN_CHUNK - 1) / VK_ATTN_DEC_MIN_CHUNK), 1);
	int chunk = (n_pos + n_splits - 1) / n_splits;
	n_splits  = (n_pos + chunk - 1) / chunk;

	size_t		part_bytes = (size_t)n_wg * n_splits * (size_t)(head_dim + 2) * sizeof(float);
	status_code s = vk_ensure_scratch(self, &p->attn_part_buf, &p->attn_part_cap, part_bytes);
	if (s != OK)
		return s;

	struct {
		uint32_t layer_off;
		int32_t	 pos, n_heads, n_kv_heads, head_dim, n_ctx;
		float	 scale;
		int32_t	 window, chunk, n_splits;
	} push				 = {(uint32_t)layer_off,
							pos_start,
							n_heads,
							n_kv_heads,
							head_dim,
							n_ctx,
							scale,
							window,
							chunk,
							n_splits};
	vk_buf		*bufs[5] = {as_vkbuf(q), kb, vb, as_vkbuf(&p->attn_part_buf), as_vkbuf(out)};
	VkDeviceSize offs[5] = {q->offset, 0, 0, p->attn_part_buf.offset, out->offset};
	vk_buf		*pb		 = as_vkbuf(&p->attn_part_buf);
	s = vk_dispatch_ex(p, &p->p_attn_decode, bufs, offs, NULL, 5, &push, sizeof(push),
					   (uint32_t)n_heads, (uint32_t)(n_splits * m), n_splits > 1 ? 0x8 : 0x10);
	if (s != OK || n_splits == 1)
		return s;
	struct {
		int32_t head_dim, n_splits;
	} mpush				  = {head_dim, n_splits};
	vk_buf		*mbufs[2] = {pb, as_vkbuf(out)};
	VkDeviceSize moffs[2] = {p->attn_part_buf.offset, out->offset};
	return vk_dispatch_ex(p, &p->p_attn_merge, mbufs, moffs, NULL, 2, &mpush, sizeof(mpush),
						  (uint32_t)n_wg, 1, 0x2);
}

static status_code vk_attention_impl(backend *self, const buffer *q, const buffer *k_cache,
									 const buffer *v_cache, buffer *out, int layer, int pos,
									 int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									 int flash_attn, float scale, int n_kv_heads_active,
									 const char *diag_label, int sliding_window, int attn_start,
									 int n_pos) {
	vk_priv *p = self->priv;

	int n_active		= n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int stride_head_dim = p->kv_head_dim_max > 0 ? p->kv_head_dim_max : head_dim;

	const vk_kv_handle *kh = (const vk_kv_handle *)k_cache->handle;
	const vk_kv_handle *vh = (const vk_kv_handle *)v_cache->handle;
	int					shader_layer_k;
	int					shader_layer_v;
	vk_buf			   *kb = vk_kv_layer_buf(kh, layer, &shader_layer_k);
	vk_buf			   *vb = vk_kv_layer_buf(vh, layer, &shader_layer_v);

	if (p->debug.diag) {
		static int seen[512];
		static int n_seen  = 0;
		int		   already = 0;
		for (int i = 0; i < n_seen; i++)
			if (seen[i] == layer) {
				already = 1;
				break;
			}
		if (!already && n_seen < 512) {
			seen[n_seen++] = layer;
			fprintf(stderr, "[VK_DIAG] %s path: %s\n", diag_label,
					(kb->mapped && vb->mapped) ? "cpu_mapped" : "device");
			if (sliding_window > 0) {
				fprintf(stderr,
						"[VK_DIAG] %s params: layer=%d shader_layer_k=%d shader_layer_v=%d "
						"pos=%d n_heads=%d n_kv_heads(stride)=%d n_kv_heads_active=%d "
						"n_active=%d "
						"head_dim=%d stride_head_dim=%d n_ctx=%d sliding_window=%d "
						"attn_start=%d "
						"swa_n_pos=%d\n",
						diag_label, layer, shader_layer_k, shader_layer_v, pos, n_heads, n_kv_heads,
						n_kv_heads_active, n_active, head_dim, stride_head_dim, n_ctx,
						sliding_window, attn_start, n_pos);
			} else {
				fprintf(stderr,
						"[VK_DIAG] %s params: layer=%d shader_layer_k=%d shader_layer_v=%d "
						"pos=%d n_heads=%d n_kv_heads(stride)=%d n_kv_heads_active=%d "
						"n_active=%d "
						"head_dim=%d stride_head_dim=%d n_ctx=%d\n",
						diag_label, layer, shader_layer_k, shader_layer_v, pos, n_heads, n_kv_heads,
						n_kv_heads_active, n_active, head_dim, stride_head_dim, n_ctx);
			}
		}
	}

	n_kv_heads = n_active;

	if (head_dim <= VK_ATTN_DEC_MAX_HEAD_DIM && (head_dim & 1) == 0 && p->p_attn_decode.pipeline &&
		p->p_attn_merge.pipeline)
		return vk_attention_decode(self, q, kb, vb, vk_kv_layer_off_elems(kh, layer), out, pos,
								   n_heads, n_kv_heads, head_dim, n_ctx, scale, sliding_window, 1);

	int				 n_groups = n_heads / n_kv_heads;
	vk_pipeline_set *flash_ps = NULL;
	int can_flash = flash_attn && vk_ensure_flash_pipeline(p, head_dim, n_groups, &flash_ps) == OK;

	size_t layer_off = vk_kv_layer_off_elems(kh, layer);

	if (can_flash) {
		struct {
			uint32_t layer_off;
			int32_t	 pos, n_heads, n_kv_heads, head_dim, n_ctx;
			float	 scale;
			int32_t	 stride_head_dim;
			int32_t	 attn_start;
		} push = {(uint32_t)layer_off, pos, n_heads, n_kv_heads, head_dim, n_ctx, scale, head_dim,
				  attn_start};
		uint32_t	 groups	 = (uint32_t)vk_flash_groups(p, head_dim, n_heads, n_kv_heads);
		vk_buf		*bufs[4] = {as_vkbuf(q), kb, vb, as_vkbuf(out)};
		VkDeviceSize offs[4] = {q->offset, 0, 0, out->offset};
		return vk_dispatch_ex(p, flash_ps, bufs, offs, NULL, 4, &push, sizeof(push), groups, 1,
							  1u << 3);
	}

	status_code s = vk_ensure_scores_buf(self, n_heads, n_ctx);
	if (s != OK)
		return s;
	struct {
		uint32_t layer_off;
		int32_t	 pos, n_heads, n_kv_heads, head_dim, n_ctx;
		float	 scale;
		int32_t	 stride_head_dim;
		int32_t	 attn_start;
	} push = {(uint32_t)layer_off, pos, n_heads, n_kv_heads, head_dim, n_ctx, scale, head_dim,
			  attn_start};
	vk_buf			*bufs[5] = {as_vkbuf(q), kb, vb, as_vkbuf(&p->attn_scores_buf), as_vkbuf(out)};
	vk_pipeline_set *ps		 = NULL;
	if (vk_ensure_attention_pipeline(p, head_dim, &ps) != OK)
		return ERR_UNSUPPORTED;
	VkDeviceSize offs[5] = {q->offset, 0, 0, 0, out->offset};
	return vk_dispatch_ex(p, ps, bufs, offs, NULL, 5, &push, sizeof(push), (uint32_t)n_heads, 1,
						  0x18);
}

static status_code vk_attention(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
								int n_kv_heads, int head_dim, int n_ctx, int flash_attn,
								float scale, int n_kv_heads_active) {
	return vk_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							 head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, "attention", 0,
							 0, pos + 1);
}

static status_code vk_add_inplace(backend *self, buffer *x, const buffer *y, int n) {
	return vk_add_batch(self, x, y, n, 1);
}

static status_code vk_scale_inplace(backend *self, buffer *x, float scale, int n) {
	vk_priv *p = self->priv;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push				 = {n, ELEM_MODE_SCALE_INPLACE, scale, 0, 1};
	vk_buf		*dummy	 = vk_dummy_buf(p);
	vk_buf		*bufs[3] = {as_vkbuf(x), dummy, dummy};
	VkDeviceSize offs[3] = {x->offset, 0, 0};
	uint32_t	 groups	 = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, 1, 0x1);
}

static status_code vk_softcap(backend *self, buffer *x, float cap, int n) {
	vk_priv *p = self->priv;
	if (cap <= 0.0f || n <= 0)
		return OK;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push				 = {n, ELEM_MODE_SOFTCAP, cap, 0, 1};
	vk_buf		*dummy	 = vk_dummy_buf(p);
	vk_buf		*bufs[3] = {as_vkbuf(x), dummy, dummy};
	VkDeviceSize offs[3] = {x->offset, 0, 0};
	uint32_t	 groups	 = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, 1, 0x1);
}

static status_code vk_split_qgate(backend *self, const buffer *mixed, buffer *q, buffer *gate,
								  int n_heads, int head_dim, int n_rows) {
	vk_priv *p = self->priv;
	if (n_rows <= 0 || n_heads <= 0 || head_dim <= 0)
		return OK;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;

	int n = n_heads * head_dim;
	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push				  = {n, ELEM_MODE_SPLIT_QGATE, 0.0f, head_dim, n_rows};
	vk_buf		*bufs[3]  = {as_vkbuf(q), as_vkbuf(mixed), as_vkbuf(gate)};
	VkDeviceSize offs[3]  = {q->offset, mixed->offset, gate->offset};
	uint32_t	 groups_x = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups_x, (uint32_t)n_rows, 0x5);
}

static status_code vk_attn_output_gate(backend *self, buffer *out, const buffer *gate, int n,
									   int n_rows) {
	vk_priv *p = self->priv;
	if (n <= 0 || n_rows <= 0)
		return OK;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;

	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push				  = {n, ELEM_MODE_ATTN_OUTPUT_GATE, 0.0f, 0, n_rows};
	vk_buf		*dummy	  = vk_dummy_buf(p);
	vk_buf		*bufs[3]  = {as_vkbuf(out), as_vkbuf(gate), dummy};
	VkDeviceSize offs[3]  = {out->offset, gate->offset, 0};
	uint32_t	 groups_x = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups_x, (uint32_t)n_rows, 0x1);
}

static status_code vk_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	vk_priv *p = self->priv;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push				 = {n, ELEM_MODE_COPY, 0.0f, 0, 1};
	vk_buf		*dummy	 = vk_dummy_buf(p);
	vk_buf		*bufs[3] = {as_vkbuf(dst), as_vkbuf(src), dummy};
	VkDeviceSize offs[3] = {dst->offset, src->offset, 0};
	uint32_t	 groups	 = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, 1, 0x1);
}

static status_code vk_elementwise(backend *self, buffer *dst, const buffer *src1,
								  const buffer *src2, int n, int mode, int m) {
	vk_priv *p = self->priv;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	int max_rows = (int)p->caps.max_workgroup_count[1];
	if (m > max_rows) {
		size_t		row_bytes = (size_t)n * sizeof(float);
		status_code st		  = OK;
		for (int r = 0; r < m && st == OK; r += max_rows) {
			int			  rows = MIN(max_rows, m - r);
			buffer		  d	   = buffer_rows(dst, r, rows, row_bytes);
			buffer		  s1;
			const buffer *s1p = src1;
			if (src1 && mode != ELEM_MODE_ADD_BIAS) {
				s1	= buffer_rows(src1, r, rows, row_bytes);
				s1p = &s1;
			}
			st = vk_elementwise(self, &d, s1p, src2, n, mode, rows);
		}
		return st;
	}
	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push			= {n, mode, 0.0f, 0, m};
	vk_buf *dummy	= vk_dummy_buf(p);
	vk_buf *bufs[3] = {as_vkbuf(dst), src1 ? as_vkbuf(src1) : dummy, src2 ? as_vkbuf(src2) : dummy};
	VkDeviceSize offs[3] = {dst->offset, src1 ? src1->offset : 0, src2 ? src2->offset : 0};
	uint32_t	 groups	 = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, (uint32_t)m, 0x1);
}

static status_code vk_activate(backend *self, buffer *x, int n, int activation) {
	return vk_elementwise(self, x, NULL, NULL, n,
						  activation ? ELEM_MODE_ACT_GELU : ELEM_MODE_ACT_SILU, 1);
}

static status_code vk_add_bias(backend *self, buffer *x, const buffer *bias, int rows, int dim) {
	return vk_elementwise(self, x, bias, NULL, dim, ELEM_MODE_ADD_BIAS, rows);
}

static status_code vk_gated_add(backend *self, buffer *x, const buffer *y, const buffer *gate,
								int rows, int dim) {
	return vk_elementwise(self, x, y, gate, dim, ELEM_MODE_GATED_ADD, rows);
}

static status_code vk_nn_dispatch(backend *self, vk_nn_pipe pipe, vk_buf **bufs,
								  const VkDeviceSize *offs, int n_bufs, const void *push,
								  uint32_t push_size, uint32_t groups, uint32_t write_mask) {
	vk_priv			*p	= self->priv;
	vk_pipeline_set *ps = &p->p_nn[pipe];
	if (!ps->pipeline)
		return ERR_UNSUPPORTED;
	return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, n_bufs, push, push_size, groups, 1,
							 write_mask);
}

static status_code vk_layernorm_modulate(backend *self, const buffer *x, const buffer *scale,
										 buffer *y, int rows, int dim, float eps) {
	struct {
		int32_t dim, rows;
		float	eps;
	} push				 = {dim, rows, eps};
	vk_buf		*bufs[3] = {as_vkbuf(x), as_vkbuf(scale), as_vkbuf(y)};
	VkDeviceSize offs[3] = {x->offset, scale->offset, y->offset};
	return vk_nn_dispatch(self, VK_NN_LAYERNORM_MODULATE, bufs, offs, 3, &push, sizeof(push),
						  (uint32_t)rows, 1u << 2);
}

static status_code vk_rope_table(backend *self, buffer *x, const buffer *table, int rows, int heads,
								 int head_dim) {
	struct {
		int32_t rows, heads, head_dim;
	} push				 = {rows, heads, head_dim};
	vk_buf		*bufs[2] = {as_vkbuf(x), as_vkbuf(table)};
	VkDeviceSize offs[2] = {x->offset, table->offset};
	uint32_t	 total	 = (uint32_t)rows * heads * (head_dim / 2);
	return vk_nn_dispatch(self, VK_NN_ROPE_TABLE, bufs, offs, 2, &push, sizeof(push),
						  (total + 127) / 128, 1u << 0);
}

static status_code vk_attention_dense(backend *self, const dense_attention_desc *d) {
	if (d->head_dim > VK_DENSE_ATTN_MAX_HEAD_DIM)
		return ERR_UNSUPPORTED;
	vk_priv *p = self->priv;
	struct {
		int32_t n_heads, head_dim, n0, n1, causal;
	} push				 = {d->n_heads, d->head_dim, d->n0, d->n1, d->causal};
	vk_buf		*dummy	 = vk_dummy_buf(p);
	vk_buf		*bufs[6] = {as_vkbuf(d->q),
							as_vkbuf(d->k0),
							as_vkbuf(d->v0),
							d->n1 ? as_vkbuf(d->k1) : dummy,
							d->n1 ? as_vkbuf(d->v1) : dummy,
							as_vkbuf(d->out)};
	VkDeviceSize offs[6] = {d->q->offset,
							d->k0->offset,
							d->v0->offset,
							d->n1 ? d->k1->offset : 0,
							d->n1 ? d->v1->offset : 0,
							d->out->offset};
	return vk_nn_dispatch(self, VK_NN_ATTENTION_DENSE, bufs, offs, 6, &push, sizeof(push),
						  (uint32_t)d->n_q * d->n_heads, 1u << 5);
}

static status_code vk_im2col3x3(backend *self, const im2col_desc *d) {
	struct {
		int32_t first_pixel, n_pixels, width, in_height, in_width, channels, shift;
	} push				 = {d->first_pixel, d->n_pixels, d->width, d->in_height,
							d->in_width,	d->channels, d->shift};
	vk_buf		*bufs[2] = {as_vkbuf(d->in), as_vkbuf(d->col)};
	VkDeviceSize offs[2] = {d->in->offset, d->col->offset};
	uint32_t	 total	 = (uint32_t)d->n_pixels * 9u * (uint32_t)d->channels;
	return vk_nn_dispatch(self, VK_NN_IM2COL, bufs, offs, 2, &push, sizeof(push),
						  (total + 127) / 128, 1u << 1);
}

static status_code vk_upsample_shortcut_add(backend *self, const upsample_shortcut_desc *d) {
	struct {
		int32_t height, width, in_channels, out_channels, repeats, frames_up, factor;
	} push				 = {d->height,	d->width,	  d->in_channels, d->out_channels,
							d->repeats, d->frames_up, d->factor};
	vk_buf		*bufs[2] = {as_vkbuf(d->in), as_vkbuf(d->out)};
	VkDeviceSize offs[2] = {d->in->offset, d->out->offset};
	uint32_t	 total =
		(uint32_t)d->height * d->factor * d->width * d->factor * (uint32_t)d->out_channels;
	return vk_nn_dispatch(self, VK_NN_UPSAMPLE_SHORTCUT, bufs, offs, 2, &push, sizeof(push),
						  (total + 127) / 128, 1u << 1);
}

static status_code vk_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
								  float combine_scale) {
	vk_priv *p = self->priv;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push				 = {n, ELEM_MODE_PLE_COMBINE, combine_scale, 0, 1};
	vk_buf		*dummy	 = vk_dummy_buf(p);
	vk_buf		*bufs[3] = {as_vkbuf(ple), as_vkbuf(proj), dummy};
	VkDeviceSize offs[3] = {ple->offset, proj->offset, 0};
	uint32_t	 groups	 = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, 1, 0x1);
}

static status_code vk_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n) {
	return vk_ffn_activate_batch_scales(self, gate, up, out, n, 0, 1, 1.0f, 1.0f);
}

static status_code vk_matmul_residual(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *x, const buffer *residual, buffer *y, int n,
									  int k) {
	return vk_matmul_impl(self, w, w_type, x, residual, y, n, k);
}

static void vk_synchronize(backend *self) {
	vk_priv *p = self->priv;
	if (!p->dev)
		return;

	vk_ring_slot *r = &p->ring[p->ring_cur];
	if (p->device_lost) {
		if (r->has_work) {
			r->has_work = 0;
			vk_dirty_clear(p);
		}
		if (!p->device_lost_warned) {
			p->device_lost_warned = 1;
			ERROR("vk: vk_synchronize: device is lost; pending work "
				  "was discarded. Subsequent device dispatches will be skipped "
				  "and CPU fallbacks will be used where possible.");
		}
		return;
	}
	if (r->has_work) {
		status_code fs = vk_flush(p);
		if (fs != OK) {
			ERROR("vk: vk_synchronize: flush failed (%d=%s) -- "
				  "pending work was NOT executed",
				  (int)fs, status_str(fs));
		}
	}
}

static void vk_submit(backend *self) {
	vk_priv *p = self->priv;
	if (!p->dev || p->device_lost || !p->timeline_supported)
		return;

	vk_ring_slot *r = &p->ring[p->ring_cur];
	if (!r->has_work)
		return;

	status_code s = vk_run_cmd(p);
	if (s != OK)
		ERROR("vk: vk_submit: submit failed (%s)", status_str(s));
}

static void vk_begin_batch(backend *self) {
	vk_priv *p		= self->priv;
	p->batch_active = 1;
}

static void vk_end_batch(backend *self) {
	vk_priv *p		= self->priv;
	p->batch_active = 0;
	vk_ring_slot *r = &p->ring[p->ring_cur];
	if (p->device_lost) {
		if (r->has_work) {
			r->has_work = 0;
			vk_dirty_clear(p);
		}
		if (!p->device_lost_warned) {
			p->device_lost_warned = 1;
			ERROR("vk: vk_end_batch: device is lost; batch was discarded.");
		}
		return;
	}
	if (r->has_work) {
		status_code fs = p->tok_pipe ? vk_run_cmd(p) : vk_flush(p);
		if (fs != OK) {
			ERROR("vk: vk_end_batch: flush failed (%s) -- "
				  "batch was NOT submitted to the device",
				  status_str(fs));
		}
	}
}

static status_code vk_ensure_token_regs(vk_priv *p) {
	if (p->tok_reg_buf.buf && p->tok_rb_buf.buf)
		return OK;
	const size_t bytes = (BACKEND_TOKEN_REGS + 1) * sizeof(uint32_t);
	status_code	 s	   = vk_alloc_buffer(p, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
										 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &p->tok_reg_buf);
	if (s != OK)
		return s;
	return vk_alloc_buffer(
		p, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &p->tok_rb_buf);
}

static status_code vk_argmax_dispatch(vk_priv *p, const buffer *logits, int n, int slot) {
	status_code s = vk_ensure_token_regs(p);
	if (s != OK)
		return s;

	const int ARGMAX_WG_CHUNK	= 8192;
	const int ARGMAX_MAX_GROUPS = 256;
	int		  groups			= (n + ARGMAX_WG_CHUNK - 1) / ARGMAX_WG_CHUNK;
	if (groups > ARGMAX_MAX_GROUPS)
		groups = ARGMAX_MAX_GROUPS;
	if (groups < 1)
		groups = 1;

	size_t partial_bytes = (size_t)groups * 2 * sizeof(uint32_t);
	if (!p->argmax_partial_buf.buf || p->argmax_partial_cap < partial_bytes) {
		if (p->argmax_partial_buf.buf)
			vk_free_buffer(p, &p->argmax_partial_buf);
		VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		s = vk_alloc_buffer(p, partial_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
							&p->argmax_partial_buf);
		if (s != OK)
			return s;
		p->argmax_partial_cap = partial_bytes;
	}

	struct {
		int32_t n;
		int32_t groups;
	} push				   = {n, groups};
	vk_buf *stage1_bufs[2] = {as_vkbuf(logits), &p->argmax_partial_buf};
	s = vk_dispatch(p, &p->p_argmax, stage1_bufs, 2, &push, sizeof(push), groups);
	if (s != OK)
		return s;

	struct {
		int32_t n;
		int32_t slot;
	} reduce_push		   = {groups, slot};
	vk_buf *stage2_bufs[3] = {&p->argmax_partial_buf, &p->tok_reg_buf, &p->tok_rb_buf};
	return vk_dispatch_ex(p, &p->p_argmax_reduce, stage2_bufs, NULL, NULL, 3, &reduce_push,
						  sizeof(reduce_push), 1, 1, 0x6);
}

static uint32_t vk_token_slot_value(const vk_priv *p, int slot) {
	uint32_t result;
	memcpy(&result, (const uint32_t *)p->tok_rb_buf.mapped + slot, sizeof(result));
	return result;
}

static status_code vk_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	vk_priv	   *p = self->priv;
	status_code s = vk_argmax_dispatch(p, logits, n, BACKEND_TOKEN_REGS);
	if (s != OK)
		return s;
	s = vk_flush(p);
	if (s != OK)
		return s;
	*out_idx = (int32_t)vk_token_slot_value(p, BACKEND_TOKEN_REGS);
	return OK;
}

static status_code vk_argmax_to_reg(backend *self, const buffer *logits, int n, int reg) {
	vk_priv *p = self->priv;
	if (reg < 0 || reg >= BACKEND_TOKEN_REGS || !p->timeline_supported)
		return ERR_INVALID_ARG;
	status_code s = vk_argmax_dispatch(p, logits, n, reg);
	if (s != OK)
		return s;
	s = vk_run_cmd(p);
	if (s != OK)
		return s;
	p->tok_signal[reg] = p->last_signal_value;
	p->tok_pipe		   = 1;
	return OK;
}

static status_code vk_read_token_reg(backend *self, int reg, int32_t *out_token) {
	vk_priv *p = self->priv;
	if (reg < 0 || reg >= BACKEND_TOKEN_REGS || !p->tok_signal[reg])
		return ERR_INVALID_ARG;
	p->tok_pipe	   = 0;
	status_code ws = vk_timeline_wait(p, p->tok_signal[reg]);
	if (ws != OK)
		return ws;
	p->tok_signal[reg] = 0;
	*out_token		   = (int32_t)vk_token_slot_value(p, reg);
	return OK;
}

static status_code vk_rmsnorm_per_head(backend *self, const buffer *x, const buffer *w, buffer *y,
									   int n_heads, int head_dim, float eps) {
	return vk_rmsnorm_per_head_batch(self, x, w, y, n_heads, head_dim, eps, 1);
}

static status_code vk_rmsnorm_noweight_per_head(backend *self, const buffer *x, buffer *y,
												int n_heads, int head_dim, float eps) {
	return vk_rmsnorm_noweight_per_head_batch(self, x, y, n_heads, head_dim, eps, 1);
}

static status_code vk_rmsnorm_noweight(backend *self, const buffer *x, buffer *y, int n,
									   float eps) {
	return vk_rmsnorm_noweight_batch(self, x, y, n, eps, 1);
}

static status_code vk_rmsnorm_add_batch(backend *self, const buffer *x, const buffer *w,
										const buffer *residual, buffer *y, int n, float eps,
										float out_scale, int m) {
	vk_priv *p = self->priv;

	if ((!w->handle && w->host_ptr) || !p->p_rmsnorm_add_batch.pipeline) {
		for (int row = 0; row < m; row++) {
			buffer xr = *x, rr = *residual, yr = *y;
			xr.offset += (size_t)row * (size_t)n * sizeof(float);
			rr.offset += (size_t)row * (size_t)n * sizeof(float);
			yr.offset += (size_t)row * (size_t)n * sizeof(float);
			status_code s = vk_rmsnorm(self, &xr, w, &yr, n, eps);
			if (s != OK)
				return s;
			s = vk_add_inplace(self, &yr, &rr, n);
			if (s != OK)
				return s;
			if (out_scale != 1.0f) {
				s = vk_scale_inplace(self, &yr, out_scale, n);
				if (s != OK)
					return s;
			}
		}
		return OK;
	}
	struct {
		int32_t n;
		float	eps;
		int32_t m;
		float	os;
	} push				 = {n, eps, m, out_scale};
	vk_buf		*bufs[4] = {as_vkbuf(x), as_vkbuf(w), as_vkbuf(residual), as_vkbuf(y)};
	VkDeviceSize offs[4] = {x->offset, w->offset, residual->offset, y->offset};
	return vk_dispatch_2d_ex(p, &p->p_rmsnorm_add_batch, bufs, offs, NULL, 4, &push, sizeof(push),
							 (uint32_t)1, (uint32_t)m, 1u << 3);
}

static status_code vk_rmsnorm_add_norm_batch(backend *self, const buffer *x, const buffer *w_add,
											 const buffer *residual, buffer *y_add,
											 const buffer *w_norm, buffer *y_norm, int n, float eps,
											 float out_scale, int m) {
	vk_priv *p = self->priv;
	if (!p->p_rmsnorm_add_norm.pipeline || !p->qcore_ok || n % 32 != 0 || n > 4096 ||
		!w_add->handle || !w_norm->handle)
		return ERR_UNSUPPORTED;
	size_t		need = (size_t)m * (size_t)(n / 32) * 12 * sizeof(uint32_t);
	status_code s	 = vk_ensure_scratch(self, &p->xq_buf[0], &p->xq_cap[0], need);
	if (s != OK)
		return s;
	struct {
		int32_t n;
		float	eps;
		int32_t m;
		float	os;
	} push			= {n, eps, m, out_scale};
	vk_buf *bufs[7] = {as_vkbuf(x),		 as_vkbuf(w_add),  as_vkbuf(residual),	   as_vkbuf(y_add),
					   as_vkbuf(w_norm), as_vkbuf(y_norm), as_vkbuf(&p->xq_buf[0])};
	VkDeviceSize offs[7] = {x->offset,		w_add->offset,	residual->offset,	y_add->offset,
							w_norm->offset, y_norm->offset, p->xq_buf[0].offset};
	s = vk_dispatch_2d_ex(p, &p->p_rmsnorm_add_norm, bufs, offs, NULL, 7, &push, sizeof(push), 1,
						  (uint32_t)m, 0x68);
	if (s != OK)
		return s;
	p->xq_fused.buf = as_vkbuf(y_norm);
	p->xq_fused.off = y_norm->offset;
	p->xq_fused.k	= n;
	p->xq_fused.m	= m;
	p->xq_fused.seq = p->stream_seq;
	return OK;
}

static status_code vk_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
								  const buffer *residual, buffer *y, int n, float eps,
								  float out_scale) {
	return vk_rmsnorm_add_batch(self, x, w, residual, y, n, eps, out_scale, 1);
}

static status_code vk_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
									  buffer *out, int n, int activation) {
	return vk_ffn_activate_batch_scales(self, gate, up, out, n, activation, 1, 1.0f, 1.0f);
}

static status_code vk_rope_ext(backend *self, buffer *vec, int n_heads, int head_dim, int pos,
							   const float *rope_cos_base, const float *rope_sin_base,
							   const float *freq_factors) {
	return vk_rope_ext_batch(self, vec, n_heads, head_dim, pos, rope_cos_base, rope_sin_base,
							 freq_factors, 1);
}

static status_code vk_attention_swa(backend *self, const buffer *q, const buffer *k_cache,
									const buffer *v_cache, buffer *out, int layer, int pos,
									int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									int flash_attn, float scale, int sliding_window,
									int n_kv_heads_active) {
	int n_pos	   = pos + 1;
	int attn_start = 0;
	if (sliding_window > 0 && n_pos > sliding_window)
		attn_start = n_pos - sliding_window;
	return vk_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							 head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, "attention_swa",
							 sliding_window, attn_start, n_pos - attn_start);
}

static vk_pipeline_set *vk_matmul_batch_pipeline(vk_priv *p, uint32_t w_type, int residual,
												 int use_wide) {
	switch (w_type) {
	case GGML_TYPE_Q4_0:
		return residual ? &p->p_matmul_q4_0_res_batch : &p->p_matmul_q4_0_batch;
	case GGML_TYPE_Q4_1:
		return residual ? &p->p_matmul_q4_1_res_batch : &p->p_matmul_q4_1_batch;
	case GGML_TYPE_Q5_0:
		return residual ? &p->p_matmul_q5_0_res_batch : &p->p_matmul_q5_0_batch;
	case GGML_TYPE_Q5_1:
		return residual ? &p->p_matmul_q5_1_res_batch : &p->p_matmul_q5_1_batch;
	case GGML_TYPE_Q8_0:
		return residual ? &p->p_matmul_q8_0_res_batch : &p->p_matmul_q8_0_batch;
	case GGML_TYPE_F32:
		if (use_wide)
			return residual ? &p->p_matmul_f32_wide_res_batch : &p->p_matmul_f32_wide_batch;
		return residual ? &p->p_matmul_f32_res_batch : &p->p_matmul_f32_batch;
	case GGML_TYPE_F16:
		if (use_wide)
			return residual ? &p->p_matmul_f16_wide_res_batch : &p->p_matmul_f16_wide_batch;
		return residual ? &p->p_matmul_f16_res_batch : &p->p_matmul_f16_batch;
	case GGML_TYPE_BF16:
		if (use_wide)
			return residual ? &p->p_matmul_bf16_wide_res_batch : &p->p_matmul_bf16_wide_batch;
		return residual ? &p->p_matmul_bf16_res_batch : &p->p_matmul_bf16_batch;
	case GGML_TYPE_Q4_K:
		return residual ? &p->p_matmul_q4_k_res_batch : &p->p_matmul_q4_k_batch;
	case GGML_TYPE_Q5_K:
		return residual ? &p->p_matmul_q5_k_res_batch : &p->p_matmul_q5_k_batch;
	case GGML_TYPE_Q6_K:
		return residual ? &p->p_matmul_q6_k_res_batch : &p->p_matmul_q6_k_batch;
	case GGML_TYPE_IQ3_S:
		return residual ? &p->p_matmul_iq3_s_res_batch : &p->p_matmul_iq3_s_batch;
	case GGML_TYPE_IQ2_S:
		return residual ? &p->p_matmul_iq2_s_res_batch : &p->p_matmul_iq2_s_batch;
	case GGML_TYPE_IQ2_XXS:
		return residual ? &p->p_matmul_iq2_xxs_res_batch : &p->p_matmul_iq2_xxs_batch;
	case GGML_TYPE_IQ3_XXS:
		return residual ? &p->p_matmul_iq3_xxs_res_batch : &p->p_matmul_iq3_xxs_batch;
	case GGML_TYPE_IQ4_XS:
		return residual ? &p->p_matmul_iq4_xs_res_batch : &p->p_matmul_iq4_xs_batch;
	default:
		return NULL;
	}
}

static status_code vk_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								   buffer *y, int n, int k, int m) {
	vk_priv *p = self->priv;

	if (m <= 0)
		return ERR_INVALID_ARG;
	if (n <= 0 || k <= 0)
		return OK;

	int max_rows = (int)p->caps.max_workgroup_count[1];
	if (m > max_rows) {
		status_code st = OK;
		for (int r = 0; r < m && st == OK; r += max_rows) {
			int	   rows = MIN(max_rows, m - r);
			buffer xs	= buffer_rows(x, r, rows, (size_t)k * sizeof(float));
			buffer ys	= buffer_rows(y, r, rows, (size_t)n * sizeof(float));
			st			= vk_matmul_batch(self, w, w_type, &xs, &ys, n, k, rows);
		}
		return st;
	}

	if (!w->handle && w->host_ptr) {
		status_code st = OK;
		for (int row = 0; row < m && st == OK; row++) {
			buffer x_row =
				buffer_slice(x, (size_t)row * k * sizeof(float), (size_t)k * sizeof(float));
			buffer y_row =
				buffer_slice(y, (size_t)row * n * sizeof(float), (size_t)n * sizeof(float));
			st = vk_matmul_impl(self, w, w_type, &x_row, NULL, &y_row, n, k);
		}
		return st;
	}

	int qt = vk_qcore_type(p, w_type, k, m);
	if (qt >= 0) {
		status_code qs = vk_qcore_run(self, qt, w, x, NULL, -1, y, NULL, n, k, m);
		if (qs != ERR_UNSUPPORTED)
			return qs;
	}

	if (w_type == GGML_TYPE_IQ4_NL && p->p_matmul_iq4_nl_batch.pipeline && k > 0 && (k % 32) == 0) {
		vk_buf *dummy = vk_dummy_buf(p);
		struct {
			int32_t mode, nn, n1, kk, activation, has_residual, n2, m;
		} push				  = {0, n, 0, k, 0, 0, 0, m};
		vk_buf		*bufs[7]  = {as_vkbuf(w), as_vkbuf(x), dummy, as_vkbuf(y), dummy, dummy, dummy};
		VkDeviceSize offs[7]  = {w->offset, x->offset, 0, y->offset, 0, 0, 0};
		int			 rows	  = p->matmul_rows_per_thread;
		int			 wg		  = p->matmul_wg_size;
		uint32_t	 groups_x = (uint32_t)((n + (wg * rows) - 1) / (wg * rows));
		return vk_dispatch_2d_ex(p, &p->p_matmul_iq4_nl_batch, bufs, offs, NULL, 7, &push,
								 sizeof(push), groups_x, (uint32_t)m, 1u << 3);
	}

	int is_iqx_grid =
		(w_type == GGML_TYPE_IQ2_S || w_type == GGML_TYPE_IQ2_XXS || w_type == GGML_TYPE_IQ3_XXS);
	int is_kquant =
		(w_type == GGML_TYPE_Q4_K || w_type == GGML_TYPE_Q5_K || w_type == GGML_TYPE_Q6_K ||
		 w_type == GGML_TYPE_IQ3_S || is_iqx_grid || w_type == GGML_TYPE_IQ4_XS);
	int unified_push =
		(w_type == GGML_TYPE_Q4_0 || w_type == GGML_TYPE_Q4_K || w_type == GGML_TYPE_Q6_K ||
		 w_type == GGML_TYPE_IQ3_S || is_iqx_grid || w_type == GGML_TYPE_IQ4_XS);

	uint32_t groups_x;
	if (is_kquant) {
		groups_x = (uint32_t)((n + p->matmul_wg_size - 1) / p->matmul_wg_size);
	} else {
		groups_x = (uint32_t)vk_matmul_groups_x(p, n, w_type);
	}

	int is_dense = (w_type == GGML_TYPE_F32 || w_type == GGML_TYPE_F16 || w_type == GGML_TYPE_BF16);
	int m_per_wg = p->matmul_m_per_wg > 0 ? p->matmul_m_per_wg : 1;
	int min_total_groups = 128;
	int wide_groups_y	 = (m + m_per_wg - 1) / m_per_wg;
	int use_wide = is_dense && m_per_wg > 1 && (int)groups_x * wide_groups_y >= min_total_groups;

	vk_pipeline_set *ps = vk_matmul_batch_pipeline(p, w_type, 0, use_wide);
	if (!ps || !ps->pipeline) {
		use_wide = 0;
		ps		 = vk_matmul_batch_pipeline(p, w_type, 0, 0);
	}
	if (!ps || !ps->pipeline)
		return ERR_UNSUPPORTED;

	vk_buf		*bufs[4] = {as_vkbuf(w), as_vkbuf(x), as_vkbuf(y)};
	VkDeviceSize offs[4] = {w->offset, x->offset, y->offset};
	int			 n_bufs	 = 3;
	if (w_type == GGML_TYPE_IQ3_S && p->iq3s_grid_buf.buf) {
		bufs[n_bufs] = &p->iq3s_grid_buf;
		offs[n_bufs] = 0;
		n_bufs++;
	}
	if (is_iqx_grid && p->iqx_grid_buf.buf) {
		bufs[n_bufs] = &p->iqx_grid_buf;
		offs[n_bufs] = 0;
		n_bufs++;
	}

	uint32_t groups_y = (uint32_t)m;
	if (use_wide)
		groups_y = (uint32_t)wide_groups_y;

	if (unified_push) {
		struct {
			int32_t n0, n1, k, m;
		} push = {n, 0, k, m};
		return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, n_bufs, &push, sizeof(push), groups_x,
								 groups_y, 1u << 2);
	} else {
		struct {
			int32_t n, k, m;
		} push = {n, k, m};
		return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, n_bufs, &push, sizeof(push), groups_x,
								 groups_y, 1u << 2);
	}
}

static status_code vk_matmul_multi_batch(backend *self, const buffer **w, const uint32_t *w_types,
										 const buffer *x, buffer **y, const int *n_list, int k,
										 int n_matmuls, int m) {
	vk_priv *p = self->priv;

	if (n_matmuls < 1)
		return ERR_INVALID_ARG;
	if (n_matmuls == 1)
		return vk_matmul_batch(self, w[0], w_types[0], x, y[0], n_list[0], k, m);

	int any_cpu = 0;
	for (int i = 0; i < n_matmuls; i++) {
		if (!w[i]->handle && w[i]->host_ptr) {
			any_cpu = 1;
			break;
		}
	}
	if (any_cpu) {
		status_code st = OK;
		for (int row = 0; row < m && st == OK; row++) {
			buffer x_row =
				buffer_slice(x, (size_t)row * k * sizeof(float), (size_t)k * sizeof(float));
			for (int i = 0; i < n_matmuls && st == OK; i++) {
				buffer y_row = buffer_slice(y[i], (size_t)row * n_list[i] * sizeof(float),
											(size_t)n_list[i] * sizeof(float));
				st			 = vk_matmul(self, w[i], w_types[i], &x_row, &y_row, n_list[i], k);
			}
		}
		return st;
	}

	int qts[8];
	int all_q = n_matmuls <= 8;
	int fams  = 0;
	for (int i = 0; all_q && i < n_matmuls; i++) {
		if (n_list[i] <= 0)
			continue;
		qts[i] = vk_qcore_type(p, w_types[i], k, m);
		all_q  = qts[i] >= 0;
		if (all_q)
			fams |= 1 << vk_qfamily(qts[i]);
	}
	if (all_q && fams) {
		status_code st = OK;
		for (int f = 0; f < 2 && st == OK; f++)
			if (fams & (1 << f))
				st = vk_qcore_quantize(self, f, x, NULL, -1, k, m);
		for (int i = 0; i < n_matmuls && st == OK; i++)
			if (n_list[i] > 0)
				st = vk_qcore_matmul(p, qts[i], w[i], x, y[i], NULL, n_list[i], k, m);
		if (st != ERR_UNSUPPORTED)
			return st;
	}

	if (n_matmuls == 3 && w_types[0] == w_types[1] && w_types[1] == w_types[2] &&
		w_types[0] == GGML_TYPE_Q4_0 && p->p_matmul_q4_0_triple_batch.pipeline && k > 0) {
		int n0 = n_list[0], n1 = n_list[1], n2 = n_list[2];
		int n_max = n0 > n1 ? n0 : n1;
		n_max	  = n_max > n2 ? n_max : n2;
		struct {
			int32_t n0, n1, n2, k, m;
		} push				  = {n0, n1, n2, k, m};
		vk_buf		*bufs[7]  = {as_vkbuf(w[0]), as_vkbuf(w[1]), as_vkbuf(w[2]), as_vkbuf(x),
								 as_vkbuf(y[0]), as_vkbuf(y[1]), as_vkbuf(y[2])};
		VkDeviceSize offs[7]  = {w[0]->offset, w[1]->offset, w[2]->offset, x->offset,
								 y[0]->offset, y[1]->offset, y[2]->offset};
		int			 rows	  = p->matmul_rows_per_thread;
		int			 wg		  = p->matmul_wg_size;
		uint32_t	 groups_x = (uint32_t)((n_max + (wg * rows) - 1) / (wg * rows));
		uint32_t	 wmask	  = (1u << 4) | (1u << 5) | (1u << 6);
		return vk_dispatch_2d_ex(p, &p->p_matmul_q4_0_triple_batch, bufs, offs, NULL, 7, &push,
								 sizeof(push), groups_x, (uint32_t)m, wmask);
	}

	if (n_matmuls == 2 && w_types[0] == w_types[1]) {
		uint32_t wt = w_types[0];

		vk_pipeline_set *ps = NULL;
		switch (wt) {
		case GGML_TYPE_Q4_0:
			ps = &p->p_matmul_q4_0_dual_batch;
			break;
		case GGML_TYPE_Q4_K:
			ps = &p->p_matmul_q4_k_dual_batch;
			break;
		case GGML_TYPE_Q6_K:
			ps = &p->p_matmul_q6_k_dual_batch;
			break;
		default:
			break;
		}

		if (ps && ps->pipeline && k > 0) {
			int n0	  = n_list[0];
			int n1	  = n_list[1];
			int n_max = (n0 > n1) ? n0 : n1;
			struct {
				int32_t n0, n1, k, m;
			} push			= {n0, n1, k, m};
			vk_buf *bufs[5] = {
				as_vkbuf(w[0]), as_vkbuf(w[1]), as_vkbuf(x), as_vkbuf(y[0]), as_vkbuf(y[1]),
			};
			VkDeviceSize offs[5] = {
				w[0]->offset, w[1]->offset, x->offset, y[0]->offset, y[1]->offset,
			};
			uint32_t groups_x;
			if (wt == GGML_TYPE_Q4_K || wt == GGML_TYPE_Q6_K) {
				groups_x = (uint32_t)((n_max + p->matmul_wg_size - 1) / p->matmul_wg_size);
			} else {
				int rows = p->matmul_rows_per_thread;
				int wg	 = p->matmul_wg_size;
				groups_x = (uint32_t)((n_max + (wg * rows) - 1) / (wg * rows));
			}
			uint32_t wmask = (1u << 3) | (1u << 4);
			return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, 5, &push, sizeof(push), groups_x,
									 (uint32_t)m, wmask);
		}

		if (wt == GGML_TYPE_IQ4_NL && p->p_matmul_iq4_nl_batch.pipeline && k > 0 && (k % 32) == 0) {
			vk_buf *dummy = vk_dummy_buf(p);
			struct {
				int32_t mode, nn, n1, kk, activation, has_residual, n2, m;
			} push			= {1, n_list[0], n_list[1], k, 0, 0, 0, m};
			vk_buf *bufs[7] = {
				as_vkbuf(w[0]), as_vkbuf(w[1]), as_vkbuf(x), as_vkbuf(y[0]),
				as_vkbuf(y[1]), dummy,			dummy,
			};
			VkDeviceSize offs[7] = {
				w[0]->offset, w[1]->offset, x->offset, y[0]->offset, y[1]->offset, 0, 0,
			};
			int		 rows	  = p->matmul_rows_per_thread;
			int		 wg		  = p->matmul_wg_size;
			int		 n_max	  = (n_list[0] > n_list[1]) ? n_list[0] : n_list[1];
			uint32_t groups_x = (uint32_t)((n_max + (wg * rows) - 1) / (wg * rows));
			uint32_t wmask	  = (1u << 3) | (1u << 4);
			return vk_dispatch_2d_ex(p, &p->p_matmul_iq4_nl_batch, bufs, offs, NULL, 7, &push,
									 sizeof(push), groups_x, (uint32_t)m, wmask);
		}
	}

	status_code st = OK;
	for (int i = 0; i < n_matmuls && st == OK; i++)
		st = vk_matmul_batch(self, w[i], w_types[i], x, y[i], n_list[i], k, m);
	return st;
}

static status_code vk_rmsnorm_quant_batch(backend *self, const buffer *x, const buffer *w,
										  buffer *y, int n, float eps, int m) {
	vk_priv	   *p	 = self->priv;
	size_t		need = (size_t)m * (size_t)(n / 32) * 12 * sizeof(uint32_t);
	status_code s	 = vk_ensure_scratch(self, &p->xq_buf[0], &p->xq_cap[0], need);
	if (s != OK)
		return s;
	struct {
		int32_t n;
		float	eps;
		int32_t m;
	} push				 = {n, eps, m};
	vk_buf		*bufs[4] = {as_vkbuf(x), as_vkbuf(w), as_vkbuf(y), as_vkbuf(&p->xq_buf[0])};
	VkDeviceSize offs[4] = {x->offset, w->offset, y->offset, p->xq_buf[0].offset};
	s = vk_dispatch_2d_ex(p, &p->p_rmsnorm_sg_quant, bufs, offs, NULL, 4, &push, sizeof(push), 1,
						  (uint32_t)m, 0xC);
	if (s != OK)
		return s;
	p->xq_fused.buf = as_vkbuf(y);
	p->xq_fused.off = y->offset;
	p->xq_fused.k	= n;
	p->xq_fused.m	= m;
	p->xq_fused.seq = p->stream_seq;
	return OK;
}

static status_code vk_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									int n, float eps, int m) {
	vk_priv *p = self->priv;

	int max_rows = (int)p->caps.max_workgroup_count[1];
	if (m > max_rows) {
		status_code st = OK;
		for (int r = 0; r < m && st == OK; r += max_rows) {
			int	   rows = MIN(max_rows, m - r);
			buffer xs	= buffer_rows(x, r, rows, (size_t)n * sizeof(float));
			buffer ys	= buffer_rows(y, r, rows, (size_t)n * sizeof(float));
			st			= vk_rmsnorm_batch(self, &xs, w, &ys, n, eps, rows);
		}
		return st;
	}

	if (!w->handle && w->host_ptr) {
		backend_report_host_fallback(
			self, "rmsnorm_batch", HFB_BUF_HOST_RESIDENT,
			"norm weight is host-resident; each row downloaded, normalized on "
			"host (cpu), and re-uploaded");
		status_code st = OK;
		for (int row = 0; row < m && st == OK; row++) {
			buffer x_row =
				buffer_slice(x, (size_t)row * n * sizeof(float), (size_t)n * sizeof(float));
			buffer y_row =
				buffer_slice(y, (size_t)row * n * sizeof(float), (size_t)n * sizeof(float));
			float *x_host = xmalloc((size_t)n * sizeof(float));
			st			  = vk_buffer_read_f32(self, &x_row, x_host, n);
			if (st == OK) {
				backend *host = backend_host();
				if (!host || !host->rmsnorm) {
					free(x_host);
					return ERR_UNSUPPORTED;
				}
				buffer xb = {0}, wb = {0}, yb = {0};
				xb.handle = x_host;
				xb.owner  = host;
				xb.size	  = (size_t)n * sizeof(float);
				wb.handle = (void *)w->host_ptr;
				wb.owner  = host;
				yb.handle = x_host;
				yb.owner  = host;
				yb.size	  = (size_t)n * sizeof(float);
				st		  = host->rmsnorm(host, &xb, &wb, &yb, n, eps);
				if (st == OK)
					st = vk_buffer_write_f32(self, &y_row, x_host, n);
			}
			free(x_host);
		}
		return st;
	}

	vk_pipeline_set *ps =
		p->p_rmsnorm_sg_batch.pipeline ? &p->p_rmsnorm_sg_batch : &p->p_rmsnorm_batch;
	if (!ps->pipeline)
		return ERR_UNSUPPORTED;
	if (ps == &p->p_rmsnorm_sg_batch && p->p_rmsnorm_sg_quant.pipeline && p->qcore_ok &&
		n % 32 == 0 && n <= 4096)
		return vk_rmsnorm_quant_batch(self, x, w, y, n, eps, m);
	struct {
		int32_t n;
		float	eps;
		int32_t m;
	} push				 = {n, eps, m};
	vk_buf		*bufs[3] = {as_vkbuf(x), as_vkbuf(w), as_vkbuf(y)};
	VkDeviceSize offs[3] = {x->offset, w->offset, y->offset};
	return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, 3, &push, sizeof(push), 1, (uint32_t)m,
							 1u << 2);
}

static status_code vk_rmsnorm_noweight_batch(backend *self, const buffer *x, buffer *y, int n,
											 float eps, int m) {
	vk_priv			*p	= self->priv;
	vk_pipeline_set *ps = p->p_rmsnorm_noweight_sg_batch.pipeline ? &p->p_rmsnorm_noweight_sg_batch
																  : &p->p_rmsnorm_noweight_batch;
	if (!ps->pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n;
		float	eps;
		int32_t m;
	} push				 = {n, eps, m};
	vk_buf		*bufs[2] = {as_vkbuf(x), as_vkbuf(y)};
	VkDeviceSize offs[2] = {x->offset, y->offset};
	return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, 2, &push, sizeof(push), 1, (uint32_t)m,
							 1u << 1);
}

static status_code vk_rmsnorm_per_head_batch(backend *self, const buffer *x, const buffer *w,
											 buffer *y, int n_heads, int head_dim, float eps,
											 int m) {
	vk_priv *p = self->priv;

	if (!w->handle && w->host_ptr) {
		backend_report_host_fallback(
			self, "rmsnorm_per_head_batch", HFB_BUF_HOST_RESIDENT,
			"norm weight is host-resident; each row downloaded, normalized on "
			"host (cpu), and re-uploaded");
		int			row_stride = n_heads * head_dim;
		status_code st		   = OK;
		for (int row = 0; row < m && st == OK; row++) {
			buffer x_row  = buffer_slice(x, (size_t)row * row_stride * sizeof(float),
										 (size_t)row_stride * sizeof(float));
			buffer y_row  = buffer_slice(y, (size_t)row * row_stride * sizeof(float),
										 (size_t)row_stride * sizeof(float));
			float *x_host = xmalloc((size_t)row_stride * sizeof(float));
			st			  = vk_buffer_read_f32(self, &x_row, x_host, row_stride);
			if (st == OK) {
				backend *host = backend_host();
				if (!host || !host->rmsnorm_per_head) {
					free(x_host);
					return ERR_UNSUPPORTED;
				}
				buffer xb = {0}, wb = {0}, yb = {0};
				xb.handle = x_host;
				xb.owner  = host;
				xb.size	  = (size_t)row_stride * sizeof(float);
				wb.handle = (void *)w->host_ptr;
				wb.owner  = host;
				yb.handle = x_host;
				yb.owner  = host;
				yb.size	  = (size_t)row_stride * sizeof(float);
				st		  = host->rmsnorm_per_head(host, &xb, &wb, &yb, n_heads, head_dim, eps);
				if (st == OK)
					st = vk_buffer_write_f32(self, &y_row, x_host, row_stride);
			}
			free(x_host);
		}
		return st;
	}

	vk_pipeline_set *ps = p->p_rmsnorm_per_head_sg_batch.pipeline ? &p->p_rmsnorm_per_head_sg_batch
																  : &p->p_rmsnorm_per_head_batch;
	if (!ps->pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n_heads;
		int32_t head_dim;
		float	eps;
		int32_t m;
	} push				 = {n_heads, head_dim, eps, m};
	vk_buf		*bufs[3] = {as_vkbuf(x), as_vkbuf(w), as_vkbuf(y)};
	VkDeviceSize offs[3] = {x->offset, w->offset, y->offset};
	return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, 3, &push, sizeof(push), (uint32_t)n_heads,
							 (uint32_t)m, 1u << 2);
}

static status_code vk_rmsnorm_noweight_per_head_batch(backend *self, const buffer *x, buffer *y,
													  int n_heads, int head_dim, float eps, int m) {
	vk_priv			*p	= self->priv;
	vk_pipeline_set *ps = p->p_rmsnorm_noweight_per_head_sg_batch.pipeline
							  ? &p->p_rmsnorm_noweight_per_head_sg_batch
							  : &p->p_rmsnorm_noweight_per_head_batch;
	if (!ps->pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n_heads;
		int32_t head_dim;
		float	eps;
		int32_t m;
	} push				 = {n_heads, head_dim, eps, m};
	vk_buf		*bufs[2] = {as_vkbuf(x), as_vkbuf(y)};
	VkDeviceSize offs[2] = {x->offset, y->offset};
	return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, 2, &push, sizeof(push), (uint32_t)n_heads,
							 (uint32_t)m, 1u << 1);
}

static status_code vk_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	return vk_elementwise(self, x, y, NULL, n, ELEM_MODE_ADD_INPLACE, m);
}

static int vk_dequant_block_elems(uint32_t type, int *out_elems, int *out_block_bytes) {
	switch (type) {
	case GGML_TYPE_F32:
		*out_elems = 1, *out_block_bytes = 4;
		return 1;
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
		*out_elems = 1, *out_block_bytes = 2;
		return 1;
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_IQ4_NL:
		*out_elems = 32, *out_block_bytes = 18;
		return 1;
	case GGML_TYPE_Q4_1:
		*out_elems = 32, *out_block_bytes = 20;
		return 1;
	case GGML_TYPE_Q5_0:
		*out_elems = 32, *out_block_bytes = 22;
		return 1;
	case GGML_TYPE_Q5_1:
		*out_elems = 32, *out_block_bytes = 24;
		return 1;
	case GGML_TYPE_Q8_0:
		*out_elems = 32, *out_block_bytes = 34;
		return 1;
	case GGML_TYPE_Q4_K:
		*out_elems = 256, *out_block_bytes = 144;
		return 1;
	case GGML_TYPE_Q5_K:
		*out_elems = 256, *out_block_bytes = 176;
		return 1;
	case GGML_TYPE_Q6_K:
		*out_elems = 256, *out_block_bytes = 210;
		return 1;
	case GGML_TYPE_IQ3_S:
		*out_elems = 256, *out_block_bytes = 110;
		return 1;
	case GGML_TYPE_IQ2_XXS:
		*out_elems = 256, *out_block_bytes = 66;
		return 1;
	case GGML_TYPE_IQ3_XXS:
		*out_elems = 256, *out_block_bytes = 98;
		return 1;
	case GGML_TYPE_IQ2_S:
		*out_elems = 256, *out_block_bytes = 82;
		return 1;
	case GGML_TYPE_IQ4_XS:
		*out_elems = 256, *out_block_bytes = 136;
		return 1;
	default:
		return 0;
	}
}

static int vk_dequant_type_native(backend *self, uint32_t type, int k) {
	vk_priv *p = self->priv;
	int		 unit_elems, block_bytes;
	if (!p->p_dequant.pipeline || !vk_dequant_block_elems(type, &unit_elems, &block_bytes))
		return 0;
	if (k <= 0 || k % unit_elems != 0)
		return 0;
	if (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS)
		return p->iqx_grid_buf.buf != VK_NULL_HANDLE;
	return type != GGML_TYPE_IQ3_S || p->iq3s_grid_buf.buf != VK_NULL_HANDLE;
}

static status_code vk_dequant_weight(backend *self, uint32_t type, const void *src, int n_rows,
									 int k, buffer *out) {
	vk_priv *p = self->priv;
	if (!p->p_dequant.pipeline || k <= 0 || n_rows <= 0)
		return ERR_UNSUPPORTED;
	int unit_elems = 0, block_bytes = 0;
	if (!vk_dequant_block_elems(type, &unit_elems, &block_bytes) || k % unit_elems != 0)
		return ERR_UNSUPPORTED;
	if (type == GGML_TYPE_IQ3_S && !p->iq3s_grid_buf.buf)
		return ERR_UNSUPPORTED;
	if ((type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS) &&
		!p->iqx_grid_buf.buf)
		return ERR_UNSUPPORTED;

	const size_t src_bytes = ggml_row_size(type, (size_t)k) * (size_t)n_rows;
	const size_t dst_bytes = (size_t)k * (size_t)n_rows * sizeof(float);
	const int	 n_units   = ((n_rows * k) / unit_elems);

	VkMemoryPropertyFlags mem_flags =
		p->caps.unified_memory
			? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
			   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
							   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	buffer		sbuf = {0};
	status_code s	 = vk_buffer_alloc_scratch(self, src_bytes, &sbuf);
	vk_buf	   *dst	 = xcalloc(1, sizeof(vk_buf));
	if (s == OK)
		s = vk_alloc_buffer(p, dst_bytes, usage, mem_flags, dst);
	if (s == OK)
		s = vk_buffer_upload(self, &sbuf, src, src_bytes);
	if (s == OK) {
		struct {
			int32_t type, n_units, block_bytes, n_elems;
		} push = {(int32_t)type, n_units, block_bytes, unit_elems};
		vk_buf *grid =
			(type == GGML_TYPE_IQ3_S) ? &p->iq3s_grid_buf
			: (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS)
				? &p->iqx_grid_buf
				: vk_dummy_buf(p);
		vk_buf		*bufs[3] = {as_vkbuf(&sbuf), dst, grid};
		VkDeviceSize offs[3] = {0, 0, 0};
		s = vk_dispatch_2d_ex(p, &p->p_dequant, bufs, offs, NULL, 3, &push, sizeof(push),
							  (uint32_t)((n_units + 63) / 64), 1, 1u << 1);
	}
	if (s == OK)
		s = vk_run_cmd_sync(p);

	vk_buffer_free(self, &sbuf);
	if (s != OK) {
		if (dst->buf)
			vk_free_buffer(p, dst);
		free(dst);
		return s;
	}

	out->handle	  = dst;
	out->size	  = dst_bytes;
	out->host_ptr = dst->mapped;
	out->owner	  = self;
	return OK;
}

static status_code vk_ffn_activate_batch_scales(backend *self, const buffer *gate, const buffer *up,
												buffer *out, int n, int activation, int m, float gs,
												float us) {
	vk_priv *p = self->priv;
	if (!p->p_ffn_activate_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, activation;
		int32_t m;
		float	gs, us;
	} push				  = {n, activation, m, gs, us};
	vk_buf		*bufs[3]  = {as_vkbuf(gate), as_vkbuf(up), as_vkbuf(out)};
	VkDeviceSize offs[3]  = {gate->offset, up->offset, out->offset};
	uint32_t	 groups_x = (uint32_t)((n + 63) / 64);
	return vk_dispatch_2d_ex(p, &p->p_ffn_activate_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups_x, (uint32_t)m, 1u << 2);
}

static status_code vk_ffn_activate_fused_batch(backend *self, const buffer *fused, buffer *out,
											   int n, int activation, int m) {
	vk_priv *p = self->priv;
	if (!p->p_ffn_activate_fused_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, activation;
		int32_t m;
	} push				  = {n, activation, m};
	vk_buf		*bufs[2]  = {as_vkbuf(fused), as_vkbuf(out)};
	VkDeviceSize offs[2]  = {fused->offset, out->offset};
	uint32_t	 groups_x = (uint32_t)((n + 63) / 64);
	return vk_dispatch_2d_ex(p, &p->p_ffn_activate_fused_batch, bufs, offs, NULL, 2, &push,
							 sizeof(push), groups_x, (uint32_t)m, 1u << 1);
}

static status_code vk_rope_batch(backend *self, buffer *vec, int n_heads, int head_dim,
								 int pos_start, const float *rope_cos_base,
								 const float *rope_sin_base, int m) {
	vk_priv *p = self->priv;
	if (!p->p_rope_batch.pipeline)
		return ERR_UNSUPPORTED;
	int			half = head_dim / 2;
	status_code s	 = vk_ensure_rope_bufs(self, half, rope_cos_base, rope_sin_base);
	if (s != OK)
		return s;
	struct {
		int32_t n_heads, head_dim, pos, neox, m;
	} push				  = {n_heads, head_dim, pos_start, self->rope_neox, m};
	vk_buf		*bufs[3]  = {as_vkbuf(vec), as_vkbuf(p->rope_cos_buf_active),
							 as_vkbuf(p->rope_sin_buf_active)};
	VkDeviceSize offs[3]  = {vec->offset, 0, 0};
	uint32_t	 total	  = (uint32_t)(n_heads * half);
	uint32_t	 groups_x = (total + 63) / 64;
	return vk_dispatch_2d_ex(p, &p->p_rope_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups_x, (uint32_t)m, 0x1);
}

static status_code vk_rope_qk_batch(backend *self, buffer *q, buffer *k, int n_heads,
									int n_kv_heads, int head_dim, int pos_start,
									const float *rope_cos_base, const float *rope_sin_base, int m) {
	vk_priv *p = self->priv;
	if (!p->p_rope_qk_batch.pipeline)
		return ERR_UNSUPPORTED;
	int			half = head_dim / 2;
	status_code s	 = vk_ensure_rope_bufs(self, half, rope_cos_base, rope_sin_base);
	if (s != OK)
		return s;
	struct {
		int32_t n_heads, n_kv_heads, head_dim, pos, neox, m;
	} push				  = {n_heads, n_kv_heads, head_dim, pos_start, self->rope_neox, m};
	vk_buf		*bufs[4]  = {as_vkbuf(q), as_vkbuf(k), as_vkbuf(p->rope_cos_buf_active),
							 as_vkbuf(p->rope_sin_buf_active)};
	VkDeviceSize offs[4]  = {q->offset, k->offset, 0, 0};
	uint32_t	 total_q  = (uint32_t)(n_heads * half);
	uint32_t	 total_k  = (uint32_t)(n_kv_heads * half);
	uint32_t	 groups_x = (total_q + total_k + 63) / 64;
	return vk_dispatch_2d_ex(p, &p->p_rope_qk_batch, bufs, offs, NULL, 4, &push, sizeof(push),
							 groups_x, (uint32_t)m, 0x3);
}

static status_code vk_partial_rope_qk(backend *self, buffer *q, buffer *k, int n_heads,
									  int n_kv_heads, int head_dim, int rope_dim, int pos_start,
									  const float *rope_cos_base, const float *rope_sin_base,
									  int n_rows) {
	vk_priv *p = self->priv;
	if (n_rows <= 0 || rope_dim <= 0)
		return OK;
	if (!p->p_rope_qk_batch.pipeline)
		return ERR_UNSUPPORTED;

	int			half = head_dim / 2;
	status_code s	 = vk_ensure_rope_bufs(self, half, rope_cos_base, rope_sin_base);
	if (s != OK)
		return s;
	struct {
		int32_t n_heads, n_kv_heads, head_dim, pos, neox, m;
		int32_t rope_dim;
	} push = {n_heads, n_kv_heads, head_dim, pos_start, self->rope_neox, n_rows, rope_dim};
	vk_buf		*bufs[4]  = {as_vkbuf(q), as_vkbuf(k), as_vkbuf(p->rope_cos_buf_active),
							 as_vkbuf(p->rope_sin_buf_active)};
	VkDeviceSize offs[4]  = {q->offset, k->offset, 0, 0};
	uint32_t	 total_q  = (uint32_t)(n_heads * half);
	uint32_t	 total_k  = (uint32_t)(n_kv_heads * half);
	uint32_t	 groups_x = (total_q + total_k + 63) / 64;
	return vk_dispatch_2d_ex(p, &p->p_rope_qk_batch, bufs, offs, NULL, 4, &push, sizeof(push),
							 groups_x, (uint32_t)n_rows, 0x3);
}

static status_code vk_rope_tables_get(backend *self, int head_dim, const float *rope_cos_base,
									  const float *rope_sin_base, const float *freq_factors,
									  vk_rope_tables *out) {
	vk_priv *p	  = self->priv;
	int		 half = head_dim / 2;
	if (freq_factors) {
		int n_ctx = p->n_ctx_stored;
		if (n_ctx <= 0 || half <= 0)
			return ERR_INVALID_ARG;
		size_t table_floats = (size_t)n_ctx * (size_t)half;
		size_t need			= table_floats * 2 * sizeof(float);
		int	   table_stale	= (need > (size_t)p->rope_ff_cap || p->rope_ff_base != freq_factors ||
							   p->rope_ff_head_dim != head_dim || p->rope_ff_n_ctx != n_ctx ||
							   p->rope_ff_theta != self->rope_theta);
		if (table_stale) {
			if (need > (size_t)p->rope_ff_cap) {
				if (p->rope_ff_buf.handle) {
					vk_drain_gpu(p);
					vk_buffer_free(self, &p->rope_ff_buf);
				}
				status_code s = vk_buffer_alloc_scratch(self, need, &p->rope_ff_buf);
				if (s != OK)
					return s;
				p->rope_ff_cap	= (int)need;
				p->rope_ff_base = NULL;
			}
			double theta_scale = pow((double)self->rope_theta, -2.0 / (double)head_dim);
			float *tbl		   = xmalloc(need);
			float *cos_tab	   = tbl;
			float *sin_tab	   = tbl + table_floats;
			double base_freq   = 1.0;
			for (int j = 0; j < half; j++) {
				double fv = freq_factors[j];
				double ff = (fv >= 1e10 || fv == 0.0) ? 0.0 : fv;
				if (ff > 0.0) {
					for (int pp = 0; pp < n_ctx; pp++) {
						double angle				   = base_freq * ff * (double)pp;
						cos_tab[(size_t)pp * half + j] = (float)cos(angle);
						sin_tab[(size_t)pp * half + j] = (float)sin(angle);
					}
				} else {
					for (int pp = 0; pp < n_ctx; pp++) {
						cos_tab[(size_t)pp * half + j] = 1.0f;
						sin_tab[(size_t)pp * half + j] = 0.0f;
					}
				}
				base_freq *= theta_scale;
			}
			status_code s = vk_buffer_upload(self, &p->rope_ff_buf, tbl, need);
			free(tbl);
			if (s != OK)
				return s;
			p->rope_ff_base		= freq_factors;
			p->rope_ff_head_dim = head_dim;
			p->rope_ff_n_ctx	= n_ctx;
			p->rope_ff_theta	= self->rope_theta;
		}
		vk_buf *ffb = as_vkbuf(&p->rope_ff_buf);
		*out		= (vk_rope_tables){ffb, ffb, 0, (VkDeviceSize)table_floats * sizeof(float)};
		return OK;
	}
	status_code s = vk_ensure_rope_bufs(self, half, rope_cos_base, rope_sin_base);
	if (s != OK)
		return s;
	*out =
		(vk_rope_tables){as_vkbuf(p->rope_cos_buf_active), as_vkbuf(p->rope_sin_buf_active), 0, 0};
	return OK;
}

static status_code vk_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim,
									 int pos_start, const float *rope_cos_base,
									 const float *rope_sin_base, const float *freq_factors, int m) {
	vk_priv *p	  = self->priv;
	int		 half = head_dim / 2;
	if (freq_factors && !p->p_rope_ext_batch.pipeline)
		return ERR_UNSUPPORTED;
	vk_rope_tables t;
	status_code	   s =
		vk_rope_tables_get(self, head_dim, rope_cos_base, rope_sin_base, freq_factors, &t);
	if (s != OK)
		return s;
	struct {
		int32_t n_heads, head_dim, pos, has_ff, neox, m;
	} push				  = {n_heads, head_dim, pos_start, 0, self->rope_neox, m};
	vk_buf		*bufs[4]  = {as_vkbuf(vec), t.cos_b, t.sin_b, t.cos_b};
	VkDeviceSize offs[4]  = {vec->offset, t.cos_off, t.sin_off, 0};
	uint32_t	 groups_x = (uint32_t)((n_heads * half + 63) / 64);
	return vk_dispatch_2d_ex(p, &p->p_rope_ext_batch, bufs, offs, NULL, 4, &push, sizeof(push),
							 groups_x, (uint32_t)m, 0x1);
}

static status_code vk_qk_norm_rope(backend *self, buffer *q, buffer *k, const buffer *q_w,
								   const buffer *k_w, int n_heads, int n_kv_heads, int head_dim,
								   float eps, int pos_start, const float *rope_cos_base,
								   const float *rope_sin_base, const float *freq_factors, int m) {
	vk_priv *p = self->priv;
	if (!p->p_qk_norm_rope.pipeline || head_dim <= 0 || (head_dim & 1) || head_dim > 1024)
		return ERR_UNSUPPORTED;
	if (freq_factors && !p->p_rope_ext_batch.pipeline)
		return ERR_UNSUPPORTED;
	vk_rope_tables t;
	status_code	   s =
		vk_rope_tables_get(self, head_dim, rope_cos_base, rope_sin_base, freq_factors, &t);
	if (s != OK)
		return s;
	struct {
		int32_t n_heads, n_kv_heads, head_dim;
		float	eps;
		int32_t neox, pos0;
	} push				 = {n_heads, n_kv_heads, head_dim, eps, self->rope_neox, pos_start};
	vk_buf		*dummy	 = vk_dummy_buf(p);
	vk_buf		*bufs[6] = {as_vkbuf(q),   n_kv_heads > 0 ? as_vkbuf(k) : dummy,
							as_vkbuf(q_w), n_kv_heads > 0 ? as_vkbuf(k_w) : dummy,
							t.cos_b,	   t.sin_b};
	VkDeviceSize offs[6] = {q->offset,	 n_kv_heads > 0 ? k->offset : 0,
							q_w->offset, n_kv_heads > 0 ? k_w->offset : 0,
							t.cos_off,	 t.sin_off};
	return vk_dispatch_2d_ex(p, &p->p_qk_norm_rope, bufs, offs, NULL, 6, &push, sizeof(push),
							 (uint32_t)(n_heads + n_kv_heads), (uint32_t)m, 0x3);
}

static status_code vk_ensure_flash_pipeline_batch(vk_priv *p, int head_dim, int n_groups,
												  vk_pipeline_set **out) {
	return vk_ensure_attn_flash(
		p, VK_ATTN_FLASH_BATCH, head_dim, n_groups, shader_attention_flash_batch_spv,
		shader_attention_flash_batch_spv_len, 4, 40, "attention_flash_batch", out);
}

static status_code vk_ensure_attention_pipeline_batch(vk_priv *p, int head_dim,
													  vk_pipeline_set **out) {
	return vk_ensure_attn_spec(p, VK_ATTN_PLAIN_BATCH, head_dim, 1, shader_attention_batch_spv,
							   shader_attention_batch_spv_len, 5, 40, "attention_batch", out);
}

static status_code vk_attention_batch_impl(backend *self, const buffer *q, const buffer *k_cache,
										   const buffer *v_cache, buffer *out, int layer,
										   int pos_start, int n_heads, int n_kv_heads, int head_dim,
										   int n_ctx, int flash_attn, float scale,
										   int n_kv_heads_active, int sliding_window,
										   int attn_start, int m) {
	vk_priv *p		  = self->priv;
	int		 n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	if (n_active != n_kv_heads) {
		DEBUG("vk_attention_batch: unsupported sparse KV (n_kv_heads=%d, n_kv_heads_active=%d) "
			  "at layer=%d -- requires per-token host fallback",
			  n_kv_heads, n_active, layer);
		return ERR_UNSUPPORTED;
	}

	const vk_kv_handle *kh = (const vk_kv_handle *)k_cache->handle;
	const vk_kv_handle *vh = (const vk_kv_handle *)v_cache->handle;
	vk_buf			   *kb = vk_kv_layer_buf(kh, layer, NULL);
	vk_buf			   *vb = vk_kv_layer_buf(vh, layer, NULL);

	if (m <= VK_ATTN_DEC_MAX_M && head_dim <= VK_ATTN_DEC_MAX_HEAD_DIM && (head_dim & 1) == 0 &&
		p->p_attn_decode.pipeline && p->p_attn_merge.pipeline)
		return vk_attention_decode(self, q, kb, vb, vk_kv_layer_off_elems(kh, layer), out,
								   pos_start, n_heads, n_kv_heads, head_dim, n_ctx, scale,
								   sliding_window, m);

	int				 n_groups = n_heads / n_kv_heads;
	vk_pipeline_set *flash_ps = NULL;
	int can_flash = flash_attn && VK_ATTN_FLASH_BATCH_ENABLED &&
					vk_ensure_flash_pipeline_batch(p, head_dim, n_groups, &flash_ps) == OK;

	size_t layer_off = vk_kv_layer_off_elems(kh, layer);

	if (can_flash) {
		struct {
			uint32_t layer_off;
			int32_t	 pos, n_heads, n_kv_heads, head_dim, n_ctx;
			float	 scale;
			int32_t	 stride_head_dim;
			int32_t	 attn_start;
			int32_t	 m;
		} push = {
			(uint32_t)layer_off, pos_start, n_heads, n_kv_heads, head_dim, n_ctx, scale, head_dim,
			attn_start,			 m};
		uint32_t	 groups_x = (uint32_t)vk_flash_groups(p, head_dim, n_heads, n_kv_heads);
		vk_buf		*bufs[4]  = {as_vkbuf(q), kb, vb, as_vkbuf(out)};
		VkDeviceSize offs[4]  = {q->offset, 0, 0, out->offset};
		return vk_dispatch_2d_ex(p, flash_ps, bufs, offs, NULL, 4, &push, sizeof(push), groups_x,
								 (uint32_t)m, 0x8);
	}

	status_code s = vk_ensure_scores_buf_batch(self, n_heads, n_ctx, m);
	if (s != OK)
		return s;
	struct {
		uint32_t layer_off;
		int32_t	 pos, n_heads, n_kv_heads, head_dim, n_ctx;
		float	 scale;
		int32_t	 stride_head_dim;
		int32_t	 attn_start;
		int32_t	 m;
	} push = {(uint32_t)layer_off, pos_start, n_heads, n_kv_heads, head_dim, n_ctx, scale, head_dim,
			  attn_start,		   m};
	vk_buf			*bufs[5] = {as_vkbuf(q), kb, vb, as_vkbuf(&p->attn_scores_buf), as_vkbuf(out)};
	VkDeviceSize	 offs[5] = {q->offset, 0, 0, 0, out->offset};
	vk_pipeline_set *ps		 = NULL;
	if (vk_ensure_attention_pipeline_batch(p, head_dim, &ps) != OK) {
		DEBUG(
			"vk_attention_batch: attention_batch pipeline unavailable for head_dim=%d at layer=%d",
			head_dim, layer);
		return ERR_UNSUPPORTED;
	}
	return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, 5, &push, sizeof(push), (uint32_t)n_heads,
							 (uint32_t)m, 0x18);
}

static status_code vk_attention_batch(backend *self, const buffer *q, const buffer *k_cache,
									  const buffer *v_cache, buffer *out, int layer, int pos_start,
									  int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									  int flash_attn, float scale, int n_kv_heads_active, int m) {
	return vk_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
								   n_kv_heads, head_dim, n_ctx, flash_attn, scale,
								   n_kv_heads_active, 0, 0, m);
}

static status_code vk_attention_swa_batch(backend *self, const buffer *q, const buffer *k_cache,
										  const buffer *v_cache, buffer *out, int layer,
										  int pos_start, int n_heads, int n_kv_heads, int head_dim,
										  int n_ctx, int flash_attn, float scale,
										  int sliding_window, int n_kv_heads_active, int m) {
	int last_pos   = pos_start + m - 1;
	int n_pos	   = last_pos + 1;
	int attn_start = 0;
	if (sliding_window > 0 && n_pos > sliding_window)
		attn_start = n_pos - sliding_window;
	return vk_attention_batch_impl(self, q, k_cache, v_cache, out, layer, pos_start, n_heads,
								   n_kv_heads, head_dim, n_ctx, flash_attn, scale,
								   n_kv_heads_active, sliding_window, attn_start, m);
}

static status_code vk_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up,
										 buffer *out, int n, int activation, int m) {
	return vk_ffn_activate_batch_scales(self, gate, up, out, n, activation, m, 1.0f, 1.0f);
}

static status_code vk_moe_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n, float gate_scale, float up_scale, int use_gelu) {
	return vk_ffn_activate_batch_scales(self, gate, up, out, n, use_gelu ? 1 : 0, 1, gate_scale,
										up_scale);
}

static status_code vk_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name	  = "vulkan";
	out->priority = 100;
	out->caps = BCAP_ROPE_QK_FUSED | BCAP_MATMUL_RESIDUAL | BCAP_MULTI_MATMUL | BCAP_RMSNORM_ADD |
				BCAP_MATMUL_FFN_DOWN | BCAP_MOE_EXPERT_RESIDENT | BCAP_KV_SHIFT | BCAP_QK_NORM_ROPE;
	out->probe							 = vk_probe;
	out->device_count					 = vk_device_count;
	out->device_identity				 = vk_device_identity;
	out->init							 = vk_init;
	out->free							 = vk_free;
	out->buffer_alloc_weight			 = vk_buffer_alloc_weight;
	out->buffer_alloc_scratch			 = vk_buffer_alloc_scratch;
	out->buffer_free					 = vk_buffer_free;
	out->buffer_read_f32				 = vk_buffer_read_f32;
	out->buffer_write_f32				 = vk_buffer_write_f32;
	out->mem_available					 = vk_mem_available;
	out->mem_total						 = vk_mem_total;
	out->kv_alloc						 = vk_kv_alloc;
	out->kv_free						 = vk_kv_free;
	out->kv_put							 = vk_kv_put;
	out->kv_put_batch					 = vk_kv_put_batch;
	out->kv_shift						 = vk_kv_shift;
	out->embd_lookup					 = vk_embd_lookup;
	out->rmsnorm						 = vk_rmsnorm;
	out->matmul							 = vk_matmul;
	out->matmul_type_native				 = vk_matmul_type_native;
	out->argmax_to_reg					 = vk_argmax_to_reg;
	out->read_token_reg					 = vk_read_token_reg;
	out->qk_norm_rope					 = vk_qk_norm_rope;
	out->dequant_type_native			 = vk_dequant_type_native;
	out->dequant_weight					 = vk_dequant_weight;
	out->matmul_residual				 = vk_matmul_residual;
	out->matmul_multi					 = vk_matmul_multi;
	out->matmul_ffn_down				 = vk_matmul_ffn_down;
	out->matmul_ffn_down_batch			 = vk_matmul_ffn_down_batch;
	out->rope							 = vk_rope;
	out->rope_qk						 = vk_rope_qk;
	out->attention						 = vk_attention;
	out->add_inplace					 = vk_add_inplace;
	out->scale_inplace					 = vk_scale_inplace;
	out->softcap						 = vk_softcap;
	out->split_qgate					 = vk_split_qgate;
	out->attn_output_gate				 = vk_attn_output_gate;
	out->partial_rope_qk				 = vk_partial_rope_qk;
	out->buffer_alloc_from_host			 = vk_buffer_alloc_from_host;
	out->moe_expert_ffn					 = vk_moe_expert_ffn;
	out->moe_experts_batch				 = vk_moe_experts_batch;
	out->moe_activate					 = vk_moe_activate;
	out->copy_buffer					 = vk_copy_buffer;
	out->ple_combine					 = vk_ple_combine;
	out->ffn_activate					 = vk_ffn_activate;
	out->argmax							 = vk_argmax;
	out->synchronize					 = vk_synchronize;
	out->set_gpu_profile				 = vk_set_gpu_profile;
	out->submit							 = vk_submit;
	out->begin_batch					 = vk_begin_batch;
	out->end_batch						 = vk_end_batch;
	out->rmsnorm_per_head				 = vk_rmsnorm_per_head;
	out->rmsnorm_noweight				 = vk_rmsnorm_noweight;
	out->rmsnorm_noweight_per_head		 = vk_rmsnorm_noweight_per_head;
	out->rmsnorm_add					 = vk_rmsnorm_add;
	out->rmsnorm_add_batch				 = vk_rmsnorm_add_batch;
	out->rmsnorm_add_norm_batch			 = vk_rmsnorm_add_norm_batch;
	out->ffn_activate_ex				 = vk_ffn_activate_ex;
	out->rope_ext						 = vk_rope_ext;
	out->attention_swa					 = vk_attention_swa;
	out->matmul_batch					 = vk_matmul_batch;
	out->matmul_multi_batch				 = vk_matmul_multi_batch;
	out->rmsnorm_batch					 = vk_rmsnorm_batch;
	out->rmsnorm_noweight_batch			 = vk_rmsnorm_noweight_batch;
	out->rmsnorm_per_head_batch			 = vk_rmsnorm_per_head_batch;
	out->rmsnorm_noweight_per_head_batch = vk_rmsnorm_noweight_per_head_batch;
	out->add_batch						 = vk_add_batch;
	out->layernorm_modulate				 = vk_layernorm_modulate;
	out->gated_add						 = vk_gated_add;
	out->add_bias						 = vk_add_bias;
	out->activate						 = vk_activate;
	out->rope_table						 = vk_rope_table;
	out->attention_dense				 = vk_attention_dense;
	out->im2col3x3						 = vk_im2col3x3;
	out->upsample_shortcut_add			 = vk_upsample_shortcut_add;
	out->ffn_activate_batch				 = vk_ffn_activate_batch;
	out->ffn_activate_fused_batch		 = vk_ffn_activate_fused_batch;
	out->rope_batch						 = vk_rope_batch;
	out->rope_qk_batch					 = vk_rope_qk_batch;
	out->rope_ext_batch					 = vk_rope_ext_batch;
	out->attention_batch				 = vk_attention_batch;
	out->attention_swa_batch			 = vk_attention_swa_batch;
	return OK;
}

BACKEND_REGISTER("vulkan", vk_ctor)
