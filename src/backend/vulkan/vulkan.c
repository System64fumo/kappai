#include "backend/backend.h"
#include "backend/cpu/scalar/quants.h"
#include "common.h"
#include "log.h"
#include "shaders_embedded.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(expr)																		\
	do {																					\
		VkResult _r = (expr);																\
		if (_r != VK_SUCCESS) {																\
			ERROR("vk: %s returned %d at %s:%d", #expr, (int)_r, __FILE__, __LINE__);              \
			if (p && _r == VK_ERROR_DEVICE_LOST)											\
				p->device_lost = 1;															\
			if (p && p->debug.abort_on_error)												\
				abort();																	\
			return ERR_INTERNAL;															\
		}																					\
	} while (0)

#define VK_DESC_CACHE_CAP 256
#define VK_MAX_BINDINGS 8
#define VK_KV_MAGIC 0x564b4b56u
#define VK_RING_DEPTH 4

enum elem_mode {
	ELEM_MODE_COPY = 0,
	ELEM_MODE_ADD_INPLACE = 1,
	ELEM_MODE_SCALE_INPLACE = 2,
	ELEM_MODE_PLE_COMBINE = 3,
};

typedef struct {
	VkDescriptorSet set;
	VkBuffer bufs[VK_MAX_BINDINGS];
	VkDeviceSize offs[VK_MAX_BINDINGS];
	int valid;
	uint64_t last_used;
	uint64_t hash;
} vk_desc_slot;

typedef struct {
	VkPipeline pipeline;
	VkPipelineLayout layout;
	VkDescriptorSetLayout set_layout;
	int n_bindings;
	vk_desc_slot desc_cache[VK_DESC_CACHE_CAP];
	const char *name;
} vk_pipeline_set;

typedef struct {
	VkBuffer buf;
	VkDeviceMemory mem;
	size_t size;
	void *mapped;
} vk_buf;

typedef struct {
	int chunked;
	int n_chunks;
	vk_buf single;
	vk_buf *chunks;
	size_t per_layer_size;
	int n_kv_layers;
	size_t *layer_off_elems;
	size_t *layer_bytes;
} vk_kv_store;

typedef struct {
	uint32_t magic;
	vk_kv_store store;
} vk_kv_handle;

typedef struct {
	VkCommandBuffer cmd;
	int recording;
	int has_work;
	uint64_t signal_value;
	int submitted;
} vk_ring_slot;

typedef struct {
	VkInstance instance;
	VkDebugUtilsMessengerEXT debug_messenger;
	VkPhysicalDevice phys;
	VkDevice dev;
	uint32_t queue_family;
	VkQueue queue;
	VkCommandPool cmd_pool;
	VkCommandBuffer cmd;
	VkDescriptorPool desc_pool;
	VkFence fence;

	int timeline_supported;
	VkSemaphore timeline;
	uint64_t timeline_value;
	uint64_t timeline_completed;
	vk_ring_slot ring[VK_RING_DEPTH];
	int ring_cur;

	VkPhysicalDeviceMemoryProperties mem_props;

	struct {
		uint32_t vendor_id;
		uint32_t device_id;
		uint32_t max_shared_memory;
		uint32_t max_workgroup_size[3];
		uint32_t max_workgroup_invocations;
		VkDeviceSize storage_buffer_offset_alignment;
		VkDeviceSize max_storage_buffer_range;
		uint32_t subgroup_size;
		int unified_memory;
		int is_mali;
		int is_power_vr;
		int is_adreno;
		int is_radv;
		int is_nvidia;
		int is_intel;
		int is_amd;
		int supports_subgroup_basic;
		int supports_subgroup_vote;
		int supports_subgroup_arithmetic;
		int supports_subgroup_ballot;
		int supports_int8;
		int supports_integer_dot_product;
	} caps;

	int pending_dispatches;
	int cmd_recording;
	uint64_t desc_tick;
	uint64_t total_desc_allocs;

	VkPipeline last_pipeline;
	VkDescriptorSet last_desc_set;
	int last_desc_pipeline_match;

#define VK_DEAD_BUF_CAP 64
	VkBuffer dead_bufs[VK_DEAD_BUF_CAP];
	int dead_buf_count;

	VkDescriptorSet pending_free_sets[VK_DESC_CACHE_CAP * 4];
	int pending_free_count;
	uint64_t diag_desc_allocs;
	uint64_t diag_buf_allocs;

#define VK_DIRTY_MAX 128
	VkBuffer dirty_bufs[VK_DIRTY_MAX];
	int dirty_count;

	VkQueryPool query_pool;
	int query_cap;
	int query_count;
	const char *query_names[512];
	float timestamp_period;
	int profiling;

	int matmul_wg_size;
	int matmul_rows_per_thread;
	int matmul_tile_k;

	vk_pipeline_set p_attention;
	vk_pipeline_set p_attention_big;
	int attention_big_ready;
#define VK_FLASH_CACHE_CAP 4
	vk_pipeline_set p_attention_flash[VK_FLASH_CACHE_CAP];
	int flash_head_dim[VK_FLASH_CACHE_CAP];
	int flash_n_groups[VK_FLASH_CACHE_CAP];
	int flash_unsupported[VK_FLASH_CACHE_CAP];
	int flash_count;
	vk_pipeline_set p_kv_put;
	vk_pipeline_set p_embd_lookup;
	vk_pipeline_set p_argmax;

	vk_pipeline_set p_matmul_q4_0_batch;
	vk_pipeline_set p_matmul_q4_0_res_batch;
	vk_pipeline_set p_matmul_q4_0_dual_batch;
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
	vk_pipeline_set p_matmul_f32_batch;
	vk_pipeline_set p_matmul_f32_res_batch;
	vk_pipeline_set p_matmul_iq4_nl_batch;
	vk_pipeline_set p_rmsnorm_batch;
	vk_pipeline_set p_rmsnorm_sg_batch;
	vk_pipeline_set p_rmsnorm_noweight_batch;
	vk_pipeline_set p_rmsnorm_noweight_sg_batch;
	vk_pipeline_set p_rmsnorm_per_head_batch;
	vk_pipeline_set p_rmsnorm_per_head_sg_batch;
	vk_pipeline_set p_rmsnorm_noweight_per_head_batch;
	vk_pipeline_set p_rmsnorm_noweight_per_head_sg_batch;
	vk_pipeline_set p_rmsnorm_add_batch;
	vk_pipeline_set p_rope_batch;
	vk_pipeline_set p_rope_ext_batch;
	vk_pipeline_set p_rope_qk_batch;
	vk_pipeline_set p_attention_batch;
	vk_pipeline_set p_attention_big_batch;
	int				attention_big_batch_ready;
	vk_pipeline_set p_attention_flash_batch[VK_FLASH_CACHE_CAP];
	int				flash_batch_head_dim[VK_FLASH_CACHE_CAP];
	int				flash_batch_n_groups[VK_FLASH_CACHE_CAP];
	int				flash_batch_unsupported[VK_FLASH_CACHE_CAP];
	int				flash_batch_count;
	vk_pipeline_set p_ffn_activate_batch;
	vk_pipeline_set p_elementwise_batch;

	buffer rope_cos_buf;
	buffer rope_sin_buf;
	buffer rope_ff_buf;
	int rope_buf_cap;
	int rope_ff_cap;

	buffer rope_cos_buf_alt;
	buffer rope_sin_buf_alt;
	int rope_buf_cap_alt;
	int rope_half_stored_alt;
	const float *rope_cos_base_alt;
	const float *rope_sin_base_alt;

	int n_ctx_stored;
	int kv_head_dim_max;

	const float *rope_cos_base;
	const float *rope_sin_base;
	const float *rope_ff_base;
	int rope_ff_pos;
	int rope_ff_head_dim;
	float rope_ff_theta;
	int rope_half_stored;
	int rope_n_ctx_stored;

	buffer *rope_cos_buf_active;
	buffer *rope_sin_buf_active;

	buffer attn_scores_buf;
	int attn_scores_cap;

	vk_buf staging_buf;
	size_t staging_cap;

	vk_buf argmax_out_buf;

	vk_buf dummy_buf;

	vk_buf iq3s_grid_buf;

	void **kv_handles;
	int kv_handle_count;
	int kv_handle_cap;

	int kquant_broken[10];
	int kquant_detect_done;
	int device_lost;

	int device_lost_warned;

	int vk_batch_limit;

#define VK_SCRATCH_POOL_CAP 32
	vk_buf *scratch_pool[VK_SCRATCH_POOL_CAP];
	size_t scratch_pool_sizes[VK_SCRATCH_POOL_CAP];
	int scratch_pool_count;

	int batch_active;

	struct {
		int diag;
		int abort_on_error;
		int validate;
		int gpu_attention;
		int gpu_attention_checked;
	} debug;

} vk_priv;

typedef struct {
	uint32_t w_type;
	const char *name;
	size_t block_bytes;
	int block_elems;
	size_t d_off;
	int has_dmin;
	size_t dmin_off;
} kquant_probe_fmt;

static const kquant_probe_fmt KQUANT_PROBE_FORMATS[] = {
		{GGML_TYPE_Q4_K, "q4_K", sizeof(q4_k_block), 256, 0, 1, 2},
		{GGML_TYPE_Q5_K, "q5_K", sizeof(q5_k_block), 256, 0, 1, 2},
		{GGML_TYPE_Q6_K, "q6_K", sizeof(q6_k_block), 256, 208, 0, 0},
		{GGML_TYPE_Q4_0, "q4_0", sizeof(q4_0_block), 32, 0, 0, 0},
		{GGML_TYPE_Q4_1, "q4_1", sizeof(q4_1_block), 32, 0, 1, 2},
		{GGML_TYPE_Q5_0, "q5_0", sizeof(q5_0_block), 32, 0, 0, 0},
		{GGML_TYPE_Q5_1, "q5_1", sizeof(q5_1_block), 32, 0, 1, 2},
		{GGML_TYPE_Q8_0, "q8_0", sizeof(q8_0_block), 32, 0, 0, 0},
		{GGML_TYPE_IQ4_NL, "iq4_nl", sizeof(iq4_nl_block), 32, 0, 0, 0},
		{GGML_TYPE_IQ3_S, "iq3_s", sizeof(iq3_s_block), 256, 0, 0, 0},
};

#define N_KQUANT_PROBE_FORMATS																								 \
	((int)(sizeof(KQUANT_PROBE_FORMATS) / sizeof(KQUANT_PROBE_FORMATS[0])))

static void vk_invalidate_desc_cache_for_buf(vk_priv *p, VkBuffer freed);

static status_code vk_ensure_staging_cap(vk_priv *p, size_t need);
static void vk_kv_store_free(vk_priv *p, vk_kv_store *store);

static int g_probed = -1;

static status_code vk_probe(void) {
	if (g_probed >= 0)
		return g_probed == 1 ? OK : ERR_UNSUPPORTED;

	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
													 .apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
															.pApplicationInfo = &app};
	VkInstance inst;
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

static uint32_t find_memory_type(vk_priv *p, uint32_t type_bits, VkMemoryPropertyFlags want) {
	for (uint32_t i = 0; i < p->mem_props.memoryTypeCount; i++) {
		if (!(type_bits & (1u << i)))
			continue;
		if ((p->mem_props.memoryTypes[i].propertyFlags & want) == want)
			return i;
	}
	return UINT32_MAX;
}

static status_code vk_alloc_buffer(vk_priv *p, size_t size, VkBufferUsageFlags usage,
								   VkMemoryPropertyFlags mem_flags, vk_buf *out) {
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
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VK_CHECK(vkCreateBuffer(p->dev, &bci, NULL, &out->buf));

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(p->dev, out->buf, &req);

	uint32_t mt = find_memory_type(p, req.memoryTypeBits, mem_flags);
	if (mt == UINT32_MAX) {
		vkDestroyBuffer(p->dev, out->buf, NULL);
		return ERR_UNSUPPORTED;
	}

	VkMemoryAllocateInfo mai = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = req.size,
			.memoryTypeIndex = mt,
	};
	if (vkAllocateMemory(p->dev, &mai, NULL, &out->mem) != VK_SUCCESS) {
		vkDestroyBuffer(p->dev, out->buf, NULL);
		return ERR_OUT_OF_MEMORY;
	}
	vkBindBufferMemory(p->dev, out->buf, out->mem, 0);
	out->size = size;

	out->mapped = NULL;
	VkMemoryPropertyFlags actual_flags = p->mem_props.memoryTypes[mt].propertyFlags;
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
	vkFreeMemory(p->dev, b->mem, NULL);
	b->buf = VK_NULL_HANDLE;
	b->mem = VK_NULL_HANDLE;
	b->size = 0;
}

static status_code vk_flush(vk_priv *p);

static void vk_queue_desc_free(vk_priv *p, VkDescriptorSet set) {
	if (set == VK_NULL_HANDLE)
		return;
	if (p->pending_free_count >=
			(int)(sizeof(p->pending_free_sets) / sizeof(p->pending_free_sets[0]))) {
		status_code fs = vk_flush(p);
		if (fs != OK) {
			ERROR("vk: queue_desc_free flush failed (status=%d), dropping %d frees", (int)fs,
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
	vkFreeDescriptorSets(p->dev, p->desc_pool, (uint32_t)p->pending_free_count,
											 p->pending_free_sets);
	p->pending_free_count = 0;
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

				&p->p_attention,
				&p->p_attention_flash[0],
				&p->p_attention_flash[1],
				&p->p_attention_flash[2],
				&p->p_attention_flash[3],
				&p->p_kv_put,
				&p->p_embd_lookup,
				&p->p_argmax,
			&p->p_matmul_q4_0_batch,
			&p->p_matmul_q4_0_res_batch,
			&p->p_matmul_q4_0_dual_batch,
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
			&p->p_matmul_f32_batch,
			&p->p_matmul_f32_res_batch,
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
			&p->p_rope_batch,
			&p->p_rope_ext_batch,
			&p->p_rope_qk_batch,
			&p->p_attention_batch,
			&p->p_attention_big_batch,
			&p->p_attention_flash_batch[0],
			&p->p_attention_flash_batch[1],
			&p->p_attention_flash_batch[2],
			&p->p_attention_flash_batch[3],
			&p->p_ffn_activate_batch,
			&p->p_elementwise_batch,
		};
		const int n_pipelines = (int)(sizeof(pipelines) / sizeof(pipelines[0]));
		for (int i = 0; i < n_pipelines; i++) {
			vk_pipeline_set *ps = pipelines[i];
			if (!ps->pipeline)
				continue;
			int n = ps->n_bindings;
			for (int j = 0; j < VK_DESC_CACHE_CAP; j++) {
				vk_desc_slot *s = &ps->desc_cache[j];
				if (!s->valid)
					continue;
				for (int k = 0; k < n; k++) {
					if (s->bufs[k] == freed) {
						vk_queue_desc_free(p, s->set);
						s->valid = 0;
						s->set = VK_NULL_HANDLE;
						break;
					}
				}
			}
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

static void vk_report_query_results(vk_priv *p) {
	if (!p->profiling || p->query_count <= 0)
		return;
	uint64_t ts[512];
	VkResult qr =
		vkGetQueryPoolResults(p->dev, p->query_pool, 0, (uint32_t)p->query_count, sizeof(ts), ts,
			sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	if (qr != VK_SUCCESS)
		return;
	double totals[256] = {0};
	int counts[256] = {0};
	const char *names[256] = {0};
	int n_names = 0;
	for (int i = 0; i < p->query_count; i += 2) {
		double ns = (double)(ts[i + 1] - ts[i]) * p->timestamp_period;
		const char *nm = p->query_names[i / 2];
		int slot = -1;
		for (int j = 0; j < n_names; j++)
			if (names[j] == nm) {
				slot = j;
				break;
			}
		if (slot < 0 && n_names < 256) {
			slot = n_names++;
			names[slot] = nm;
		}
		if (slot >= 0) {
			totals[slot] += ns;
			counts[slot]++;
		}
	}
	fprintf(stderr, "[GPU_PROFILE] batch of %d dispatches:\n", p->query_count / 2);
	for (int j = 0; j < n_names; j++) {
		fprintf(stderr, "       %-20s calls=%-4d total=%.3fms avg=%.3fms\n", names[j], counts[j],
				totals[j] / 1e6, totals[j] / counts[j] / 1e6);
	}
}

static status_code vk_ring_begin(vk_priv *p, int idx) {
	vk_ring_slot *r = &p->ring[idx];
	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	VK_CHECK(vkBeginCommandBuffer(r->cmd, &bi));
	if (p->profiling && p->query_pool) {
		vkCmdResetQueryPool(r->cmd, p->query_pool, 0, (uint32_t)p->query_cap);
	}
	r->recording = 1;
	r->has_work = 0;
	r->submitted = 0;
	p->cmd = r->cmd;
	p->cmd_recording = 1;
	p->pending_dispatches = 0;
	p->last_pipeline = VK_NULL_HANDLE;
	p->last_desc_set = VK_NULL_HANDLE;
	p->last_desc_pipeline_match = 0;
	return OK;
}

static status_code vk_timeline_wait(vk_priv *p, uint64_t value) {
	if (value <= p->timeline_completed)
		return OK;
	VkSemaphoreWaitInfo wi = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
			.semaphoreCount = 1,
			.pSemaphores = &p->timeline,
			.pValues = &value,
	};
	VK_CHECK(vkWaitSemaphores(p->dev, &wi, UINT64_MAX));
	p->timeline_completed = value;
	return OK;
}

static void vk_ring_invalidate(vk_priv *p, int idx) {
	vk_ring_slot *r = &p->ring[idx];
	(void)vkResetCommandBuffer(r->cmd, 0);
	r->recording = 0;
	r->has_work = 0;
	r->submitted = 0;
	if (p->cmd == r->cmd) {
		p->cmd = VK_NULL_HANDLE;
		p->cmd_recording = 0;
		p->last_pipeline = VK_NULL_HANDLE;
		p->last_desc_set = VK_NULL_HANDLE;
		p->last_desc_pipeline_match = 0;
	}
	p->pending_dispatches = 0;
	p->dirty_count = 0;
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
						" try setting VK_BATCH_LIMIT=4 to reduce per-buffer dispatch count",
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

		VkTimelineSemaphoreSubmitInfo tsi = {
				.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
				.signalSemaphoreValueCount = 1,
				.pSignalSemaphoreValues = &r->signal_value,
		};
		VkSubmitInfo si = {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.pNext = &tsi,
				.commandBufferCount = 1,
				.pCommandBuffers = &r->cmd,
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &p->timeline,
		};
		VkResult submit_res = vkQueueSubmit(p->queue, 1, &si, VK_NULL_HANDLE);
		if (submit_res != VK_SUCCESS) {
			ERROR("vk: vkQueueSubmit returned %d at %s:%d", (int)submit_res, __FILE__, __LINE__);
			if (submit_res == VK_ERROR_DEVICE_LOST)
				p->device_lost = 1;
			if (p && p->debug.abort_on_error)
				abort();
			vk_ring_invalidate(p, idx);
			return ERR_INTERNAL;
		}
		r->submitted = 1;
		r->recording = 0;

		if (wait_now || p->profiling) {
				status_code ws = vk_timeline_wait(p, r->signal_value);
				if (ws != OK) {
					vk_ring_invalidate(p, idx);
					return ws;
			}
			vk_report_query_results(p);

			vk_flush_pending_desc_frees(p);
		}
		return OK;
	}

	VkSubmitInfo si = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &r->cmd,
	};
	vkResetFences(p->dev, 1, &p->fence);
	{
		VkResult submit_res = vkQueueSubmit(p->queue, 1, &si, p->fence);
		if (submit_res != VK_SUCCESS) {
			ERROR("vk: vkQueueSubmit returned %d at %s:%d", (int)submit_res, __FILE__, __LINE__);
			if (submit_res == VK_ERROR_DEVICE_LOST)
				p->device_lost = 1;
			if (p && p->debug.abort_on_error)
				abort();
			vk_ring_invalidate(p, idx);
			return ERR_INTERNAL;
		}
	}
	r->submitted = 1;
	r->recording = 0;
	{
		VkResult wait_res = vkWaitForFences(p->dev, 1, &p->fence, VK_TRUE, UINT64_MAX);
			if (wait_res != VK_SUCCESS) {
				if (wait_res == VK_ERROR_DEVICE_LOST)
					p->device_lost = 1;
				ERROR("vk: vkWaitForFences returned %d", (int)wait_res);
				vk_ring_invalidate(p, idx);
				return ERR_INTERNAL;
			}
	}
	vk_report_query_results(p);
	vk_flush_pending_desc_frees(p);
	return OK;
}

static status_code vk_run_cmd(vk_priv *p) {
	int idx = p->ring_cur;
	status_code s = vk_ring_submit(p, idx, 1);
	if (s != OK)
		return s;

	p->ring_cur = (idx + 1) % VK_RING_DEPTH;
	vk_ring_slot *nr = &p->ring[p->ring_cur];
	if (nr->recording && nr->has_work) {
		status_code ss = vk_ring_submit(p, p->ring_cur, 1);
		if (ss != OK)
			return ss;
	}
	if (!nr->recording) {
		status_code bs = vk_ring_begin(p, p->ring_cur);
		if (bs != OK)
			return bs;
	}
	p->dirty_count = 0;
	return OK;
}

static status_code vk_flush(vk_priv *p) {
	if (p->device_lost)
		return ERR_INTERNAL;

	int idx = p->ring_cur;
	vk_ring_slot *r = &p->ring[idx];

	if (!r->has_work) {
		if (r->submitted) {
			status_code ws = vk_timeline_wait(p, r->signal_value);
			if (ws != OK)
				return ws;
		}
		if (!r->recording)
			return vk_ring_begin(p, idx);
		return OK;
	}

	return vk_run_cmd(p);
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
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = spv_len,
			.pCode = spv,
	};
	VkShaderModule mod;
	VK_CHECK(vkCreateShaderModule(p->dev, &smci, NULL, &mod));

	VkDescriptorSetLayoutBinding bindings[8];
	for (int i = 0; i < n_bindings; i++) {
		bindings[i] = (VkDescriptorSetLayoutBinding){
				.binding = (uint32_t)i,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		};
	}
	VkDescriptorSetLayoutCreateInfo dslci = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32_t)n_bindings,
			.pBindings = bindings,
	};
	if (vkCreateDescriptorSetLayout(p->dev, &dslci, NULL, &out->set_layout) != VK_SUCCESS) {
		vkDestroyShaderModule(p->dev, mod, NULL);
		out->set_layout = VK_NULL_HANDLE;
		out->layout = VK_NULL_HANDLE;
		out->pipeline = VK_NULL_HANDLE;
		return ERR_INTERNAL;
	}

	VkPushConstantRange pcr = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size};
	VkPipelineLayoutCreateInfo plci = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &out->set_layout,
			.pushConstantRangeCount = push_size > 0 ? 1u : 0u,
			.pPushConstantRanges = push_size > 0 ? &pcr : NULL,
	};
	if (vkCreatePipelineLayout(p->dev, &plci, NULL, &out->layout) != VK_SUCCESS) {
		vkDestroyDescriptorSetLayout(p->dev, out->set_layout, NULL);
		vkDestroyShaderModule(p->dev, mod, NULL);
		out->set_layout = VK_NULL_HANDLE;
		out->layout = VK_NULL_HANDLE;
		out->pipeline = VK_NULL_HANDLE;
		return ERR_INTERNAL;
	}

	VkSpecializationMapEntry map_entries[8];
	int n_spec = 0;
	if (spec_data && spec_size > 0) {
		n_spec = (int)(spec_size / 4);
		if (n_spec > 8)
			n_spec = 8;
		for (int i = 0; i < n_spec; i++) {
			map_entries[i] = (VkSpecializationMapEntry){
					.constantID = (uint32_t)i,
					.offset = (uint32_t)(i * 4),
					.size = 4,
			};
		}
	}

	VkSpecializationInfo spec_info = {0};
	if (spec_data && spec_size > 0) {
		spec_info.mapEntryCount = (uint32_t)n_spec;
		spec_info.pMapEntries = map_entries;
		spec_info.dataSize = spec_size;
		spec_info.pData = spec_data;
	}

	VkComputePipelineCreateInfo cpci = {
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage =
					{
							.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
							.stage = VK_SHADER_STAGE_COMPUTE_BIT,
							.module = mod,
							.pName = "main",
				.pSpecializationInfo = (spec_data && spec_size > 0) ? &spec_info : NULL,
					},
			.layout = out->layout,
	};
	VkResult pr = vkCreateComputePipelines(p->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &out->pipeline);
	vkDestroyShaderModule(p->dev, mod, NULL);
	if (pr != VK_SUCCESS) {
		ERROR("vk: vkCreateComputePipelines failed (result=%d).", (int)pr);
		ERROR("vk:       Device maxComputeSharedMemorySize=%u bytes.", p->caps.max_shared_memory);
		ERROR("vk:	 If the shader uses > %u bytes of shared memory, reduce the "
					"tile size.",
					p->caps.max_shared_memory);
		vkDestroyPipelineLayout(p->dev, out->layout, NULL);
		vkDestroyDescriptorSetLayout(p->dev, out->set_layout, NULL);
		out->set_layout = VK_NULL_HANDLE;
		out->layout = VK_NULL_HANDLE;
		out->pipeline = VK_NULL_HANDLE;
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
			ERROR("vk: vk_destroy_pipeline: flush failed (status=%d)", (int)fs);
		}
	}
	if (r->has_work) {
		r->has_work = 0;
		p->dirty_count = 0;
	}
	vk_flush_pending_desc_frees(p);
	vkDestroyPipeline(p->dev, ps->pipeline, NULL);
	vkDestroyPipelineLayout(p->dev, ps->layout, NULL);
	vkDestroyDescriptorSetLayout(p->dev, ps->set_layout, NULL);
	memset(ps, 0, sizeof(*ps));
}

static int vk_dirty_check(vk_priv *p, const VkBuffer *bufs, int n_bufs) {
	for (int i = 0; i < n_bufs; i++) {
		for (int j = 0; j < p->dirty_count; j++) {
			if (p->dirty_bufs[j] == bufs[i])
				return 1;
		}
	}
	return 0;
}

static inline uint64_t vk_desc_hash(const VkBuffer *bufs, const VkDeviceSize *offs, int n_bufs) {
	uint64_t h = 1469598103934665603ULL;
	for (int i = 0; i < n_bufs; i++) {
		h ^= (uint64_t)bufs[i];
		h *= 1099511628211ULL;
		h ^= (uint64_t)offs[i];
		h *= 1099511628211ULL;
	}
	return h;
}

static void vk_dirty_add(vk_priv *p, const VkBuffer *bufs, int n_bufs, uint32_t write_mask) {
	for (int i = 0; i < n_bufs; i++) {
		if (!(write_mask & (1u << i)))
			continue;
		if (p->dirty_count >= VK_DIRTY_MAX) {
			VkMemoryBarrier barrier = {
					.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
					.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
			};
			vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
								 NULL);
			p->dirty_count = 0;
		}
		int dup = 0;
		for (int j = 0; j < p->dirty_count; j++) {
			if (p->dirty_bufs[j] == bufs[i]) {
				dup = 1;
				break;
			}
		}
		if (!dup)
			p->dirty_bufs[p->dirty_count++] = bufs[i];
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

	if (!p->ring[p->ring_cur].recording) {
		status_code bs = vk_ring_begin(p, p->ring_cur);
		if (bs != OK)
			return bs;
	}
	int flush_limit;
	if (p->caps.is_mali || p->caps.is_power_vr || p->caps.is_adreno) {
		flush_limit = 8;
	} else {
		flush_limit = p->batch_active ? 4096 : 256;
	}
	if (p->vk_batch_limit > 0)
		flush_limit = p->vk_batch_limit;
	if (p->pending_dispatches >= flush_limit) {
		status_code fs = vk_flush(p);
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

	VkBuffer key[VK_MAX_BINDINGS];
	VkDeviceSize key_offs[VK_MAX_BINDINGS];
	for (int i = 0; i < n_bufs; i++) {
		key[i] = bufs[i]->buf;
		key_offs[i] = offs ? offs[i] : 0;
	}

	if (p->dirty_count > 0 && vk_dirty_check(p, key, n_bufs)) {
		VkMemoryBarrier barrier = {
				.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		};
		vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
							 NULL);
		p->dirty_count = 0;
	}

	const size_t key_bytes = sizeof(VkBuffer) * (size_t)n_bufs;
	const size_t off_bytes = sizeof(VkDeviceSize) * (size_t)n_bufs;
	uint64_t key_hash = vk_desc_hash(key, key_offs, n_bufs);

	VkDescriptorSet set = VK_NULL_HANDLE;
	int miss_slot = -1;
	int lru_slot = 0;
	uint64_t lru_tick = UINT64_MAX;

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
		uint32_t probe = (uint32_t)(key_hash % VK_DESC_CACHE_CAP);
		for (int i = 0; i < VK_DESC_CACHE_CAP; i++) {
			uint32_t idx = (probe + (uint32_t)i) % VK_DESC_CACHE_CAP;
			vk_desc_slot *s = &ps->desc_cache[idx];
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
				s->set = VK_NULL_HANDLE;
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
			s->set = VK_NULL_HANDLE;
			s->valid = 0;
		}

		VkDescriptorSetAllocateInfo dsai = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = p->desc_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &ps->set_layout,
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
		VkWriteDescriptorSet writes[VK_MAX_BINDINGS];
		for (int i = 0; i < n_bufs; i++) {
			binfo[i] = (VkDescriptorBufferInfo){
					.buffer = key[i],
					.offset = key_offs[i],
					.range = ranges ? ranges[i] : VK_WHOLE_SIZE,
			};
			writes[i] = (VkWriteDescriptorSet){
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = new_set,
					.dstBinding = (uint32_t)i,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &binfo[i],
			};
		}
		vkUpdateDescriptorSets(p->dev, (uint32_t)n_bufs, writes, 0, NULL);

		s->set = new_set;
		memcpy(s->bufs, key, key_bytes);
		memcpy(s->offs, key_offs, off_bytes);
		s->hash = key_hash;
		s->valid = 1;
		p->desc_tick++;
		s->last_used = p->desc_tick;
		set = new_set;
	}

have_set:;

	if (p->last_pipeline != ps->pipeline) {
		vkCmdBindPipeline(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipeline);
		p->last_pipeline = ps->pipeline;
		p->last_desc_pipeline_match = 0;
	}
	if (!p->last_desc_pipeline_match || p->last_desc_set != set) {
		vkCmdBindDescriptorSets(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, 1, &set, 0,
								NULL);
		p->last_desc_set = set;
		p->last_desc_pipeline_match = 1;
	}
	if (push_size > 0)
		vkCmdPushConstants(p->cmd, ps->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);

	int q0 = -1;
	if (p->profiling && p->query_count + 2 <= p->query_cap) {
		q0 = p->query_count;
		p->query_names[q0 / 2] = ps->name ? ps->name : "?";
		vkCmdWriteTimestamp(p->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p->query_pool, q0);
		p->query_count += 2;
	}

	vkCmdDispatch(p->cmd, groups_x, groups_y, 1);

	if (q0 >= 0) {
		vkCmdWriteTimestamp(p->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->query_pool, q0 + 1);
	}

	vk_dirty_add(p, key, n_bufs, write_mask);

	p->pending_dispatches++;
	p->ring[p->ring_cur].has_work = 1;
	return OK;
}

static status_code vk_dispatch(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs, int n_bufs,
							   const void *push, uint32_t push_size, uint32_t groups_x) {
	uint32_t wmask = (n_bufs > 0) ? (1u << (n_bufs - 1)) : 0;
	return vk_dispatch_ex(p, ps, bufs, NULL, NULL, n_bufs, push, push_size, groups_x, 1, wmask);
}

static status_code vk_dispatch_masked(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs, int n_bufs,
									  const void *push, uint32_t push_size, uint32_t groups_x,
									  uint32_t write_mask) {
	return vk_dispatch_ex(p, ps, bufs, NULL, NULL, n_bufs, push, push_size, groups_x, 1,
						  write_mask);
}

static status_code vk_dispatch_2d(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs, int n_bufs,
								  const void *push, uint32_t push_size, uint32_t groups_x,
								  uint32_t groups_y) {
	uint32_t wmask = (n_bufs > 0) ? (1u << (n_bufs - 1)) : 0;
	return vk_dispatch_ex(p, ps, bufs, NULL, NULL, n_bufs, push, push_size, groups_x, groups_y,
						  wmask);
}

static status_code vk_dispatch_2d_masked(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs, int n_bufs,
										 const void *push, uint32_t push_size, uint32_t groups_x,
										 uint32_t groups_y, uint32_t write_mask) {
	return vk_dispatch_ex(p, ps, bufs, NULL, NULL, n_bufs, push, push_size, groups_x, groups_y,
						  write_mask);
}

static status_code vk_dispatch_2d_ex(vk_priv *p, vk_pipeline_set *ps, vk_buf **bufs,
									 const VkDeviceSize *offs, const VkDeviceSize *ranges,
									 int n_bufs, const void *push, uint32_t push_size,
									 uint32_t groups_x, uint32_t groups_y, uint32_t write_mask) {
	return vk_dispatch_ex(p, ps, bufs, offs, ranges, n_bufs, push, push_size, groups_x, groups_y,
						  write_mask);
}

static VkBool32 vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT	  severity,
									VkDebugUtilsMessageTypeFlagsEXT type,
									const VkDebugUtilsMessengerCallbackDataEXT *data,
									void *user_data) {
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

static status_code vk_init(backend *self, int device_index) {
	vk_priv *p = xcalloc(1, sizeof(vk_priv));
	self->priv = p;

	p->debug.diag = getenv("VK_DIAG") != NULL;
	p->debug.validate = getenv("VK_VALIDATE") != NULL;
	p->debug.abort_on_error = getenv("VK_ABORT_ON_ERROR") != NULL;

	{
		const char *bl = getenv("VK_BATCH_LIMIT");
		if (bl && *bl) {
			long v = strtol(bl, NULL, 10);
			if (v > 0 && v < 100000)
				p->vk_batch_limit = (int)v;
		}
	}

	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
													 .apiVersion = VK_API_VERSION_1_3};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
															.pApplicationInfo = &app};
	const char *validation_layer = "VK_LAYER_KHRONOS_validation";
	const char *debug_ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
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
		ici.enabledLayerCount = 1;
		ici.ppEnabledLayerNames = &validation_layer;
		ici.enabledExtensionCount = 1;
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
					.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
					.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
														 VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
														 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
														 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
					.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
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
		device_index = 0;

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

	int has_timeline_ext = 0;
	int has_dot_product_ext = 0;
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
				}
			}
			free(exts);
		}
	}

	VkPhysicalDeviceProperties dev_props_for_version;
	vkGetPhysicalDeviceProperties(p->phys, &dev_props_for_version);
	uint32_t device_api_version = dev_props_for_version.apiVersion;
	int device_supports_v2_queries = (device_api_version >= VK_API_VERSION_1_1);

	VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dot_product_feat = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
	};
	VkPhysicalDeviceVulkan12Features vk12_feat = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = &dot_product_feat,
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

	const char *device_exts[2];
	uint32_t n_device_exts = 0;
	if (timeline_available && has_timeline_ext && device_api_version < VK_API_VERSION_1_2) {
		device_exts[n_device_exts++] = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
	}
	if (dot_product_available && has_dot_product_ext && device_api_version < VK_API_VERSION_1_3) {
		device_exts[n_device_exts++] = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
	}

	float prio = 1.0f;
	VkDeviceQueueCreateInfo dqci = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = p->queue_family,
			.queueCount = 1,
			.pQueuePriorities = &prio,
	};
	VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR enable_dot_product = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
			.shaderIntegerDotProduct = dot_product_available ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceVulkan12Features enable_vk12 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = &enable_dot_product,
			.shaderInt8 = int8_available ? VK_TRUE : VK_FALSE,
			.timelineSemaphore = timeline_available ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceTimelineSemaphoreFeatures enable_timeline = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
			.pNext = &enable_dot_product,
			.timelineSemaphore = timeline_available ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceFeatures2 enable_feat2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = (device_api_version >= VK_API_VERSION_1_2) ? (void *)&enable_vk12
									 : (void *)&enable_timeline,
	};
	VkDeviceCreateInfo dci = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = &enable_feat2,
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &dqci,
			.enabledExtensionCount = n_device_exts,
			.ppEnabledExtensionNames = n_device_exts ? device_exts : NULL,
	};
	if (vkCreateDevice(p->phys, &dci, NULL, &p->dev) != VK_SUCCESS) {
		timeline_available = 0;
		int8_available = 0;
		dot_product_available = 0;
		dci.pNext = NULL;
		dci.enabledExtensionCount = 0;
		dci.ppEnabledExtensionNames = NULL;
		if (vkCreateDevice(p->phys, &dci, NULL, &p->dev) != VK_SUCCESS)
			return ERR_INTERNAL;
	}
	p->timeline_supported = timeline_available;
	p->caps.supports_int8 = int8_available;
	p->caps.supports_integer_dot_product = dot_product_available;

	vkGetDeviceQueue(p->dev, p->queue_family, 0, &p->queue);

	VkPhysicalDeviceProperties props = dev_props_for_version;
	p->caps.vendor_id = props.vendorID;
	p->caps.device_id = props.deviceID;
	p->caps.max_shared_memory = props.limits.maxComputeSharedMemorySize;
	p->caps.max_workgroup_size[0] = props.limits.maxComputeWorkGroupSize[0];
	p->caps.max_workgroup_size[1] = props.limits.maxComputeWorkGroupSize[1];
	p->caps.max_workgroup_size[2] = props.limits.maxComputeWorkGroupSize[2];
	p->caps.max_workgroup_invocations		= props.limits.maxComputeWorkGroupInvocations;
	p->caps.storage_buffer_offset_alignment = props.limits.minStorageBufferOffsetAlignment;
	p->caps.max_storage_buffer_range = props.limits.maxStorageBufferRange;

	p->caps.is_mali = (props.vendorID == 0x13B5);
	p->caps.is_power_vr = (props.vendorID == 0x1010);
	p->caps.is_adreno = (props.vendorID == 0x5143);
	p->caps.is_nvidia = (props.vendorID == 0x10DE);
	p->caps.is_intel = (props.vendorID == 0x8086);
	p->caps.is_amd = (props.vendorID == 0x1002);
	p->caps.is_radv = (props.vendorID == 0x1002);

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
	p->caps.subgroup_size = subgroup_props.subgroupSize;
	VkSubgroupFeatureFlags sf = subgroup_props.supportedOperations;
	p->caps.supports_subgroup_basic = (sf & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
	p->caps.supports_subgroup_vote = (sf & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0;
	p->caps.supports_subgroup_arithmetic = (sf & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
	p->caps.supports_subgroup_ballot = (sf & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0;
#endif

	p->caps.unified_memory = 0;
	if (p->caps.is_mali || p->caps.is_power_vr || p->caps.is_adreno) {
		p->caps.unified_memory = 1;
	} else if (p->caps.is_intel) {
		for (uint32_t i = 0; i < p->mem_props.memoryTypeCount; i++) {
			VkMemoryPropertyFlags f = p->mem_props.memoryTypes[i].propertyFlags;
			if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
					(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				p->caps.unified_memory = 1;
				break;
			}
		}
	}

	log_tag("VLK", "device: %s", props.deviceName);

	if (p->caps.is_mali || p->caps.is_power_vr || p->caps.is_adreno) {
		p->matmul_wg_size = 32;
		p->matmul_rows_per_thread = 2;
		p->matmul_tile_k = 1024;
	} else {
		p->matmul_wg_size = 64;
		p->matmul_rows_per_thread = 4;
		p->matmul_tile_k = 2048;
	}
	if (p->caps.is_mali) {
		if (p->caps.subgroup_size <= 8) {
			p->matmul_wg_size = 32;
			p->matmul_rows_per_thread = 1;
			p->matmul_tile_k = 512;
		} else {
			p->matmul_wg_size = 96;
			p->matmul_rows_per_thread = 1;
			p->matmul_tile_k = 1152;
		}
	}
	if (p->caps.is_adreno) {
		p->matmul_wg_size = 128;
		p->matmul_rows_per_thread = 1;
		p->matmul_tile_k = 256;
	}
	if (p->caps.is_radv || p->caps.is_amd) {
		p->matmul_wg_size = 128;
		p->matmul_rows_per_thread = 1;
		p->matmul_tile_k = 2048;
	}

	VkCommandPoolCreateInfo cpci = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = p->queue_family,
	};
	if (vkCreateCommandPool(p->dev, &cpci, NULL, &p->cmd_pool) != VK_SUCCESS)
		return ERR_INTERNAL;

	{
		VkCommandBuffer cmds[VK_RING_DEPTH];
		VkCommandBufferAllocateInfo cbai = {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = p->cmd_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = VK_RING_DEPTH,
		};
		if (vkAllocateCommandBuffers(p->dev, &cbai, cmds) != VK_SUCCESS)
			return ERR_INTERNAL;
		for (int i = 0; i < VK_RING_DEPTH; i++) {
			p->ring[i].cmd = cmds[i];
		}
	}

	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	if (vkCreateFence(p->dev, &fci, NULL, &p->fence) != VK_SUCCESS)
		return ERR_INTERNAL;

	if (p->timeline_supported) {
		VkSemaphoreTypeCreateInfo type_ci = {
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
				.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
				.initialValue = 0,
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
	p->timeline_value = 0;
	p->timeline_completed = 0;

	VkDescriptorPoolSize dps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
															.descriptorCount = 65536};
	VkDescriptorPoolCreateInfo dpci = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
			.maxSets = 8192,
			.poolSizeCount = 1,
			.pPoolSizes = &dps,
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
	uint32_t spec_matmul[3] = {
			(uint32_t)p->matmul_wg_size,
			(uint32_t)p->matmul_rows_per_thread,
			(uint32_t)p->matmul_tile_k,
	};

	uint32_t kquant_tile_k = (uint32_t)p->matmul_tile_k;
	if (kquant_tile_k < 256)
		kquant_tile_k = 256;
	if (kquant_tile_k % 256 != 0)
		kquant_tile_k += 256 - (kquant_tile_k % 256);
	if (p->caps.max_shared_memory > 0) {
		double budget = (double)p->caps.max_shared_memory * 0.75;
		double bytes_per_elem = 4.0 + 4.0 + (4.0 / 256.0) + (4.0 * 16.0 / 256.0);
		uint32_t max_tile = (uint32_t)(budget / bytes_per_elem);
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

	uint32_t iq3s_tile_k = kquant_tile_k;
	if (p->caps.is_mali || p->caps.is_power_vr || p->caps.is_adreno) {
		iq3s_tile_k = 512;
	}
	{
		const char *tk = getenv("VK_IQ3S_TILE_K");
		if (tk && *tk) {
			long v = strtol(tk, NULL, 10);
			if (v >= 256 && (v % 256) == 0 && v < 100000)
				iq3s_tile_k = (uint32_t)v;
		}
	}
	uint32_t spec_matmul_iq3s[3] = {
			(uint32_t)p->matmul_wg_size,
			(uint32_t)p->matmul_rows_per_thread,
			iq3s_tile_k,
	};

	s				  = vk_create_pipeline(p, shader_attention_spv, shader_attention_spv_len, 5, 36,
										   &p->p_attention);
	if (s != OK)
		return s;
	p->p_attention.name = "attention";
	p->flash_count = 0;
	memset(p->p_attention_flash, 0, sizeof(p->p_attention_flash));
	s = vk_create_pipeline(p, shader_kv_put_spv, shader_kv_put_spv_len, 4, 32, &p->p_kv_put);
	if (s != OK)
		return s;
	p->p_kv_put.name = "kv_put";
	s = vk_create_pipeline(p, shader_embd_lookup_spv, shader_embd_lookup_spv_len, 3, 16,
						   &p->p_embd_lookup);
	if (s != OK)
		return s;
	p->p_embd_lookup.name = "embd_lookup";
	s = vk_create_pipeline(p, shader_argmax_spv, shader_argmax_spv_len, 2, 4, &p->p_argmax);
	if (s != OK)
		return s;
	p->p_argmax.name = "argmax";
	{
		VkBufferUsageFlags grid_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VkMemoryPropertyFlags grid_flags =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		if (p->caps.unified_memory)
			grid_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		s = vk_alloc_buffer(p, sizeof(iq3s_grid), grid_usage, grid_flags, &p->iq3s_grid_buf);
		if (s != OK) {
			WARN("vulkan: failed to allocate IQ3_S grid buffer -- "
					 "IQ3_S GPU path will be disabled");
			p->iq3s_grid_buf.buf = VK_NULL_HANDLE;
		} else {
			memcpy(p->iq3s_grid_buf.mapped, iq3s_grid, sizeof(iq3s_grid));
		}
	}

	s = vk_create_pipeline_spec(p, shader_matmul_f32_batch_spv, shader_matmul_f32_batch_spv_len, 3,
								12, spec_matmul, sizeof(spec_matmul), &p->p_matmul_f32_batch);
	if (s != OK)
		WARN("vulkan: failed to create matmul_f32_batch pipeline -- "
			 "batched prefill will fall back to per-token dispatch");
	else
		p->p_matmul_f32_batch.name = "matmul_f32_batch";

	s = vk_create_pipeline_spec(p, shader_matmul_f32_residual_batch_spv,
								shader_matmul_f32_residual_batch_spv_len, 4, 12, spec_matmul,
								sizeof(spec_matmul), &p->p_matmul_f32_res_batch);
	if (s != OK)
		WARN("vulkan: failed to create matmul_f32_res_batch pipeline");
	else
		p->p_matmul_f32_res_batch.name = "matmul_f32_res_batch";

#define VK_CREATE_BATCH_MATMUL(qname, spv_prefix, n_bind, push_sz, spec_arr)                       \
	do {                                                                                           \
		s = vk_create_pipeline_spec(p, shader_##spv_prefix##_batch_spv,                            \
									shader_##spv_prefix##_batch_spv_len, n_bind, push_sz,          \
									spec_arr, sizeof(spec_arr), &p->p_matmul_##qname##_batch);     \
		if (s != OK)                                                                               \
			WARN("vulkan: failed to create " #spv_prefix "_batch pipeline");                       \
		else                                                                                       \
			p->p_matmul_##qname##_batch.name = #spv_prefix "_batch";                               \
	} while (0)

#define VK_CREATE_BATCH_MATMUL_RES(qname, spv_prefix, n_bind, push_sz, spec_arr)                   \
	do {                                                                                           \
		s = vk_create_pipeline_spec(p, shader_##spv_prefix##_residual_batch_spv,                   \
									shader_##spv_prefix##_residual_batch_spv_len, n_bind, push_sz, \
									spec_arr, sizeof(spec_arr), &p->p_matmul_##qname##_res_batch); \
		if (s != OK)                                                                               \
			WARN("vulkan: failed to create " #spv_prefix "_res_batch pipeline");                   \
		else                                                                                       \
			p->p_matmul_##qname##_res_batch.name = #spv_prefix "_res_batch";                       \
	} while (0)

#define VK_CREATE_BATCH_MATMUL_DUAL(qname, spv_prefix, n_bind, push_sz, spec_arr)                  \
	do {                                                                                           \
		s = vk_create_pipeline_spec(                                                               \
			p, shader_##spv_prefix##_dual_batch_spv, shader_##spv_prefix##_dual_batch_spv_len,     \
			n_bind, push_sz, spec_arr, sizeof(spec_arr), &p->p_matmul_##qname##_dual_batch);       \
		if (s != OK)                                                                               \
			WARN("vulkan: failed to create " #spv_prefix "_dual_batch pipeline");                  \
		else                                                                                       \
			p->p_matmul_##qname##_dual_batch.name = #spv_prefix "_dual_batch";                     \
	} while (0)

	VK_CREATE_BATCH_MATMUL(q4_0, matmul_q4_0, 3, 16, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES(q4_0, matmul_q4_0, 4, 16, spec_matmul);
	VK_CREATE_BATCH_MATMUL_DUAL(q4_0, matmul_q4_0, 5, 16, spec_matmul);

	VK_CREATE_BATCH_MATMUL(q4_1, matmul_q4_1, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES(q4_1, matmul_q4_1, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL(q5_0, matmul_q5_0, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES(q5_0, matmul_q5_0, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL(q5_1, matmul_q5_1, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES(q5_1, matmul_q5_1, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL(q8_0, matmul_q8_0, 3, 12, spec_matmul);
	VK_CREATE_BATCH_MATMUL_RES(q8_0, matmul_q8_0, 4, 12, spec_matmul);

	VK_CREATE_BATCH_MATMUL(q4_k, matmul_q4_k, 3, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_RES(q4_k, matmul_q4_k, 4, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_DUAL(q4_k, matmul_q4_k, 5, 16, spec_matmul_kquant);

	VK_CREATE_BATCH_MATMUL(q5_k, matmul_q5_k, 3, 12, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_RES(q5_k, matmul_q5_k, 4, 12, spec_matmul_kquant);

	VK_CREATE_BATCH_MATMUL(q6_k, matmul_q6_k, 3, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_RES(q6_k, matmul_q6_k, 4, 16, spec_matmul_kquant);
	VK_CREATE_BATCH_MATMUL_DUAL(q6_k, matmul_q6_k, 5, 16, spec_matmul_kquant);

	VK_CREATE_BATCH_MATMUL(iq3_s, matmul_iq3_s, 4, 16, spec_matmul_iq3s);
	VK_CREATE_BATCH_MATMUL_RES(iq3_s, matmul_iq3_s, 5, 16, spec_matmul_iq3s);

#undef VK_CREATE_BATCH_MATMUL
#undef VK_CREATE_BATCH_MATMUL_RES
#undef VK_CREATE_BATCH_MATMUL_DUAL

	s = vk_create_pipeline_spec(p, shader_matmul_iq4_nl_batch_spv,
								shader_matmul_iq4_nl_batch_spv_len, 7, 32, spec_matmul,
								sizeof(spec_matmul), &p->p_matmul_iq4_nl_batch);
	if (s != OK)
		WARN("vulkan: failed to create matmul_iq4_nl_batch pipeline");
	else
		p->p_matmul_iq4_nl_batch.name = "matmul_iq4_nl_batch";

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
							   12, &p->p_rmsnorm_add_batch);
		if (s == OK)
			p->p_rmsnorm_add_batch.name = "rmsnorm_add_batch";
	}

	s = vk_create_pipeline(p, shader_rope_batch_spv, shader_rope_batch_spv_len, 3, 20,
						   &p->p_rope_batch);
	if (s == OK)
		p->p_rope_batch.name = "rope_batch";
	s = vk_create_pipeline(p, shader_rope_ext_batch_spv, shader_rope_ext_batch_spv_len, 4, 24,
						   &p->p_rope_ext_batch);
	if (s == OK)
		p->p_rope_ext_batch.name = "rope_ext_batch";
	s = vk_create_pipeline(p, shader_rope_qk_batch_spv, shader_rope_qk_batch_spv_len, 4, 24,
						   &p->p_rope_qk_batch);
	if (s == OK)
		p->p_rope_qk_batch.name = "rope_qk_batch";

	s = vk_create_pipeline(p, shader_attention_batch_spv, shader_attention_batch_spv_len, 5, 40,
						   &p->p_attention_batch);
	if (s == OK)
		p->p_attention_batch.name = "attention_batch";
	p->attention_big_batch_ready = 0;
	p->flash_batch_count		 = 0;
	memset(p->p_attention_flash_batch, 0, sizeof(p->p_attention_flash_batch));
	memset(p->flash_batch_head_dim, 0, sizeof(p->flash_batch_head_dim));
	memset(p->flash_batch_n_groups, 0, sizeof(p->flash_batch_n_groups));
	memset(p->flash_batch_unsupported, 1, sizeof(p->flash_batch_unsupported));

	s = vk_create_pipeline(p, shader_ffn_activate_batch_spv, shader_ffn_activate_batch_spv_len, 3,
						   12, &p->p_ffn_activate_batch);
	if (s == OK)
		p->p_ffn_activate_batch.name = "ffn_activate_batch";

	s = vk_create_pipeline(p, shader_elementwise_batch_spv, shader_elementwise_batch_spv_len, 3, 20,
						   &p->p_elementwise_batch);
	if (s == OK)
		p->p_elementwise_batch.name = "elementwise_batch";

	return OK;
}

static void vk_free(backend *self) {
	vk_priv *p = self->priv;
	if (!p)
		return;
	if (p->dev) {
		vkDeviceWaitIdle(p->dev);

		vk_destroy_pipeline(p, &p->p_attention);
		vk_destroy_pipeline(p, &p->p_attention_big);
		for (int fi = 0; fi < VK_FLASH_CACHE_CAP; fi++)
			vk_destroy_pipeline(p, &p->p_attention_flash[fi]);
		vk_destroy_pipeline(p, &p->p_kv_put);
		vk_destroy_pipeline(p, &p->p_embd_lookup);
		vk_destroy_pipeline(p, &p->p_argmax);

		vk_destroy_pipeline(p, &p->p_matmul_q4_0_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_0_res_batch);
		vk_destroy_pipeline(p, &p->p_matmul_q4_0_dual_batch);
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
		vk_destroy_pipeline(p, &p->p_matmul_f32_batch);
		vk_destroy_pipeline(p, &p->p_matmul_f32_res_batch);
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
		vk_destroy_pipeline(p, &p->p_rope_batch);
		vk_destroy_pipeline(p, &p->p_rope_ext_batch);
		vk_destroy_pipeline(p, &p->p_rope_qk_batch);
		vk_destroy_pipeline(p, &p->p_attention_batch);
		vk_destroy_pipeline(p, &p->p_attention_big_batch);
		for (int fi = 0; fi < VK_FLASH_CACHE_CAP; fi++)
			vk_destroy_pipeline(p, &p->p_attention_flash_batch[fi]);
		vk_destroy_pipeline(p, &p->p_ffn_activate_batch);
		vk_destroy_pipeline(p, &p->p_elementwise_batch);

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
		if (p->staging_buf.buf) {
			vk_free_buffer(p, &p->staging_buf);
			p->staging_cap = 0;
		}
		if (p->argmax_out_buf.buf) {
			vk_free_buffer(p, &p->argmax_out_buf);
		}
		if (p->dummy_buf.buf) {
			vk_free_buffer(p, &p->dummy_buf);
		}
		if (p->iq3s_grid_buf.buf) {
			vk_free_buffer(p, &p->iq3s_grid_buf);
		}

		for (int i = 0; i < p->scratch_pool_count; i++) {
			vk_free_buffer(p, p->scratch_pool[i]);
			free(p->scratch_pool[i]);
		}
		p->scratch_pool_count = 0;

		free(p->kv_handles);
		if (p->desc_pool)
			vkDestroyDescriptorPool(p->dev, p->desc_pool, NULL);
		if (p->fence)
			vkDestroyFence(p->dev, p->fence, NULL);
		if (p->timeline)
			vkDestroySemaphore(p->dev, p->timeline, NULL);
		if (p->cmd_pool)
			vkDestroyCommandPool(p->dev, p->cmd_pool, NULL);
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
		VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VkMemoryPropertyFlags flags =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		vk_alloc_buffer(p, 16, usage, flags, &p->dummy_buf);
	}
	return &p->dummy_buf;
}

static void vk_kv_registry_add(vk_priv *p, void *h) {
	if (p->kv_handle_count == p->kv_handle_cap) {
		p->kv_handle_cap = p->kv_handle_cap ? p->kv_handle_cap * 2 : 8;
		p->kv_handles	 = xrealloc(p->kv_handles, (size_t)p->kv_handle_cap * sizeof(void *));
	}
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

static status_code vk_buffer_alloc_scratch(backend *self, size_t size, buffer *out) {
	vk_priv *p = self->priv;

	for (int i = 0; i < p->scratch_pool_count; i++) {
		if (p->scratch_pool_sizes[i] >= size) {
			vk_buf *b = p->scratch_pool[i];
			p->scratch_pool[i] = p->scratch_pool[p->scratch_pool_count - 1];
			p->scratch_pool_sizes[i] = p->scratch_pool_sizes[p->scratch_pool_count - 1];
			p->scratch_pool_count--;

			out->handle = b;
			out->size = size;
			out->host_ptr = b->mapped;
			out->owner = self;
			return OK;
		}
	}

	vk_buf *b = xcalloc(1, sizeof(vk_buf));

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
	out->handle = b;
	out->size = size;
	out->host_ptr = b->mapped;
	out->owner = self;

	return OK;
}

static status_code vk_buffer_alloc_weight(backend *self, const tensor_desc *desc, buffer *out) {
	vk_priv *p = self->priv;
	size_t	 size = (desc->n_dims == 1) ? ggml_row_size(desc->type, desc->dims[0])
										: ggml_row_size(desc->type, desc->dims[0]) * desc->dims[1];

	for (int i = 0; i < p->scratch_pool_count; i++) {
		if (p->scratch_pool_sizes[i] >= size) {
			vk_buf *b = p->scratch_pool[i];
			p->scratch_pool[i] = p->scratch_pool[p->scratch_pool_count - 1];
			p->scratch_pool_sizes[i] = p->scratch_pool_sizes[p->scratch_pool_count - 1];
			p->scratch_pool_count--;

			if (b->mapped) {
				memcpy(b->mapped, desc->host_data, size);
			} else {
				status_code s = vk_ensure_staging_cap(p, size);
				if (s != OK) {
					return s;
				}
				memcpy(p->staging_buf.mapped, desc->host_data, size);
				VkBufferCopy copy = {.dstOffset = 0, .size = size};
				vkCmdCopyBuffer(p->cmd, p->staging_buf.buf, b->buf, 1, &copy);
				s = vk_run_cmd(p);
				if (s != OK)
					return s;
			}

			out->handle = b;
			out->size = size;
			out->host_ptr = desc->host_data;
			out->owner = self;
			return OK;
		}
	}

	vk_buf *b = xcalloc(1, sizeof(vk_buf));

	VkBufferUsageFlags weight_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
																		VK_BUFFER_USAGE_TRANSFER_DST_BIT |
																		VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	if (p->caps.unified_memory) {
		VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
																	VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
																	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		status_code s = vk_alloc_buffer(p, size, weight_usage, flags, b);
		if (s != OK) {
			s = vk_alloc_buffer(
				p, size, weight_usage,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b);
		}
		if (s != OK) {
			ERROR("vk: weight alloc failed for %zu bytes on UMA device -- "
						"per-BO size limit may be exceeded. Model may be too "
						"large for this GPU.",
						size);
			free(b);
			return s;
		}

		memcpy(b->mapped, desc->host_data, size);
	} else {
		status_code s =
			vk_alloc_buffer(p, size, weight_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b);
		if (s != OK) {
			s = vk_alloc_buffer(
				p, size, weight_usage,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b);
		}
		if (s != OK) {
			free(b);
			return s;
		}

		s = vk_ensure_staging_cap(p, size);
		if (s != OK) {
			vk_free_buffer(p, b);
			free(b);
			return s;
		}

		memcpy(p->staging_buf.mapped, desc->host_data, size);

		VkBufferCopy copy = {.size = size};
		vkCmdCopyBuffer(p->cmd, p->staging_buf.buf, b->buf, 1, &copy);
		status_code rs = vk_run_cmd(p);
		if (rs != OK) {
			vk_free_buffer(p, b);
			free(b);
			return rs;
		}
	}

	out->handle = b;
	out->size = size;
	out->host_ptr = desc->host_data;
	out->owner = self;
	return OK;
}

static void vk_buffer_free(backend *self, buffer *buf) {
	vk_priv *p = self->priv;
	if (!buf || !buf->handle)
		return;
	if (vk_kv_registry_remove(p, buf->handle)) {
		status_code fs = vk_flush(p);
		if (fs != OK) {
			ERROR("vk: vk_buffer_free: flush failed (status=%d)", (int)fs);
		}
		vk_kv_handle *kh = (vk_kv_handle *)buf->handle;
		vk_kv_store_free(p, &kh->store);
		free(kh);
		buf->handle = NULL;
		buf->size = 0;
		return;
	}
	vk_buf *b = as_vkbuf(buf);

	if (p->scratch_pool_count < VK_SCRATCH_POOL_CAP) {
		p->scratch_pool[p->scratch_pool_count] = b;
		p->scratch_pool_sizes[p->scratch_pool_count] = b->size;
		p->scratch_pool_count++;
		buf->handle = NULL;
		buf->size = 0;
		return;
	}

	vk_free_buffer(p, b);
	free(b);
	buf->handle = NULL;
	buf->size = 0;
}

static void vk_kv_free(backend *self, buffer *k, buffer *v) {
	vk_buffer_free(self, k);
	vk_buffer_free(self, v);
}

static status_code vk_ensure_staging_cap(vk_priv *p, size_t need) {
	if (need <= p->staging_cap && p->staging_buf.buf)
		return OK;

	if (p->staging_buf.buf) {
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

static status_code vk_buffer_upload(backend *self, buffer *buf, const void *host_src, size_t size) {
	vk_priv *p = self->priv;
	vk_buf *vb = as_vkbuf(buf);
	size_t off = buf->offset;

	if (vb->mapped) {
		vk_ring_slot *r = &p->ring[p->ring_cur];
		if (r->has_work && !p->device_lost) {
			status_code fs = vk_flush(p);
			if (fs != OK)
				return fs;
		}
		if (r->has_work) {
			r->has_work = 0;
			p->dirty_count = 0;
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

	VkBufferCopy copy = {.dstOffset = off, .size = size};
	vkCmdCopyBuffer(p->cmd, p->staging_buf.buf, vb->buf, 1, &copy);
	return vk_run_cmd(p);
}

static status_code vk_buf_download_raw(backend *self, vk_buf *vb, size_t off, void *host_dst,
									   size_t size) {
	vk_priv *p = self->priv;

	if (vb->mapped) {
		vk_ring_slot *r = &p->ring[p->ring_cur];
		if (r->has_work && !p->device_lost) {
			status_code fs = vk_flush(p);
			if (fs != OK)
				return fs;
		}
		if (r->has_work) {
			r->has_work = 0;
			p->dirty_count = 0;
		}
		memcpy(host_dst, (uint8_t *)vb->mapped + off, size);
		return OK;
	}

	if (p->device_lost)
		return ERR_INTERNAL;

	if (p->dirty_count > 0) {
		VkMemoryBarrier barrier = {
				.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		};
		vkCmdPipelineBarrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
		p->dirty_count = 0;
	}

	status_code s = vk_ensure_staging_cap(p, size);
	if (s != OK)
		return s;

	VkBufferCopy copy = {.srcOffset = off, .size = size};
	vkCmdCopyBuffer(p->cmd, vb->buf, p->staging_buf.buf, 1, &copy);

	status_code rs = vk_run_cmd(p);
	if (rs != OK)
		return rs;

	memcpy(host_dst, p->staging_buf.mapped, size);
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

static status_code vk_kv_store_alloc(vk_priv *p, size_t total, size_t per_layer_size,
									 int n_kv_layers, const size_t *layer_bytes,
									 size_t *layer_off_elems, vk_kv_store *out) {
	memset(out, 0, sizeof(*out));
	out->n_kv_layers = n_kv_layers > 0 ? n_kv_layers : 1;
	out->layer_off_elems = NULL;
	out->layer_bytes = NULL;
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
	VkBufferUsageFlags kv_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
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
			return OK;
		}
		if (n_kv_layers > 1) {
			WARN("vulkan: single KV allocation of %zu bytes failed -- "
					 "retrying as %d per-layer chunks of %zu bytes each "
					 "(common on UMA drivers with per-BO size limits)",
					 total, n_kv_layers, per_layer_for_checks);
			needs_chunking = 1;
		} else {
			ERROR("kv_alloc: failed to allocate %zu bytes for KV cache "
						"(single layer, cannot chunk further). The GPU may not "
						"have enough memory for this context length. Try a "
						"smaller --ctx-size or --accel cpu.",
						total);
			return ERR_OUT_OF_MEMORY;
		}
	}

	if (p->caps.max_storage_buffer_range > 0 &&
			per_layer_for_checks > p->caps.max_storage_buffer_range) {
		ERROR("kv_alloc: a single KV layer needs %zu bytes, which still "
					"exceeds this device's maxStorageBufferRange (%llu bytes); "
					"this model's context/head configuration cannot fit on this "
					"GPU. Try a smaller --ctx-size or run with --accel cpu.",
			  per_layer_for_checks, (unsigned long long)p->caps.max_storage_buffer_range);
		return ERR_OUT_OF_MEMORY;
	}

	int n = out->n_kv_layers;
	vk_buf *chunks = xcalloc((size_t)n, sizeof(vk_buf));
	for (int i = 0; i < n; i++) {
		size_t layer_sz = layer_bytes ? layer_bytes[i] : per_layer_size;
		status_code s		 = vk_alloc_buffer(p, layer_sz, kv_usage, kv_preferred, &chunks[i]);
		if (s != OK)
			s = vk_alloc_buffer(p, layer_sz, kv_usage, kv_fallback, &chunks[i]);
		if (s != OK) {
			for (int j = 0; j < i; j++)
				vk_free_buffer(p, &chunks[j]);
			free(chunks);
			ERROR("kv_alloc: failed to allocate KV layer %d/%d (%zu bytes). "
						"Available GPU memory may be insufficient for this model "
						"and context length. Try a smaller --ctx-size or --accel cpu.",
						i, n, layer_sz);
			return s;
		}
	}
	out->chunked = 1;
	out->n_chunks = n;
	out->chunks = chunks;
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
		*shader_layer_out = 0;
		return &h->store.chunks[idx];
	}
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

static size_t vk_kv_layer_n_elems(const vk_kv_handle *h, int layer) {
	if (layer < 0)
		layer = 0;
	vk_kv_store *s = (vk_kv_store *)&h->store;
	if (s->n_kv_layers < 1)
		return s->per_layer_size / sizeof(uint16_t);
	if (layer >= s->n_kv_layers)
		layer = s->n_kv_layers - 1;
	if (s->layer_bytes)
		return s->layer_bytes[layer] / sizeof(uint16_t);
	return s->per_layer_size / sizeof(uint16_t);
}

static status_code vk_kv_alloc(backend *self, const kv_desc *desc, buffer *k_out, buffer *v_out) {
	vk_priv *p = self->priv;
	p->n_ctx_stored = desc->n_ctx;
	p->kv_head_dim_max = desc->head_dim;

	int n_kv_layers = desc->n_kv_layers > 0 ? desc->n_kv_layers : 1;
	int has_layer_dims = desc->layer_head_dim && desc->layer_n_kv_heads && n_kv_layers > 0;

	size_t *layer_bytes = NULL;
	size_t *layer_off_elems = NULL;
	size_t per_layer_uniform = 0;
	size_t total;
	if (has_layer_dims) {
		layer_bytes = xcalloc((size_t)n_kv_layers, sizeof(size_t));
		layer_off_elems = xcalloc((size_t)n_kv_layers + 1, sizeof(size_t));
		size_t acc = 0;
		for (int i = 0; i < n_kv_layers; i++) {
			size_t elems =
				(size_t)desc->layer_n_kv_heads[i] * desc->n_ctx * (size_t)desc->layer_head_dim[i];
			layer_bytes[i] = elems * sizeof(uint16_t);
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
	kh->magic = VK_KV_MAGIC;
	vh->magic = VK_KV_MAGIC;

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

	k_out->handle = kh;
	k_out->size = total;
	k_out->host_ptr = NULL;
	k_out->owner = self;
	v_out->handle = vh;
	v_out->size = total;
	v_out->host_ptr = NULL;
	v_out->owner = self;
	vk_kv_registry_add(p, kh);
	vk_kv_registry_add(p, vh);
	return OK;
}

static status_code vk_kv_put(backend *self, buffer *k, buffer *v, int layer, int pos,
							 const buffer *k_in, const buffer *v_in, int n_kv_heads, int head_dim,
							 int n_ctx, int n_kv_heads_active) {
	vk_priv *p = self->priv;
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	const vk_kv_handle *kh = (const vk_kv_handle *)k->handle;
	const vk_kv_handle *vh = (const vk_kv_handle *)v->handle;
	int shader_layer_k;
	int shader_layer_v;
	vk_buf *kb = vk_kv_layer_buf(kh, layer, &shader_layer_k);
	vk_buf *vb = vk_kv_layer_buf(vh, layer, &shader_layer_v);

	struct {
		uint32_t layer_off;
		int32_t pos_start, n_kv_heads, head_dim, n_ctx, stride_head_dim;
		int32_t in_row_stride, m_tokens;
	} push					 = {(uint32_t)vk_kv_layer_off_elems(kh, layer),
								pos,
								n_kv_heads,
								head_dim,
								n_ctx,
								head_dim,
								n_kv_heads * head_dim,
								1};
	vk_buf *bufs[4] = {kb, vb, as_vkbuf(k_in), as_vkbuf(v_in)};
	VkDeviceSize offs[4]	 = {0, 0, k_in->offset, v_in->offset};
	int total_pairs = n_active * (head_dim / 2);
	uint32_t groups = (uint32_t)((total_pairs + 127) / 128);
	return vk_dispatch_ex(p, &p->p_kv_put, bufs, offs, NULL, 4, &push, sizeof(push), groups, 1,
						  0x3);
}

static status_code vk_kv_put_batch(backend *self, buffer *k, buffer *v, int layer, int pos_start,
								   const buffer *k_in, const buffer *v_in, int in_row_stride,
								   int n_kv_heads, int head_dim, int n_ctx, int n_kv_heads_active,
								   int m) {
	vk_priv *p = self->priv;
	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	const vk_kv_handle *kh = (const vk_kv_handle *)k->handle;
	const vk_kv_handle *vh = (const vk_kv_handle *)v->handle;
	int shader_layer_k;
	int shader_layer_v;
	vk_buf *kb = vk_kv_layer_buf(kh, layer, &shader_layer_k);
	vk_buf *vb = vk_kv_layer_buf(vh, layer, &shader_layer_v);

	struct {
		uint32_t layer_off;
		int32_t pos_start, n_kv_heads, head_dim, n_ctx, stride_head_dim;
		int32_t in_row_stride, m_tokens;
	} push					 = {(uint32_t)vk_kv_layer_off_elems(kh, layer),
								pos_start,
								n_kv_heads,
								head_dim,
								n_ctx,
								head_dim,
								in_row_stride > 0 ? in_row_stride : n_kv_heads * head_dim,
								m};
	vk_buf *bufs[4] = {kb, vb, as_vkbuf(k_in), as_vkbuf(v_in)};
	VkDeviceSize offs[4]	 = {0, 0, k_in->offset, v_in->offset};
	int total_pairs = n_active * (head_dim / 2);
	uint64_t total = (uint64_t)total_pairs * (uint64_t)m;
	uint32_t groups = (uint32_t)((total + 127) / 128);
	return vk_dispatch_ex(p, &p->p_kv_put, bufs, offs, NULL, 4, &push, sizeof(push), groups, 1,
						  0x3);
}

static status_code vk_embd_lookup(backend *self, const buffer *tok_embd, uint32_t tok_embd_type,
								  int token, int dim, buffer *x_out) {
	vk_priv *p = self->priv;

	if (!tok_embd->handle && tok_embd->host_ptr) {
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
		} push = {token, dim, shader_type, (int)row_stride};
		vk_buf *dummy = vk_dummy_buf(p);
		vk_buf *grid_buf =
			(tok_embd_type == GGML_TYPE_IQ3_S && p->iq3s_grid_buf.buf) ? &p->iq3s_grid_buf : dummy;
		vk_buf *bufs[3] = {as_vkbuf(tok_embd), as_vkbuf(x_out), grid_buf};
		VkDeviceSize offs[3] = {tok_embd->offset, x_out->offset, 0};
		uint32_t groups = (uint32_t)((dim + 63) / 64);
		return vk_dispatch_ex(p, &p->p_embd_lookup, bufs, offs, NULL, 3, &push, sizeof(push),
							  groups, 1, 0x2);
	}

	size_t row_stride = ggml_row_size(tok_embd_type, dim);
	const uint8_t *embd_host;
	if (tok_embd->host_ptr) {
		embd_host = (const uint8_t *)tok_embd->host_ptr + ((size_t)token * row_stride);
	} else {
		return ERR_UNSUPPORTED;
	}

	float *dst_host = xmalloc((size_t)dim * sizeof(float));
	switch (tok_embd_type) {
	case GGML_TYPE_F32:
		memcpy(dst_host, embd_host, (size_t)dim * sizeof(float));
		break;
	case GGML_TYPE_F16:
		dequant_f16_row(embd_host, dim, dst_host);
		break;
	case GGML_TYPE_IQ3_S:
		dequant_iq3_s_row(embd_host, (size_t)dim / 256, dst_host);
		break;
	default:
		free(dst_host);
		return ERR_UNSUPPORTED;
	}

	status_code s = vk_buffer_write_f32(self, x_out, dst_host, dim);
	free(dst_host);
	return s;
}

static vk_pipeline_set *vk_matmul_batch_pipeline(vk_priv *p, uint32_t w_type, int residual);
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
static status_code vk_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up,
										 buffer *out, int n, int activation, int m);
static status_code vk_add_batch(backend *self, buffer *x, const buffer *y, int n, int m);

static status_code vk_rmsnorm(backend *self, const buffer *x, const buffer *w, buffer *y, int n,
							  float eps) {
	return vk_rmsnorm_batch(self, x, w, y, n, eps, 1);
}

static int kquant_index(uint32_t w_type) {
	switch (w_type) {
	case GGML_TYPE_Q4_K:
		return 0;
	case GGML_TYPE_Q5_K:
		return 1;
	case GGML_TYPE_Q6_K:
		return 2;
	case GGML_TYPE_IQ4_NL:
		return 8;
	case GGML_TYPE_IQ3_S:
		return 9;
	default:
		return -1;
	}
}

static uint32_t kquant_probe_rng_step(uint32_t *s) {
	*s ^= *s << 13;
	*s ^= *s >> 17;
	*s ^= *s << 5;
	return *s;
}

static int kquant_probe_shape(backend *self, const kquant_probe_fmt *fmt, int n, int k,
							  uint32_t seed, float *out_gpu0, float *out_cpu0,
							  status_code *out_status) {
	int blocks_per_row = k / fmt->block_elems;
	size_t row_bytes = fmt->block_bytes * (size_t)blocks_per_row;
	size_t total_bytes = row_bytes * (size_t)n;

	uint8_t *wbuf = xcalloc(1, total_bytes);
	uint32_t rng = seed ? seed : 0x9E3779B9u;
	for (size_t i = 0; i < total_bytes; i++)
		wbuf[i] = (uint8_t)(kquant_probe_rng_step(&rng) & 0xFF);

	int n_blocks = (int)((size_t)n * blocks_per_row);
	for (int b = 0; b < n_blocks; b++) {
		uint8_t *bp = wbuf + ((size_t)b * fmt->block_bytes);
		float	 d_val = 0.0005f + (0.02f * ((kquant_probe_rng_step(&rng) % 997) / 997.0f));
		uint16_t d16 = f32_to_f16(d_val);
		memcpy(bp + fmt->d_off, &d16, 2);
		if (fmt->has_dmin) {
			float	 dmin_val = 0.0002f + (0.005f * ((kquant_probe_rng_step(&rng) % 997) / 997.0f));
			uint16_t dmin16 = f32_to_f16(dmin_val);
			memcpy(bp + fmt->dmin_off, &dmin16, 2);
		}
	}

	float *x = xmalloc((size_t)k * sizeof(float));
	for (int i = 0; i < k; i++) {
		int32_t v = (int32_t)(kquant_probe_rng_step(&rng) % 2001) - 1000;
		x[i] = (float)v / 1000.0f;
	}

	tensor_desc wd = {
			.host_data = wbuf,
			.type = fmt->w_type,
			.n_dims = 2,
			.dims = {(uint64_t)k, (uint64_t)n},
	};
	buffer w_gpu = {0};
	buffer x_gpu = {0};
	buffer y_gpu = {0};
	self->buffer_alloc_weight(self, &wd, &w_gpu);
	self->buffer_alloc_scratch(self, (size_t)k * sizeof(float), &x_gpu);
	self->buffer_alloc_scratch(self, (size_t)n * sizeof(float), &y_gpu);
	self->buffer_write_f32(self, &x_gpu, x, k);

	status_code s = self->matmul(self, &w_gpu, fmt->w_type, &x_gpu, &y_gpu, n, k);
	float *y = xcalloc((size_t)n, sizeof(float));
	if (s == OK) {
		if (self->synchronize)
			self->synchronize(self);
		vk_priv *p = self->priv;
		if (p->device_lost)
			s = ERR_INTERNAL;
		else
			self->buffer_read_f32(self, &y_gpu, y, n);
	}

	float *y_cpu = xcalloc((size_t)n, sizeof(float));
	matmul_generic_f32(wbuf, fmt->w_type, x, y_cpu, n, k);

	int broken = 0;
	if (s != OK) {
		broken = 1;
	} else {
		int n_zero_vs_nonzero = 0;
		for (int i = 0; i < n; i++) {
			if (!isfinite(y[i])) {
				broken = 1;
				break;
			}
			if (y[i] == 0.0f && fabsf(y_cpu[i]) > 1e-6f) {
				n_zero_vs_nonzero++;
				continue;
			}
			float abs_err = fabsf(y[i] - y_cpu[i]);
			float scale = fmaxf(fabsf(y_cpu[i]), fabsf(y[i]));
			if (scale > 1e-3f) {
				if (abs_err / scale > 0.5f) {
					broken = 1;
					break;
				}
			} else if (abs_err > 1.0f) {
				broken = 1;
				break;
			}
		}
		if (n_zero_vs_nonzero > n / 2)
			broken = 1;
	}

	if (out_gpu0)
		*out_gpu0 = y[0];
	if (out_cpu0)
		*out_cpu0 = y_cpu[0];
	if (out_status)
		*out_status = s;

	free(wbuf);
	free(x);
	free(y);
	free(y_cpu);
	self->buffer_free(self, &w_gpu);
	self->buffer_free(self, &x_gpu);
	self->buffer_free(self, &y_gpu);
	return broken;
}

static void vk_kquant_detect(backend *self) {
	vk_priv *p = self->priv;
	if (p->kquant_detect_done)
		return;
	p->kquant_detect_done = 1;

	int saved_batch = p->batch_active;
	p->batch_active = 0;

	struct {
		int n, k;
	} shapes[] = {
			{8, 256},
			{32, 256},
			{64, 512},
			{17, 256},
	};
	int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

	for (int fi = 0; fi < N_KQUANT_PROBE_FORMATS; fi++) {
		const kquant_probe_fmt *fmt = &KQUANT_PROBE_FORMATS[fi];
		int idx = kquant_index(fmt->w_type);
		if (idx < 0)
			continue;

		if (p->kquant_broken[idx])
			continue;

		for (int si = 0; si < n_shapes; si++) {
			if (shapes[si].k % fmt->block_elems != 0)
				continue;

			float gpu0 = 0.0f;
			float cpu0 = 0.0f;
			status_code st = OK;
			uint32_t seed = 0xA5A5u + ((uint32_t)fi * 131u) + ((uint32_t)si * 17u);
			int			broken =
				kquant_probe_shape(self, fmt, shapes[si].n, shapes[si].k, seed, &gpu0, &cpu0, &st);
			if (broken) {
				p->kquant_broken[idx] = 1;
				WARN("vulkan: %s matmul shader broken on this GPU -- "
						 "falling back to CPU (shape N=%d,K=%d gpu[0]=%f cpu[0]=%f "
						 "status=%d)",
						 fmt->name, shapes[si].n, shapes[si].k, gpu0, cpu0, st);
				break;
			}
			if (p->device_lost) {
				ERROR("vulkan: device lost during shader capability probe -- "
							"aborting remaining probes, all quant formats will fall back to "
							"CPU");
				for (int j = 0; j < 10; j++)
					p->kquant_broken[j] = 1;
				goto probe_done;
			}
		}
	}

probe_done:
	p->batch_active = saved_batch;
}

static int vk_kquant_should_fallback(vk_priv *p, uint32_t w_type) {
	int idx = kquant_index(w_type);
	if (idx < 0)
		return 0;
	return p->kquant_broken[idx];
}

static vk_pipeline_set *vk_matmul_pipeline(vk_priv *p, uint32_t w_type, int residual) {
	return vk_matmul_batch_pipeline(p, w_type, residual);
	}

static int vk_matmul_type_native(backend *self, uint32_t w_type) {
	vk_priv *p = self->priv;
	if (w_type == GGML_TYPE_IQ4_NL)
		return !vk_kquant_should_fallback(p, GGML_TYPE_IQ4_NL) &&
			   p->p_matmul_iq4_nl_batch.pipeline != NULL;
	vk_pipeline_set *ps = vk_matmul_pipeline(p, w_type, 0);
	if (!ps || !ps->pipeline)
		return 0;
	if (kquant_index(w_type) >= 0 && vk_kquant_should_fallback(p, w_type))
		return 0;
	return 1;
}

static status_code vk_matmul_impl(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								  const buffer *residual, buffer *y, int n, int k) {
	vk_priv *p = self->priv;
	int has_residual = (residual != NULL);

	if (!w->handle && w->host_ptr) {
		float	   *x_host = xmalloc((size_t)k * sizeof(float));
		status_code s	   = vk_buffer_read_f32(self, x, x_host, k);
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
		float		 *y_host = xmalloc((size_t)n * sizeof(float));
		quant_scratch qs	 = {0};
		switch (w_type) {
		case GGML_TYPE_Q4_K:
			matmul_q4_k_q8_k_f32(w->host_ptr, x_host, y_host, n, k, &qs);
			break;
		case GGML_TYPE_Q5_K:
			matmul_q5_k_q8_k_f32(w->host_ptr, x_host, y_host, n, k, &qs);
			break;
		case GGML_TYPE_Q6_K:
			matmul_q6_k_q8_f32(w->host_ptr, x_host, y_host, n, k, &qs);
			break;
		default:
			matmul_generic_f32(w->host_ptr, w_type, x_host, y_host, n, k);
			break;
		}
		free(qs.q8_buf);
		if (has_residual)
			for (int i = 0; i < n; i++)
				y_host[i] += r_host[i];
		s = vk_buffer_write_f32(self, y, y_host, n);
		free(x_host);
		free(r_host);
		free(y_host);
		return s;
	}

	if (kquant_index(w_type) >= 0 && !p->kquant_detect_done)
		vk_kquant_detect(self);

	if (vk_kquant_should_fallback(p, w_type) && w->host_ptr) {
		float *x_host = xmalloc((size_t)k * sizeof(float));
		status_code s = vk_buffer_read_f32(self, x, x_host, k);
		if (s != OK) {
			free(x_host);
			return s;
		}
		float *r_host = NULL;
		if (has_residual) {
			r_host = xmalloc((size_t)n * sizeof(float));
			s = vk_buffer_read_f32(self, residual, r_host, n);
			if (s != OK) {
				free(x_host);
				free(r_host);
				return s;
			}
		}
		float *y_host = xmalloc((size_t)n * sizeof(float));
		quant_scratch qs = {0};
		switch (w_type) {
		case GGML_TYPE_Q4_K:
			matmul_q4_k_q8_k_f32(w->host_ptr, x_host, y_host, n, k, &qs);
			break;
		case GGML_TYPE_Q5_K:
			matmul_q5_k_q8_k_f32(w->host_ptr, x_host, y_host, n, k, &qs);
			break;
		case GGML_TYPE_Q6_K:
			matmul_q6_k_q8_f32(w->host_ptr, x_host, y_host, n, k, &qs);
			break;
		default:
			matmul_generic_f32(w->host_ptr, w_type, x_host, y_host, n, k);
			break;
		}
		free(qs.q8_buf);
		if (has_residual)
			for (int i = 0; i < n; i++)
				y_host[i] += r_host[i];
		s = vk_buffer_write_f32(self, y, y_host, n);
		free(x_host);
		free(r_host);
		free(y_host);
		return s;
	}

	vk_pipeline_set *ps = vk_matmul_pipeline(p, w_type, has_residual);

	if (w_type == GGML_TYPE_IQ4_NL && !vk_kquant_should_fallback(p, GGML_TYPE_IQ4_NL) &&
		p->p_matmul_iq4_nl_batch.pipeline && k > 0 && (k % 32) == 0) {
		vk_buf *dummy = vk_dummy_buf(p);
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
		float *x_host = xmalloc((size_t)k * sizeof(float));
		status_code s = vk_buffer_read_f32(self, x, x_host, k);
		if (s != OK) {
			free(x_host);
			return s;
		}
		float *r_host = NULL;
		if (has_residual) {
			r_host = xmalloc((size_t)n * sizeof(float));
			s = vk_buffer_read_f32(self, residual, r_host, n);
			if (s != OK) {
				free(x_host);
				free(r_host);
				return s;
			}
		}
		float *y_host = xmalloc((size_t)n * sizeof(float));
		switch (w_type) {
		case GGML_TYPE_BF16:
			matmul_bf16_f32(w->host_ptr, x_host, y_host, n, k);
			break;
		default:
			matmul_generic_f32(w->host_ptr, w_type, x_host, y_host, n, k);
			break;
		}
		if (has_residual)
			for (int i = 0; i < n; i++)
				y_host[i] += r_host[i];
		s = vk_buffer_write_f32(self, y, y_host, n);
		free(x_host);
		free(r_host);
		free(y_host);
		return s;
	}

	int is_iq3_s = (w_type == GGML_TYPE_IQ3_S);
	int is_kquant = (w_type == GGML_TYPE_Q4_K || w_type == GGML_TYPE_Q5_K ||
									 w_type == GGML_TYPE_Q6_K || w_type == GGML_TYPE_IQ3_S);
	int unified_push = (w_type == GGML_TYPE_Q4_0 || w_type == GGML_TYPE_Q4_K ||
											w_type == GGML_TYPE_Q6_K || w_type == GGML_TYPE_IQ3_S);

	vk_buf *bufs[5];
	VkDeviceSize offs[5];
	int n_bufs;
	if (has_residual) {
		bufs[0] = as_vkbuf(w);
		offs[0] = w->offset;
		bufs[1] = as_vkbuf(x);
		offs[1] = x->offset;
		bufs[2] = as_vkbuf(residual);
		offs[2] = residual->offset;
		bufs[3] = as_vkbuf(y);
		offs[3] = y->offset;
		n_bufs = 4;
	} else {
		bufs[0] = as_vkbuf(w);
		offs[0] = w->offset;
		bufs[1] = as_vkbuf(x);
		offs[1] = x->offset;
		bufs[2] = as_vkbuf(y);
		offs[2] = y->offset;
		n_bufs = 3;
	}
	if (is_iq3_s && p->iq3s_grid_buf.buf) {
		bufs[n_bufs] = &p->iq3s_grid_buf;
		offs[n_bufs] = 0;
		n_bufs++;
	}
	uint32_t groups;
	if (is_kquant) {
		groups = (uint32_t)((n + (int)p->matmul_wg_size - 1) / (int)p->matmul_wg_size);
	} else {
		int rows = p->matmul_rows_per_thread;
		int wg = p->matmul_wg_size;
		groups = (uint32_t)((n + (wg * rows) - 1) / (wg * rows));
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

	int rows = p->matmul_rows_per_thread;
	int wg = p->matmul_wg_size;
	int n_max = (mode == 1) ? ((n > n1) ? n : n1) : n;
	uint32_t groups = (uint32_t)((n_max + (wg * rows) - 1) / (wg * rows));
	uint32_t wmask = (mode == 1) ? ((1u << 3) | (1u << 4)) : (1u << 3);
	return vk_dispatch_ex(p, &p->p_matmul_iq4_nl_batch, bufs, offs, NULL, 7, &push, sizeof(push),
						  groups, 1, wmask);
}

static status_code vk_matmul_multi(backend *self, const buffer **w, const uint32_t *w_types,
								   const buffer *x, buffer **y, const int *n_list, int k,
																	 int n_matmuls) {
	return vk_matmul_multi_batch(self, w, w_types, x, y, n_list, k, n_matmuls, 1);
}

static status_code vk_matmul_ffn_down(backend *self, const buffer *w, uint32_t w_type,
									  const buffer *gate, const buffer *up, buffer *y, int n, int k,
																			int activation) {
	vk_priv *p = self->priv;

	if (w_type == GGML_TYPE_IQ4_NL) {
		if (!p->kquant_detect_done)
			vk_kquant_detect(self);
		if (!vk_kquant_should_fallback(p, GGML_TYPE_IQ4_NL) && p->p_matmul_iq4_nl_batch.pipeline &&
			k > 0 && (k % 32) == 0) {
			return vk_dispatch_iq4_nl_unified(p, 2, w, gate, up, y, NULL, n, 0, k, activation, 0);
		}
	}

	float *gate_host = xmalloc((size_t)k * sizeof(float));
	float *up_host = xmalloc((size_t)k * sizeof(float));
	float *act_host = xmalloc((size_t)k * sizeof(float));
	float *y_host = xmalloc((size_t)n * sizeof(float));

	status_code s = vk_buffer_read_f32(self, gate, gate_host, k);
	if (s != OK)
		goto ffn_down_cleanup;
	s = vk_buffer_read_f32(self, up, up_host, k);
	if (s != OK)
		goto ffn_down_cleanup;

	for (int i = 0; i < k; i++) {
		float g = gate_host[i];
		float act = (activation == 1) ? gelu_tanh(g) : silu(g);
		act_host[i] = act * up_host[i];
	}

	if (w->host_ptr) {
		quant_scratch qs = {0};
		switch (w_type) {
		case GGML_TYPE_Q4_K:
			matmul_q4_k_q8_k_f32(w->host_ptr, act_host, y_host, n, k, &qs);
			break;
		case GGML_TYPE_Q5_K:
			matmul_q5_k_q8_k_f32(w->host_ptr, act_host, y_host, n, k, &qs);
			break;
		case GGML_TYPE_Q6_K:
			matmul_q6_k_q8_f32(w->host_ptr, act_host, y_host, n, k, &qs);
			break;
		default:
			matmul_generic_f32(w->host_ptr, w_type, act_host, y_host, n, k);
			break;
		}
		free(qs.q8_buf);
	} else {
		s = ERR_UNSUPPORTED;
		goto ffn_down_cleanup;
	}

	s = vk_buffer_write_f32(self, y, y_host, n);

ffn_down_cleanup:
	free(gate_host);
	free(up_host);
	free(act_host);
	free(y_host);
	return s;
}

static status_code vk_alloc_rope_buf(vk_priv *p, size_t size, buffer *out) {
	vk_buf *b = xcalloc(1, sizeof(vk_buf));
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

	out->handle = b;
	out->size = size;
	out->host_ptr = NULL;
	return OK;
}

static status_code vk_ensure_rope_bufs(backend *self, int half, const float *cos_base,
																			 const float *sin_base) {
	vk_priv *p = self->priv;
	int n_ctx = p->n_ctx_stored;
	if (n_ctx <= 0 || half <= 0)
		return ERR_INVALID_ARG;

	int need = n_ctx * half;

	buffer *cos_buf;
	buffer *sin_buf;
	int *cap;
	int *half_stored;
	const float **cos_base_p;
	const float **sin_base_p;

	if (p->rope_cos_base == cos_base && p->rope_sin_base == sin_base) {
		cos_buf = &p->rope_cos_buf;
		sin_buf = &p->rope_sin_buf;
		cap = &p->rope_buf_cap;
		half_stored = &p->rope_half_stored;
		cos_base_p = &p->rope_cos_base;
		sin_base_p = &p->rope_sin_base;
	} else if (p->rope_cos_base_alt == cos_base && p->rope_sin_base_alt == sin_base) {
		cos_buf = &p->rope_cos_buf_alt;
		sin_buf = &p->rope_sin_buf_alt;
		cap = &p->rope_buf_cap_alt;
		half_stored = &p->rope_half_stored_alt;
		cos_base_p = &p->rope_cos_base_alt;
		sin_base_p = &p->rope_sin_base_alt;
	} else if (p->rope_cos_base == NULL) {
		cos_buf = &p->rope_cos_buf;
		sin_buf = &p->rope_sin_buf;
		cap = &p->rope_buf_cap;
		half_stored = &p->rope_half_stored;
		cos_base_p = &p->rope_cos_base;
		sin_base_p = &p->rope_sin_base;
	} else {
		if (p->rope_cos_base_alt == NULL) {
			cos_buf = &p->rope_cos_buf_alt;
			sin_buf = &p->rope_sin_buf_alt;
			cap = &p->rope_buf_cap_alt;
			half_stored = &p->rope_half_stored_alt;
			cos_base_p = &p->rope_cos_base_alt;
			sin_base_p = &p->rope_sin_base_alt;
		} else {
			cos_buf = &p->rope_cos_buf;
			sin_buf = &p->rope_sin_buf;
			cap = &p->rope_buf_cap;
			half_stored = &p->rope_half_stored;
			cos_base_p = &p->rope_cos_base;
			sin_base_p = &p->rope_sin_base;
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
		*cos_base_p = cos_base;
		*sin_base_p = sin_base;
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

static status_code vk_ensure_scores_buf(backend *self, int n_heads, int n_ctx) {
	vk_priv *p = self->priv;
	int needed = n_heads * n_ctx;
	if (needed <= p->attn_scores_cap)
		return OK;

	if (p->attn_scores_buf.handle) {
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

static status_code vk_ensure_flash_pipeline(vk_priv *p, int head_dim, int n_groups,
																						vk_pipeline_set **out) {
	for (int i = 0; i < p->flash_count; i++) {
		if (p->flash_head_dim[i] == head_dim && p->flash_n_groups[i] == n_groups) {
			if (p->flash_unsupported[i])
				return ERR_UNSUPPORTED;
			*out = &p->p_attention_flash[i];
			return OK;
		}
	}
	if (p->flash_count >= VK_FLASH_CACHE_CAP)
		return ERR_UNSUPPORTED;

	int slot = p->flash_count;

	int lanes_per_head = 32;
	while (lanes_per_head > 1 && head_dim % lanes_per_head != 0)
		lanes_per_head >>= 1;

	int local_size = lanes_per_head * n_groups;
	int max_wg_inv = (int)p->caps.max_workgroup_invocations;
	int max_wg_size = (int)p->caps.max_workgroup_size[0];
	if (max_wg_inv > 0 && local_size > max_wg_inv) {
		p->flash_head_dim[slot] = head_dim;
		p->flash_n_groups[slot] = n_groups;
		p->flash_unsupported[slot] = 1;
		p->flash_count++;
		return ERR_UNSUPPORTED;
	}
	if (max_wg_size > 0 && local_size > max_wg_size) {
		p->flash_head_dim[slot] = head_dim;
		p->flash_n_groups[slot] = n_groups;
		p->flash_unsupported[slot] = 1;
		p->flash_count++;
		return ERR_UNSUPPORTED;
	}
	{
		const int TILE_T = 16;
		size_t	  red_tile_bytes = (size_t)n_groups * (size_t)lanes_per_head * sizeof(float);
		size_t	  kv_tile_bytes	 = 2 * (size_t)TILE_T * (size_t)head_dim * sizeof(float);
		size_t shared_bytes = kv_tile_bytes + red_tile_bytes;
		if (p->caps.max_shared_memory > 0 && shared_bytes > p->caps.max_shared_memory) {
			p->flash_head_dim[slot] = head_dim;
			p->flash_n_groups[slot] = n_groups;
			p->flash_unsupported[slot] = 1;
			p->flash_count++;
			return ERR_UNSUPPORTED;
		}
	}

	if (n_groups < 1 || head_dim > 256 || local_size > 1024 || head_dim % lanes_per_head != 0 ||
		head_dim / lanes_per_head > 16) {
		p->flash_head_dim[slot] = head_dim;
		p->flash_n_groups[slot] = n_groups;
		p->flash_unsupported[slot] = 1;
		p->flash_count++;
		return ERR_UNSUPPORTED;
	}

	uint32_t	spec_data[4] = {(uint32_t)head_dim, (uint32_t)n_groups, (uint32_t)lanes_per_head,
								(uint32_t)local_size};
	status_code s =
		vk_create_pipeline_spec(p, shader_attention_flash_spv, shader_attention_flash_spv_len, 4,
								36, spec_data, sizeof(spec_data), &p->p_attention_flash[slot]);
	if (s != OK) {
		WARN("flash attention pipeline creation failed for head_dim=%d "
				 "n_groups=%d, falling back",
				 head_dim, n_groups);
		p->flash_head_dim[slot] = head_dim;
		p->flash_n_groups[slot] = n_groups;
		p->flash_unsupported[slot] = 1;
		p->flash_count++;
		return s;
	}
	p->p_attention_flash[slot].name = "attention_flash";
	p->flash_head_dim[slot] = head_dim;
	p->flash_n_groups[slot] = n_groups;
	p->flash_unsupported[slot] = 0;
	p->flash_count++;
	*out = &p->p_attention_flash[slot];
	return OK;
}

static status_code vk_ensure_attention_big_pipeline(vk_priv *p, int head_dim,
																										vk_pipeline_set **out) {
	const size_t TILE_T = 8;
	size_t q_bytes = (size_t)512 * sizeof(float);
	size_t kv_tile_bytes = TILE_T * (size_t)512 * sizeof(float);
	size_t shared_bytes = q_bytes + kv_tile_bytes;
	if (p->caps.max_shared_memory > 0 && shared_bytes > p->caps.max_shared_memory) {
		WARN("attention_big pipeline needs %zu bytes shared (device has %u), "
				 "head_dim=%d",
				 shared_bytes, p->caps.max_shared_memory, head_dim);
		return ERR_UNSUPPORTED;
	}
	if (p->attention_big_ready) {
		*out = &p->p_attention_big;
		return OK;
	}
	status_code s = vk_create_pipeline(p, shader_attention_big_spv, shader_attention_big_spv_len, 5,
									   36, &p->p_attention_big);
	if (s != OK) {
		WARN("attention_big pipeline creation failed for head_dim=%d", head_dim);
		return s;
	}
	p->p_attention_big.name = "attention_big";
	p->attention_big_ready = 1;
	*out = &p->p_attention_big;
	return OK;
}

static status_code vk_attention_host_fallback(backend *self, const buffer *q, buffer *out,
											  vk_buf *kb, vk_buf *vb, size_t layer_off_elems,
											  size_t layer_n_elems, int n_heads, int n_kv_heads,
											  int n_active, int head_dim, int n_ctx, float scale,
											  int attn_start, int n_pos) {
	(void)n_kv_heads;
	int n_groups = (n_heads + n_active - 1) / n_active;
	int q_total = n_heads * head_dim;
	float *qf = xmalloc((size_t)q_total * sizeof(float));
	float *outf = xmalloc((size_t)q_total * sizeof(float));
	status_code st;
	st = self->buffer_read_f32(self, q, qf, q_total);
	if (st) {
		free(qf);
		free(outf);
		return st;
	}

	size_t kvh_stride = (size_t)n_ctx * head_dim;
	size_t dl_bytes = layer_n_elems * sizeof(uint16_t);

	uint16_t *kd_base_buf = xmalloc(dl_bytes);
	uint16_t *vd_base_buf = xmalloc(dl_bytes);
	st = vk_buf_download_raw(self, kb, layer_off_elems * sizeof(uint16_t), kd_base_buf, dl_bytes);
	if (st == OK)
		st = vk_buf_download_raw(self, vb, layer_off_elems * sizeof(uint16_t), vd_base_buf,
								 dl_bytes);
	if (st != OK) {
		free(qf);
		free(outf);
		free(kd_base_buf);
		free(vd_base_buf);
		return st;
	}

	float *k_slice = xmalloc((size_t)n_pos * head_dim * sizeof(float));
	float *v_slice = xmalloc((size_t)n_pos * head_dim * sizeof(float));
	float *scores = xmalloc((size_t)n_pos * sizeof(float));
	int cur_kvh = -1;

	for (int h = 0; h < n_heads; h++) {
		int kvh = h / n_groups;
		if (kvh != cur_kvh) {
			cur_kvh = kvh;
			for (int t = 0; t < n_pos; t++) {
				size_t kv_off = ((size_t)kvh * kvh_stride) + ((size_t)(attn_start + t) * head_dim);
				uint16_t *kd = kd_base_buf + kv_off;
				uint16_t *vd = vd_base_buf + kv_off;
				float *ks = k_slice + ((size_t)t * head_dim);
				float *vs = v_slice + ((size_t)t * head_dim);
				for (int d = 0; d < head_dim; d++) {
					ks[d] = f16_to_f32(kd[d]);
					vs[d] = f16_to_f32(vd[d]);
				}
			}
		}
		{
			float *qh = qf + ((size_t)h * head_dim);
			float *out_h = outf + ((size_t)h * head_dim);
			for (int t = 0; t < n_pos; t++) {
				float *kt = k_slice + ((size_t)t * head_dim);
				float dot = 0.0f;
				int d = 0;
				int hd_main = head_dim - (head_dim % 8);
				for (; d < hd_main; d += 8)
					dot += (qh[d] * kt[d]) + (qh[d + 1] * kt[d + 1]) + (qh[d + 2] * kt[d + 2]) +
						   (qh[d + 3] * kt[d + 3]) + (qh[d + 4] * kt[d + 4]) +
						   (qh[d + 5] * kt[d + 5]) + (qh[d + 6] * kt[d + 6]) +
						   (qh[d + 7] * kt[d + 7]);
				for (; d < head_dim; d++)
					dot += qh[d] * kt[d];
				scores[t] = dot * scale;
			}
			softmax_masked(scores, n_pos);
			for (int d = 0; d < head_dim; d++)
				out_h[d] = 0.0f;
			for (int t = 0; t < n_pos; t++) {
				float sv = scores[t];
				float *vt = v_slice + ((size_t)t * head_dim);
				for (int d = 0; d < head_dim; d++)
					out_h[d] += sv * vt[d];
			}
		}
	}
	free(k_slice);
	free(v_slice);
	free(scores);
	free(kd_base_buf);
	free(vd_base_buf);
	st = self->buffer_write_f32(self, out, outf, q_total);
	free(qf);
	free(outf);
	return st;
}

static status_code vk_attention_impl(backend *self, const buffer *q, const buffer *k_cache,
									const buffer *v_cache, buffer *out, int layer, int pos,
									int n_heads, int n_kv_heads, int head_dim, int n_ctx,
									int flash_attn, float scale, int n_kv_heads_active,
									const char *diag_label, int sliding_window, int attn_start,
									int n_pos, int dump_debug) {
	vk_priv *p = self->priv;

	int n_active = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;
	int stride_head_dim = p->kv_head_dim_max > 0 ? p->kv_head_dim_max : head_dim;

	const vk_kv_handle *kh = (const vk_kv_handle *)k_cache->handle;
	const vk_kv_handle *vh = (const vk_kv_handle *)v_cache->handle;
	int shader_layer_k;
	int shader_layer_v;
	vk_buf *kb = vk_kv_layer_buf(kh, layer, &shader_layer_k);
	vk_buf *vb = vk_kv_layer_buf(vh, layer, &shader_layer_v);

	if (p->debug.diag) {
		static int seen[512];
		static int n_seen = 0;
		int already = 0;
		for (int i = 0; i < n_seen; i++)
			if (seen[i] == layer) {
				already = 1;
				break;
			}
		if (!already && n_seen < 512) {
			seen[n_seen++] = layer;
			fprintf(stderr, "[VK_DIAG] %s path: %s\n", diag_label,
							(kb->mapped && vb->mapped) ? "cpu_mapped" : "gpu");
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

	int need_host_path = (n_active != n_kv_heads);

	if (need_host_path) {
		size_t layer_off = vk_kv_layer_off_elems(kh, layer);
		size_t layer_n = vk_kv_layer_n_elems(kh, layer);
		return vk_attention_host_fallback(self, q, out, kb, vb, layer_off, layer_n, n_heads,
										  n_kv_heads, n_active, head_dim, n_ctx, scale, attn_start,
										  n_pos);
	}

	int n_groups = n_heads / n_kv_heads;
	vk_pipeline_set *flash_ps = NULL;
	int can_flash = flash_attn && vk_ensure_flash_pipeline(p, head_dim, n_groups, &flash_ps) == OK;

	size_t layer_off = vk_kv_layer_off_elems(kh, layer);

	if (can_flash) {
		struct {
			uint32_t layer_off;
			int32_t pos, n_heads, n_kv_heads, head_dim, n_ctx;
			float scale;
			int32_t stride_head_dim;
			int32_t attn_start;
		} push = {(uint32_t)layer_off, pos, n_heads, n_kv_heads, head_dim, n_ctx, scale, head_dim,
				  attn_start};
		uint32_t groups = (uint32_t)n_kv_heads;
		vk_buf *bufs[4] = {as_vkbuf(q), kb, vb, as_vkbuf(out)};

		static int dbg_dumped = 0;
		int want_dump = dump_debug && getenv("VK_ATTN_DEBUG") != NULL && !dbg_dumped && pos <= 1;
		if (want_dump) {
			dbg_dumped = 1;
			int n_pos_dbg = pos + 1;
			int q_total = n_heads * head_dim;
			float *qf = xmalloc((size_t)q_total * sizeof(float));
			status_code qs = self->buffer_read_f32(self, q, qf, q_total);
			fprintf(stderr,
					"[VK_ATTN_DEBUG] === pre-dispatch dump: layer=%d pos=%d n_heads=%d "
					"n_kv_heads=%d head_dim=%d stride_head_dim=%d ===\n",
					layer, pos, n_heads, n_kv_heads, head_dim, stride_head_dim);
			if (qs == OK) {
				fprintf(stderr, "[VK_ATTN_DEBUG] Q[head=0][0:%d] =", head_dim);
				for (int i = 0; i < head_dim; i++)
					fprintf(stderr, " %+.6f", qf[i]);
				fprintf(stderr, "\n");
			} else {
				fprintf(stderr, "[VK_ATTN_DEBUG] Q read failed: %d\n", (int)qs);
			}
			free(qf);

			size_t layer_off_dump = vk_kv_layer_off_elems(kh, layer);
			size_t layer_n_dump = vk_kv_layer_n_elems(kh, layer);
			size_t dl_bytes = layer_n_dump * sizeof(uint16_t);
			uint16_t *kd_dump = xmalloc(dl_bytes);
			uint16_t *vd_dump = xmalloc(dl_bytes);
			status_code ks =
				vk_buf_download_raw(self, kb, layer_off_dump * sizeof(uint16_t), kd_dump, dl_bytes);
			status_code vs =
				vk_buf_download_raw(self, vb, layer_off_dump * sizeof(uint16_t), vd_dump, dl_bytes);
			if (ks == OK && vs == OK) {
				for (int t = 0; t <= pos; t++) {
					size_t kv_off = (size_t)t * head_dim;
					fprintf(stderr, "[VK_ATTN_DEBUG] K[kvh=0][t=%d][0:%d] =", t, head_dim);
					for (int i = 0; i < head_dim; i++)
						fprintf(stderr, " %+.6f", f16_to_f32(kd_dump[kv_off + i]));
					fprintf(stderr, "\n");
					fprintf(stderr, "[VK_ATTN_DEBUG] V[kvh=0][t=%d][0:%d] =", t, head_dim);
					for (int i = 0; i < head_dim; i++)
						fprintf(stderr, " %+.6f", f16_to_f32(vd_dump[kv_off + i]));
					fprintf(stderr, "\n");
				}
			} else {
				fprintf(stderr, "[VK_ATTN_DEBUG] K/V read failed: k=%d v=%d\n", (int)ks, (int)vs);
			}
			free(kd_dump);
			free(vd_dump);
			(void)n_pos_dbg;
		}

		status_code disp_status = vk_dispatch(p, flash_ps, bufs, 4, &push, sizeof(push), groups);

		if (want_dump && disp_status == OK) {
			if (self->synchronize)
				self->synchronize(self);
			int q_total = n_heads * head_dim;
			float *outf = xmalloc((size_t)q_total * sizeof(float));
			status_code os = self->buffer_read_f32(self, out, outf, q_total);
			if (os == OK) {
				fprintf(stderr, "[VK_ATTN_DEBUG] GPU out[head=0][0:8] =");
				for (int i = 0; i < 8 && i < head_dim; i++)
					fprintf(stderr, " %+.6f", outf[i]);
				fprintf(stderr, "\n[VK_ATTN_DEBUG] === end dump ===\n");
			} else {
				fprintf(stderr, "[VK_ATTN_DEBUG] out read failed: %d\n", (int)os);
			}
			free(outf);
		}

		return disp_status;
	}

	status_code s = vk_ensure_scores_buf(self, n_heads, n_ctx);
	if (s != OK)
		return s;
	struct {
		uint32_t layer_off;
		int32_t pos, n_heads, n_kv_heads, head_dim, n_ctx;
		float scale;
		int32_t stride_head_dim;
		int32_t attn_start;
	} push = {(uint32_t)layer_off, pos, n_heads, n_kv_heads, head_dim, n_ctx, scale, head_dim,
			  attn_start};
	vk_buf *bufs[5] = {as_vkbuf(q), kb, vb, as_vkbuf(&p->attn_scores_buf), as_vkbuf(out)};
	if (head_dim > 256) {
		vk_pipeline_set *ps = NULL;
		if (vk_ensure_attention_big_pipeline(p, head_dim, &ps) != OK)
			return ERR_UNSUPPORTED;
		return vk_dispatch_masked(p, ps, bufs, 5, &push, sizeof(push), (uint32_t)n_heads, 0x18);
	}
	return vk_dispatch_masked(p, &p->p_attention, bufs, 5, &push, sizeof(push), (uint32_t)n_heads,
							  0x18);
}

static status_code vk_attention(backend *self, const buffer *q, const buffer *k_cache,
								const buffer *v_cache, buffer *out, int layer, int pos, int n_heads,
								int n_kv_heads, int head_dim, int n_ctx, int flash_attn,
								float scale, int n_kv_heads_active) {
	return vk_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							 head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, "attention", 0,
							 0, pos + 1, 1);
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
		float scale;
		int32_t _pad;
		int32_t m;
	} push				 = {n, ELEM_MODE_SCALE_INPLACE, scale, 0, 1};
	vk_buf *dummy = vk_dummy_buf(p);
	vk_buf *bufs[3] = {as_vkbuf(x), dummy, dummy};
	VkDeviceSize offs[3] = {x->offset, 0, 0};
	uint32_t groups = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, 1, 0x1);
}

static status_code vk_copy_buffer(backend *self, const buffer *src, buffer *dst, int n) {
	vk_priv *p = self->priv;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, mode;
		float scale;
		int32_t _pad;
		int32_t m;
	} push				 = {n, ELEM_MODE_COPY, 0.0f, 0, 1};
	vk_buf *dummy = vk_dummy_buf(p);
	vk_buf *bufs[3] = {as_vkbuf(dst), as_vkbuf(src), dummy};
	VkDeviceSize offs[3] = {dst->offset, src->offset, 0};
	uint32_t groups = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, 1, 0x1);
}

static status_code vk_ple_combine(backend *self, buffer *ple, const buffer *proj, int n,
																	float combine_scale) {
	vk_priv *p = self->priv;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, mode;
		float scale;
		int32_t _pad;
		int32_t m;
	} push				 = {n, ELEM_MODE_PLE_COMBINE, combine_scale, 0, 1};
	vk_buf *dummy = vk_dummy_buf(p);
	vk_buf *bufs[3] = {as_vkbuf(ple), as_vkbuf(proj), dummy};
	VkDeviceSize offs[3] = {ple->offset, proj->offset, 0};
	uint32_t groups = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups, 1, 0x1);
}

static status_code vk_ffn_activate(backend *self, const buffer *gate, const buffer *up, buffer *out,
								   int n) {
	return vk_ffn_activate_batch(self, gate, up, out, n, 0, 1);
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
			p->dirty_count = 0;
		}
		if (!p->device_lost_warned) {
			p->device_lost_warned = 1;
			ERROR("vk: vk_synchronize: device is lost; pending GPU work "
						"was discarded. Subsequent GPU dispatches will be skipped "
						"and CPU fallbacks will be used where possible.");
		}
		return;
	}
	if (r->has_work) {
		status_code fs = vk_flush(p);
		if (fs != OK) {
			ERROR("vk: vk_synchronize: flush failed (status=%d) -- "
						"pending GPU work was NOT executed",
						(int)fs);
		}
	}
}

static void vk_begin_batch(backend *self) {
	vk_priv *p = self->priv;
	p->batch_active = 1;
}

static void vk_end_batch(backend *self) {
	vk_priv *p = self->priv;
	p->batch_active = 0;
	vk_ring_slot *r = &p->ring[p->ring_cur];
	if (p->device_lost) {
		if (r->has_work) {
			r->has_work = 0;
			p->dirty_count = 0;
		}
		if (!p->device_lost_warned) {
			p->device_lost_warned = 1;
			ERROR("vk: vk_end_batch: device is lost; batch was discarded.");
		}
		return;
	}
	if (r->has_work) {
		status_code fs = vk_flush(p);
		if (fs != OK) {
			ERROR("vk: vk_end_batch: flush failed (status=%d) -- "
						"batch was NOT submitted to the GPU",
						(int)fs);
		}
	}
}

static status_code vk_argmax(backend *self, const buffer *logits, int n, int32_t *out_idx) {
	vk_priv *p = self->priv;

	if (!p->argmax_out_buf.buf) {
		VkBufferUsageFlags usage =
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		VkMemoryPropertyFlags flags =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		status_code s = vk_alloc_buffer(p, 256, usage, flags, &p->argmax_out_buf);
		if (s != OK)
			return s;
	}

	struct {
		int32_t n;
	} push = {n};
	vk_buf *bufs[2] = {as_vkbuf(logits), &p->argmax_out_buf};
	status_code s = vk_dispatch(p, &p->p_argmax, bufs, 2, &push, sizeof(push), 1);
	if (s != OK)
		return s;

	s = vk_flush(p);
	if (s != OK)
		return s;

	uint32_t result;
	memcpy(&result, p->argmax_out_buf.mapped, sizeof(uint32_t));
	*out_idx = (int32_t)result;
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

static status_code vk_rmsnorm_add(backend *self, const buffer *x, const buffer *w,
								  const buffer *residual, buffer *y, int n, float eps) {
	vk_priv *p = self->priv;

	if (!w->handle && w->host_ptr) {
		status_code s = vk_rmsnorm(self, x, w, y, n, eps);
		if (s != OK)
			return s;
		return vk_add_inplace(self, y, residual, n);
	}

	if (!p->p_rmsnorm_add_batch.pipeline) {
		status_code s = vk_rmsnorm(self, x, w, y, n, eps);
		if (s != OK)
			return s;
		return vk_add_inplace(self, y, residual, n);
	}
	struct {
		int32_t n;
		float eps;
		int32_t m;
	} push				 = {n, eps, 1};
	vk_buf *bufs[4] = {as_vkbuf(x), as_vkbuf(w), as_vkbuf(residual), as_vkbuf(y)};
	VkDeviceSize offs[4] = {x->offset, w->offset, residual->offset, y->offset};
	return vk_dispatch_2d_ex(p, &p->p_rmsnorm_add_batch, bufs, offs, NULL, 4, &push, sizeof(push),
							 1, 1, 1u << 3);
}

static status_code vk_ffn_activate_ex(backend *self, const buffer *gate, const buffer *up,
									  buffer *out, int n, int activation) {
	return vk_ffn_activate_batch(self, gate, up, out, n, activation, 1);
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
	int n_pos = pos + 1;
	int attn_start = 0;
	if (sliding_window > 0 && n_pos > sliding_window)
		attn_start = n_pos - sliding_window;
	return vk_attention_impl(self, q, k_cache, v_cache, out, layer, pos, n_heads, n_kv_heads,
							 head_dim, n_ctx, flash_attn, scale, n_kv_heads_active, "attention_swa",
							 sliding_window, attn_start, n_pos - attn_start, 0);
}

static vk_pipeline_set *vk_matmul_batch_pipeline(vk_priv *p, uint32_t w_type, int residual) {
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
		return residual ? &p->p_matmul_f32_res_batch : &p->p_matmul_f32_batch;
	case GGML_TYPE_Q4_K:
		return residual ? &p->p_matmul_q4_k_res_batch : &p->p_matmul_q4_k_batch;
	case GGML_TYPE_Q5_K:
		return residual ? &p->p_matmul_q5_k_res_batch : &p->p_matmul_q5_k_batch;
	case GGML_TYPE_Q6_K:
		return residual ? &p->p_matmul_q6_k_res_batch : &p->p_matmul_q6_k_batch;
	case GGML_TYPE_IQ3_S:
		return residual ? &p->p_matmul_iq3_s_res_batch : &p->p_matmul_iq3_s_batch;
	default:
		return NULL;
	}
}

static status_code vk_matmul_batch(backend *self, const buffer *w, uint32_t w_type, const buffer *x,
								   buffer *y, int n, int k, int m) {
	vk_priv *p = self->priv;

	if (m <= 0)
		return ERR_INVALID_ARG;

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

	if (w_type == GGML_TYPE_IQ4_NL && !vk_kquant_should_fallback(p, GGML_TYPE_IQ4_NL) &&
		p->p_matmul_iq4_nl_batch.pipeline && k > 0 && (k % 32) == 0) {
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

	if (kquant_index(w_type) >= 0 && !p->kquant_detect_done)
		vk_kquant_detect(self);

	if (vk_kquant_should_fallback(p, w_type))
		return ERR_UNSUPPORTED;

	vk_pipeline_set *ps = vk_matmul_batch_pipeline(p, w_type, 0);
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

	int is_kquant	 = (w_type == GGML_TYPE_Q4_K || w_type == GGML_TYPE_Q5_K ||
						w_type == GGML_TYPE_Q6_K || w_type == GGML_TYPE_IQ3_S);
	int unified_push = (w_type == GGML_TYPE_Q4_0 || w_type == GGML_TYPE_Q4_K ||
						w_type == GGML_TYPE_Q6_K || w_type == GGML_TYPE_IQ3_S);

	uint32_t groups_x;
	if (is_kquant) {
		groups_x = (uint32_t)((n + p->matmul_wg_size - 1) / p->matmul_wg_size);
	} else {
		int rows = p->matmul_rows_per_thread;
		int wg	 = p->matmul_wg_size;
		groups_x = (uint32_t)((n + (wg * rows) - 1) / (wg * rows));
	}

	if (unified_push) {
		struct {
			int32_t n0, n1, k, m;
		} push = {n, 0, k, m};
		return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, n_bufs, &push, sizeof(push), groups_x,
								 (uint32_t)m, 1u << 2);
	} else {
		struct {
			int32_t n, k, m;
		} push = {n, k, m};
		return vk_dispatch_2d_ex(p, ps, bufs, offs, NULL, n_bufs, &push, sizeof(push), groups_x,
								 (uint32_t)m, 1u << 2);
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

	if (n_matmuls == 2 && w_types[0] == w_types[1]) {
		uint32_t wt = w_types[0];
		if (kquant_index(wt) >= 0 && !p->kquant_detect_done)
			vk_kquant_detect(self);
		int broken = vk_kquant_should_fallback(p, wt);

		vk_pipeline_set *ps = NULL;
		if (!broken) {
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

		if (wt == GGML_TYPE_IQ4_NL && !broken && p->p_matmul_iq4_nl_batch.pipeline && k > 0 &&
			(k % 32) == 0) {
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
	for (int i = 0; i < n_matmuls && st == OK; i++) {
		st = vk_matmul_batch(self, w[i], w_types[i], x, y[i], n_list[i], k, m);
	}
	return st;
}

static status_code vk_rmsnorm_batch(backend *self, const buffer *x, const buffer *w, buffer *y,
									int n, float eps, int m) {
	vk_priv *p = self->priv;

	if (!w->handle && w->host_ptr) {
		status_code st = OK;
		for (int row = 0; row < m && st == OK; row++) {
			buffer x_row =
				buffer_slice(x, (size_t)row * n * sizeof(float), (size_t)n * sizeof(float));
			buffer y_row =
				buffer_slice(y, (size_t)row * n * sizeof(float), (size_t)n * sizeof(float));
			float *x_host = xmalloc((size_t)n * sizeof(float));
			st			  = vk_buffer_read_f32(self, &x_row, x_host, n);
			if (st == OK) {
				rmsnorm(x_host, (const float *)w->host_ptr, x_host, n, eps);
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
		int			row_stride = n_heads * head_dim;
		status_code st		   = OK;
		for (int row = 0; row < m && st == OK; row++) {
			buffer x_row = buffer_slice(x, (size_t)row * row_stride * sizeof(float),
										(size_t)row_stride * sizeof(float));
			buffer y_row = buffer_slice(y, (size_t)row * row_stride * sizeof(float),
										(size_t)row_stride * sizeof(float));
			float *x_host = xmalloc((size_t)row_stride * sizeof(float));
			st			  = vk_buffer_read_f32(self, &x_row, x_host, row_stride);
			if (st == OK) {
				rmsnorm_per_head(x_host, (const float *)w->host_ptr, x_host, n_heads, head_dim,
								 eps);
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
	} push			= {n_heads, head_dim, eps, m};
	vk_buf *bufs[3] = {as_vkbuf(x), as_vkbuf(w), as_vkbuf(y)};
	return vk_dispatch_2d_ex(p, ps, bufs, NULL, NULL, 3, &push, sizeof(push), (uint32_t)n_heads,
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
	} push			= {n_heads, head_dim, eps, m};
	vk_buf *bufs[2] = {as_vkbuf(x), as_vkbuf(y)};
	return vk_dispatch_2d_ex(p, ps, bufs, NULL, NULL, 2, &push, sizeof(push), (uint32_t)n_heads,
							 (uint32_t)m, 1u << 1);
}

static status_code vk_add_batch(backend *self, buffer *x, const buffer *y, int n, int m) {
	vk_priv *p = self->priv;
	if (!p->p_elementwise_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, mode;
		float	scale;
		int32_t _pad;
		int32_t m;
	} push				  = {n, ELEM_MODE_ADD_INPLACE, 0.0f, 0, m};
	vk_buf		*dummy	  = vk_dummy_buf(p);
	vk_buf		*bufs[3]  = {as_vkbuf(x), as_vkbuf(y), dummy};
	VkDeviceSize offs[3]  = {x->offset, y->offset, 0};
	uint32_t	 groups_x = (uint32_t)((n + 127) / 128);
	return vk_dispatch_2d_ex(p, &p->p_elementwise_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups_x, (uint32_t)m, 0x1);
}

static status_code vk_ffn_activate_batch(backend *self, const buffer *gate, const buffer *up,
										 buffer *out, int n, int activation, int m) {
	vk_priv *p = self->priv;
	if (!p->p_ffn_activate_batch.pipeline)
		return ERR_UNSUPPORTED;
	struct {
		int32_t n, activation;
		int32_t m;
	} push				  = {n, activation, m};
	vk_buf		*bufs[3]  = {as_vkbuf(gate), as_vkbuf(up), as_vkbuf(out)};
	VkDeviceSize offs[3]  = {gate->offset, up->offset, out->offset};
	uint32_t	 groups_x = (uint32_t)((n + 63) / 64);
	return vk_dispatch_2d_ex(p, &p->p_ffn_activate_batch, bufs, offs, NULL, 3, &push, sizeof(push),
							 groups_x, (uint32_t)m, 1u << 2);
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

static status_code vk_rope_ext_batch(backend *self, buffer *vec, int n_heads, int head_dim,
									 int pos_start, const float *rope_cos_base,
									 const float *rope_sin_base, const float *freq_factors, int m) {
	vk_priv *p	  = self->priv;
	int		 half = head_dim / 2;

	if (freq_factors) {
		if (!p->p_rope_ext_batch.pipeline)
			return ERR_UNSUPPORTED;
		int n_ctx = p->n_ctx_stored;
		if (n_ctx <= 0 || half <= 0)
			return ERR_INVALID_ARG;
		size_t table_floats = (size_t)n_ctx * (size_t)half;
		size_t need			= table_floats * 2 * sizeof(float);
		int table_stale = (need > (size_t)p->rope_ff_cap || p->rope_ff_base != freq_factors ||
						   p->rope_ff_head_dim != head_dim || p->rope_ff_theta != self->rope_theta);
		if (table_stale) {
			if (need > (size_t)p->rope_ff_cap) {
				if (p->rope_ff_buf.handle)
					vk_buffer_free(self, &p->rope_ff_buf);
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
			p->rope_ff_theta	= self->rope_theta;
		}
		struct {
			int32_t n_heads, head_dim, pos, has_ff, neox, m;
		} push				  = {n_heads, head_dim, pos_start, 0, self->rope_neox, m};
		VkDeviceSize offs[4]  = {0, 0, (VkDeviceSize)table_floats * sizeof(float), 0};
		uint32_t	 total	  = (uint32_t)(n_heads * half);
		uint32_t	 groups_x = (total + 63) / 64;
		vk_buf		*ffb	  = as_vkbuf(&p->rope_ff_buf);
		vk_buf		*fbufs[4] = {as_vkbuf(vec), ffb, ffb, ffb};
		return vk_dispatch_2d_ex(p, &p->p_rope_ext_batch, fbufs, offs, NULL, 4, &push, sizeof(push),
								 groups_x, (uint32_t)m, 0x1);
	}

	status_code s = vk_ensure_rope_bufs(self, half, rope_cos_base, rope_sin_base);
	if (s != OK)
		return s;
	struct {
		int32_t n_heads, head_dim, pos, has_ff, neox, m;
	} push			  = {n_heads, head_dim, pos_start, 0, self->rope_neox, m};
	vk_buf	*bufs[4]  = {as_vkbuf(vec), as_vkbuf(p->rope_cos_buf_active),
						 as_vkbuf(p->rope_sin_buf_active), as_vkbuf(p->rope_cos_buf_active)};
	uint32_t total	  = (uint32_t)(n_heads * half);
	uint32_t groups_x = (total + 63) / 64;
	return vk_dispatch_2d_masked(p, &p->p_rope_ext_batch, bufs, 4, &push, sizeof(push), groups_x,
								 (uint32_t)m, 0x1);
}

static status_code vk_ensure_scores_buf_batch(backend *self, int n_heads, int n_ctx, int m) {
	vk_priv *p		= self->priv;
	int		 needed = m * n_heads * n_ctx;
	if (needed <= p->attn_scores_cap)
		return OK;
	if (p->attn_scores_buf.handle) {
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

static status_code vk_ensure_flash_pipeline_batch(vk_priv *p, int head_dim, int n_groups,
												  vk_pipeline_set **out) {
	for (int i = 0; i < p->flash_batch_count; i++) {
		if (p->flash_batch_head_dim[i] == head_dim && p->flash_batch_n_groups[i] == n_groups) {
			if (p->flash_batch_unsupported[i])
				return ERR_UNSUPPORTED;
			*out = &p->p_attention_flash_batch[i];
			return OK;
		}
	}
	if (p->flash_batch_count >= VK_FLASH_CACHE_CAP)
		return ERR_UNSUPPORTED;

	int slot		   = p->flash_batch_count;
	int lanes_per_head = 32;
	while (lanes_per_head > 1 && head_dim % lanes_per_head != 0)
		lanes_per_head >>= 1;
	int local_size	= lanes_per_head * n_groups;
	int max_wg_inv	= (int)p->caps.max_workgroup_invocations;
	int max_wg_size = (int)p->caps.max_workgroup_size[0];
	if (max_wg_inv > 0 && local_size > max_wg_inv) {
		p->flash_batch_head_dim[slot]	 = head_dim;
		p->flash_batch_n_groups[slot]	 = n_groups;
		p->flash_batch_unsupported[slot] = 1;
		p->flash_batch_count++;
		return ERR_UNSUPPORTED;
	}
	if (max_wg_size > 0 && local_size > max_wg_size) {
		p->flash_batch_head_dim[slot]	 = head_dim;
		p->flash_batch_n_groups[slot]	 = n_groups;
		p->flash_batch_unsupported[slot] = 1;
		p->flash_batch_count++;
		return ERR_UNSUPPORTED;
	}
	{
		const int TILE_T		 = 16;
		size_t	  red_tile_bytes = (size_t)n_groups * (size_t)lanes_per_head * sizeof(float);
		size_t	  kv_tile_bytes	 = 2 * (size_t)TILE_T * (size_t)head_dim * sizeof(float);
		size_t	  shared_bytes	 = kv_tile_bytes + red_tile_bytes;
		if (p->caps.max_shared_memory > 0 && shared_bytes > p->caps.max_shared_memory) {
			p->flash_batch_head_dim[slot]	 = head_dim;
			p->flash_batch_n_groups[slot]	 = n_groups;
			p->flash_batch_unsupported[slot] = 1;
			p->flash_batch_count++;
			return ERR_UNSUPPORTED;
		}
	}
	if (n_groups < 1 || head_dim > 256 || local_size > 1024 || head_dim % lanes_per_head != 0 ||
		head_dim / lanes_per_head > 16) {
		p->flash_batch_head_dim[slot]	 = head_dim;
		p->flash_batch_n_groups[slot]	 = n_groups;
		p->flash_batch_unsupported[slot] = 1;
		p->flash_batch_count++;
		return ERR_UNSUPPORTED;
	}
	uint32_t	spec_data[4] = {(uint32_t)head_dim, (uint32_t)n_groups, (uint32_t)lanes_per_head,
								(uint32_t)local_size};
	status_code s = vk_create_pipeline_spec(p, shader_attention_flash_batch_spv,
											shader_attention_flash_batch_spv_len, 4, 40, spec_data,
											sizeof(spec_data), &p->p_attention_flash_batch[slot]);
	if (s != OK) {
		p->flash_batch_head_dim[slot]	 = head_dim;
		p->flash_batch_n_groups[slot]	 = n_groups;
		p->flash_batch_unsupported[slot] = 1;
		p->flash_batch_count++;
		return s;
	}
	p->p_attention_flash_batch[slot].name = "attention_flash_batch";
	p->flash_batch_head_dim[slot]		  = head_dim;
	p->flash_batch_n_groups[slot]		  = n_groups;
	p->flash_batch_unsupported[slot]	  = 0;
	p->flash_batch_count++;
	*out = &p->p_attention_flash_batch[slot];
	return OK;
}

static status_code vk_ensure_attention_big_pipeline_batch(vk_priv *p, int head_dim,
														  vk_pipeline_set **out) {
	if (p->attention_big_batch_ready) {
		*out = &p->p_attention_big_batch;
		return OK;
	}
	status_code s =
		vk_create_pipeline(p, shader_attention_big_batch_spv, shader_attention_big_batch_spv_len, 5,
						   40, &p->p_attention_big_batch);
	if (s != OK) {
		WARN("attention_big_batch pipeline creation failed for head_dim=%d", head_dim);
		return s;
	}
	p->p_attention_big_batch.name = "attention_big_batch";
	p->attention_big_batch_ready  = 1;
	*out						  = &p->p_attention_big_batch;
	return OK;
}

static status_code vk_attention_batch_impl(backend *self, const buffer *q, const buffer *k_cache,
										   const buffer *v_cache, buffer *out, int layer,
										   int pos_start, int n_heads, int n_kv_heads, int head_dim,
										   int n_ctx, int flash_attn, float scale,
										   int n_kv_heads_active, int sliding_window,
										   int attn_start, int m) {
	(void)sliding_window;
	vk_priv *p				 = self->priv;
	int		 n_active		 = n_kv_heads_active > 0 ? n_kv_heads_active : n_kv_heads;

	if (n_active != n_kv_heads) {
		DEBUG("vk_attention_batch: unsupported sparse KV (n_kv_heads=%d, n_kv_heads_active=%d) "
			  "at layer=%d -- requires per-token host fallback",
			  n_kv_heads, n_active, layer);
		return ERR_UNSUPPORTED;
	}

	const vk_kv_handle *kh = (const vk_kv_handle *)k_cache->handle;
	const vk_kv_handle *vh = (const vk_kv_handle *)v_cache->handle;
	int					shader_layer_k, shader_layer_v;
	vk_buf			   *kb = vk_kv_layer_buf(kh, layer, &shader_layer_k);
	vk_buf			   *vb = vk_kv_layer_buf(vh, layer, &shader_layer_v);

	int				 n_groups = n_heads / n_kv_heads;
	vk_pipeline_set *flash_ps = NULL;
	int can_flash = flash_attn && p->p_attention_flash_batch[0].pipeline != VK_NULL_HANDLE &&
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
		uint32_t groups_x = (uint32_t)n_kv_heads;
		vk_buf	*bufs[4]  = {as_vkbuf(q), kb, vb, as_vkbuf(out)};
		return vk_dispatch_2d(p, flash_ps, bufs, 4, &push, sizeof(push), groups_x, (uint32_t)m);
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
	vk_buf *bufs[5] = {as_vkbuf(q), kb, vb, as_vkbuf(&p->attn_scores_buf), as_vkbuf(out)};
	if (head_dim > 256) {
		vk_pipeline_set *ps = NULL;
		if (vk_ensure_attention_big_pipeline_batch(p, head_dim, &ps) != OK) {
			DEBUG("vk_attention_batch: big_batch pipeline unavailable for head_dim=%d at layer=%d",
				  head_dim, layer);
			return ERR_UNSUPPORTED;
		}
		return vk_dispatch_2d_masked(p, ps, bufs, 5, &push, sizeof(push), (uint32_t)n_heads,
									 (uint32_t)m, 0x18);
	}
	if (!p->p_attention_batch.pipeline) {
		DEBUG("vk_attention_batch: attention_batch pipeline not initialized at layer=%d", layer);
		return ERR_UNSUPPORTED;
	}
	return vk_dispatch_2d_masked(p, &p->p_attention_batch, bufs, 5, &push, sizeof(push),
								 (uint32_t)n_heads, (uint32_t)m, 0x18);
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

static status_code vk_ctor(backend *out) {
	memset(out, 0, sizeof(*out));
	out->name	  = "vulkan";
	out->priority = 100;
	out->caps  = BCAP_ROPE_QK_FUSED | BCAP_MATMUL_RESIDUAL | BCAP_MULTI_MATMUL | BCAP_RMSNORM_ADD |
				 BCAP_MATMUL_FFN_DOWN;
	out->probe = vk_probe;
	out->init  = vk_init;
	out->free  = vk_free;
	out->buffer_alloc_weight	   = vk_buffer_alloc_weight;
	out->buffer_alloc_scratch	   = vk_buffer_alloc_scratch;
	out->buffer_free			   = vk_buffer_free;
	out->buffer_read_f32		   = vk_buffer_read_f32;
	out->buffer_write_f32		   = vk_buffer_write_f32;
	out->kv_alloc				   = vk_kv_alloc;
	out->kv_free				   = vk_kv_free;
	out->kv_put					   = vk_kv_put;
	out->kv_put_batch			   = vk_kv_put_batch;
	out->embd_lookup			   = vk_embd_lookup;
	out->rmsnorm				   = vk_rmsnorm;
	out->matmul					   = vk_matmul;
	out->matmul_type_native		   = vk_matmul_type_native;
	out->matmul_residual		   = vk_matmul_residual;
	out->matmul_multi			   = vk_matmul_multi;
	out->matmul_ffn_down		   = vk_matmul_ffn_down;
	out->rope					   = vk_rope;
	out->rope_qk				   = vk_rope_qk;
	out->attention				   = vk_attention;
	out->add_inplace			   = vk_add_inplace;
	out->scale_inplace			   = vk_scale_inplace;
	out->copy_buffer = vk_copy_buffer;
	out->ple_combine = vk_ple_combine;
	out->ffn_activate = vk_ffn_activate;
	out->argmax = vk_argmax;
	out->synchronize = vk_synchronize;
	out->begin_batch = vk_begin_batch;
	out->end_batch = vk_end_batch;
	out->rmsnorm_per_head = vk_rmsnorm_per_head;
	out->rmsnorm_noweight = vk_rmsnorm_noweight;
	out->rmsnorm_noweight_per_head = vk_rmsnorm_noweight_per_head;
	out->rmsnorm_add = vk_rmsnorm_add;
	out->ffn_activate_ex = vk_ffn_activate_ex;
	out->rope_ext = vk_rope_ext;
	out->attention_swa = vk_attention_swa;

	out->matmul_batch					 = vk_matmul_batch;
	out->matmul_multi_batch				 = vk_matmul_multi_batch;
	out->rmsnorm_batch					 = vk_rmsnorm_batch;
	out->rmsnorm_noweight_batch			 = vk_rmsnorm_noweight_batch;
	out->rmsnorm_per_head_batch			 = vk_rmsnorm_per_head_batch;
	out->rmsnorm_noweight_per_head_batch = vk_rmsnorm_noweight_per_head_batch;
	out->add_batch						 = vk_add_batch;
	out->ffn_activate_batch				 = vk_ffn_activate_batch;
	out->rope_batch						 = vk_rope_batch;
	out->rope_qk_batch					 = vk_rope_qk_batch;
	out->rope_ext_batch					 = vk_rope_ext_batch;
	out->attention_batch				 = vk_attention_batch;
	out->attention_swa_batch			 = vk_attention_swa_batch;
	return OK;
}

BACKEND_REGISTER("vulkan", vk_ctor)