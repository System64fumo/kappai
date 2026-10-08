CC      ?= gcc
GLSLC   ?= glslc
OBJCOPY ?= objcopy
STRIP   ?= strip

SRC_DIR := src
OUT_DIR := build
OBJ_DIR := $(OUT_DIR)/src

BUILD ?= debug
CPU_ARCH_OPT ?= 1
BACKENDS ?=
TSAN ?= 0

HOST_ARCH ?= $(shell uname -m)

.DEFAULT_GOAL := all
-include $(OUT_DIR)/config.mk

VALID_BUILDS := debug release
ifeq ($(filter $(BUILD),$(VALID_BUILDS)),)
  $(error Invalid BUILD='$(BUILD)'. Valid options: $(VALID_BUILDS))
endif

AVAILABLE_BACKENDS := $(sort $(notdir $(patsubst %/,%,$(filter-out %/cpu/,$(wildcard $(SRC_DIR)/backend/*/)))))
BACKENDS_TOKENS := $(subst ",,$(BACKENDS))
REQUESTED_BACKENDS := $(strip $(shell echo '$(BACKENDS_TOKENS)' | tr ',' ' '))
UNKNOWN_BACKENDS := $(filter-out $(AVAILABLE_BACKENDS),$(REQUESTED_BACKENDS))
ifneq ($(UNKNOWN_BACKENDS),)
  $(error Unknown backend(s): $(UNKNOWN_BACKENDS). Available backends: $(AVAILABLE_BACKENDS))
endif
HAS_VULKAN  := $(filter vulkan,$(REQUESTED_BACKENDS))
HAS_OPENGL := $(filter opengl,$(REQUESTED_BACKENDS))

CONFIG_FILE := $(OUT_DIR)/config.mk
CONFIG_AGNOSTIC_GOALS := clean format tidy print-config backends-help config
BUILD_GOALS := $(filter-out $(CONFIG_AGNOSTIC_GOALS),$(if $(MAKECMDGOALS),$(MAKECMDGOALS),all))
ifneq ($(BUILD_GOALS),)
  ifeq ($(wildcard $(CONFIG_FILE)),)
    $(error No build configuration found. Run 'make config' first)
  endif
endif

ifeq ($(HOST_ARCH),aarch64)
  DETECTED_CACHE_LINE := $(shell cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size 2>/dev/null)
  DETECTED_L1D_KB     := $(shell cat /sys/devices/system/cpu/cpu0/cache/index0/size 2>/dev/null | tr -dc '0-9')
  DETECTED_L2_KB      := $(shell for d in /sys/devices/system/cpu/cpu0/cache/index*; do \
                           if grep -qxE '(Unified)' $$d/type 2>/dev/null && [ "$$(cat $$d/level)" -gt 1 ]; then \
                             cat $$d/size | tr -dc '0-9'; break; fi; done)
endif
ifeq ($(HOST_ARCH),x86_64)
  DETECTED_CACHE_LINE := $(shell getconf LEVEL1_DCACHE_LINESIZE 2>/dev/null || echo 64)
  DETECTED_L1D_KB     := $(shell getconf LEVEL1_DCACHE_SIZE 2>/dev/null | tr -dc '0-9')
  DETECTED_L2_KB      := $(shell getconf LEVEL2_CACHE_SIZE 2>/dev/null | tr -dc '0-9')
endif
DETECTED_ARCH_FLAGS := $(shell $(CC) -### -E - -march=native 2>&1 | sed -rn '/cc1/!d;s/(\")|(^.* - )//g;s/ -dumpbase -//g;p')

ifeq ($(origin ARCH_FLAGS),undefined)
  ifneq ($(strip $(MACHINE_ARCH_FLAGS)),)
    ARCH_FLAGS := $(MACHINE_ARCH_FLAGS)
  else
    ARCH_FLAGS := $(DETECTED_ARCH_FLAGS)
  endif
endif

ifeq ($(wildcard $(OUT_DIR)/config.mk),)
  KAI_CACHE_LINE     := $(DETECTED_CACHE_LINE)
  KAI_L1D_KB         := $(DETECTED_L1D_KB)
  KAI_L2_KB          := $(DETECTED_L2_KB)
  MACHINE_ARCH_FLAGS := $(DETECTED_ARCH_FLAGS)
endif

BASE_FLAGS := -std=c11 -D_DEFAULT_SOURCE
DEP_FLAGS  := -MMD -MP
MATH_FLAGS := -fno-math-errno -fno-trapping-math -fno-signed-zeros -fcx-limited-range
WARN_FLAGS := -Wall -Wextra -Wformat=2

ifeq ($(TSAN),1)
  SANITIZE_FLAGS := -fsanitize=thread
else
  SANITIZE_FLAGS := -fsanitize=address,undefined -fno-sanitize-recover=undefined
endif

ifeq ($(CPU_ARCH_OPT),1)
  ifneq ($(strip $(KAI_CACHE_LINE)),)
    CFLAGS += -DCACHE_LINE_BYTES=$(KAI_CACHE_LINE)
  endif
  ifneq ($(strip $(KAI_L1D_KB)),)
    CFLAGS += -DL1D_SIZE_BYTES=$(shell echo $$(( $(KAI_L1D_KB) * 1024 )))
  endif
  ifneq ($(strip $(KAI_L2_KB)),)
    CFLAGS += -DL2_SIZE_BYTES=$(shell echo $$(( $(KAI_L2_KB) * 1024 )))
    ifeq ($(shell test $(KAI_L2_KB) -lt 256 && echo yes),yes)
      CFLAGS += -DPREFETCH_LOCALITY=0
    endif
  endif
endif

ifeq ($(BUILD),debug)
  CFLAGS  := $(BASE_FLAGS) $(DEP_FLAGS) -O2 -g -fno-omit-frame-pointer \
             $(SANITIZE_FLAGS) $(WARN_FLAGS) $(MATH_FLAGS) $(ARCH_FLAGS)
  LDFLAGS := -lm -lpthread $(SANITIZE_FLAGS)
else
  CFLAGS  := $(BASE_FLAGS) $(DEP_FLAGS) -O3 -funroll-loops -funroll-all-loops \
             -ftree-vectorize -fvect-cost-model=unlimited -fivopts -fweb \
             -frename-registers -fprefetch-loop-arrays -g \
             $(MATH_FLAGS) $(ARCH_FLAGS)
  LDFLAGS := -lm -lpthread
  LTO_FLAGS := -flto=$(shell nproc)
endif

define split_debug
	@$(OBJCOPY) --only-keep-debug $1 $1.debug
	@$(STRIP) --strip-unneeded $1
	@$(OBJCOPY) --add-gnu-debuglink=$1.debug $1
	@echo "  DBG     $1.debug"
endef

ifeq ($(BUILD),release)
  SPLIT_DEBUG = $(call split_debug,$@)
endif

ifneq ($(HAS_VULKAN),)
  VK_BACKEND_INCLUDES := -I$(OBJ_DIR)/backend/vulkan -I$(SRC_DIR)/backend/vulkan
endif

ifneq ($(HAS_OPENGL),)
  GL_BACKEND_INCLUDES := -I$(OBJ_DIR)/backend/opengl -I$(SRC_DIR)/backend/opengl
endif

LIB_SRCS := \
	$(wildcard $(SRC_DIR)/*.c) \
	$(wildcard $(SRC_DIR)/models/*.c) \
	$(wildcard $(SRC_DIR)/moe/*.c) \
	$(SRC_DIR)/backend/backend.c

BACKEND_DIR     := $(OUT_DIR)/backends
BACKEND_OBJ_DIR := $(OUT_DIR)/backend_obj
BACKEND_CFLAGS  := $(CFLAGS) -fvisibility=hidden

SCALAR_CORE_OBJS := \
	$(BACKEND_OBJ_DIR)/backend/cpu/scalar/core.o \
	$(BACKEND_OBJ_DIR)/backend/cpu/scalar/quants.o

SCALAR_BACKEND_OBJS := $(SCALAR_CORE_OBJS)

CPU_ARCH_DIR :=
ifeq ($(CPU_ARCH_OPT),1)
  ifeq ($(HOST_ARCH),aarch64)
    CPU_ARCH_DIR := aarch64
  endif
  ifeq ($(HOST_ARCH),x86_64)
    CPU_ARCH_DIR := x86_64
  endif
endif

SCALAR_BACKEND := $(BACKEND_DIR)/libkappai_cpu_scalar.so
BACKEND_LIBS   := $(SCALAR_BACKEND)
BACKEND_OBJS   := $(SCALAR_BACKEND_OBJS)

ifneq ($(CPU_ARCH_DIR),)
  ARCH_BACKEND_OBJS := $(SCALAR_CORE_OBJS) \
                       $(BACKEND_OBJ_DIR)/backend/cpu/$(CPU_ARCH_DIR)/core.o \
                       $(BACKEND_OBJ_DIR)/backend/cpu/$(CPU_ARCH_DIR)/quants.o
  ARCH_BACKEND := $(BACKEND_DIR)/libkappai_cpu_$(CPU_ARCH_DIR).so
  BACKEND_LIBS += $(ARCH_BACKEND)
  BACKEND_OBJS += $(ARCH_BACKEND_OBJS)
endif

ifneq ($(HAS_VULKAN),)
  VK_BACKEND_OBJS := $(BACKEND_OBJ_DIR)/backend/vulkan/vulkan.o
  VK_BACKEND := $(BACKEND_DIR)/libkappai_vulkan.so
  BACKEND_LIBS += $(VK_BACKEND)
  BACKEND_OBJS += $(VK_BACKEND_OBJS)
endif

ifneq ($(HAS_OPENGL),)
  GL_BACKEND_OBJS := \
	$(BACKEND_OBJ_DIR)/backend/opengl/opengl.o \
	$(BACKEND_OBJ_DIR)/backend/opengl/gl_context.o
  GL_BACKEND := $(BACKEND_DIR)/libkappai_opengl.so
  BACKEND_LIBS += $(GL_BACKEND)
  BACKEND_OBJS += $(GL_BACKEND_OBJS)
endif

ifneq ($(HAS_VULKAN),)
  BACKEND_CFLAGS_VULKAN := $(BACKEND_CFLAGS) $(VK_BACKEND_INCLUDES)
  $(VK_BACKEND_OBJS): BACKEND_CFLAGS := $(BACKEND_CFLAGS_VULKAN)
endif

ifneq ($(HAS_OPENGL),)
  BACKEND_CFLAGS_OPENGL := $(BACKEND_CFLAGS) $(GL_BACKEND_INCLUDES)
  $(GL_BACKEND_OBJS): BACKEND_CFLAGS := $(BACKEND_CFLAGS_OPENGL)
endif

TEST_SRCS   := $(wildcard $(SRC_DIR)/test/*.c)
SERVER_SRCS := $(wildcard $(SRC_DIR)/server/*.c)
SERVER_LIBS := -ljson-c -lmicrohttpd
CLI_SRCS    := $(wildcard $(SRC_DIR)/cli/*.c)
HEADERS     = $(shell find $(SRC_DIR) -type f \( -name "*.h" -o -name "*.hpp" \))
ALL_SRCS    = $(shell find $(SRC_DIR) -type f \( -name "*.c" -o -name "*.cpp" \))
ALL_SHADERS = $(shell find $(SRC_DIR) -type f \( -name "*.comp" -o -name "*.glsl" -o -name "*.inc" \))

LIB_OBJS     := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(LIB_SRCS))
TEST_OBJ_DIR := $(OBJ_DIR)/test
TEST_OBJS    := $(patsubst $(SRC_DIR)/test/%.c,$(TEST_OBJ_DIR)/%.o,$(TEST_SRCS))
SERVER_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SERVER_SRCS))
CLI_OBJS     := $(patsubst $(SRC_DIR)/cli/%.c,$(OBJ_DIR)/cli/%.o,$(CLI_SRCS))

ENGINE      := $(OUT_DIR)/libkappai.so
CLI_BIN     := $(OUT_DIR)/kappai-cli
SERVER_BIN  := $(OUT_DIR)/kappai-server
TEST_BIN    := $(OUT_DIR)/kappai-test
MONITOR_BIN := $(OUT_DIR)/kappai-monitor

SHIPPED_BINS := $(CLI_BIN) $(SERVER_BIN) $(TEST_BIN) $(ENGINE) $(BACKEND_LIBS)

ENGINE_LDFLAGS := -L$(OUT_DIR) -lkappai -Wl,-rpath,'$$ORIGIN' $(LDFLAGS)

BUILD_DIRS := $(BACKEND_DIR) $(sort $(dir $(LIB_OBJS) $(TEST_OBJS) $(SERVER_OBJS) $(CLI_OBJS)))

FORMAT_FLAGS := -i -style=file
TIDY_LOG     := $(OUT_DIR)/tidy.log

VK_SHADERS_DIR := $(SRC_DIR)/backend/vulkan/shaders
VK_INC_FILES   := $(filter %.glsl %.inc,$(wildcard $(VK_SHADERS_DIR)/*.glsl) $(wildcard $(VK_SHADERS_DIR)/*.inc))
VK_COMP_DEPS  := $(wildcard $(VK_SHADERS_DIR)/*.comp)

MATMUL_BATCH           := matmul_q4_0 matmul_q4_1 matmul_q5_0 matmul_q5_1 matmul_q8_0 matmul_q4_k matmul_q5_k matmul_q6_k matmul_iq3_s matmul_f32 matmul_f16 matmul_bf16
MATMUL_NMAT_DUAL_BATCH := matmul_q4_0 matmul_q4_k matmul_q6_k
MATMUL_NMAT_TRIPLE_BATCH := matmul_q4_0

RMSNORM_VARIANTS := rmsnorm_noweight rmsnorm_sg rmsnorm_noweight_sg \
                    rmsnorm_per_head rmsnorm_per_head_sg rmsnorm_add \
                    rmsnorm_noweight_per_head rmsnorm_noweight_per_head_sg
RMSNORM_ALL := rmsnorm $(RMSNORM_VARIANTS)

rmsnorm_FLAGS                      := -DHAS_WEIGHT
rmsnorm_noweight_FLAGS             :=
rmsnorm_sg_FLAGS                   := -DHAS_WEIGHT -DUSE_SUBGROUP
rmsnorm_noweight_sg_FLAGS          := -DUSE_SUBGROUP
rmsnorm_per_head_FLAGS             := -DHAS_WEIGHT -DPER_HEAD
rmsnorm_per_head_sg_FLAGS          := -DHAS_WEIGHT -DPER_HEAD -DUSE_SUBGROUP
rmsnorm_add_FLAGS                  := -DHAS_WEIGHT -DUSE_SUBGROUP -DADD_RESIDUAL
rmsnorm_noweight_per_head_FLAGS    := -DPER_HEAD
rmsnorm_noweight_per_head_sg_FLAGS := -DPER_HEAD -DUSE_SUBGROUP

VK_NONBATCH_SPVS := \
	$(OBJ_DIR)/backend/vulkan/argmax.spv \
	$(OBJ_DIR)/backend/vulkan/argmax_reduce.spv \
	$(OBJ_DIR)/backend/vulkan/moe_gather.spv \
	$(OBJ_DIR)/backend/vulkan/moe_combine.spv \
	$(OBJ_DIR)/backend/vulkan/attention.spv \
	$(OBJ_DIR)/backend/vulkan/attention_flash.spv \
	$(OBJ_DIR)/backend/vulkan/embd_lookup.spv \
	$(OBJ_DIR)/backend/vulkan/kv_put.spv

SHADER_SPVS := \
	$(VK_NONBATCH_SPVS) \
	$(foreach s,$(MATMUL_BATCH),$(OBJ_DIR)/backend/vulkan/$(s)_batch.spv) \
	$(foreach s,$(MATMUL_BATCH),$(OBJ_DIR)/backend/vulkan/$(s)_residual_batch.spv) \
	$(foreach s,$(MATMUL_NMAT_DUAL_BATCH),$(OBJ_DIR)/backend/vulkan/$(s)_dual_batch.spv) \
	$(foreach s,$(MATMUL_NMAT_TRIPLE_BATCH),$(OBJ_DIR)/backend/vulkan/$(s)_triple_batch.spv) \
	$(foreach s,$(RMSNORM_ALL),$(OBJ_DIR)/backend/vulkan/$(s)_batch.spv) \
	$(OBJ_DIR)/backend/vulkan/rope_batch.spv \
	$(OBJ_DIR)/backend/vulkan/rope_ext_batch.spv \
	$(OBJ_DIR)/backend/vulkan/rope_qk_batch.spv \
	$(OBJ_DIR)/backend/vulkan/attention_batch.spv \
	$(OBJ_DIR)/backend/vulkan/attention_flash_batch.spv \
	$(OBJ_DIR)/backend/vulkan/dequant.spv \
	$(OBJ_DIR)/backend/vulkan/ffn_activate_batch.spv \
	$(OBJ_DIR)/backend/vulkan/ffn_activate_fused_batch.spv \
	$(OBJ_DIR)/backend/vulkan/elementwise_batch.spv \
	$(OBJ_DIR)/backend/vulkan/matmul_iq4_nl_batch.spv

SHADERS_H := $(OBJ_DIR)/backend/vulkan/shaders_embedded.h

ifneq ($(HAS_VULKAN),)
  $(BACKEND_OBJ_DIR)/backend/vulkan/vulkan.o: $(SHADERS_H)

  $(OBJ_DIR)/backend/vulkan/%.spv: $(VK_SHADERS_DIR)/%.comp $(VK_INC_FILES) $(VK_COMP_DEPS) | $(OUT_DIR)
	@mkdir -p $(dir $@)
	@echo "  GLSLC   $@"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SHADERS_DIR) $< -o $@

  define VK_VARIANT_RULE
  $(OBJ_DIR)/backend/vulkan/$(1)$(2).spv: $(VK_SHADERS_DIR)/$(3).comp $(VK_INC_FILES) $(VK_COMP_DEPS) | $(OUT_DIR)
	@mkdir -p $$(dir $$@)
	@echo "  GLSLC   $(1)$(2).spv"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SHADERS_DIR) $(4) $$< -o $$@
  endef

  define RMSNORM_VARIANT_RULE
  $(OBJ_DIR)/backend/vulkan/$(1)$(2).spv: $(VK_SHADERS_DIR)/rmsnorm.comp $(VK_INC_FILES) $(VK_COMP_DEPS) | $(OUT_DIR)
	@mkdir -p $$(dir $$@)
	@echo "  GLSLC   $(1)$(2).spv  [$$(strip $$($(1)_FLAGS) $(3))]"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SHADERS_DIR) $$(strip $$($(1)_FLAGS) $(3)) $$< -o $$@
  endef

  $(foreach s,$(MATMUL_BATCH),$(eval $(call VK_VARIANT_RULE,$(s),_batch,$(s),-DBATCHED)))
  $(foreach s,$(MATMUL_BATCH),$(eval $(call VK_VARIANT_RULE,$(s),_residual_batch,$(s),-DHAS_RESIDUAL -DBATCHED)))
  $(foreach s,$(MATMUL_NMAT_DUAL_BATCH),$(eval $(call VK_VARIANT_RULE,$(s),_dual_batch,$(s),-DNMAT_DUAL -DBATCHED)))
  $(foreach s,$(MATMUL_NMAT_TRIPLE_BATCH),$(eval $(call VK_VARIANT_RULE,$(s),_triple_batch,$(s),-DNMAT_TRIPLE -DBATCHED)))
  $(foreach s,$(RMSNORM_ALL),$(eval $(call RMSNORM_VARIANT_RULE,$(s),_batch,-DBATCHED)))
  $(eval $(call VK_VARIANT_RULE,rope,_batch,rope,-DBATCHED))
  $(eval $(call VK_VARIANT_RULE,rope,_ext_batch,rope,-DHAS_FF -DBATCHED))
  $(eval $(call VK_VARIANT_RULE,rope,_qk_batch,rope_qk,-DBATCHED))
  $(eval $(call VK_VARIANT_RULE,attention,_batch,attention,-DBATCHED))
  $(eval $(call VK_VARIANT_RULE,attention,_flash_batch,attention_flash,-DBATCHED))
  $(eval $(call VK_VARIANT_RULE,ffn,_activate_batch,ffn_activate,-DBATCHED))
  $(eval $(call VK_VARIANT_RULE,elementwise,_batch,elementwise,-DBATCHED))
  $(eval $(call VK_VARIANT_RULE,matmul_iq4_nl,_batch,matmul_iq4_nl,-DBATCHED))

  $(SHADERS_H): $(SHADER_SPVS) | $(OUT_DIR)
	@echo "  GEN     $@"
	@printf '#ifndef SHADERS_H\n#define SHADERS_H\n\n#include <stdint.h>\n#include <stddef.h>\n\n' > $@
	@for spv in $^; do \
	        name=$$(basename $$spv .spv); \
	        varname=$$(echo "shader_$${name}_spv" | tr '-' '_'); \
	        printf 'static const uint32_t %s[] = {\n' "$$varname" >> $@; \
	        od -v -An -tx4 "$$spv" | sed 's/[0-9a-f]\{8\}/0x&,/g' >> $@; \
	        printf '};\n' >> $@; \
	        printf 'static const size_t %s_len = sizeof(%s);\n\n' "$$varname" "$$varname" >> $@; \
	done
	@printf '#endif\n' >> $@
endif

GL_SHADERS_DIR := $(SRC_DIR)/backend/opengl/shaders
GL_COMP_FILES  := $(sort $(wildcard $(GL_SHADERS_DIR)/*.comp))
GL_SHADERS_H   := $(OBJ_DIR)/backend/opengl/shaders_embedded.h

ifneq ($(HAS_OPENGL),)
  $(BACKEND_OBJ_DIR)/backend/opengl/opengl.o: $(GL_SHADERS_H)

  $(GL_SHADERS_H): $(GL_COMP_FILES)
	@mkdir -p $(dir $@)
	@echo "  GEN     $@"
	@printf '#ifndef SHADERS_H\n#define SHADERS_H\n\n' > $@
	@for comp in $^; do \
	        name=$$(basename $$comp .comp); \
	        varname="gl_shader_$${name}_src"; \
	        printf 'static const char %s[] = {\n' "$$varname" >> $@; \
	        od -v -An -tu1 "$$comp" | tr -s ' ' '\n' | grep -v '^$$' | sed 's/$$/,/' | tr '\n' ' ' >> $@; \
	        printf '0x00\n};\n\n' >> $@; \
	done
	@printf '#endif\n' >> $@
endif

.PHONY: all cli kappai-test server monitor clean print-config format tidy backends-help config

all: $(BACKEND_LIBS) cli kappai-test server

config: $(OUT_DIR) $(CONFIG_FILE)
	@echo "Build configuration created:"
	@cat $(CONFIG_FILE)

$(CONFIG_FILE): | $(OUT_DIR)
	@echo "Generating build configuration..."
	@printf 'BUILD = %s\n' "$(BUILD)" > $@
	@printf 'BACKENDS = "%s"\n' "$(sort $(REQUESTED_BACKENDS))" >> $@
	@printf 'CPU_ARCH_OPT = %s\n' "$(CPU_ARCH_OPT)" >> $@
	@printf 'TSAN = %s\n' "$(if $(filter 1,$(TSAN)),1,0)" >> $@
	@printf 'HOST_ARCH = %s\n' "$(HOST_ARCH)" >> $@
	@printf 'MACHINE_ARCH_FLAGS = %s\n' "$(DETECTED_ARCH_FLAGS)" >> $@
	@printf 'KAI_CACHE_LINE = %s\n' "$(DETECTED_CACHE_LINE)" >> $@
	@printf 'KAI_L1D_KB = %s\n' "$(DETECTED_L1D_KB)" >> $@
	@printf 'KAI_L2_KB = %s\n' "$(DETECTED_L2_KB)" >> $@

backends-help:
	@echo "optional backends (via BACKENDS=...): $(AVAILABLE_BACKENDS)"
	@echo "cpu backends are always built as shared libraries:"
	@echo "  libkappai_cpu_scalar.so   - portable scalar reference implementation"
	@echo "  libkappai_cpu_$(HOST_ARCH).so - $(HOST_ARCH)-optimized implementation (when CPU_ARCH_OPT=1)"
	@echo "backend libraries are installed to $(BACKEND_DIR) and dlopen()ed at runtime;"
	@echo "set KAPPAI_BACKEND_PATH to load backend libraries from another directory"
	@echo "usage: make config BACKENDS=$(if $(AVAILABLE_BACKENDS),$(firstword $(AVAILABLE_BACKENDS)),vulkan)$(if $(word 2,$(AVAILABLE_BACKENDS)),$(,)$(word 2,$(AVAILABLE_BACKENDS)),)"

monitor: $(MONITOR_BIN)
$(MONITOR_BIN): $(SRC_DIR)/monitor/viewer.c | $(OUT_DIR) $(CONFIG_FILE)
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	@$(CC) -O2 -g -Wall -Wextra -MMD -MP -MF $(MONITOR_BIN).d -I$(SRC_DIR) $< -o $@ -lncurses -ljson-c

cli: $(CLI_BIN)
$(CLI_BIN): $(CLI_OBJS) | $(ENGINE) $(BACKEND_LIBS)
	@echo "  LD      $@"
	@$(CC) $(CFLAGS) $(CLI_OBJS) $(ENGINE_LDFLAGS) -ljson-c -o $@
	$(SPLIT_DEBUG)

server: $(SERVER_BIN)
$(SERVER_BIN): $(SERVER_OBJS) | $(ENGINE) $(BACKEND_LIBS)
	@echo "  LD      $@"
	@$(CC) $(CFLAGS) $(SERVER_OBJS) $(ENGINE_LDFLAGS) $(SERVER_LIBS) -o $@
	$(SPLIT_DEBUG)

kappai-test: $(TEST_BIN)
$(TEST_BIN): $(TEST_OBJS) | $(ENGINE) $(BACKEND_LIBS)
	@echo "  LD      $@"
	@$(CC) $(CFLAGS) $(TEST_OBJS) $(ENGINE_LDFLAGS) -ljson-c -o $@
	$(SPLIT_DEBUG)

$(ENGINE): $(LIB_OBJS)
	@echo "  LD      $@"
	@$(CC) -shared -Wl,-soname,libkappai.so $(CFLAGS) $^ $(LDFLAGS) -ljson-c -ldl -o $@
	$(SPLIT_DEBUG)

$(BACKEND_OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OUT_DIR) $(CONFIG_FILE)
	@mkdir -p $(dir $@)
	@echo "  CC(b)   $<"
	@$(CC) $(BACKEND_CFLAGS) $(LTO_FLAGS) -fPIC -I$(SRC_DIR) -c $< -o $@

$(SCALAR_BACKEND): $(SCALAR_BACKEND_OBJS) | $(ENGINE)
	@echo "  LD(b)   $@"
	@$(CC) -shared $(BACKEND_CFLAGS) $(SCALAR_BACKEND_OBJS) \
		-L$(OUT_DIR) -lkappai -Wl,-rpath,'$$ORIGIN/..' $(LDFLAGS) $(LTO_FLAGS) -o $@
	$(SPLIT_DEBUG)

ifneq ($(CPU_ARCH_DIR),)
$(ARCH_BACKEND): $(ARCH_BACKEND_OBJS) | $(ENGINE)
	@echo "  LD(b)   $@"
	@$(CC) -shared $(BACKEND_CFLAGS) $(ARCH_BACKEND_OBJS) \
		-L$(OUT_DIR) -lkappai -Wl,-rpath,'$$ORIGIN/..' $(LDFLAGS) $(LTO_FLAGS) -o $@
	$(SPLIT_DEBUG)
endif

ifneq ($(HAS_VULKAN),)
$(VK_BACKEND): $(VK_BACKEND_OBJS) | $(ENGINE)
	@echo "  LD(b)   $@"
	@$(CC) -shared $(BACKEND_CFLAGS) $(VK_BACKEND_OBJS) \
		-L$(OUT_DIR) -lkappai -Wl,-rpath,'$$ORIGIN/..' $(LDFLAGS) -lvulkan -o $@
	$(SPLIT_DEBUG)
endif

ifneq ($(HAS_OPENGL),)
$(GL_BACKEND): $(GL_BACKEND_OBJS) | $(ENGINE)
	@echo "  LD(b)   $@"
	@$(CC) -shared $(BACKEND_CFLAGS) $(GL_BACKEND_OBJS) \
		-L$(OUT_DIR) -lkappai -Wl,-rpath,'$$ORIGIN/..' $(LDFLAGS) -lEGL -lGLESv2 -lgbm -o $@
	$(SPLIT_DEBUG)
endif

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OUT_DIR) $(CONFIG_FILE)
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -fPIC -I$(SRC_DIR) -c $< -o $@

$(OUT_DIR):
	@mkdir -p $(BUILD_DIRS)

clean:
	rm -rf $(OUT_DIR)

print-config:
	@echo "BUILD              = $(BUILD)"
	@echo "OUT_DIR            = $(OUT_DIR)"
	@echo "CC                 = $(CC)"
	@echo "ARCH_FLAGS         = $(ARCH_FLAGS)"
	@echo "MACHINE_ARCH_FLAGS = $(MACHINE_ARCH_FLAGS)"
	@echo "CFLAGS             = $(CFLAGS)"
	@echo "LDFLAGS            = $(LDFLAGS)"
	@echo "CPU_ARCH_OPT       = $(CPU_ARCH_OPT)"
	@echo "TSAN               = $(if $(filter 1,$(TSAN)),1,0)"
	@echo "SANITIZE_FLAGS     = $(SANITIZE_FLAGS)"
	@echo "SPLIT_DEBUG        = $(if $(SPLIT_DEBUG),1,0) ($(if $(SPLIT_DEBUG),symbols+line info in .debug sidecars,inline -g))"
	@echo "Cache line         = $(if $(KAI_CACHE_LINE),$(KAI_CACHE_LINE) B,64 B (generic default))"
	@echo "L1D / L2           = $(if $(KAI_L1D_KB),$(KAI_L1D_KB)K,generic) / $(if $(KAI_L2_KB),$(KAI_L2_KB)K,generic)"
	@echo "AVAILABLE_BACKENDS = $(AVAILABLE_BACKENDS)"
	@echo "BACKENDS           = $(REQUESTED_BACKENDS)"
	@echo "CPU_ARCH_DIR       = $(if $(CPU_ARCH_DIR),$(CPU_ARCH_DIR),(none: scalar library only))"
	@echo "BACKEND_LIBS       = $(BACKEND_LIBS)"
	@echo "LIB_SRCS           = $(LIB_SRCS)"

format:
	@which clang-format >/dev/null 2>&1 || { echo "clang-format not found"; exit 1; }
	@printf '%s\n' $(ALL_SRCS) $(HEADERS) $(ALL_SHADERS) | \
	xargs -P $$(nproc) -I {} sh -c ' \
		echo "  FMT     {}"; \
		clang-format $(FORMAT_FLAGS) {}; \
		if [ -s {} ] && [ "$$(tail -c1 {})" != "" ]; then \
			printf "\n" >> {}; \
			echo "  EOL     {}"; \
		fi; \
		chmod 644 {}; \
	'

NON_HOST_CPU_ARCHS := $(filter-out $(HOST_ARCH),aarch64 x86_64)
TIDY_SRCS = $(filter-out $(foreach a,$(NON_HOST_CPU_ARCHS),$(SRC_DIR)/backend/cpu/$(a)/%),$(ALL_SRCS))

tidy: | $(OUT_DIR)
	@which clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not found"; exit 1; }
	@echo "  TIDY    -> $(TIDY_LOG)"
	@: > $(TIDY_LOG)
	@printf '%s\n' $(TIDY_SRCS) | \
		xargs -P $$(nproc) \
		-I {} sh -c ' \
			echo "  TIDY    {}"; \
			clang-tidy --config-file=.clang-tidy {} -- $(filter-out $(ARCH_FLAGS) -fvect-cost-model=%,$(CFLAGS)) -march=native -I$(SRC_DIR) >> $(TIDY_LOG) 2>&1 || true \
		'
	@echo "  TIDY    done, see $(TIDY_LOG)"

-include $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(SERVER_OBJS:.o=.d) $(CLI_OBJS:.o=.d) $(BACKEND_OBJS:.o=.d) $(MONITOR_BIN).d
