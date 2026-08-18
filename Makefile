CC      ?= gcc
GLSLC   ?= glslc

SRC_DIR := src
OUT_DIR := build
OBJ_DIR := $(OUT_DIR)/src

.DEFAULT_GOAL := all

# --- Build mode: debug | release-rdbg | release ---
BUILD ?= release-rdbg

VALID_BUILDS := debug release-rdbg release
ifeq ($(filter $(BUILD),$(VALID_BUILDS)),)
  $(error Invalid BUILD='$(BUILD)'. Valid options: $(VALID_BUILDS))
endif

ifeq ($(BUILD),release)
  RANLIB := gcc-ranlib
endif

HOST_ARCH := $(shell uname -m)

CPU_ARCH_OPT ?= 1

AVAILABLE_BACKENDS := $(sort $(notdir $(patsubst %/,%,$(filter-out %/cpu/,$(wildcard $(SRC_DIR)/backend/*/)))))

comma := ,
BACKENDS ?=
REQUESTED_BACKENDS := $(strip $(subst $(comma), ,$(BACKENDS)))
UNKNOWN_BACKENDS := $(filter-out $(AVAILABLE_BACKENDS),$(REQUESTED_BACKENDS))

ifneq ($(UNKNOWN_BACKENDS),)
  $(error Unknown backend(s): $(UNKNOWN_BACKENDS). Available backends: $(AVAILABLE_BACKENDS))
endif

HAS_VULKAN := $(filter vulkan,$(REQUESTED_BACKENDS))

BASE_FLAGS := -std=c11 -D_DEFAULT_SOURCE
DEP_FLAGS  := -MMD -MP

ifeq ($(BUILD),release)
  ARCH_FLAGS ?= -march=native
else
  ARCH_FLAGS ?=
endif

MATH_FLAGS := -fno-math-errno -fno-trapping-math -fno-signed-zeros \
	      -fcx-limited-range

# Sanitizers default to ON for debug (bug-hunting), OFF elsewhere. Set
# SANITIZE=1 to force-enable them in any build, e.g. make BUILD=release-rdbg SANITIZE=1.
SANITIZE ?= 0
ifeq ($(BUILD),debug)
  SANITIZE := 1
endif
ifeq ($(SANITIZE),1)
  ifeq ($(TSAN),1)
    SANITIZE_FLAGS := -fsanitize=thread,undefined -fno-sanitize-recover=undefined
  else
    SANITIZE_FLAGS := -fsanitize=address,undefined -fno-sanitize-recover=undefined
  endif
else
  SANITIZE_FLAGS :=
endif

ifeq ($(BUILD),debug)
  CFLAGS  := $(BASE_FLAGS) $(DEP_FLAGS) -O0 -g3 -ggdb3 -fno-omit-frame-pointer \
	     $(SANITIZE_FLAGS) \
	     -Wall -Wextra -Wformat=2 -Wshadow -Wstrict-prototypes \
	     -DDEBUG_BUILD=1
  LDFLAGS := -lm -lpthread $(SANITIZE_FLAGS)

else ifeq ($(BUILD),release-rdbg)
  CFLAGS  := $(BASE_FLAGS) $(DEP_FLAGS) -O3 -g3 -ggdb3 -fno-omit-frame-pointer \
	     $(SANITIZE_FLAGS) \
	     -Wall -Wextra -Wformat=2 \
	     $(MATH_FLAGS) $(ARCH_FLAGS) \
	     -DRELEASE_DBG=1
  LDFLAGS := -lm -lpthread $(SANITIZE_FLAGS)

else
  CFLAGS  := $(BASE_FLAGS) $(DEP_FLAGS) -O3 -flto -funroll-loops -funroll-all-loops \
	     -ftree-vectorize -fvect-cost-model=unlimited -fivopts -fweb \
	     -frename-registers -fprefetch-loop-arrays \
	     $(MATH_FLAGS) $(ARCH_FLAGS) -DNDEBUG
  LDFLAGS := -lm -lpthread -flto
endif

ifneq ($(HAS_VULKAN),)
  CFLAGS  += -DBACKEND_VULKAN -I$(OBJ_DIR)/backend/vulkan -I$(SRC_DIR)/backend/vulkan
  LDFLAGS += -lvulkan
endif

LIB_SRCS := \
	$(wildcard $(SRC_DIR)/*.c) \
	$(wildcard $(SRC_DIR)/models/*.c) \
	$(wildcard $(SRC_DIR)/moe/*.c) \
	$(SRC_DIR)/backend/backend.c \
	$(wildcard $(SRC_DIR)/backend/cpu/scalar/*.c)

ifeq ($(CPU_ARCH_OPT),1)
  ifeq ($(HOST_ARCH),aarch64)
    LIB_SRCS += $(SRC_DIR)/backend/cpu/aarch64/quants.c
    LIB_SRCS += $(SRC_DIR)/backend/cpu/aarch64/core.c
  endif
  ifeq ($(HOST_ARCH),x86_64)
    LIB_SRCS += $(SRC_DIR)/backend/cpu/x86_64/quants.c
    LIB_SRCS += $(SRC_DIR)/backend/cpu/x86_64/core.c
  endif
endif

ifneq ($(HAS_VULKAN),)
  LIB_SRCS += $(SRC_DIR)/backend/vulkan/vulkan.c
endif

TEST_SRCS := $(wildcard $(SRC_DIR)/test/*.c)
HEADERS   := $(shell find $(SRC_DIR) -type f \( -name "*.h" -o -name "*.hpp" \))

LIB_OBJS      := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(LIB_SRCS))
TEST_OBJ_DIR  := $(OBJ_DIR)/test
TEST_OBJS     := $(patsubst $(SRC_DIR)/test/%.c,$(TEST_OBJ_DIR)/%.o,$(TEST_SRCS))

ENGINE      := $(OUT_DIR)/libkappai.so
CLI_BIN     := $(OUT_DIR)/kappai-cli
TEST_BIN    := $(OUT_DIR)/test
MONITOR_BIN := $(OUT_DIR)/kappai-monitor

BUILD_DIRS := $(OBJ_DIR)/backend/cpu/scalar $(OBJ_DIR)/backend/cpu/aarch64 \
	      $(OBJ_DIR)/backend/cpu/x86_64 \
	      $(OBJ_DIR)/backend/vulkan \
	      $(OBJ_DIR)/cli $(OBJ_DIR)/moe $(OBJ_DIR)/monitor \
	      $(OBJ_DIR)/models $(OBJ_DIR)/test

CONFIG_STAMP := $(OUT_DIR)/.build-config
CONFIG_SIG   := BUILD=$(BUILD) BACKENDS=$(sort $(REQUESTED_BACKENDS)) CPU_ARCH_OPT=$(CPU_ARCH_OPT) HOST_ARCH=$(HOST_ARCH)
PREV_SIG     := $(shell cat $(CONFIG_STAMP) 2>/dev/null)

CONFIG_AGNOSTIC_GOALS := clean format tidy print-config backends-help
BUILD_GOALS := $(filter-out $(CONFIG_AGNOSTIC_GOALS),$(if $(MAKECMDGOALS),$(MAKECMDGOALS),all))

ifneq ($(BUILD_GOALS),)
  ifneq ($(PREV_SIG),)
    ifneq ($(PREV_SIG),$(CONFIG_SIG))
      $(info Previous build config: $(PREV_SIG))
      $(info Requested build config: $(CONFIG_SIG))
      $(error $(OUT_DIR) was built with different settings. Run 'make clean' first, or match the previous settings)
    endif
  endif
endif

VK_SHADERS_DIR := $(SRC_DIR)/backend/vulkan/shaders
VK_SHADER_FILES := $(wildcard $(VK_SHADERS_DIR)/*.comp)
VK_INC_FILES    := $(wildcard $(VK_SHADERS_DIR)/*.glsl) $(wildcard $(VK_SHADERS_DIR)/*.inc)

MATMUL_DUAL := matmul_q4_0 matmul_q4_1 matmul_q5_0 matmul_q5_1 matmul_q8_0 matmul_q4_k matmul_q5_k matmul_q6_k matmul_iq3_s matmul_f32

MATMUL_NMAT_DUAL := matmul_q4_0 matmul_q4_k matmul_q6_k

RMSNORM_VARIANTS := rmsnorm_noweight rmsnorm_sg rmsnorm_noweight_sg \
	            rmsnorm_per_head rmsnorm_per_head_sg rmsnorm_add \
	            rmsnorm_noweight_per_head rmsnorm_noweight_per_head_sg

RMSNORM_ALL := rmsnorm $(RMSNORM_VARIANTS)

ROPE_EXT_VARIANTS := rope_ext

rmsnorm_FLAGS                        := -DHAS_WEIGHT
rmsnorm_noweight_FLAGS               :=
rmsnorm_sg_FLAGS                     := -DHAS_WEIGHT -DUSE_SUBGROUP
rmsnorm_noweight_sg_FLAGS            := -DUSE_SUBGROUP
rmsnorm_per_head_FLAGS               := -DHAS_WEIGHT -DPER_HEAD
rmsnorm_per_head_sg_FLAGS            := -DHAS_WEIGHT -DPER_HEAD -DUSE_SUBGROUP
rmsnorm_add_FLAGS                    := -DHAS_WEIGHT -DUSE_SUBGROUP -DADD_RESIDUAL
rmsnorm_noweight_per_head_FLAGS      := -DPER_HEAD
rmsnorm_noweight_per_head_sg_FLAGS   := -DPER_HEAD -DUSE_SUBGROUP

SHADER_SPVS := \
	$(patsubst $(VK_SHADERS_DIR)/%.comp,$(OBJ_DIR)/backend/vulkan/%.spv,$(VK_SHADER_FILES)) \
	$(foreach s,$(MATMUL_DUAL),$(OBJ_DIR)/backend/vulkan/$(s)_residual.spv) \
	$(foreach s,$(MATMUL_NMAT_DUAL),$(OBJ_DIR)/backend/vulkan/$(s)_dual.spv) \
	$(foreach s,$(RMSNORM_VARIANTS),$(OBJ_DIR)/backend/vulkan/$(s).spv) \
	$(foreach s,$(ROPE_EXT_VARIANTS),$(OBJ_DIR)/backend/vulkan/$(s).spv)

SHADERS_H := $(OBJ_DIR)/backend/vulkan/shaders_embedded.h

ifneq ($(HAS_VULKAN),)
  $(OBJ_DIR)/backend/vulkan/vulkan.o: $(SHADERS_H)

  $(OBJ_DIR)/backend/vulkan/%.spv: $(VK_SHADERS_DIR)/%.comp $(VK_INC_FILES) | $(OUT_DIR)
	@mkdir -p $(dir $@)
	@echo "  GLSLC   $@"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SHADERS_DIR) $< -o $@

  define MATMUL_RES_RULE
  $(OBJ_DIR)/backend/vulkan/$(1)_residual.spv: $(VK_SHADERS_DIR)/$(1).comp $(VK_INC_FILES) | $(OUT_DIR)
	@mkdir -p $$(dir $$@)
	@echo "  GLSLC   $(1)_residual.spv"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SHADERS_DIR) -DHAS_RESIDUAL $$< -o $$@
  endef
  $(foreach s,$(MATMUL_DUAL),$(eval $(call MATMUL_RES_RULE,$(s))))

  define MATMUL_DUAL_RULE
  $(OBJ_DIR)/backend/vulkan/$(1)_dual.spv: $(VK_SHADERS_DIR)/$(1).comp $(VK_INC_FILES) | $(OUT_DIR)
	@mkdir -p $$(dir $$@)
	@echo "  GLSLC   $(1)_dual.spv"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SHADERS_DIR) -DNMAT_DUAL $$< -o $$@
  endef
  $(foreach s,$(MATMUL_NMAT_DUAL),$(eval $(call MATMUL_DUAL_RULE,$(s))))

  define RMSNORM_RULE
  $(OBJ_DIR)/backend/vulkan/$(1).spv: $(VK_SHADERS_DIR)/rmsnorm.comp $(VK_INC_FILES) | $(OUT_DIR)
	@mkdir -p $$(dir $$@)
	@echo "  GLSLC   $(1).spv  [$$($(1)_FLAGS)]"
	@$$(GLSLC) -O --target-env=vulkan1.1 -I$$(VK_SHADERS_DIR) $$($(1)_FLAGS) $$< -o $$@
  endef
  $(foreach s,$(RMSNORM_ALL),$(eval $(call RMSNORM_RULE,$(s))))

  $(OBJ_DIR)/backend/vulkan/rope_ext.spv: $(VK_SHADERS_DIR)/rope.comp $(VK_INC_FILES) | $(OUT_DIR)
	@mkdir -p $(dir $@)
	@echo "  GLSLC   rope_ext.spv"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SHADERS_DIR) -DHAS_FF $< -o $@

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

FORMAT_FLAGS := -i -style=file

TIDY_LOG  := $(OUT_DIR)/tidy.log
TIDY_SRCS := $(LIB_SRCS) $(SRC_DIR)/cli/main.c $(TEST_SRCS)

.PHONY: all cli test monitor clean print-config format tidy backends-help

all: cli test

backends-help:
	@echo "available backends: $(AVAILABLE_BACKENDS)"
	@echo "usage: make BACKENDS=$(if $(AVAILABLE_BACKENDS),$(firstword $(AVAILABLE_BACKENDS)),vulkan)$(if $(word 2,$(AVAILABLE_BACKENDS)),$(comma)$(word 2,$(AVAILABLE_BACKENDS)),)"

$(CONFIG_STAMP): | $(OUT_DIR)
	@printf '%s' "$(CONFIG_SIG)" > $@

monitor: $(MONITOR_BIN)
$(MONITOR_BIN): $(SRC_DIR)/monitor/viewer.c | $(OUT_DIR) $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	@$(CC) -O2 -g -Wall -Wextra -I$(SRC_DIR) $< -o $@ -lncurses -ljson-c

cli: $(CLI_BIN)
$(CLI_BIN): $(SRC_DIR)/cli/main.c $(ENGINE)
	@echo "  LD      $@"
	@$(CC) $(CFLAGS) -I$(SRC_DIR) $< -L$(OUT_DIR) -lkappai -Wl,-rpath,'$$ORIGIN' -lm -lpthread -o $@

test: $(TEST_BIN)
$(TEST_BIN): $(TEST_OBJS) $(ENGINE)
	@echo "  LD      $@"
	@$(CC) $(CFLAGS) -I$(SRC_DIR) $(TEST_OBJS) -L$(OUT_DIR) -lkappai -Wl,-rpath,'$$ORIGIN' -lm -lpthread -o $@

$(TEST_OBJ_DIR)/%.o: $(SRC_DIR)/test/%.c | $(TEST_OBJ_DIR) $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

$(TEST_OBJ_DIR):
	@mkdir -p $@

$(ENGINE): $(LIB_OBJS)
	@echo "  LD      $@"
	@$(CC) -shared $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OUT_DIR) $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -fPIC -I$(SRC_DIR) -c $< -o $@

# x86_64 SIMD kernels need explicit ISA flags (AVX2/F16C/FMA are not baseline x86-64).
ifeq ($(HOST_ARCH),x86_64)
  $(OBJ_DIR)/backend/cpu/x86_64/quants.o: CFLAGS += -mavx2 -mf16c -mfma
  $(OBJ_DIR)/backend/cpu/x86_64/core.o: CFLAGS += -mavx2 -mf16c -mfma
endif

$(OUT_DIR):
	@mkdir -p $(BUILD_DIRS)

clean:
	rm -rf $(OUT_DIR)

print-config:
	@echo "BUILD              = $(BUILD)"
	@echo "OUT_DIR            = $(OUT_DIR)"
	@echo "CC                 = $(CC)"
	@echo "CFLAGS             = $(CFLAGS)"
	@echo "LDFLAGS            = $(LDFLAGS)"
	@echo "CPU_ARCH_OPT       = $(CPU_ARCH_OPT)"
	@echo "SANITIZE           = $(SANITIZE)"
	@echo "AVAILABLE_BACKENDS = $(AVAILABLE_BACKENDS)"
	@echo "BACKENDS           = $(REQUESTED_BACKENDS)"
	@echo "LIB_SRCS           = $(LIB_SRCS)"

format:
	@which clang-format >/dev/null 2>&1 || { echo "clang-format not found"; exit 1; }
	@for f in $(TIDY_SRCS) $(HEADERS); do \
	        echo "  FMT     $$f"; \
	        clang-format $(FORMAT_FLAGS) $$f; \
	done

tidy: | $(OUT_DIR)
	@which clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not found"; exit 1; }
	@echo "  TIDY    -> $(TIDY_LOG)"
	@: > $(TIDY_LOG)
	@printf '%s\n' $(TIDY_SRCS) | \
		xargs -P $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) \
		-I {} sh -c ' \
			echo "  TIDY    {}"; \
			echo "===== {} =====" >> $(TIDY_LOG); \
			clang-tidy --config-file=.clang-tidy {} -- $(filter-out -fvect-cost-model=unlimited,$(CFLAGS)) -I$(SRC_DIR) >> $(TIDY_LOG) 2>&1 || true \
		'
	@echo "  TIDY    done, see $(TIDY_LOG)"

-include $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d)