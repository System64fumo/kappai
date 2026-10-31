SRC_DIR     := src
OUT_DIR     := build
OBJ_DIR     := $(OUT_DIR)/src
CONFIG_FILE := $(OUT_DIR)/config.mk

PKG_CONFIG ?= pkg-config
GLSLC      ?= glslc
OBJCOPY    ?= objcopy
STRIP      ?= strip

BUILD        ?= debug
BACKENDS     ?=
CPU_ARCH_OPT ?= 1
TSAN         ?= 0
TASK_BACKEND ?= auto

.DEFAULT_GOAL := all
comma := ,
space := $(subst ,, )

ifeq ($(filter config,$(MAKECMDGOALS)),)
-include $(CONFIG_FILE)
endif

ifneq ($(filter-out config clean help format,$(or $(MAKECMDGOALS),all)),)
  ifeq ($(wildcard $(CONFIG_FILE)),)
    $(error No build configuration found. Run 'make config' first)
  endif
endif

ifeq ($(filter $(BUILD),debug release),)
  $(error Invalid BUILD='$(BUILD)'. Valid options: debug release)
endif
ifeq ($(filter $(TASK_BACKEND),auto ucontext pthread),)
  $(error Invalid TASK_BACKEND='$(TASK_BACKEND)'. Valid options: auto ucontext pthread)
endif

AVAILABLE_BACKENDS := $(sort $(notdir $(patsubst %/,%,$(filter-out %/cpu/,$(wildcard $(SRC_DIR)/backend/*/)))))
BACKEND_LIST       := $(sort $(subst $(comma), ,$(subst ",,$(BACKENDS))))
ifneq ($(filter-out $(AVAILABLE_BACKENDS),$(BACKEND_LIST)),)
  $(error Unknown backend(s): $(filter-out $(AVAILABLE_BACKENDS),$(BACKEND_LIST)). Available: $(AVAILABLE_BACKENDS))
endif
HAS_VULKAN := $(filter vulkan,$(BACKEND_LIST))
HAS_OPENGL := $(filter opengl,$(BACKEND_LIST))

UNAME     := $(shell uname -sm)
HOST_OS   := $(word 1,$(UNAME))
HOST_ARCH := $(word 2,$(UNAME))
CPU_ARCH  := $(if $(filter arm64,$(HOST_ARCH)),aarch64,$(HOST_ARCH))

SHLIB_SUFFIX    := .so
SHLIB_LDFLAGS   := -shared
RPATH_ORIGIN    := $$ORIGIN
ENGINE_ID_FLAGS := -Wl,-soname,libkappai$(SHLIB_SUFFIX)
DL_LIBS         := -ldl
ifeq ($(HOST_OS),Darwin)
  SHLIB_SUFFIX    := .dylib
  SHLIB_LDFLAGS   := -dynamiclib
  RPATH_ORIGIN    := @loader_path
  ENGINE_ID_FLAGS := -Wl,-install_name,@rpath/libkappai$(SHLIB_SUFFIX)
endif
ifneq ($(filter Darwin FreeBSD OpenBSD NetBSD DragonFly,$(HOST_OS)),)
  DL_LIBS :=
endif

cc_option  = $(shell printf 'int main(void) { return 0; }\n' | $(CC) -Werror $(1) -x c -c -o /dev/null - >/dev/null 2>&1 && printf '%s' '$(1)')
pkg       = $(shell $(PKG_CONFIG) --cflags --libs $(1) 2>/dev/null)
pkg_libs  = $(filter -l% -L% -Wl%,$(1))
pkg_flags = $(filter-out -l% -L% -Wl%,$(1))

PKG_ALL    := $(or $(call pkg,json-c libmicrohttpd ncurses),-ljson-c -lmicrohttpd -lncurses)
PKG_CFLAGS := $(call pkg_flags,$(PKG_ALL)) $(addprefix -I,$(patsubst %/json-c,%,$(filter %/json-c,$(PKG_ALL))))
PKG_LIBS   := $(call pkg_libs,$(PKG_ALL))

DETECTED_ARCH_FLAGS = $(shell $(CC) -### -E - -march=native 2>&1 | sed -rn '/cc1/!d;s/(\")|(^.* - )//g;s/ -dumpbase -//g;p')

ifeq ($(origin ARCH_FLAGS),undefined)
  ARCH_FLAGS := $(or $(call cc_option,$(MACHINE_ARCH_FLAGS)),$(DETECTED_ARCH_FLAGS))
endif

BASE_FLAGS := -std=c11 -D_DEFAULT_SOURCE -D_FILE_OFFSET_BITS=64 -MMD -MP \
              $(PKG_CFLAGS)
MATH_FLAGS := -fno-math-errno -fno-trapping-math -fno-signed-zeros

ifeq ($(BUILD),debug)
  SANITIZE_FLAGS := $(if $(filter 1,$(TSAN)),-fsanitize=thread,-fsanitize=address,undefined -fno-sanitize-recover=undefined)
  CFLAGS  := $(BASE_FLAGS) -DKAPPAI_DEBUG -O2 -g -fno-omit-frame-pointer $(SANITIZE_FLAGS) \
             -Wall -Wextra -Wformat=2 $(MATH_FLAGS) $(ARCH_FLAGS)
  LDFLAGS := -lm -lpthread $(SANITIZE_FLAGS)
else
  RELEASE_FLAGS := -funroll-loops -funroll-all-loops -ftree-vectorize -fvect-cost-model=unlimited \
                   -fivopts -fweb -frename-registers -fprefetch-loop-arrays
  RELEASE_FLAGS := $(or $(call cc_option,$(RELEASE_FLAGS)),$(foreach f,$(RELEASE_FLAGS),$(call cc_option,$(f))))
  LTO_FLAGS ?= $(call cc_option,-flto=auto)
  CFLAGS  := $(BASE_FLAGS) -O3 $(RELEASE_FLAGS) -g $(MATH_FLAGS) $(ARCH_FLAGS)
  LDFLAGS := -lm -lpthread
  ifeq ($(HOST_OS),Linux)
    SPLIT_DEBUG = $(call split_debug,$@)
  endif
endif

define split_debug
	@$(OBJCOPY) --only-keep-debug $1 $1.debug
	@$(STRIP) --strip-unneeded $1
	@$(OBJCOPY) --add-gnu-debuglink=$1.debug $1
	@echo "  DBG     $1.debug"
endef

HAVE_UCONTEXT :=
ifneq ($(filter ucontext autorelease,$(TASK_BACKEND) $(TASK_BACKEND)$(BUILD)),)
  HAVE_UCONTEXT := $(shell printf '\043include <ucontext.h>\nstatic void entry(void) {}\nint main(void) { ucontext_t a = {0}, b = {0}; if (getcontext(&a)) return 1; makecontext(&a, entry, 0); return swapcontext(&a, &b); }\n' | $(CC) -std=c11 -D_DEFAULT_SOURCE -D_XOPEN_SOURCE=700 -D_DARWIN_C_SOURCE -Werror=implicit-function-declaration -Werror=deprecated-declarations -x c - -o /dev/null >/dev/null 2>&1 && echo 1)
  ifeq ($(HAVE_UCONTEXT)$(TASK_BACKEND),ucontext)
    $(error Native ucontext APIs are unavailable to $(CC); use TASK_BACKEND=pthread)
  endif
endif
TASK_CFLAGS := $(if $(HAVE_UCONTEXT),-DKAPPAI_TASK_UCONTEXT=1)

objs = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(1))

LIB_SRCS := $(wildcard $(SRC_DIR)/*.c $(SRC_DIR)/recipe/*.c $(SRC_DIR)/models/*.c \
                       $(SRC_DIR)/moe/*.c $(SRC_DIR)/diffusion/*.c) $(SRC_DIR)/backend/backend.c
SERVER_SHARED  := $(OBJ_DIR)/server/openai.o $(OBJ_DIR)/server/task.o
LIB_OBJS       := $(call objs,$(LIB_SRCS))
CLI_OBJS       := $(call objs,$(wildcard $(SRC_DIR)/cli/*.c))
SERVER_OBJS    := $(call objs,$(wildcard $(SRC_DIR)/server/*.c))
DIFFUSION_OBJS := $(call objs,$(wildcard $(SRC_DIR)/diffusion/cli/*.c))
TUI_OBJS       := $(call objs,$(wildcard $(SRC_DIR)/tui/*.c)) $(SERVER_SHARED)
TEST_OBJS      := $(call objs,$(wildcard $(SRC_DIR)/test/*.c)) $(OBJ_DIR)/server/task.o

$(SERVER_SHARED) $(OBJ_DIR)/test/test_server_task.o: CFLAGS += $(TASK_CFLAGS)

ENGINE          := $(OUT_DIR)/libkappai$(SHLIB_SUFFIX)
BACKEND_DIR     := $(OUT_DIR)/backends
BACKEND_OBJ_DIR := $(OUT_DIR)/backend_obj
BACKEND_CFLAGS  := $(CFLAGS) -fvisibility=hidden
ENGINE_LDFLAGS  := -L$(OUT_DIR) -lkappai -Wl,-rpath,'$(RPATH_ORIGIN)' $(LDFLAGS)
BACKEND_LDFLAGS := $(SHLIB_LDFLAGS) -L$(OUT_DIR) -lkappai -Wl,-rpath,'$(RPATH_ORIGIN)/..' $(LDFLAGS) $(LTO_FLAGS)

backend_objs = $(addprefix $(BACKEND_OBJ_DIR)/backend/,$(1))
cpu_objs     = $(call backend_objs,cpu/common.o cpu/nn_ops.o cpu/$(1)/core.o cpu/$(1)/quants.o)

BACKEND_LIBS :=
BACKEND_OBJS :=

define add_backend
BACKEND_LIBS += $$(BACKEND_DIR)/libkappai_$(1)$$(SHLIB_SUFFIX)
BACKEND_OBJS += $(2)
$$(BACKEND_DIR)/libkappai_$(1)$$(SHLIB_SUFFIX): $(2) | $$(ENGINE)
	@mkdir -p $$(@D)
	@echo "  LD(b)   $$@"
	@$$(CC) $$(BACKEND_CFLAGS) $(2) $$(BACKEND_LDFLAGS) $(3) -o $$@
	$$(SPLIT_DEBUG)
endef

$(eval $(call add_backend,cpu_scalar,$(call cpu_objs,scalar),))

ifeq ($(CPU_ARCH_OPT),1)
  ifneq ($(filter aarch64 x86_64,$(CPU_ARCH)),)
    $(eval $(call add_backend,cpu_$(CPU_ARCH),$(call cpu_objs,$(CPU_ARCH)),))
  endif
endif

ifneq ($(HAS_VULKAN),)
  VK_PKG  := $(or $(call pkg,vulkan),-lvulkan)
  VK_OBJS := $(call backend_objs,vulkan/vulkan.o)
  $(eval $(call add_backend,vulkan,$(VK_OBJS),$(call pkg_libs,$(VK_PKG))))
  $(VK_OBJS): BACKEND_CFLAGS += -I$(OBJ_DIR)/backend/vulkan -I$(SRC_DIR)/backend/vulkan $(call pkg_flags,$(VK_PKG))
endif

ifneq ($(HAS_OPENGL),)
  GL_PKG  := $(or $(call pkg,egl glesv2 gbm),-lEGL -lGLESv2 -lgbm)
  GL_OBJS := $(call backend_objs,opengl/opengl.o opengl/gl_context.o)
  $(eval $(call add_backend,opengl,$(GL_OBJS),$(call pkg_libs,$(GL_PKG))))
  $(GL_OBJS): BACKEND_CFLAGS += -I$(OBJ_DIR)/backend/opengl -I$(SRC_DIR)/backend/opengl $(call pkg_flags,$(GL_PKG))
endif

ifneq ($(HAS_VULKAN),)

VK_SRC      := $(SRC_DIR)/backend/vulkan/shaders
VK_OUT      := $(OBJ_DIR)/backend/vulkan
VK_INCLUDES := $(wildcard $(VK_SRC)/*.glsl $(VK_SRC)/*.inc)
VK_SPVS     :=

define vk_rule
VK_SPVS += $(VK_OUT)/$(1).spv
$(VK_OUT)/$(1).spv: $(VK_SRC)/$(2).comp $(VK_INCLUDES)
	@mkdir -p $$(@D)
	@echo "  GLSLC   $(1).spv"
	@$(GLSLC) -O --target-env=vulkan1.1 -I$(VK_SRC) $(3) $$< -o $$@
endef

vk  = $(eval $(call vk_rule,$(1),$(2),$(3)))
rms = $(call vk,rmsnorm$(1)_batch,rmsnorm,$(2) -DBATCHED)


VK_PLAIN := argmax argmax_reduce moe_gather moe_combine attention attention_flash embd_lookup \
            kv_put kv_shift_gather q8_quant attn_decode attn_merge qk_norm_rope dequant \
            ffn_activate_fused_batch layernorm_modulate rope_table attention_dense im2col \
            upsample_shortcut
VK_MATMUL := matmul_q4_0 matmul_q4_1 matmul_q5_0 matmul_q5_1 matmul_q8_0 matmul_q4_k matmul_q5_k \
             matmul_q6_k matmul_iq3_s matmul_f32 matmul_f16 matmul_bf16 matmul_iq2_s matmul_iq2_xxs \
             matmul_iq3_xxs matmul_iq4_xs
VK_QCORE := q4_0 q4_1 q5_0 q5_1 q8_0 iq4_nl
VK_KQ    := q4_k q5_k q6_k
VK_DENSE := f32 f16 bf16

VK_TYPES := $(VK_QCORE) $(VK_KQ) $(VK_DENSE)
VK_DEFS  := $(join $(addsuffix :,$(VK_TYPES)),$(shell echo $(VK_TYPES) | tr a-z A-Z))
def       = -D$(patsubst $(1):%,%,$(filter $(1):%,$(VK_DEFS)))

$(foreach s,$(VK_PLAIN),$(call vk,$(s),$(s),))
$(foreach s,$(VK_MATMUL),$(call vk,$(s)_batch,$(s),-DBATCHED))
$(foreach s,$(VK_MATMUL),$(call vk,$(s)_residual_batch,$(s),-DHAS_RESIDUAL -DBATCHED))
$(foreach s,matmul_q4_0 matmul_q4_k matmul_q6_k,$(call vk,$(s)_dual_batch,$(s),-DNMAT_DUAL -DBATCHED))
$(call vk,matmul_q4_0_triple_batch,matmul_q4_0,-DNMAT_TRIPLE -DBATCHED)
$(call vk,matmul_iq4_nl_batch,matmul_iq4_nl,-DBATCHED)
$(foreach t,$(VK_QCORE),$(call vk,q_matvec_$(t),q_matvec,$(call def,$(t))))
$(foreach t,$(VK_QCORE),$(call vk,q_matmul_$(t),q_matmul,$(call def,$(t))))
$(foreach t,$(VK_KQ),$(call vk,kq_matvec_$(t),kq_matvec,$(call def,$(t))))
$(foreach t,$(VK_DENSE),$(call vk,dense_matvec_$(t),dense_matvec,$(call def,$(t))))
$(call vk,q8k_quant,q8_quant,-DQ8K)
$(call vk,rope_batch,rope,-DBATCHED)
$(call vk,rope_ext_batch,rope,-DHAS_FF -DBATCHED)
$(call vk,rope_qk_batch,rope_qk,-DBATCHED)
$(call vk,attention_batch,attention,-DBATCHED)
$(call vk,attention_flash_batch,attention_flash,-DBATCHED)
$(call vk,ffn_activate_batch,ffn_activate,-DBATCHED)
$(call vk,elementwise_batch,elementwise,-DBATCHED)

$(call rms,,-DHAS_WEIGHT)
$(call rms,_noweight,)
$(call rms,_sg,-DHAS_WEIGHT -DUSE_SUBGROUP)
$(call rms,_noweight_sg,-DUSE_SUBGROUP)
$(call rms,_per_head,-DHAS_WEIGHT -DPER_HEAD)
$(call rms,_per_head_sg,-DHAS_WEIGHT -DPER_HEAD -DUSE_SUBGROUP)
$(call rms,_add,-DHAS_WEIGHT -DUSE_SUBGROUP -DADD_RESIDUAL)
$(call rms,_sg_quant,-DHAS_WEIGHT -DUSE_SUBGROUP -DQUANT_OUT)
$(call rms,_add_norm,-DHAS_WEIGHT -DUSE_SUBGROUP -DADD_RESIDUAL -DNORM2 -DQUANT_OUT)
$(call rms,_noweight_per_head,-DPER_HEAD)
$(call rms,_noweight_per_head_sg,-DPER_HEAD -DUSE_SUBGROUP)

VK_HEADER := $(VK_OUT)/shaders_embedded.h

$(VK_OBJS): $(VK_HEADER)

$(VK_HEADER): $(VK_SPVS)
	@mkdir -p $(@D)
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

ifneq ($(HAS_OPENGL),)

GL_SRC    := $(SRC_DIR)/backend/opengl/shaders
GL_HEADER := $(OBJ_DIR)/backend/opengl/shaders_embedded.h

$(GL_OBJS): $(GL_HEADER)

$(GL_HEADER): $(sort $(wildcard $(GL_SRC)/*.comp $(GL_SRC)/*.glsl))
	@mkdir -p $(@D)
	@echo "  GEN     $@"
	@printf '#ifndef SHADERS_H\n#define SHADERS_H\n\n' > $@
	@for comp in $^; do \
	        name=$$(basename $$comp); \
	        name=$${name%.*}; \
	        varname="gl_shader_$${name}_src"; \
	        printf 'static const char %s[] = {\n' "$$varname" >> $@; \
	        od -v -An -tu1 "$$comp" | tr -s ' ' '\n' | grep -v '^$$' | sed 's/$$/,/' | tr '\n' ' ' >> $@; \
	        printf '0x00\n};\n\n' >> $@; \
	done
	@printf '#endif\n' >> $@

endif

.PHONY: all backends libkappai kappai kappai-cli kappai-server kappai-diffusion kappai-test kappai-monitor
.PHONY: config help clean format tidy

all: backends kappai kappai-cli kappai-server kappai-diffusion
backends: $(BACKEND_LIBS)
libkappai: $(ENGINE)

define link_binary
$(1): $(OUT_DIR)/$(1)
$(OUT_DIR)/$(1): $(2) | $$(ENGINE) $$(BACKEND_LIBS)
	@echo "  LD      $$@"
	@$$(CC) $$(CFLAGS) $(2) $$(ENGINE_LDFLAGS) $(3) -o $$@
	$$(SPLIT_DEBUG)
endef

$(eval $(call link_binary,kappai,$(TUI_OBJS),$(PKG_LIBS)))
$(eval $(call link_binary,kappai-cli,$(CLI_OBJS),$(PKG_LIBS)))
$(eval $(call link_binary,kappai-server,$(SERVER_OBJS),$(PKG_LIBS)))
$(eval $(call link_binary,kappai-diffusion,$(DIFFUSION_OBJS),$(PKG_LIBS)))
$(eval $(call link_binary,kappai-test,$(TEST_OBJS),$(PKG_LIBS)))

kappai-monitor: $(OUT_DIR)/kappai-monitor
$(OUT_DIR)/kappai-monitor: $(SRC_DIR)/monitor/viewer.c $(CONFIG_FILE)
	@echo "  CC      $<"
	@$(CC) $(BASE_FLAGS) -O2 -g -Wall -Wextra -MF $@.d -I$(SRC_DIR) $< -o $@ $(PKG_LIBS)

$(ENGINE): $(LIB_OBJS)
	@echo "  LD      $@"
	@$(CC) $(SHLIB_LDFLAGS) $(ENGINE_ID_FLAGS) $(CFLAGS) $^ $(LDFLAGS) $(PKG_LIBS) $(DL_LIBS) -o $@
	$(SPLIT_DEBUG)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(CONFIG_FILE)
	@mkdir -p $(@D)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -fPIC -I$(SRC_DIR) -c $< -o $@

$(BACKEND_OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(CONFIG_FILE)
	@mkdir -p $(@D)
	@echo "  CC(b)   $<"
	@$(CC) $(BACKEND_CFLAGS) $(LTO_FLAGS) -fPIC -I$(SRC_DIR) -c $< -o $@

-include $(sort $(LIB_OBJS:.o=.d) $(CLI_OBJS:.o=.d) $(SERVER_OBJS:.o=.d) $(DIFFUSION_OBJS:.o=.d) \
                $(TUI_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(BACKEND_OBJS:.o=.d)) $(OUT_DIR)/kappai-monitor.d

config:
	@mkdir -p $(OUT_DIR)
	@printf 'BUILD = %s\nBACKENDS = %s\nCPU_ARCH_OPT = %s\nTSAN = %s\nTASK_BACKEND = %s\nMACHINE_ARCH_FLAGS = %s\n' \
		'$(BUILD)' '$(BACKEND_LIST)' '$(CPU_ARCH_OPT)' '$(TSAN)' '$(TASK_BACKEND)' '$(ARCH_FLAGS)' > $(CONFIG_FILE).new
	@if cmp -s $(CONFIG_FILE).new $(CONFIG_FILE) 2>/dev/null; then rm $(CONFIG_FILE).new; \
		else mv $(CONFIG_FILE).new $(CONFIG_FILE); fi
	@cat $(CONFIG_FILE)

help:
	@echo "make config [BUILD=debug|release] [BACKENDS=$(subst $(space),$(comma),$(AVAILABLE_BACKENDS))]"
	@echo "            [CPU_ARCH_OPT=0|1] [TSAN=0|1] [TASK_BACKEND=auto|ucontext|pthread]"
	@echo "make                  kappai, kappai-cli, kappai-server, kappai-diffusion"
	@echo "make kappai-test      backend regression tests"
	@echo "make kappai-monitor   monitor viewer"
	@echo "make clean | format | tidy"

clean:
	rm -rf $(OUT_DIR)

NPROC       = $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
ALL_SRCS    = $(shell find $(SRC_DIR) -type f \( -name "*.c" -o -name "*.cpp" \))
ALL_FILES   = $(ALL_SRCS) $(shell find $(SRC_DIR) -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.comp" -o -name "*.glsl" -o -name "*.inc" \))
TIDY_LOG    := $(OUT_DIR)/tidy.log
TIDY_SRCS    = $(filter-out $(foreach a,$(filter-out $(CPU_ARCH),aarch64 x86_64),$(SRC_DIR)/backend/cpu/$(a)/%),$(ALL_SRCS))

format:
	@command -v clang-format >/dev/null 2>&1 || { echo "clang-format not found"; exit 1; }
	@printf '%s\n' $(ALL_FILES) | xargs -P $(NPROC) -I {} sh -c ' \
		echo "  FMT     {}"; \
		clang-format -i -style=file {}; \
		if [ -s {} ] && [ "$$(tail -c1 {})" != "" ]; then printf "\n" >> {}; fi; \
		chmod 644 {}'

tidy:
	@command -v clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not found"; exit 1; }
	@mkdir -p $(OUT_DIR)
	@: > $(TIDY_LOG)
	@printf '%s\n' $(TIDY_SRCS) | xargs -P $(NPROC) -n 1 sh -c ' \
		echo "  TIDY    $$1"; \
		case "$$1" in \
		*/opengl/*) inc="-I$(OBJ_DIR)/backend/opengl";; \
		*/vulkan/*) inc="-I$(OBJ_DIR)/backend/vulkan";; \
		esac; \
		clang-tidy --config-file=.clang-tidy "$$1" -- $(filter-out -fvect-cost-model=%,$(CFLAGS)) $(ARCH_FLAGS) $$inc -I$(SRC_DIR) >> $(TIDY_LOG) 2>&1 || true' sh
	@echo "  TIDY    done, see $(TIDY_LOG)"
