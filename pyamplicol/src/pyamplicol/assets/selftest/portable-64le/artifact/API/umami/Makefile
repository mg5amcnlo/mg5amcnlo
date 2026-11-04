# SPDX-License-Identifier: 0BSD
RUSTICOL_CONFIG ?= rusticol-config
RUSTICOL_CONFIG_PATH := $(shell \
	if command -v "$(RUSTICOL_CONFIG)" >/dev/null 2>&1; then \
		command -v "$(RUSTICOL_CONFIG)"; \
	elif [ "$(RUSTICOL_CONFIG)" = rusticol-config ]; then \
		directory="$(CURDIR)"; \
		while [ "$$directory" != / ]; do \
			candidate="$$directory/.venv/bin/rusticol-config"; \
			if [ -x "$$candidate" ]; then printf '%s\n' "$$candidate"; break; fi; \
			directory=$$(dirname "$$directory"); \
		done; \
	fi)
ifeq ($(strip $(RUSTICOL_CONFIG_PATH)),)
$(error rusticol-config is unavailable; activate the pyAmpliCol environment or set RUSTICOL_CONFIG=/absolute/path)
endif
include providers.mk
CC ?= cc
# Make's abspath/notdir functions split on spaces; the shell preserves a path.
ARTIFACT_DIR := $(shell cd ../.. && pwd -P)
BUILD_DIR ?= $(shell dirname "$(ARTIFACT_DIR)")/.pyamplicol-api-build/$(shell basename "$(ARTIFACT_DIR)")/umami
empty :=
space := $(empty) $(empty)
make_path = $(subst $(space),\$(space),$(1))
BUILD_TARGET := $(call make_path,$(BUILD_DIR))
CFLAGS ?= -O2
CFLAGS += -std=c11 -Wall -Wextra -pedantic
SDK_LIBRARY := $(shell "$(RUSTICOL_CONFIG_PATH)" --library)
ifeq ($(strip $(SDK_LIBRARY)),)
$(error rusticol-config could not resolve the native library)
endif
SDK_CFLAGS := $(shell "$(RUSTICOL_CONFIG_PATH)" --cflags)
SDK_LIBS := $(shell "$(RUSTICOL_CONFIG_PATH)" --libs)
SDK_LIBRARY_TARGET := $(call make_path,$(SDK_LIBRARY))
HOST_SYSTEM := $(shell uname -s)
ifeq ($(HOST_SYSTEM),Darwin)
SHARED_FLAGS = -dynamiclib -Wl,-install_name,@rpath/$(shell basename "$@") -Wl,-exported_symbols_list,umami.exports.macos
DRIVER_FLAGS = -Wl,-rpath,@loader_path
else ifeq ($(HOST_SYSTEM),Linux)
SHARED_FLAGS = -shared -Wl,-z,defs -Wl,-soname,$(shell basename "$@") -Wl,--version-script=umami.exports
DRIVER_FLAGS = -Wl,-rpath,'$$ORIGIN'
else
$(error UMAMI shared-library examples support Linux and macOS)
endif
ifeq ($(UMAMI_SINGLE_PROVIDER),1)
LIBRARY_NAMES := libumami.so
DRIVER_NAMES := umami_driver
RUN_DRIVER := $(BUILD_DIR)/umami_driver
else
LIBRARY_NAMES := $(addprefix libumami_,$(addsuffix .so,$(UMAMI_PROVIDERS)))
DRIVER_NAMES := $(addprefix umami_driver_,$(UMAMI_PROVIDERS))
PROVIDER ?= $(firstword $(UMAMI_PROVIDERS))
RUN_DRIVER := $(BUILD_DIR)/umami_driver_$(PROVIDER)
endif
LIBRARIES := $(addprefix $(BUILD_TARGET)/,$(LIBRARY_NAMES))
DRIVERS := $(addprefix $(BUILD_TARGET)/,$(DRIVER_NAMES))

.PHONY: all shared run clean
all: $(LIBRARIES) $(DRIVERS)
shared: $(LIBRARIES)

$(BUILD_TARGET):
	mkdir -p "$@"

# Explicit generated rules also work with spaces in directories on older Make
# versions, whose pattern-rule handling does not preserve escaped spaces.
define provider_rules
$(BUILD_TARGET)/libumami$(2).so: umami.c provider_$(1).h umami_provider.h umami.h umami.exports umami.exports.macos $(SDK_LIBRARY_TARGET) | $(BUILD_TARGET)
	$$(CC) $$(CPPFLAGS) $$(SDK_CFLAGS) $$(CFLAGS) -fPIC -DUMAMI_PROVIDER_HEADER='"provider_$(1).h"' $$(SHARED_FLAGS) "$$<" -o "$$@" $$(LDFLAGS) $$(SDK_LIBS)

$(BUILD_TARGET)/umami_driver$(2): umami_driver.c $(BUILD_TARGET)/libumami$(2).so provider_$(1).h umami_provider.h umami.h | $(BUILD_TARGET)
	$$(CC) $$(CPPFLAGS) $$(CFLAGS) -DUMAMI_PROVIDER_HEADER='"provider_$(1).h"' "$$<" "$(BUILD_DIR)/libumami$(2).so" -o "$$@" $$(LDFLAGS) $$(DRIVER_FLAGS)
endef
ifeq ($(UMAMI_SINGLE_PROVIDER),1)
$(eval $(call provider_rules,p0,))
else
$(foreach provider,$(UMAMI_PROVIDERS),$(eval $(call provider_rules,$(provider),_$(provider))))
endif

run: $(call make_path,$(RUN_DRIVER))
	"$(RUN_DRIVER)" "$(ARTIFACT_DIR)" $(ARGS)

clean:
	$(RM) $(foreach name,$(LIBRARY_NAMES) $(DRIVER_NAMES),"$(BUILD_DIR)/$(name)")
