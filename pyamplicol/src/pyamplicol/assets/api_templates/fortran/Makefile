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
ifeq ($(origin FC), default)
FC := gfortran
endif
FC ?= gfortran
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/fortran
TARGET := $(BUILD_DIR)/check_standalone
RUSTICOL_FORTRAN := $(shell "$(RUSTICOL_CONFIG_PATH)" --fortran-source)
FFLAGS ?= -O2
FFLAGS += -std=f2008 -ffree-line-length-none -Wall -Wextra
LDLIBS += $(shell "$(RUSTICOL_CONFIG_PATH)" --libs)

.PHONY: all check_standalone clean run

all: $(TARGET)

check_standalone: $(TARGET)

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGET): check_standalone.f90 | $(BUILD_DIR)
	test -f "$(RUSTICOL_FORTRAN)"
	$(FC) $(FFLAGS) -J"$(BUILD_DIR)" "$(RUSTICOL_FORTRAN)" "$<" -o "$@" $(LDLIBS)

run: $(TARGET)
	cd "$(ARTIFACT_DIR)" && "$(TARGET)" $(ARGS)

clean:
	$(RM) "$(TARGET)" "$(BUILD_DIR)/rusticol.mod"
