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
RUSTC ?= rustc
RUST_SCRIPT ?= rust-script
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/rust
TARGET := $(BUILD_DIR)/check_standalone
RUSTFLAGS ?= -C opt-level=2
RUSTICOL_RUST_SOURCE := $(shell "$(RUSTICOL_CONFIG_PATH)" --rust-source)
RUSTICOL_RUSTFLAGS := $(shell "$(RUSTICOL_CONFIG_PATH)" --rustflags)

.PHONY: all check_standalone clean run run-script

all: $(TARGET)

check_standalone: $(TARGET)

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGET): check_standalone.rs | $(BUILD_DIR)
	RUSTICOL_RUST_SOURCE="$(RUSTICOL_RUST_SOURCE)" \
		$(RUSTC) --edition=2021 $(RUSTFLAGS) "$<" -o "$@" $(RUSTICOL_RUSTFLAGS)

run: $(TARGET)
	cd "$(ARTIFACT_DIR)" && "$(TARGET)" $(ARGS)

run-script:
	cd "$(ARTIFACT_DIR)" && \
		RUSTICOL_RUST_SOURCE="$(RUSTICOL_RUST_SOURCE)" \
		CARGO_ENCODED_RUSTFLAGS="$$("$(RUSTICOL_CONFIG_PATH)" --cargo-rustflags)" \
		$(RUST_SCRIPT) "API/rust/check_standalone.rs" -- $(ARGS)

clean:
	$(RM) "$(TARGET)"
