# SPDX-License-Identifier: 0BSD
"""Compatibility import for the standalone, metadata-only SDK query utility."""

from rusticol_config import SdkInfo, SdkUnavailableError, load_sdk_info, main

__all__ = ["SdkInfo", "SdkUnavailableError", "load_sdk_info", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
