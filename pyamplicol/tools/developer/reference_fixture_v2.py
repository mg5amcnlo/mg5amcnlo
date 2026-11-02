#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compatibility entry point for the strict reference-fixture v2 reader."""

from __future__ import annotations

import sys
from pathlib import Path

DEVELOPER_ROOT = Path(__file__).resolve().parent
if str(DEVELOPER_ROOT) not in sys.path:
    sys.path.insert(0, str(DEVELOPER_ROOT))

from reference_fixture import *  # type: ignore[import-not-found] # noqa: E402,F403
from reference_fixture import __all__  # noqa: E402,F401
