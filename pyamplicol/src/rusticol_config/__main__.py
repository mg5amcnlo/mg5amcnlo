# SPDX-License-Identifier: 0BSD
"""Support ``python -m rusticol_config`` without importing pyAmpliCol."""

from . import main

raise SystemExit(main())
