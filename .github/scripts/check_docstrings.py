#!/usr/bin/env python3
"""Fail if madspace's pybind11 bindings carry no docstrings, i.e. the docs
build compiled madspace with ENABLE_DOCS=OFF. The other build guards (page
counts, ':glob:' fallback) can't catch this: the OFF stub still writes a
docstrings.hpp whose doc() returns "", so pages exist but carry no prose.
"""
import sys

sys.path.insert(0, "madspace/install")
import madspace  # noqa: E402

CHECK = ("Mapping", "Diagram", "FunctionBuilder", "Cuts")
bad = [n for n in CHECK if len((getattr(madspace, n).__doc__ or "").strip()) < 50]
if bad:
    sys.exit(f"empty docstrings for {bad}: madspace was built with ENABLE_DOCS=OFF")
print(f"docstring check OK for {CHECK}")
