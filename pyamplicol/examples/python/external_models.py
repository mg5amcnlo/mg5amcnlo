# SPDX-License-Identifier: 0BSD
"""Resolve serialized-JSON and trusted-UFO Standard Model sources."""

from __future__ import annotations

import argparse

from pyamplicol import ModelSource


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sm_json", help="Path to the serialized SM JSON model")
    parser.add_argument("sm_ufo", help="Path to a trusted UFO model named sm")
    parser.add_argument("--ufo-restriction", default="restrict_default.dat")
    return parser


def main() -> int:
    args = _parser().parse_args()
    sources = {
        "sm-json": ModelSource.from_path(args.sm_json),
        "sm-ufo": ModelSource.from_path(
            args.sm_ufo,
            restriction=args.ufo_restriction,
        ),
    }
    for name, source in sources.items():
        print(f"{name}: kind={source.kind} path={source.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
