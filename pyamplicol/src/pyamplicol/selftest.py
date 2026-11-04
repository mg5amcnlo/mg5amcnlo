# SPDX-License-Identifier: 0BSD
"""Fast installed-distribution self-test entry point."""

from __future__ import annotations

import json

from .diagnostics import run_self_test


def main() -> int:
    report = run_self_test()
    print(
        json.dumps(
            {
                "package_version": report.package_version,
                "ok": report.ok,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status,
                        "detail": check.detail,
                    }
                    for check in report.checks
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
