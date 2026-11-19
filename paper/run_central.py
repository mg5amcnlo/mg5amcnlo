#!/usr/bin/env python3
"""Regenerate the four central MC@NLO samples and their shower plots."""

from run_g_variations import run_campaign


if __name__ == "__main__":
    run_campaign({"C": 4}, "central_runs_fixed", __doc__)
