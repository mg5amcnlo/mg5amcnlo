#!/usr/bin/env python3
"""Rerun the 12 matching-scale variations with hard-scale weights, sequentially."""

from run_g_variations import run_campaign


if __name__ == "__main__":
    run_campaign({"S": 12}, "scale_runs_fixed", __doc__)
