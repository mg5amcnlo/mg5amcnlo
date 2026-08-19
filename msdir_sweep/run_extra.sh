#!/bin/bash
# The two legs outside the (spinmode x ordering x fresh/reuse) matrix.
cd "$(dirname "$0")/.."
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
python3 msdir_sweep/sweep_use_old_dir.py
python3 msdir_sweep/sweep_cross_spinmode.py onshell madspin PA madspin_v1 none
