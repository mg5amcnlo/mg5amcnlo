#!/bin/bash
# usage: run_sweep.sh [spinmode ...]   (no argument = the whole matrix)
cd "$(dirname "$0")/.."
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
exec python3 msdir_sweep/sweep.py "$@"
