#!/bin/bash
# Generate the p p > t t~ sample the ms_dir sweep runs on.
cd "$(dirname "$0")/.."
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
./bin/mg5_aMC -f msdir_sweep/gen.mg5 > msdir_sweep/work/gen.log 2>&1
echo "EXIT=$?"
tail -20 msdir_sweep/work/gen.log
