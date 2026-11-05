#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job madspace_tests of gpu_runner_ci.yml.
# The madspace unit tests (madspace/tests) on the GPU: MADSPACE_TEST_DEVICE=$BACKEND puts
# their torch tensors on the GPU, so that madspace runs on its GPU runtime.
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, MADSPACE_PREFIX, WORKDIR).
# This runs the python3 of the modules, not the venv of build_madspace (same interpreter,
# which madspace was built with): EasyBuild finds some packages of the modules (e.g.
# typing_extensions of Python-bundle-PyPI, which torch imports) through EBPYTHONPREFIXES,
# which only the python3 of the modules reads (its sitecustomize), not a venv made from it.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }
START=$(date +%s)

section "Environment"
if [ -n "$MODULES" ]; then source "$HERE/load_modules.sh"; fi
python3 --version
case $BACKEND in
    cuda) GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p) ;;
    hip)  GPU_NAME=$(rocminfo 2> /dev/null | awk -F': *' '/Marketing Name/ {name=$2} /Device Type: *GPU/ {print name; exit}' || true) ;;
    *)    echo "::error::the madspace tests on the GPU need a GPU backend, not '$BACKEND'"; exit 2 ;;
esac
# test requirements the modules lack (pytest, scipy) are kept per python version, after
# the packages of the modules in the path
DEPS=${GLOBALSCRATCH:-$HOME}/mg7-gpu-ci/cache/test-deps-$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
export PYTHONPATH=$MADSPACE_PREFIX${PYTHONPATH:+:$PYTHONPATH}:$DEPS
MISSING=$(python3 -c 'import importlib.util as u; print(" ".join(m for m in ("pytest", "scipy") if u.find_spec(m) is None))')
if [ -n "$MISSING" ]; then
    echo "Installing $MISSING into $DEPS"
    python3 -m pip install --quiet --target="$DEPS" $MISSING
fi
# manneback has only a CPU build of PyTorch among its modules: then a CUDA wheel, kept in
# the cache, goes before it in the path
if [ "$BACKEND" = cuda ] && ! python3 -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' 2> /dev/null; then
    TORCH_WHEEL=${TORCH_WHEEL:-torch==2.8.0}
    TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
    TORCH_DIR=$DEPS-$(printf '%s %s' "$TORCH_WHEEL" "$TORCH_INDEX" | sha1sum | cut -c1-8)
    if [ ! -e "$TORCH_DIR/.ready" ]; then
        echo "The torch of the modules has no CUDA: installing $TORCH_WHEEL from $TORCH_INDEX into $TORCH_DIR"
        rm -rf "$TORCH_DIR"
        python3 -m pip install --quiet --target="$TORCH_DIR" --index-url "$TORCH_INDEX" "$TORCH_WHEEL"
        touch "$TORCH_DIR/.ready"
    fi
    export PYTHONPATH=$TORCH_DIR:$PYTHONPATH
fi
python3 -c 'import torch; print("torch", torch.__version__, "GPU available:", torch.cuda.is_available())'

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
section "madspace tests on $BACKEND ($GPU_NAME)"
STATUS=0
MADSPACE_TEST_DEVICE=$BACKEND python3 -m pytest "$REPO/madspace/tests" -ra -p no:cacheprovider \
    --junitxml=junit.xml 2>&1 | tee tests.log || STATUS=1

section "Summary"
{
    echo "tests=$(grep -E '^=+ .*(passed|failed|error|skipped).* =+$' tests.log | tail -1 | sed 's/^=* *//; s/ *=*$//')"
    echo "node=$(uname -n)"
    echo "gpu=${GPU_NAME:-none}"
    echo "walltime=$(( $(date +%s) - START ))s"
} > summary.txt
cat summary.txt
exit $STATUS
