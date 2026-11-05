#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job acceptance of gpu_runner_ci.yml.
# The acceptance tests that need a GPU, through tests/test_manager.py. Such a test skips
# without nvcc/hipcc or without a device, which on a GPU node is a failure (test_manager
# exits 1 on a skip, and summary.txt says which).
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, GPU_ARCH, VENV, WORKDIR); no madspace
# and no PDF set needed (standalone outputs only). GPU_TESTS: the tests to run, space
# separated (test names, or test_manager expressions), by default the list below.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }

# the good helicities chosen at umami_initialize, with device buffers
DEFAULT_TESTS="test_standalone_umami_first_call_gpu"

section "Environment"
if [ -n "$MODULES" ]; then source "$HERE/load_modules.sh"; fi
source "$VENV/bin/activate"
python3 --version
case $BACKEND in
    cuda) nvidia-smi
          GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p) ;;
    hip)  rocm-smi || true
          GPU_NAME=$(rocminfo 2> /dev/null | awk -F': *' '/Marketing Name/ {name=$2} /Device Type: *GPU/ {print name; exit}' || true) ;;
    *)    echo "::error::the GPU acceptance tests need a GPU backend, not '$BACKEND'"; exit 2 ;;
esac
export GPU_NAME
if [ -n "$GPU_ARCH" ]; then
    export MADGRAPH_CUDA_ARCHITECTURE=$GPU_ARCH MADGRAPH_HIP_ARCHITECTURE=$GPU_ARCH
fi
# the tests call make without -j
export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK:-4}"

# default configuration, as .github/actions/checkout_mg5 does for the other CI jobs
cp "$REPO/input/.mg7_configuration_default.txt" "$REPO/input/mg7_configuration.txt"
cp "$REPO/Template/LO/Source/.make_opts" "$REPO/Template/LO/Source/make_opts"
cat >> "$REPO/input/mg7_configuration.txt" << EOF
auto_update = 0
automatic_html_opening = False
notification_center = False
nb_core = ${SLURM_CPUS_PER_TASK:-4}
EOF

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
STATUS=0
: > summary.txt
for test in ${GPU_TESTS:-$DEFAULT_TESTS}; do
    section "$test"
    log=$WORKDIR/${test//[^A-Za-z0-9_.-]/_}.log
    if (cd "$REPO" && python3 ./tests/test_manager.py -p A -t0 -v 2 "$test") 2>&1 | tee "$log"; then
        result=ok
    else
        result=FAILED
        STATUS=1
    fi
    # unittest's own verdict ("Ran 1 test in 95.2s", "OK"), and why a test skipped: the
    # verdict of test_manager's runner says OK for a skip
    ran=$(grep -E '^Ran [0-9]+ tests? in ' "$log" | tail -1 || true)
    verdict=$(grep -E '^(OK|FAILED)( \(.*\))?$' "$log" | tail -1 || true)
    skipped=$(grep -oE '\.\.\. skipped .*' "$log" | sed 's/^\.\.\. //' | tail -1 || true)
    echo "$test=$result (${ran:-no test ran}, ${verdict:-no verdict}${skipped:+, $skipped})" >> summary.txt
done
{
    echo "node=$(hostname)"
    echo "gpu=${GPU_NAME:-none}"
    echo "backend=$BACKEND"
} >> summary.txt

section "Summary"
cat summary.txt
exit $STATUS
