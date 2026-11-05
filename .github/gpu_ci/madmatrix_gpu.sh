#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job madmatrix of gpu_runner_ci.yml.
# The GPU backend of madmatrix against its CPU backend, with check_sa.exe: see
# madmatrix_checks.py for what is checked.
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, GPU_ARCH, VENV, WORKDIR); no madspace
# and no PDF set needed (standalone outputs only).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }

section "Environment"
if [ -n "$MODULES" ]; then source "$HERE/load_modules.sh"; fi
source "$VENV/bin/activate"
python3 --version
case $BACKEND in
    cuda) nvidia-smi
          GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p) ;;
    hip)  rocm-smi || true
          GPU_NAME=$(rocminfo 2> /dev/null | awk -F': *' '/Marketing Name/ {name=$2} /Device Type: *GPU/ {print name; exit}' || true) ;;
    *)    echo "::error::the madmatrix checks need a GPU backend, not '$BACKEND'"; exit 2 ;;
esac
export GPU_NAME
if [ -n "$GPU_ARCH" ]; then
    export MADGRAPH_CUDA_ARCHITECTURE=$GPU_ARCH MADGRAPH_HIP_ARCHITECTURE=$GPU_ARCH
fi

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
python3 "$HERE/madmatrix_checks.py" --repo "$REPO" --backend "$BACKEND" 2>&1 | tee checks.log || STATUS=1

section "Summary"
cat summary.txt 2> /dev/null || echo "no summary.txt"
exit $STATUS
