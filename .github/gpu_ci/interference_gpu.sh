#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job interference of gpu_runner_ci.yml.
# The squared split orders (interference) on the GPU backend of madmatrix and in mg7:
# see interference_checks.py for what is checked.
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, GPU_ARCH, VENV, MADSPACE_PREFIX,
# WORKDIR, CACHE_DIR). The mg7 check runs with the PDF set its reference was made with,
# whatever the workflow's pdf_set: interference_checks.py names it and it is downloaded
# into the cache if needed. INTERFERENCE_EVENTS sets its number of events (5000).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }
CACHE_DIR=${CACHE_DIR:-${GLOBALSCRATCH:-$HOME}/mg7-gpu-ci/cache}
mkdir -p "$CACHE_DIR"

section "Environment"
if [ -n "$MODULES" ]; then source "$HERE/load_modules.sh"; fi
source "$VENV/bin/activate"
python3 --version
case $BACKEND in
    cuda) nvidia-smi
          GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p) ;;
    hip)  rocm-smi || true
          GPU_NAME=$(rocminfo 2> /dev/null | awk -F': *' '/Marketing Name/ {name=$2} /Device Type: *GPU/ {print name; exit}' || true) ;;
    *)    echo "::error::the interference checks need a GPU backend, not '$BACKEND'"; exit 2 ;;
esac
export GPU_NAME
if [ -n "$GPU_ARCH" ]; then
    export MADGRAPH_CUDA_ARCHITECTURE=$GPU_ARCH MADGRAPH_HIP_ARCHITECTURE=$GPU_ARCH
fi

# madspace goes where the mg7 runtime looks for it: <MadGraph7>/madspace/install
rm -rf "$REPO/madspace/install"
ln -s "$MADSPACE_PREFIX" "$REPO/madspace/install"
# default configuration, as .github/actions/checkout_mg5 does for the other CI jobs
cp "$REPO/input/.mg7_configuration_default.txt" "$REPO/input/mg7_configuration.txt"
cp "$REPO/Template/LO/Source/.make_opts" "$REPO/Template/LO/Source/make_opts"
cat >> "$REPO/input/mg7_configuration.txt" << EOF
auto_update = 0
automatic_html_opening = False
notification_center = False
nb_core = ${SLURM_CPUS_PER_TASK:-4}
EOF

export LHAPDF_DATA_PATH=$CACHE_DIR/lhapdf
REFERENCE_PDF=$(python3 "$HERE/interference_checks.py" --print-reference-pdf)
if [ ! -d "$LHAPDF_DATA_PATH/$REFERENCE_PDF" ]; then
    echo "Downloading the PDF set $REFERENCE_PDF"
    mkdir -p "$LHAPDF_DATA_PATH"
    curl -fsSL --retry 3 "https://lhapdfsets.web.cern.ch/current/$REFERENCE_PDF.tar.gz" | tar -xz -C "$LHAPDF_DATA_PATH"
fi

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
STATUS=0
python3 "$HERE/interference_checks.py" --repo "$REPO" --backend "$BACKEND" \
    --events "${INTERFERENCE_EVENTS:-5000}" 2>&1 | tee checks.log || STATUS=1

section "Summary"
cat summary.txt 2> /dev/null || echo "no summary.txt"
exit $STATUS
