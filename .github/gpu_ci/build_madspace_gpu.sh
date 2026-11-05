#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job build_madspace of gpu_runner_ci.yml.
# Builds madspace with the CUDA or HIP backend, for the GPU of this node only, and
# keeps it on the cluster: it is rebuilt only when the madspace sources, the backend,
# the GPU architecture or the loaded module versions change.
# Environment:
#   BACKEND    cuda | hip
#   MODULES    modules to load (see load_modules.sh)
#   GPU_ARCH   CUDA compute capability (e.g. 80) or HIP target (e.g. gfx942);
#              empty = the GPU of this node
#   CACHE_DIR  cache root on the cluster (default $GLOBALSCRATCH/mg7-gpu-ci/cache)
# Writes venv=<python venv> and madspace_prefix=<install prefix> to $GITHUB_OUTPUT.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/load_modules.sh"
CACHE_DIR=${CACHE_DIR:-${GLOBALSCRATCH:-$HOME}/mg7-gpu-ci/cache}
mkdir -p "$CACHE_DIR"

case $BACKEND in
    cuda)
        nvidia-smi
        ARCH=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n 1p | tr -d .)}
        GPU_FLAGS=(-Ccmake.define.ENABLE_CUDA=ON "-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=$ARCH") ;;
    hip)
        rocm-smi || true
        ARCH=${GPU_ARCH:-$(rocm_agent_enumerator 2> /dev/null | grep -v gfx000 | sort -u | sed -n 1p || true)}
        GPU_FLAGS=(-Ccmake.define.ENABLE_HIP=ON "-Ccmake.define.CMAKE_HIP_ARCHITECTURES=$ARCH")
        ROCM=${ROCM_PATH:-$(hipconfig --rocmpath)}
        export CMAKE_PREFIX_PATH=$ROCM${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH} ;;
    *)
        echo "::error::unknown BACKEND '$BACKEND' (expected cuda or hip)"; exit 2 ;;
esac
if [ -z "$ARCH" ]; then
    echo "::error::could not detect the GPU architecture, set the gpu_arch input"; exit 2
fi
echo "Backend $BACKEND, GPU architecture $ARCH"

# python environment, per set of modules actually loaded (with their versions, so that
# an unversioned name like "ROCm" triggers a rebuild when its default version changes)
ENV_KEY=$(printf '%s\n' "$(module -t list 2>&1 | sort)" "$(python3 -VV)" | sha1sum | cut -c1-12)
ENV_DIR=$CACHE_DIR/env-$ENV_KEY
VENV=$ENV_DIR/venv
if [ ! -e "$VENV/.ready" ]; then
    echo "Creating the python environment $VENV"
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip packaging
    touch "$VENV/.ready"
fi

# madspace build (per madspace source tree, backend and architecture)
SRC=$(git -C "$REPO" rev-parse HEAD:madspace)
PREFIX=$ENV_DIR/madspace-$SRC-$BACKEND-$ARCH
if [ -e "$PREFIX/.ready" ]; then
    echo "Reusing the madspace build $PREFIX"
    touch "$PREFIX" # keeps it out of the 30-day clean-up below
else
    echo "Building madspace into $PREFIX"
    rm -rf "$PREFIX"
    # EasyBuild modules (e.g. PyTorch) bring an older pybind11 whose include directory is in
    # CPATH, which the compiler searches before the pybind11 3.x fetched by madspace's CMake:
    # drop it for this build
    for var in CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
        if [ -n "${!var}" ]; then
            export "$var=$(printf '%s' "${!var}" | tr ':' '\n' | { grep -v '/pybind11/' || true; } | paste -sd: -)"
        fi
    done
    CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-8} \
        "$VENV/bin/python" -m pip install --target="$PREFIX" "$REPO/madspace" \
        "${GPU_FLAGS[@]}" -Ccmake.define.ENABLE_OPENBLAS=ON -Ccmake.build-type=Release
    touch "$PREFIX/.ready"
fi
# fails if the GPU backend library was not built or cannot be loaded
PYTHONPATH=$PREFIX "$VENV/bin/python" -c "import madspace as ms; print('madspace device:', ms.${BACKEND}_device(0))"

# forget the builds not used for 30 days
find "$CACHE_DIR" -mindepth 2 -maxdepth 2 -name 'madspace-*' -type d -mtime +30 -print -exec rm -rf {} + || true

if [ -n "$GITHUB_OUTPUT" ]; then
    echo "venv=$VENV" >> "$GITHUB_OUTPUT"
    echo "madspace_prefix=$PREFIX" >> "$GITHUB_OUTPUT"
fi
