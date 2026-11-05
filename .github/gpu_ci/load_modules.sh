# Sourced by the scripts running on the self-hosted runner (GPU node).
# Loads the modules of $MODULES (space separated), in that order, after 'module purge',
# and checks that the python in PATH is recent enough for mg7 (tomllib: python >= 3.11).
# On failure it lists the modules that may have been meant and exits.
module purge
for mod in $MODULES; do
    if ! module load "$mod"; then
        echo "::error::cannot load module '$mod'. Possibly relevant modules on $(hostname):"
        module -t avail 2>&1 | grep -i -E 'cuda|rocm|hip|python|gcc|cmake|releases' | sort -u || true
        exit 2
    fi
done
module list 2>&1
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "::error::mg7 needs python >= 3.11, found $(python3 --version 2>&1): add a Python module to MODULES"
    exit 2
fi
