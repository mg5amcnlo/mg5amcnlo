#!/usr/bin/env bash
#
# Content check for the HEPTools cache tree.
#
# "./bin/mg5_aMC cmd" exits 0 even when the "install <tool>" it contains
# failed (e.g. the boost tarball came down truncated, so bootstrap.sh/b2
# never ran and eMELA aborted).  A warm_cache job therefore ends green and
# actions/cache happily saves a cache with the tool missing, which only shows
# up much later as a broken acceptance test.
#
# Every path asserted below mirrors what HEPToolInstaller.py installs
# ('install_path' = <prefix>/<tool>, libraries looked up in lib/ or lib64/)
# and what the .github/actions/restore_heptools_* actions then hand to MG5
# through input/mg5_configuration.txt.
#
# Usage: check_heptools.sh <component> [<component> ...]
#   HEPTOOLS_PREFIX  install prefix to inspect  (default ~/.cache/HEPtools)
#   CHECK_MODE       fatal (default) | report

set -o pipefail

HEP="${HEPTOOLS_PREFIX:-}"
[ -n "$HEP" ] || HEP="$HOME/.cache/HEPtools"
MODE="${CHECK_MODE:-fatal}"

missing=()

need_file() { [ -f "$1" ] || missing+=("missing file: $1"); }
need_exec() { [ -x "$1" ] || missing+=("missing executable: $1"); }
need_dir()  { [ -d "$1" ] || missing+=("missing directory: $1"); }

# need_glob <description> <pattern> [<pattern> ...]
# Satisfied as soon as one pattern matches something (used for the lib/lib64
# ambiguity and for library extensions that differ between builds).
need_glob() {
    local desc="$1"; shift
    local pattern
    for pattern in "$@"; do
        if compgen -G "$pattern" > /dev/null 2>&1; then
            return 0
        fi
    done
    missing+=("missing $desc (no match for: $*)")
}

check_lhapdf() {
    need_exec "$HEP/lhapdf6_py3/bin/lhapdf-config"
    need_exec "$HEP/bin/lhapdf-config"
    need_glob "libLHAPDF" "$HEP/lhapdf6_py3/lib/libLHAPDF.*" "$HEP/lhapdf6_py3/lib64/libLHAPDF.*"
    # the two PDF sets the NLO acceptance tests need are unpacked by the
    # warm_cache lhapdf job itself, not by the installer
    need_dir "$HEP/lhapdf6_py3/share/LHAPDF/cteq6l1"
    need_dir "$HEP/lhapdf6_py3/share/LHAPDF/NNPDF23_nlo_as_0118_qed"
}

check_boost() {
    # exactly what HEPToolInstaller.find_dependency('boost') looks for: without
    # it eMELA re-downloads and rebuilds boost from scratch
    need_glob "libboost_system" \
        "$HEP/boost/lib/libboost_system.*" "$HEP/boost/lib/libboost_system-mt.*" \
        "$HEP/boost/lib64/libboost_system.*" "$HEP/boost/lib64/libboost_system-mt.*"
    need_file "$HEP/boost/include/boost/version.hpp"
}

check_pythia8() {
    # a network-failed pythia8 install leaves an incomplete pythia8/ dir; MG5
    # then nullifies pythia8_path at startup, leaving the parton shower
    # unavailable and breaking the rivet/contur acceptance tests
    need_file "$HEP/pythia8/include/Pythia8/Pythia.h"
    need_exec "$HEP/pythia8/bin/pythia8-config"
    need_glob "libpythia8" "$HEP/pythia8/lib/libpythia8.*" "$HEP/pythia8/lib64/libpythia8.*"
    need_dir  "$HEP/hepmc"
    need_file "$HEP/MG5aMC_PY8_interface/MG5aMC_PY8_interface"
}

check_emela() {
    # bin/eMELA-config is the symlink finalize_installation() creates and the
    # path restore_heptools_emela writes into mg5_configuration.txt
    need_exec "$HEP/bin/eMELA-config"
    need_glob "eMELA grids" "$HEP/EMELA/share/eMELA/*"
}

check_contur() {
    # a missing fastjet-config means NLO subprocesses cannot compile
    # (fastjet/ClusterSequence.hh)
    need_exec "$HEP/fastjet/bin/fastjet-config"
    need_glob "libfastjet" "$HEP/fastjet/lib/libfastjet.*" "$HEP/fastjet/lib64/libfastjet.*"
    need_glob "libRivet"   "$HEP/rivet/lib/libRivet.*"     "$HEP/rivet/lib64/libRivet.*"
    need_glob "libYODA"    "$HEP/yoda/lib/libYODA.*"       "$HEP/yoda/lib64/libYODA.*"
    need_dir  "$HEP/contur"
    need_dir  "$HEP/hepmc3"
}

check_looptools() {
    need_file "$HEP/CutTools/lib/libcts.a"
    need_file "$HEP/CutTools/lib/mpmodule.mod"
    need_file "$HEP/IREGI/src/libiregi.a"
    need_glob "libcollier" "$HEP/collier/libcollier.*" \
        "$HEP/collier/lib/libcollier.*" "$HEP/collier/lib64/libcollier.*"
    need_glob "libninja"   "$HEP/ninja/lib/libninja.*" "$HEP/ninja/lib64/libninja.*"
}

if [ "$#" -eq 0 ]; then
    echo "::error::check_heptools: no component requested"
    exit 1
fi

echo "Checking HEPTools content under $HEP for: $*"
for component in "$@"; do
    case "$component" in
        lhapdf|boost|pythia8|emela|contur|looptools) "check_$component" ;;
        *) echo "::error::check_heptools: unknown component '$component'"; exit 1 ;;
    esac
done

if [ "${#missing[@]}" -eq 0 ]; then
    echo "All requested components are complete."
    [ -n "${GITHUB_OUTPUT:-}" ] && echo "valid=true" >> "$GITHUB_OUTPUT"
    exit 0
fi

level=error
[ "$MODE" = "fatal" ] || level=warning
for entry in "${missing[@]}"; do
    echo "::${level}::HEPTools cache incomplete ($*): ${entry}"
done
[ -n "${GITHUB_OUTPUT:-}" ] && echo "valid=false" >> "$GITHUB_OUTPUT"

if [ "$MODE" = "fatal" ]; then
    echo "Refusing to treat this build as successful: the cache would be saved incomplete."
    exit 1
fi
exit 0
