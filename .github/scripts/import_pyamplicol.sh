#!/usr/bin/env bash
# Import (or re-sync) the history of github.com/MadGraphTeam/pyamplicol into
# the pyamplicol/ subdirectory. The filter is deterministic, so re-running it
# on a newer upstream main reproduces the SHAs of commits imported earlier and
# a merge only brings in the new ones.
#
# Usage: .github/scripts/import_pyamplicol.sh [upstream-ref]
# Leaves the filtered history in the local branch pyamplicol-import; merge it
# with `git merge pyamplicol-import` (first import: --allow-unrelated-histories).
# Requires git-filter-repo (set FILTER_REPO to override the command).
set -euo pipefail

UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/MadGraphTeam/pyamplicol.git}
REF=${1:-main}
FILTER_REPO=${FILTER_REPO:-git-filter-repo}
MG_ROOT=$(git rev-parse --show-toplevel)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git clone --quiet --no-local --single-branch --branch "$REF" "$UPSTREAM_URL" "$WORK/pyamplicol"
cd "$WORK/pyamplicol"
UPSTREAM_SHA=$(git rev-parse HEAD)

# Prepared-model packs are rebuilt in CI and shipped as one release archive;
# reports and papers are linked from the archived upstream repository.
$FILTER_REPO --force \
    --path-glob '*.pyamplicol-model' --invert-paths \
    --path-glob '*.pdf' --invert-paths \
    --to-subdirectory-filter pyamplicol

git -C "$MG_ROOT" fetch --quiet --no-tags "$WORK/pyamplicol" "+HEAD:refs/heads/pyamplicol-import"
echo "Imported $UPSTREAM_URL $REF at $UPSTREAM_SHA into branch pyamplicol-import"
