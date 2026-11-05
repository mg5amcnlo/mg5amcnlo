#!/usr/bin/env bash
# Publish (or remove) one version's built docs on the gh-pages branch, then
# regenerate the site-level metadata (versions.json, the root redirect,
# stable/) from what is actually on the branch afterwards.
#
# Usage: deploy_docs.sh <version-dir> <publish|remove>
#
# Required env: GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_WORKSPACE, GITHUB_SHA.
# Optional env: DOCS_BRANCH (default gh-pages), DOCS_CNAME (custom domain;
# omit to not write a CNAME file yet -- see the staged domain rollout).
set -euo pipefail

VERSION_DIR="${1:?usage: deploy_docs.sh <version-dir> <publish|remove>}"
ACTION="${2:?usage: deploy_docs.sh <version-dir> <publish|remove>}"
case "$ACTION" in publish|remove) ;; *) echo "action must be publish or remove, got: $ACTION" >&2; exit 2 ;; esac

BRANCH="${DOCS_BRANCH:-gh-pages}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="${RUNNER_TEMP:-/tmp}/gh-pages-site"
SRC="$GITHUB_WORKSPACE/docs/build/html"
REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

if [ "$ACTION" = publish ] && [ ! -d "$SRC" ]; then
  echo "::error::$SRC does not exist -- build the docs before deploying" >&2
  exit 1
fi

rm -rf "$SITE"
git config --global user.name  "github-actions[bot]"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"

if git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
  git clone --quiet --depth 1 --branch "$BRANCH" --single-branch "$REMOTE" "$SITE"
else
  # gh-pages does not exist yet: bootstrap it as an orphan branch off whatever
  # the default branch's clone gives us, then clear its tree.
  echo "branch $BRANCH does not exist yet; bootstrapping it"
  git clone --quiet "$REMOTE" "$SITE"
  git -C "$SITE" checkout --quiet --orphan "$BRANCH"
  git -C "$SITE" rm -rqf . >/dev/null 2>&1 || true
fi

stage() {
  # rm-then-copy: a page removed upstream must not linger from a stale copy.
  rm -rf "${SITE:?}/$VERSION_DIR"
  if [ "$ACTION" = publish ]; then
    mkdir -p "$SITE/$VERSION_DIR"
    cp -a "$SRC/." "$SITE/$VERSION_DIR/"
  fi
  [ -n "${DOCS_CNAME:-}" ] && printf '%s\n' "$DOCS_CNAME" > "$SITE/CNAME"
  python3 "$SCRIPT_DIR/gen_site_index.py" "$SITE"
  git -C "$SITE" add -A
}

stage
if git -C "$SITE" diff --cached --quiet; then
  echo "nothing to do"
  exit 0
fi
git -C "$SITE" commit --quiet -m "docs: ${ACTION} ${VERSION_DIR} (${GITHUB_SHA:0:7})"

# Race handling: `concurrency:` in the workflow serialises our own runs, but a
# manual push (or a second workflow run kicked off outside our control) can
# still land in between. Do NOT rebase or amend onto their tip -- that would
# republish OUR whole tree and could delete a version dir they just added.
# Instead take THEIR tree as the new base and re-apply only our own directory
# and the regenerated metadata on top of it.
for delay in 0 5 15 30 60; do
  if [ "$delay" != 0 ]; then
    echo "push rejected, retrying in ${delay}s..."
    sleep "$delay"
  fi
  if git -C "$SITE" push --quiet origin "HEAD:refs/heads/$BRANCH"; then
    echo "pushed $ACTION of $VERSION_DIR to $BRANCH"
    exit 0
  fi
  # Explicit ref -> FETCH_HEAD, always updated regardless of the single-branch
  # clone's configured refspec (unlike origin/$BRANCH, which may not be).
  git -C "$SITE" fetch --quiet --depth 1 origin "$BRANCH"
  git -C "$SITE" reset --hard --quiet FETCH_HEAD
  stage
  if git -C "$SITE" diff --cached --quiet; then
    echo "already up to date after rebase"
    exit 0
  fi
  git -C "$SITE" commit --quiet -m "docs: ${ACTION} ${VERSION_DIR} (${GITHUB_SHA:0:7})"
done

echo "::error::could not push to $BRANCH after 5 attempts" >&2
exit 1
