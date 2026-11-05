#!/usr/bin/env python3
"""Regenerate the gh-pages site's cross-version metadata from the version
directories actually present, so the site is self-consistent by construction
and adding/removing a version needs no hand-maintained file.

Run inside a checkout of the gh-pages branch, after the deploy script has
added or removed the directory for the version being published:

    gen_site_index.py <site-dir>

Writes/updates, all at the site root:
  - stable/          a byte-identical copy of the newest non-prerelease
                      release (git dedupes identical blobs across commits, so
                      this does not meaningfully grow the branch)
  - versions.json     read at runtime by docs/source/_templates/sidebar/
                      version-switcher.html
  - index.html        meta-refresh to ./stable/ (or ./latest/ before the
                      first release exists)
  - 404.html          rewrites Read the Docs' old /en/<version>/... paths
"""

import json
import re
import shutil
import sys
from pathlib import Path

SEMVER = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-(.+))?$")


def sort_key(tag: str):
    """Descending version order, with a prerelease sorting just before the
    release it precedes (so v0.2.0-alpha < v0.2.0, matching intuition even
    though we don't currently publish prereleases)."""
    major, minor, patch, pre = SEMVER.match(tag).groups()
    return (int(major), int(minor), int(patch), (0, pre) if pre else (1, ""))


def discover(site: Path):
    dirs = {
        p.name
        for p in site.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    }
    tags = sorted((d for d in dirs if SEMVER.match(d)), key=sort_key, reverse=True)
    stable_tags = [t for t in tags if not SEMVER.match(t).group(4)]
    return dirs, tags, stable_tags


def sync_stable(site: Path, stable_tags: list[str]) -> str | None:
    """Keep stable/ a byte-identical copy of the newest release, so it gets a
    permanent deep-linkable URL. Returns the tag it now mirrors, or None if
    there is no release (e.g. only "latest" exists, or the last release was
    just removed) -- in which case any leftover stable/ is deleted too."""
    stable_dir = site / "stable"
    if stable_dir.is_symlink() or stable_dir.exists():
        shutil.rmtree(stable_dir)
    if not stable_tags:
        return None
    newest = stable_tags[0]
    shutil.copytree(site / newest, stable_dir)
    return newest


def write_versions_json(
    site: Path, dirs: set[str], tags: list[str], default: str, newest_stable: str | None
):
    entries = []
    if "latest" in dirs:
        entries.append({"name": "latest", "title": "latest (main)", "url": "latest/"})
    for tag in tags:
        prerelease = bool(SEMVER.match(tag).group(4))
        title = f"{tag} (stable)" if tag == newest_stable else tag
        entries.append(
            {"name": tag, "title": title, "url": f"{tag}/", "prerelease": prerelease}
        )
    (site / "versions.json").write_text(
        json.dumps({"default": default, "versions": entries}, indent=2) + "\n"
    )


def write_redirects(site: Path, default: str):
    (site / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MadGraph7 documentation</title>
<link rel="canonical" href="https://docs.madgraph.org/{default}/">
<meta http-equiv="refresh" content="0; url=./{default}/">
<script>location.replace("./{default}/" + location.hash);</script>
</head>
<body><p>Redirecting to <a href="./{default}/">{default}</a>.</p></body>
</html>
"""
    )

    # Read the Docs served /en/<latest|stable|vX.Y.Z>/<page>; keep those links
    # alive after the domain moves by rewriting them client-side and falling
    # through to the new layout.
    (site / "404.html").write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>Not found</title>
<script>
var m = location.pathname.match(/^\\/(?:en|[a-z]{{2}})\\/(latest|stable|v[\\d.]+(?:-[^/]+)?)\\/(.*)$/);
if (m) {{
  location.replace("/" + m[1] + "/" + m[2] + location.hash);
}} else {{
  location.replace("/{default}/" + location.hash);
}}
</script>
<p>Redirecting&hellip;</p>
"""
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <site-dir>", file=sys.stderr)
        return 2
    site = Path(argv[1])

    dirs, tags, stable_tags = discover(site)
    newest_stable = sync_stable(site, stable_tags)
    if newest_stable:
        dirs.add("stable")
        default = "stable"
    elif "latest" in dirs:
        default = "latest"
    elif tags:
        default = tags[0]
    else:
        print("gen_site_index: no version directories found; nothing to index")
        return 0

    write_versions_json(site, dirs, tags, default, newest_stable)
    write_redirects(site, default)
    (site / ".nojekyll").touch()

    print(
        f"gen_site_index: default={default} "
        f"stable->{newest_stable or '(none)'} "
        f"versions={tags + (['latest'] if 'latest' in dirs else [])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
