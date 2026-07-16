"""Flatten the generated page's local video refs for GitHub Pages.

site/index.html is written for ``file://`` inside a checkout, so it points at
videos with repo-relative paths (``../cube_x.mp4``). Published at the gh-pages
root those escape above /sim2sim/ and 404, so here each local ref is resolved
against the site dir, copied next to index.html, and rewritten to a flat name.
Refs whose file is absent (the *.mp4 are gitignored) and absolute video_url
refs are both left untouched.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

SITE_HTML = Path("examples/cube_drop/site/index.html")


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    html = SITE_HTML.read_text()
    site_dir = SITE_HTML.parent

    copied, missing = [], []

    def repl(m: re.Match) -> str:
        ref = m.group(1)
        if ref.startswith(("http://", "https://", "//")):
            return m.group(0)
        target = (site_dir / ref).resolve()
        if not target.is_file():
            missing.append(ref)
            return m.group(0)
        shutil.copy2(target, out / target.name)
        copied.append(target.name)
        return f'src="{target.name}"'

    html = re.sub(r'src="([^"]+\.mp4)"', repl, html)
    (out / "index.html").write_text(html)

    for n in copied:
        print(f"[publish] copied {n}")
    for n in missing:
        print(f"[publish] WARNING: no local file for {n} — card will 404")
    print(f"[publish] {len(copied)} videos, {len(missing)} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
