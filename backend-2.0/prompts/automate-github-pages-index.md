# Automate the GitHub Pages lesson index

## Feature context

The root `index.html` contains 25 hard-coded lesson links. New HTML lessons under `Claude/week*/day*/` would not appear until that file is manually edited. Add a deterministic standard-library generator and a GitHub Actions workflow so a pushed lesson automatically refreshes the shareable homepage.

## Exact files and changes

### 1. `index.html`

At current line 29, immediately after `<main>`, add:

```html
    <!-- LESSONS:START -->
```

At current line 63, immediately before `</main>`, add:

```html
    <!-- LESSONS:END -->
```

The existing lesson sections between the markers are the generator's initial output. No other HTML or CSS changes are allowed.

### 2. Create `scripts/generate_index.py`

Create the file at line 1 with this implementation:

```python
#!/usr/bin/env python3
"""Regenerate the lesson cards in the GitHub Pages index."""

from __future__ import annotations

import html
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
START = "    <!-- LESSONS:START -->"
END = "    <!-- LESSONS:END -->"
WEEK_NAMES = {
    1: "Embeddings",
    2: "Vector Databases",
    3: "Retrieval-Augmented Generation",
    4: "Agent Architecture",
}


class TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.parts.append(data)


def lesson_details(path: Path) -> tuple[int, tuple[int, int], str, str, str]:
    relative = path.relative_to(ROOT).as_posix()
    week_match = re.fullmatch(r"week(\d+)", path.parent.parent.name)
    day_match = re.fullmatch(r"day(\d+)(?:-(\d+))?", path.parent.name)
    lesson_match = re.match(r"(\d+)([a-z]?)", path.stem, re.IGNORECASE)
    if not (week_match and day_match and lesson_match):
        raise ValueError(f"Unexpected lesson path: {relative}")

    parser = TitleParser()
    parser.feed(path.read_text(encoding="utf-8"))
    raw_title = html.unescape("".join(parser.parts).strip())
    if not raw_title:
        raise ValueError(f"Missing <title> in {relative}")

    label = f"{lesson_match.group(1)}{lesson_match.group(2).upper()}"
    title = re.sub(rf"^{re.escape(label)}\s*[—–-]\s*", "", raw_title).strip()
    day_sort = (int(day_match.group(1)), int(day_match.group(2) or day_match.group(1)))
    return int(week_match.group(1)), day_sort, label, title, relative


def render() -> str:
    grouped: dict[int, list[tuple[tuple[int, int], str, str, str]]] = defaultdict(list)
    for path in ROOT.glob("Claude/week*/day*/*.html"):
        week, day_sort, label, title, relative = lesson_details(path)
        grouped[week].append((day_sort, label, title, relative))

    sections: list[str] = []
    for week in sorted(grouped):
        suffix = f" · {WEEK_NAMES[week]}" if week in WEEK_NAMES else ""
        lines = [f'    <section><h2>Week {week}{suffix}</h2><div class="lessons">']
        for _, label, title, relative in sorted(grouped[week]):
            lines.append(
                f'      <a href="{html.escape(relative, quote=True)}">'
                f'<span class="number">Lesson {html.escape(label)}</span>'
                f'<span class="title">{html.escape(title)}</span></a>'
            )
        lines.append("    </div></section>")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def main() -> None:
    current = INDEX.read_text(encoding="utf-8")
    if current.count(START) != 1 or current.count(END) != 1:
        raise ValueError("index.html must contain one start marker and one end marker")
    before, remainder = current.split(START, 1)
    _, after = remainder.split(END, 1)
    INDEX.write_text(f"{before}{START}\n{render()}\n{END}{after}", encoding="utf-8")


if __name__ == "__main__":
    main()
```

### 3. Create `.github/workflows/update-lesson-index.yml`

Create the file at line 1 with:

```yaml
name: Update lesson index

on:
  push:
    branches: [main]
    paths:
      - "Claude/**/*.html"
      - "scripts/generate_index.py"
  workflow_dispatch:

permissions:
  contents: write

jobs:
  update-index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Generate index
        run: python3 scripts/generate_index.py
      - name: Commit updated index
        run: |
          if git diff --quiet -- index.html; then
            exit 0
          fi
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add index.html
          git commit -m "docs: update lesson index"
          git push
```

## Verification

1. Run `python3 scripts/generate_index.py` twice and confirm the second run produces no diff.
2. Parse `index.html`; confirm every generated `href` exists and all 25 lesson files appear exactly once.
3. Copy the repository to a temporary directory, add a `Claude/week5/day1/0025-test.html` fixture with a `<title>`, run the generator, and confirm Week 5 and Lesson 0025 appear without changing the source repository.
4. Run `git diff --check`.
5. Confirm the workflow YAML has `contents: write` and only triggers automatically for lesson HTML or generator changes on `main`.

## Constraints

- Use only the Python standard library.
- Keep relative URLs so each lesson remains directly shareable under the GitHub Pages project URL.
- Do not modify lesson files.
- A repository setting or organization policy can still prevent GitHub Actions from pushing; if that occurs, enable read/write workflow permissions in repository Actions settings.
