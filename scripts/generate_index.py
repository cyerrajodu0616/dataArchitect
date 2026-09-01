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
