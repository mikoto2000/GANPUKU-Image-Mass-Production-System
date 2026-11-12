"""v2 反映後の微修正（v3）。

1. 目の状態・形容タグ（closed / glossy / shiny など）を色として誤認しない
2. 色付き虹彩タグを全撤去し、実際の目の色タグの直後にのみ再挿入
3. 服の柄（floral print など）がある行は、対応するネガティブを外す
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"

# 色ではない目の状態・形容タグ
EYE_STATE = {
    "closed", "half", "wide", "narrow", "empty", "glowing", "sparkling",
    "sharp", "round", "large", "small", "big", "one", "both", "open",
    "detailed", "beautiful", "pretty", "tareme", "tsurime",
    "glossy", "shiny", "bright", "clear", "deep", "pale", "dark",
    "light", "vivid", "soft", "gentle", "intense", "sleepy", "teary",
}

EYE_RE = re.compile(r"^(?P<color>[a-z][a-z \-]*?) eyes$", re.IGNORECASE)
# 色付き虹彩タグ（generic な "detailed gradient iris" は対象外）
COLORED_IRIS_RE = re.compile(r"^detailed [a-z][a-z \-]* gradient iris$", re.IGNORECASE)

PATTERN_MAP = {
    "floral print": "floral pattern",
    "polka dot": "polka dot",
    "striped": "striped",
    "plaid": "plaid",
    "checkered": "checkered",
}


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def dedupe(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def eye_color(tags: list[str]) -> tuple[int, str] | None:
    """実際の目の色タグの位置と色名を返す。状態タグは除外。"""
    for i, t in enumerate(tags):
        m = EYE_RE.match(t.strip())
        if not m:
            continue
        color = m.group("color").strip()
        if color.lower() in EYE_STATE:
            continue
        return i, color
    return None


def fix_positive(pos: str) -> str:
    tags = split_tags(pos)
    # 色付き虹彩タグを全撤去（位置がずれたものも含む）
    tags = [t for t in tags if not COLORED_IRIS_RE.match(t)]

    found = eye_color(tags)
    if found:
        idx, color = found
        tags.insert(idx + 1, "detailed " + color + " gradient iris")

    return ", ".join(dedupe(tags))


def fix_negative(neg: str, pos: str) -> str:
    tags = split_tags(neg)
    low = pos.lower()
    drop: set[str] = set()
    for pattern, neg_tag in PATTERN_MAP.items():
        if pattern in low:
            drop.add(neg_tag.lower())
    if drop:
        tags = [t for t in tags if t.lower() not in drop]
    return ", ".join(dedupe(tags))


def main() -> int:
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames
        rows = list(reader)

    changed = 0
    for r in rows:
        pos = fix_positive(r["positive"])
        neg = fix_negative(r["negative"], pos)
        if pos != r["positive"] or neg != r["negative"]:
            changed += 1
            print("fix " + r["id"])
        r["positive"] = pos
        r["negative"] = neg

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print("修正=" + str(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
