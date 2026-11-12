"""アニメ塗り（セル塗り）を外し、参照画像の淡いブラシ塗りへ寄せる。

- positive: スタイル固定部から soft cel shading / finished anime illustration /
  modern anime style / light novel cover を撤去
- positive: airbrush (medium) + smooth gradient shading + soft shading で
  セル塗りではなく連続階調の塗りを指定
- negative: cel shading / flat shading / hard shading / anime coloring /
  sharp shadows / hard edges を追加してセル塗りを抑える
- キャラ部（1girl 以降）は維持
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BAK_PATH = ROOT / "prompts" / "prompts.csv.bak9"

STYLE = [
    "masterpiece", "best quality", "(full color:1.2)",
    "illustration", "digital painting", "airbrush (medium)",
    "smooth gradient shading", "soft shading", "detailed shading",
    "pastel color palette", "pale color",
    "bright high-key lighting", "cool soft shadows",
    "translucent skin", "pink blush", "glossy eyes", "detailed gradient iris",
    "(simple background:1.2)", "plain white background",
    "clean composition", "negative space", "minimal decoration",
]

# セル塗り・アニメ塗りを抑えるネガティブ
CEL_NEG = [
    "cel shading", "flat shading", "hard shading", "anime coloring",
    "sharp shadows", "hard edges", "posterized", "color banding",
]

# 旧スタイル固定部に含まれていたタグ（撤去対象）
OLD_STYLE = [
    "finished anime illustration", "modern anime style", "light novel cover",
    "soft digital painting", "soft cel shading",
]


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


def build_positive(old: str) -> str:
    tags = split_tags(old)
    start = next((i for i, t in enumerate(tags) if t.lower().startswith("1girl")), None)
    char = tags[start:] if start is not None else []
    return ", ".join(dedupe(STYLE + char))


def build_negative(old: str) -> str:
    tags = split_tags(old)
    # 既存の cel 系があれば除去してから追加（重複防止）
    tags = [t for t in tags if t.lower() not in {c.lower() for c in CEL_NEG}]
    return ", ".join(dedupe(tags + CEL_NEG))


def main() -> int:
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields:
        print("ヘッダがない")
        return 2

    if not BAK_PATH.exists():
        shutil.copy2(CSV_PATH, BAK_PATH)

    changed = 0
    for r in rows:
        pos = build_positive(r["positive"])
        neg = build_negative(r["negative"])
        if pos != r["positive"] or neg != r["negative"]:
            changed += 1
        r["positive"] = pos
        r["negative"] = neg

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print("行数=" + str(len(rows)) + " 更新=" + str(changed))
    print("backup=" + str(BAK_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
