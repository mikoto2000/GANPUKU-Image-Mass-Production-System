"""水彩の柔らかさを戻しつつ、セピア線画化を防ぐスタイルへ差し替える。

背景:
  bak7 で watercolor (medium) + traditional media + thin lineart を入れたら
  セピアの装飾線画になった。Mue の指摘どおり、watercolor は「紙に描いた
  伝統水彩画」として効きやすく、traditional media と lineart が増幅していた。

方針:
  - positive: watercolor (medium) を弱めの重みで残す。traditional media と
    thin lineart は入れない（セピア化と線画ジャンル化の原因）
  - positive: full color / translucent colors / pastel で淡いフルカラーを明示
  - negative: watercolor painting と traditional media を撤去（直接衝突）
  - negative: sepia / brown tint / monochrome / ink drawing / hatching は維持
    （セピア事故の再発防止）
  - キャラ部（1girl 以降）は維持

使い方:
  python scripts/set_watercolor_style.py            # dry-run（差分表示のみ）
  python scripts/set_watercolor_style.py --apply    # 適用
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BAK_PATH = ROOT / "prompts" / "prompts.csv.bak10"

# 水彩寄りのスタイル固定部（1girl より前に置く）
STYLE = [
    "masterpiece", "best quality", "(full color:1.2)",
    "illustration", "(watercolor (medium):0.9)", "digital painting",
    "translucent colors", "pale color", "pastel color palette",
    "soft gradient shading", "soft shading", "soft lighting",
    "bright high-key lighting", "cool soft shadows",
    "translucent skin", "pink blush", "glossy eyes", "detailed gradient iris",
    "(simple background:1.2)", "plain white background",
    "clean composition", "negative space", "minimal decoration",
]

# 水彩と直接衝突するのでネガティブから外すタグ
NEG_REMOVE = ["watercolor painting", "traditional media"]

# セピア線画事故の再発防止として必ず入れておくタグ
NEG_ENSURE = [
    "sepia", "brown tint", "monochrome", "grayscale",
    "ink drawing", "pen drawing", "hatching", "crosshatching",
    "ornament", "decorative", "detailed background", "scenery",
    "cel shading", "flat shading", "hard shading", "anime coloring",
    "sharp shadows", "hard edges", "posterized", "color banding",
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
    drop = {t.lower() for t in NEG_REMOVE}
    tags = [t for t in tags if t.lower() not in drop]
    return ", ".join(dedupe(tags + NEG_ENSURE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="実際に書き込む")
    args = ap.parse_args()

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields:
        print("ヘッダがない")
        return 2

    changed = 0
    for r in rows:
        pos = build_positive(r["positive"])
        neg = build_negative(r["negative"])
        if pos != r["positive"] or neg != r["negative"]:
            changed += 1
        r["positive"] = pos
        r["negative"] = neg

    print("行数=" + str(len(rows)) + " 変更=" + str(changed))
    print("--- 001 positive ---")
    print(rows[0]["positive"])
    print("--- 001 negative ---")
    print(rows[0]["negative"])

    if not args.apply:
        print("\n[dry-run] 書き込みなし。適用するには --apply を付けてね。")
        return 0

    if not BAK_PATH.exists():
        shutil.copy2(CSV_PATH, BAK_PATH)
        print("backup=" + str(BAK_PATH))

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print("applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
