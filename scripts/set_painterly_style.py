"""prompts.csv の画風を「アニメ塗り」から「イラストレーターの塗り」へ差し替える。

- positive: cel shading 系ブロックを painterly 系へ置換
- negative: realistic shading を外し、cel shading / flat color を抑える
- 背景: 白 or 抽象シンプルへ正規化（風景タグは除去）
- 元ファイルは prompts.csv.bak2 に退避（既存があれば保持）
- 2 回以上実行しても二重付与しない

使い方:
    .\\venv\\Scripts\\python.exe scripts\\set_painterly_style.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BACKUP_PATH = ROOT / "prompts" / "prompts.csv.bak2"

# 旧: アニメ塗り（セル塗り）ブロック
OLD_STYLE_TAGS = ["illustration", "anime coloring", "cel shading", "clean lineart", "vibrant colors"]
# 新: イラストレーターの塗り（柔らかい陰影＋筆致）
NEW_STYLE_TAGS = [
    "illustration",
    "painterly",
    "digital painting",
    "soft shading",
    "detailed shading",
    "brush strokes",
    "refined lineart",
    "rich colors",
]

# 旧ネガティブ（realistic shading が柔らかい塗りを潰していた）
OLD_NEG_TAGS = ["realistic", "photorealistic", "3d", "cgi", "render", "photo", "realistic shading"]
# 新ネガティブ（写真化は防ぐが、塗りの柔らかさは残す）
NEW_NEG_TAGS = [
    "photorealistic",
    "3d",
    "cgi",
    "render",
    "photo",
    "cel shading",
    "flat color",
    "flat shading",
    "anime coloring",
    "simple shading",
]

# 背景と衝突する風景・照明タグ
SCENERY_TAGS = {
    "outdoors", "indoors", "scenery", "city", "cityscape", "night", "sunset",
    "beach", "ocean", "forest", "sky", "cloudy sky", "cherry blossoms",
    "snow", "ruins", "window", "sunlight", "dappled sunlight", "neon lights",
    "depth of field",
}

# 風景だった行は、色味だけ残した抽象背景へ置き換える
ABSTRACT_BG = {
    "041": ("abstract background, pink background, simple background", "桜・屋外→抽象背景（ピンク）"),
    "042": ("abstract background, dark background, simple background", "夜の街・ネオン→抽象背景（暗色）"),
    "043": ("abstract background, warm lighting, simple background", "室内・窓光→抽象背景（暖色）"),
    "044": ("abstract background, blue background, simple background", "海辺→抽象背景（青）"),
    "045": ("abstract background, green background, simple background", "森・木漏れ日→抽象背景（緑）"),
    "046": ("abstract background, grey background, simple background", "遺跡・ファンタジー→抽象背景（灰）"),
    "047": ("abstract background, white background, simple background", "雪景色→抽象背景（白）"),
}

BG_DEFAULT = "simple background, white background"


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def join_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def apply_style(tags: list[str]) -> list[str]:
    lower = [t.lower() for t in tags]
    if "painterly" in lower:
        return tags

    idx = None
    for i, t in enumerate(tags):
        if t.lower() in OLD_STYLE_TAGS:
            idx = i
            break

    tags = [t for t in tags if t.lower() not in OLD_STYLE_TAGS]

    if idx is None:
        idx = 0
        for i, t in enumerate(tags):
            if t.lower() == "absurdres":
                idx = i + 1
                break

    tags[idx:idx] = NEW_STYLE_TAGS
    return tags


def apply_negative(tags: list[str]) -> list[str]:
    lower = [t.lower() for t in tags]
    if "cel shading" in lower:
        return tags

    tags = [t for t in tags if t.lower() not in OLD_NEG_TAGS]
    for t in NEW_NEG_TAGS:
        if t.lower() not in [x.lower() for x in tags]:
            tags.append(t)
    return tags


def normalize_background(tags: list[str], row_id: str) -> list[str]:
    tags = [t for t in tags if t.lower() not in SCENERY_TAGS]
    tags = [t for t in tags if not t.lower().endswith("background")]

    if row_id in ABSTRACT_BG:
        tags.extend(split_tags(ABSTRACT_BG[row_id][0]))
    else:
        tags.extend(split_tags(BG_DEFAULT))
    return tags


def main() -> int:
    if not CSV_PATH.exists():
        print("見つからない: " + str(CSV_PATH))
        return 2

    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print("バックアップ: " + str(BACKUP_PATH))
    else:
        print("バックアップは既存のため保持: " + str(BACKUP_PATH))

    head = CSV_PATH.read_bytes()[:8192]
    lineterm = "\r\n" if b"\r\n" in head else "\n"

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("ヘッダ行がない: " + str(CSV_PATH))
            return 2
        rows = list(reader)

    changed = 0
    for row in rows:
        pos = (row.get("positive") or "").strip()
        if not pos:
            continue

        row_id = (row.get("id") or "").strip()
        tags = apply_style(split_tags(pos))
        tags = normalize_background(tags, row_id)
        row["positive"] = join_tags(tags)

        if row_id in ABSTRACT_BG:
            row["note"] = ABSTRACT_BG[row_id][1]

        neg = (row.get("negative") or "").strip()
        row["negative"] = join_tags(apply_negative(split_tags(neg)))

        changed += 1

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator=lineterm)
        writer.writeheader()
        writer.writerows(rows)

    print("更新 " + str(changed) + " 行 / 全 " + str(len(rows)) + " 行")
    print("改行コード: " + repr(lineterm))

    for sample_id in ("001", "041", "042"):
        for row in rows:
            if (row.get("id") or "").strip() == sample_id:
                print("")
                print("--- id=" + sample_id + " ---")
                print("positive: " + str(row.get("positive")))
                print("negative: " + str(row.get("negative")))
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
