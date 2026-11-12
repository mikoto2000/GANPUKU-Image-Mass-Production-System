"""prompts.csv の画風を参照画像 HWAC6ZqasAAkFhT.jpg のタッチへ差し替える。

参照画像の特徴:
- 細くシャープな線画（crisp / sharp lineart）
- アニメ調だが平坦ではなく、グラデーションのある丁寧な陰影
- 髪の毛の細かい描写とツヤ
- 白背景

- positive: painterly 系ブロックを sharp lineart 系へ置換
- negative: painterly / brush strokes を抑え、flat color 系は維持
- 背景: simple background, white background を維持
- 元ファイルは prompts.csv.bak5 に退避（既存があれば保持）
- 2 回以上実行しても二重付与しない

使い方:
    .\\venv\\Scripts\\python.exe scripts\\set_reference_style.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BACKUP_PATH = ROOT / "prompts" / "prompts.csv.bak5"

# 旧: イラストレーターの塗り（柔らかい陰影＋筆致）
OLD_STYLE_TAGS = [
    "illustration",
    "painterly",
    "digital painting",
    "soft shading",
    "detailed shading",
    "brush strokes",
    "refined lineart",
    "rich colors",
]

# 新: 参照画像のタッチ（シャープな線画＋グラデーション陰影＋髪の描写）
NEW_STYLE_TAGS = [
    "illustration",
    "sharp lineart",
    "crisp lineart",
    "detailed lineart",
    "soft shading",
    "gradient shading",
    "detailed shading",
    "detailed hair",
    "glossy hair",
    "vibrant colors",
]

# 旧ネガティブ（cel shading / anime coloring を抑えていた）
OLD_NEG_TAGS = [
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

# 新ネガティブ（写真化と筆致を抑える。アニメ調は残す）
NEW_NEG_TAGS = [
    "photorealistic",
    "3d",
    "cgi",
    "render",
    "photo",
    "painterly",
    "brush strokes",
    "impasto",
    "oil painting",
    "watercolor",
    "sketch",
    "rough lineart",
    "messy lineart",
    "flat color",
    "flat shading",
    "simple shading",
]

BG_DEFAULT = "simple background, white background"
MARKER = "sharp lineart"


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def join_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def apply_style(tags: list[str]) -> list[str]:
    lower = [t.lower() for t in tags]
    if MARKER in lower:
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
    if "painterly" in lower:
        return tags

    tags = [t for t in tags if t.lower() not in OLD_NEG_TAGS]
    for t in NEW_NEG_TAGS:
        if t.lower() not in [x.lower() for x in tags]:
            tags.append(t)
    return tags


def normalize_background(tags: list[str]) -> list[str]:
    tags = [t for t in tags if not t.lower().endswith("background")]
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

        tags = apply_style(split_tags(pos))
        tags = normalize_background(tags)
        row["positive"] = join_tags(tags)

        neg = (row.get("negative") or "").strip()
        row["negative"] = join_tags(apply_negative(split_tags(neg)))

        changed += 1

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator=lineterm)
        writer.writeheader()
        writer.writerows(rows)

    print("更新 " + str(changed) + " 行 / 全 " + str(len(rows)) + " 行")
    print("改行コード: " + repr(lineterm))

    for sample_id in ("001", "031", "101"):
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
