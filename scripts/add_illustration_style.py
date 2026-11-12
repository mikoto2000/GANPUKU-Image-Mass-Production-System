"""prompts.csv の全行に「イラスト寄り」のスタイルタグを付与する。

- positive: 品質タグの直後にスタイルブロックを挿入
- negative: 写実・3D 寄りを抑えるタグを末尾に追加
- 元ファイルは prompts.csv.bak に退避（既存があれば保持）
- 2 回以上実行しても二重付与しない

使い方:
    .\\venv\\Scripts\\python.exe scripts\\add_illustration_style.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BACKUP_PATH = ROOT / "prompts" / "prompts.csv.bak"

# 品質タグの直後に差し込む画風ブロック。
# Illustrious 系は画風タグの効きが強いので、ここで「アニメ塗り＋線画」を固定する。
QUALITY_PREFIX = "masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
STYLE_BLOCK = "illustration, anime coloring, cel shading, clean lineart, vibrant colors, "

# 写実・3D 方向へ流れるのを抑えるネガティブ。
NEG_ADD = "realistic, photorealistic, 3d, cgi, render, photo, realistic shading"

# 二重付与の判定に使うマーカー。
STYLE_MARKER = "cel shading"


def main() -> int:
    if not CSV_PATH.exists():
        print("見つからない: " + str(CSV_PATH))
        return 2

    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print("バックアップ: " + str(BACKUP_PATH))
    else:
        print("バックアップは既存のため保持: " + str(BACKUP_PATH))

    # 元ファイルの改行コードを引き継ぐ。
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
    skipped = 0
    for row in rows:
        pos = (row.get("positive") or "").strip()
        if not pos:
            continue
        if STYLE_MARKER in pos:
            skipped += 1
            continue

        if pos.startswith(QUALITY_PREFIX):
            row["positive"] = QUALITY_PREFIX + STYLE_BLOCK + pos[len(QUALITY_PREFIX):]
        else:
            row["positive"] = STYLE_BLOCK + pos

        neg = (row.get("negative") or "").strip()
        if not neg:
            row["negative"] = NEG_ADD
        elif "photorealistic" not in neg:
            row["negative"] = neg + ", " + NEG_ADD

        changed += 1

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator=lineterm)
        writer.writeheader()
        writer.writerows(rows)

    print("更新 " + str(changed) + " 行 / 既適用 " + str(skipped) + " 行 / 全 " + str(len(rows)) + " 行")
    print("改行コード: " + repr(lineterm))

    if rows:
        print("")
        print("--- 例: id=" + str(rows[0].get("id")) + " ---")
        print("positive: " + str(rows[0].get("positive")))
        print("negative: " + str(rows[0].get("negative")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
