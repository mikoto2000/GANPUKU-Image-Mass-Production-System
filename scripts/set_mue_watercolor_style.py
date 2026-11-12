"""prompts.csv の画風を Mue の助言に合わせて更新する（淡い水彩アニメ調）。

Mue の助言の要点:
- SDXL アニメ系は文章より Danbooru 風タグ列挙が効く
- CFG を上げすぎると線が太く色が濃くなり、この淡い絵柄が死ぬ（4.5〜6）
- ① スタイル固定部は全行で 1 文字も変えない。② キャラ部だけ差し替える
- Sampler は DPM++ 2M Karras、Steps 28〜35

このスクリプトの動作:
- positive: 先頭の画風ブロックを STYLE_TAGS に置換し、1girl 以降はそのまま残す
- negative: 全行を NEG_TAGS に統一（watercolor をネガティブから除去）
- sampler / scheduler: dpmpp_2m / karras に統一
- cfg / steps / 解像度は変更しない（既に推奨範囲内）
- 元ファイルは prompts.csv.bak6 に退避
- 冪等: watercolor (medium) が既にあればスキップ

使い方:
    .\\venv\\Scripts\\python.exe scripts\\set_mue_watercolor_style.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BACKUP_PATH = ROOT / "prompts" / "prompts.csv.bak6"

# ① スタイル固定部（全行で完全に同一。ここは絶対に変えない）
STYLE_TAGS = [
    "masterpiece",
    "best quality",
    "amazing quality",
    "very aesthetic",
    "absurdres",
    "illustration",
    "watercolor (medium)",
    "traditional media",
    "pale color",
    "pastel colors",
    "soft lighting",
    "soft shading",
    "thin lineart",
    "detailed hair",
    "anime style",
]

# ネガティブ固定ブロック（全行で同一）
# 旧版の watercolor / painterly / brush strokes は水彩と衝突するため除去
NEG_TAGS = [
    "bad quality",
    "worst quality",
    "worst detail",
    "sketch",
    "censor",
    "jpeg artifacts",
    "signature",
    "watermark",
    "username",
    "blurry",
    "lowres",
    "bad anatomy",
    "bad hands",
    "extra digits",
    "fewer digits",
    "missing fingers",
    "expressionless",
    "blank stare",
    "dead eyes",
    "emotionless",
    "serious",
    "frown",
    "scowl",
    "straight-on",
    "arms at sides",
    "stiff pose",
    "t-pose",
    "photorealistic",
    "3d",
    "cgi",
    "render",
    "photo",
    "oil painting",
    "impasto",
    "gouache",
    "acrylic (medium)",
    "thick lineart",
    "bold outlines",
    "heavy shading",
    "dark colors",
    "saturated colors",
    "high contrast",
    "flat color",
    "flat shading",
    "simple shading",
    "rough lineart",
    "messy lineart",
]

MARKER = "watercolor (medium)"
SAMPLER = "dpmpp_2m"
SCHEDULER = "karras"


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def join_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def rebuild_positive(text: str) -> tuple[str, bool]:
    """画風ブロックを STYLE_TAGS に置換し、1girl 以降を維持する。

    戻り値: (新しい positive, 1girl が見つかったか)
    """
    tags = split_tags(text)
    lower = [t.lower() for t in tags]

    idx = None
    for i, t in enumerate(lower):
        if t == "1girl" or t.startswith("1girl"):
            idx = i
            break

    if idx is None:
        return text, False

    character = tags[idx:]
    return join_tags(STYLE_TAGS + character), True


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
    skipped = 0
    no_1girl: list[str] = []

    for row in rows:
        pos = (row.get("positive") or "").strip()
        if not pos:
            continue

        if MARKER in [t.lower() for t in split_tags(pos)]:
            skipped += 1
            continue

        new_pos, found = rebuild_positive(pos)
        if not found:
            no_1girl.append((row.get("id") or "").strip())
            continue

        row["positive"] = new_pos
        row["negative"] = join_tags(NEG_TAGS)
        row["sampler"] = SAMPLER
        row["scheduler"] = SCHEDULER
        changed += 1

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator=lineterm)
        writer.writeheader()
        writer.writerows(rows)

    print("更新 " + str(changed) + " 行 / 全 " + str(len(rows)) + " 行")
    print("スキップ（既に適用済み）: " + str(skipped) + " 行")
    print("1girl が見つからず未更新: " + str(len(no_1girl)) + " 行")
    if no_1girl:
        print("  対象 id: " + ", ".join(no_1girl))
    print("改行コード: " + repr(lineterm))
    print("")
    print("--- ① スタイル固定部 ---")
    print(join_tags(STYLE_TAGS))
    print("")
    print("--- ネガティブ ---")
    print(join_tags(NEG_TAGS))

    for sample_id in ("001", "031", "101"):
        for row in rows:
            if (row.get("id") or "").strip() == sample_id:
                print("")
                print("--- id=" + sample_id + " ---")
                print("positive: " + str(row.get("positive")))
                print("sampler/scheduler: " + str(row.get("sampler")) + " / " + str(row.get("scheduler")))
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
