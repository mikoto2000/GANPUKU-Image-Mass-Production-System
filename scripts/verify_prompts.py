"""verify_prompts.py — prompts.csv の整合性を検証する。

共通プロンプトファイル（common_positive.txt / common_negative.txt）が存在する場合は
自動で読み込み、CSV の行個別プロンプトと合成してから検証する。
--no-common を付けると CSV 単体で検証する。

使い方:
    .\\venv\\Scripts\\python.exe scripts\\verify_prompts.py
    .\\venv\\Scripts\\python.exe scripts\\verify_prompts.py --no-common
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
POS_PATH = ROOT / "prompts" / "common_positive.txt"
NEG_PATH = ROOT / "prompts" / "common_negative.txt"

BAD_BG = {
    "abstract background", "pink background", "dark background",
    "blue background", "green background", "grey background",
}
SCENE = {
    "rain", "umbrella", "wet", "backlighting", "rim light",
    "cinematic lighting", "depth of field", "sunflower",
}


def load_common(path: Path) -> str:
    """共通プロンプトファイルを読む。1 行 1 タグ（カンマ区切りも可）。"""
    if not path.exists():
        return ""
    tags: list[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tag in line.split(","):
                tag = tag.strip()
                if tag:
                    tags.append(tag)
    return ", ".join(tags)


def join_prompt(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="prompts.csv の整合性を検証する")
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--common-positive", default=str(POS_PATH))
    ap.add_argument("--common-negative", default=str(NEG_PATH))
    ap.add_argument("--no-common", action="store_true", help="共通ファイルを使わず CSV 単体で検証")
    args = ap.parse_args(argv)

    common_pos = "" if args.no_common else load_common(Path(args.common_positive))
    common_neg = "" if args.no_common else load_common(Path(args.common_negative))

    with open(Path(args.csv), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        rows = list(reader)

    print("行数: " + str(len(rows)))
    print("列数: " + str(len(reader.fieldnames or [])))
    print("共通ポジティブ: " + (str(len(common_pos.split(","))) + " タグ" if common_pos else "なし"))
    print("共通ネガティブ: " + (str(len(common_neg.split(","))) + " タグ" if common_neg else "なし"))

    no_white = []
    bad_bg = []
    scene = []
    outfits = Counter()

    for row in rows:
        rid = (row.get("id") or "").strip()
        pos = join_prompt(common_pos, row.get("positive") or "")
        tags = [t.strip() for t in pos.split(",") if t.strip()]
        tagset = set(tags)

        if "white background" not in tagset:
            no_white.append(rid)
        hit = tagset & BAD_BG
        if hit:
            bad_bg.append((rid, sorted(hit)))
        hit2 = tagset & SCENE
        if hit2:
            scene.append((rid, sorted(hit2)))

        try:
            num = int(rid)
        except ValueError:
            continue
        if 31 <= num <= 50 or 101 <= num <= 200:
            # 服装タグは BASE の後ろ、背景の前
            try:
                i = tags.index("full body")
                j = tags.index("simple background")
                outfits[", ".join(tags[i + 1 : j])] += 1
            except ValueError:
                outfits["<parse error>"] += 1

    print("white background 欠落: " + str(len(no_white)) + (" -> " + ",".join(no_white) if no_white else ""))
    print("抽象/色背景残存: " + str(len(bad_bg)) + (" -> " + str(bad_bg) if bad_bg else ""))
    print("風景タグ残存: " + str(len(scene)) + (" -> " + str(scene) if scene else ""))
    print("服装行数: " + str(sum(outfits.values())))
    print("服装ユニーク: " + str(len(outfits)))
    print("重複服装: " + str([k for k, v in outfits.items() if v > 1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
