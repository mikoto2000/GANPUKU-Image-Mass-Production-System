"""prompts.csv の画風更新結果を検証する。"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"

STYLE = (
    "masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
    "illustration, watercolor (medium), traditional media, pale color, "
    "pastel colors, soft lighting, soft shading, thin lineart, detailed hair, anime style"
)

OLD_TAGS = [
    "sharp lineart",
    "crisp lineart",
    "detailed lineart",
    "gradient shading",
    "glossy hair",
    "vibrant colors",
    "painterly",
    "brush strokes",
    "digital painting",
    "refined lineart",
    "rich colors",
    "cel shading",
    "anime coloring",
    "clean lineart",
]


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def main() -> int:
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    print("行数: " + str(len(rows)))
    print("列数: " + str(len(fieldnames)))
    print("")

    style_ok = 0
    style_bad: list[str] = []
    white_ok = 0
    white_bad: list[str] = []
    sampler_ok = 0
    sampler_bad: list[str] = []
    neg_ok = 0
    neg_bad: list[str] = []
    old_hits: Counter[str] = Counter()
    cfg_values: Counter[str] = Counter()
    steps_values: Counter[str] = Counter()
    size_values: Counter[str] = Counter()
    char_dup: list[str] = []

    for row in rows:
        rid = (row.get("id") or "").strip()
        pos = (row.get("positive") or "").strip()
        neg = (row.get("negative") or "").strip()
        tags = split_tags(pos)
        lower = [t.lower() for t in tags]

        # ① スタイル固定部が完全一致か
        if pos.startswith(STYLE + ","):
            style_ok += 1
        else:
            style_bad.append(rid)

        # 白背景
        if "white background" in lower:
            white_ok += 1
        else:
            white_bad.append(rid)

        # sampler / scheduler
        if (row.get("sampler") or "").strip() == "dpmpp_2m" and (row.get("scheduler") or "").strip() == "karras":
            sampler_ok += 1
        else:
            sampler_bad.append(rid)

        # ネガティブに watercolor が入っていないこと
        neg_lower = [t.lower() for t in split_tags(neg)]
        if "watercolor" not in neg_lower and "watercolor (medium)" not in neg_lower:
            neg_ok += 1
        else:
            neg_bad.append(rid)

        # 旧タグ残存
        for t in OLD_TAGS:
            if t in lower:
                old_hits[t] += 1

        # 1girl が 1 回だけか
        if lower.count("1girl") != 1:
            char_dup.append(rid + "(" + str(lower.count("1girl")) + ")")

        cfg_values[(row.get("cfg") or "").strip()] += 1
        steps_values[(row.get("steps") or "").strip()] += 1
        size_values[((row.get("width") or "").strip() + "x" + (row.get("height") or "").strip())] += 1

    print("[① スタイル固定部 完全一致] " + str(style_ok) + " / " + str(len(rows)))
    if style_bad:
        print("  不一致: " + ", ".join(style_bad[:20]))
    print("[white background] " + str(white_ok) + " / " + str(len(rows)))
    if white_bad:
        print("  欠落: " + ", ".join(white_bad[:20]))
    print("[sampler=dpmpp_2m / scheduler=karras] " + str(sampler_ok) + " / " + str(len(rows)))
    if sampler_bad:
        print("  不一致: " + ", ".join(sampler_bad[:20]))
    print("[negative に watercolor なし] " + str(neg_ok) + " / " + str(len(rows)))
    if neg_bad:
        print("  混入: " + ", ".join(neg_bad[:20]))
    print("[1girl が 1 回] " + str(len(rows) - len(char_dup)) + " / " + str(len(rows)))
    if char_dup:
        print("  異常: " + ", ".join(char_dup[:20]))
    print("")
    print("[旧タグ残存]")
    if old_hits:
        for t, c in old_hits.items():
            print("  " + t + ": " + str(c))
    else:
        print("  なし")
    print("")
    print("[cfg] " + str(dict(cfg_values)))
    print("[steps] " + str(dict(steps_values)))
    print("[size] " + str(dict(size_values)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
