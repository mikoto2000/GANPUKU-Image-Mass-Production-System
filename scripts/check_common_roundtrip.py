"""check_common_roundtrip.py — 共通ファイル分離前後のプロンプト一致を検証する。

バックアップ CSV（分離前）と、現在の CSV + 共通ファイル（分離後）を比較し、
全行で positive / negative が完全一致するかを確認する。

使い方:
    .\\venv\\Scripts\\python.exe scripts\\check_common_roundtrip.py
    .\\venv\\Scripts\\python.exe scripts\\check_common_roundtrip.py --backup prompts\\prompts.csv.bak_20261106_114730
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
POS_PATH = ROOT / "prompts" / "common_positive.txt"
NEG_PATH = ROOT / "prompts" / "common_negative.txt"


def load_common(path: Path) -> str:
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, strict=True))


def join_prompt(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="共通ファイル分離前後の一致を検証する")
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--backup", required=True, help="分離前のバックアップ CSV")
    ap.add_argument("--common-positive", default=str(POS_PATH))
    ap.add_argument("--common-negative", default=str(NEG_PATH))
    args = ap.parse_args(argv)

    common_pos = load_common(Path(args.common_positive))
    common_neg = load_common(Path(args.common_negative))

    before = read_rows(Path(args.backup))
    after = read_rows(Path(args.csv))

    print("分離前: " + str(len(before)) + " 行 / 分離後: " + str(len(after)) + " 行")
    if len(before) != len(after):
        print("行数が一致しない")
        return 1

    mismatches: list[str] = []
    for b, a in zip(before, after):
        rid = (b.get("id") or "").strip()
        if rid != (a.get("id") or "").strip():
            mismatches.append(rid + ": id 不一致")
            continue
        pos_before = (b.get("positive") or "").strip()
        pos_after = join_prompt(common_pos, (a.get("positive") or "").strip())
        neg_before = (b.get("negative") or "").strip()
        neg_after = join_prompt(common_neg, (a.get("negative") or "").strip())
        if pos_before != pos_after:
            mismatches.append(rid + ": positive 不一致")
        if neg_before != neg_after:
            mismatches.append(rid + ": negative 不一致")

    if mismatches:
        print("不一致 " + str(len(mismatches)) + " 件")
        for m in mismatches[:20]:
            print("  " + m)
        return 1

    print("全 " + str(len(before)) + " 行で positive / negative が完全一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
