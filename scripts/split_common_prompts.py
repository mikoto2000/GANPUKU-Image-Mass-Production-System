"""split_common_prompts.py — prompts.csv から全行共通のプロンプトを分離する。

positive / negative の両方について、全行で一致する先頭タグ列を抽出し、
prompts\\common_positive.txt / prompts\\common_negative.txt へ書き出す。
--apply を付けると CSV 側から共通部分を取り除く（元 CSV はバックアップ）。

使い方:
    .\\venv\\Scripts\\python.exe scripts\\split_common_prompts.py
    .\\venv\\Scripts\\python.exe scripts\\split_common_prompts.py --apply
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
POS_PATH = ROOT / "prompts" / "common_positive.txt"
NEG_PATH = ROOT / "prompts" / "common_negative.txt"


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def common_prefix(tag_lists: list[list[str]]) -> list[str]:
    """全行で一致する先頭タグ列を返す。"""
    if not tag_lists:
        return []
    n = min(len(t) for t in tag_lists)
    k = 0
    while k < n and len({t[k] for t in tag_lists}) == 1:
        k += 1
    return tag_lists[0][:k]


def write_common(path: Path, tags: list[str], title: str) -> None:
    lines = ["# " + title, "# 1 行 1 タグ。空行と # 行は無視される。"]
    lines.extend(tags)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(label: str, tag_lists: list[list[str]], prefix: list[str]) -> None:
    """共通接頭辞と、それに従わない行数を表示する。"""
    print(label + " 共通先頭タグ: " + str(len(prefix)))
    if prefix:
        print("  " + ", ".join(prefix))
    deviating = sum(1 for t in tag_lists if t[: len(prefix)] != prefix)
    print("  共通接頭辞に従わない行: " + str(deviating))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="prompts.csv から共通プロンプトを分離する")
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--positive-out", default=str(POS_PATH))
    ap.add_argument("--negative-out", default=str(NEG_PATH))
    ap.add_argument("--apply", action="store_true", help="共通部分を CSV から取り除く")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("データ行がない: " + str(csv_path))
        return 2

    pos_lists = [split_tags(r.get("positive")) for r in rows]
    neg_lists = [split_tags(r.get("negative")) for r in rows]

    pos_prefix = common_prefix(pos_lists)
    neg_prefix = common_prefix(neg_lists)

    print("行数: " + str(len(rows)))
    report("positive", pos_lists, pos_prefix)
    report("negative", neg_lists, neg_prefix)

    neg_variants = Counter(", ".join(t) for t in neg_lists)
    print("negative の種類: " + str(len(neg_variants)))
    for text, count in neg_variants.most_common(5):
        print("  " + str(count) + " 行: " + text[:120])

    if not pos_prefix and not neg_prefix:
        print("分離できる共通部分がない")
        return 0

    if not args.apply:
        print("")
        print("--apply を付けると書き出す（この時点では何も変更していない）")
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = csv_path.with_name(csv_path.name + ".bak_" + stamp)
    shutil.copy2(csv_path, backup)

    if pos_prefix:
        write_common(Path(args.positive_out), pos_prefix, "全行共通のポジティブプロンプト")
    if neg_prefix:
        write_common(Path(args.negative_out), neg_prefix, "全行共通のネガティブプロンプト")

    emptied_pos: list[str] = []
    emptied_neg: list[str] = []
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row, pos_tags, neg_tags in zip(rows, pos_lists, neg_lists):
            new_row = dict(row)
            new_row["positive"] = ", ".join(pos_tags[len(pos_prefix):])
            new_row["negative"] = ", ".join(neg_tags[len(neg_prefix):])
            rid = (row.get("id") or "").strip()
            if not new_row["positive"]:
                emptied_pos.append(rid)
            if not new_row["negative"]:
                emptied_neg.append(rid)
            writer.writerow(new_row)

    print("")
    print("バックアップ: " + str(backup))
    if pos_prefix:
        print("共通ポジティブ: " + str(args.positive_out))
    if neg_prefix:
        print("共通ネガティブ: " + str(args.negative_out))
    print("CSV を書き換えた: " + str(csv_path))
    if emptied_pos:
        print("positive が空になった行: " + ", ".join(emptied_pos))
    if emptied_neg:
        print("negative が空になった行: " + str(len(emptied_neg)) + " 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
