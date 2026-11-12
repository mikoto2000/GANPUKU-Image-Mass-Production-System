# -*- coding: utf-8 -*-
"""prompts_v7_200.csv の行を「大人判定 / 未成年判定」に分類し、タグ統計を出す。
本文（プロンプト文字列）は出力しない。統計と行番号のみ。
"""
import csv
from collections import Counter

PATH = "prompts/prompts_v7_200.csv"

MINOR_FLAGS = [
    "petite", "school uniform", "sailor uniform", "school bag",
    "elementary", "middle school", "child", "loli", "kid",
]
ADULT_FLAGS = ["mature face", "tall female", "curvy", "mature"]


def flags_in(text, words):
    return [w for w in words if w in text]


with open(PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

minor_only, adult_only, both, neither = [], [], [], []
for r in rows:
    p = r["positive"].lower()
    m = flags_in(p, MINOR_FLAGS)
    a = flags_in(p, ADULT_FLAGS)
    rid = r["id"]
    if m and a:
        both.append((rid, m, a))
    elif m:
        minor_only.append((rid, m))
    elif a:
        adult_only.append((rid, a))
    else:
        neither.append(rid)

print("総行数:", len(rows))
print("未成年判定のみ:", len(minor_only))
print("大人判定のみ:", len(adult_only))
print("両方のタグあり:", len(both))
print("どちらもなし:", len(neither))
print()
print("--- 大人判定のみのID ---")
print(", ".join(rid for rid, _ in adult_only))
print()
print("--- 両方のタグありのID（要個別判断） ---")
print(", ".join(rid for rid, _, _ in both))
print()
print("--- どちらもなしのID ---")
print(", ".join(neither))
print()

tags = Counter()
for r in rows:
    for t in r["positive"].split(","):
        t = t.strip()
        if t:
            tags[t] += 1
print("ユニークタグ数:", len(tags))
print("出現回数トップ25:")
for t, c in tags.most_common(25):
    print(f"  {c:3d}  {t}")

neg = Counter()
for r in rows:
    for t in r["negative"].split(","):
        t = t.strip()
        if t:
            neg[t] += 1
print()
print("ネガティブ ユニーク数:", len(neg))
print("ネガティブ トップ15:")
for t, c in neg.most_common(15):
    print(f"  {c:3d}  {t}")
