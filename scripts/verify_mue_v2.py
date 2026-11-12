"""v2 反映後の検証"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"

BANNED_POS = [
    "lineart", "watercolor", "traditional media", "detailed hair",
    "detailed shading", "detailed lineart", "painterly", "brush strokes",
    "impasto", "cel shading", "anime coloring", "gradient shading",
    "glossy hair", "vibrant colors", "rich colors", "ultra detailed",
    "highly detailed", "intricate", "ornament", "floral", "decorative",
]

BANNED_NEG = ["watercolor"]

with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, strict=True)
    fields = reader.fieldnames
    rows = list(reader)

print("行数=" + str(len(rows)) + " 列数=" + str(len(fields)))

bad_pos = {}
for r in rows:
    low = r["positive"].lower()
    hits = [b for b in BANNED_POS if b in low]
    if hits:
        bad_pos[r["id"]] = hits

print("positive 禁止タグ残存: " + str(len(bad_pos)) + " 行")
for k, v in list(bad_pos.items())[:10]:
    print("  " + k + ": " + ", ".join(v))

neg_water = [r["id"] for r in rows if "watercolor" in r["negative"].lower()]
print("negative に watercolor: " + str(len(neg_water)) + " 行")

no_white = [r["id"] for r in rows if "white background" not in r["positive"].lower()]
print("white background 欠落: " + str(len(no_white)) + " 行")

no_color = []
for r in rows:
    low = r["positive"].lower()
    if not re.search(r"(hair|eyes)", low):
        no_color.append(r["id"])
print("髪・目の色指定なし: " + str(len(no_color)) + " 行")

sepia_neg = [r["id"] for r in rows if "sepia" not in r["negative"].lower()]
print("negative に sepia なし: " + str(len(sepia_neg)) + " 行")

print("")
print("--- サンプル ---")
for r in rows[:2] + rows[100:102]:
    print("id=" + r["id"])
    print("  pos: " + r["positive"])
    print("  neg: " + r["negative"][:160] + "...")
