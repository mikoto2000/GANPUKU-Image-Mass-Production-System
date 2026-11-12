"""セル塗り撤去後の検証"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"

STYLE_HEAD = (
    "masterpiece, best quality, (full color:1.2), illustration, digital painting, "
    "airbrush (medium), smooth gradient shading, soft shading, detailed shading, "
    "pastel color palette, pale color, bright high-key lighting, cool soft shadows, "
    "translucent skin, pink blush, glossy eyes, detailed gradient iris, "
    "(simple background:1.2), plain white background, clean composition, "
    "negative space, minimal decoration"
)

BANNED_POS = [
    "cel shading", "flat color", "flat shading", "anime coloring",
    "finished anime illustration", "modern anime style", "light novel cover",
    "soft digital painting", "lineart", "watercolor", "painterly",
    "brush strokes", "impasto", "ornament", "decorative",
    "abstract background", "depth of field",
]

REQUIRED_NEG = [
    "cel shading", "flat shading", "hard shading", "anime coloring",
    "sharp shadows", "hard edges", "posterized", "color banding",
    "lineart", "sepia", "brown tint", "ink drawing", "hatching",
    "bad anatomy", "bad hands", "text",
]

BAD_IRIS = re.compile(
    r"detailed (closed|glossy|shiny|bright|clear|half|wide|open|big|small) gradient iris",
    re.IGNORECASE,
)

with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, strict=True)
    fields = reader.fieldnames
    rows = list(reader)

print("行数=" + str(len(rows)) + " 列数=" + str(len(fields)))

bad_style = [r["id"] for r in rows if not r["positive"].startswith(STYLE_HEAD)]
print("スタイル固定部が不一致: " + str(len(bad_style)) + " 行")
for i in bad_style[:5]:
    print("  " + i)

bad_pos = {}
for r in rows:
    low = r["positive"].lower()
    hits = [b for b in BANNED_POS if b in low]
    if hits:
        bad_pos[r["id"]] = hits
print("positive 禁止タグ残存: " + str(len(bad_pos)) + " 行")
for k, v in list(bad_pos.items())[:10]:
    print("  " + k + ": " + ", ".join(v))

bad_iris = [r["id"] for r in rows if BAD_IRIS.search(r["positive"])]
print("誤認虹彩タグ: " + str(len(bad_iris)) + " 行")

missing_neg = {}
for r in rows:
    low = r["negative"].lower()
    miss = [t for t in REQUIRED_NEG if t not in low]
    if miss:
        missing_neg[r["id"]] = miss
print("negative 必須タグ欠落: " + str(len(missing_neg)) + " 行")
for k, v in list(missing_neg.items())[:5]:
    print("  " + k + ": " + ", ".join(v))

no_white = [r["id"] for r in rows if "plain white background" not in r["positive"].lower()]
print("plain white background 欠落: " + str(len(no_white)) + " 行")

no_char = [r["id"] for r in rows if "1girl" not in r["positive"].lower()]
print("1girl 欠落: " + str(len(no_char)) + " 行")

no_hair = [r["id"] for r in rows if "hair silhouette" not in r["positive"].lower()]
print("simple hair silhouette 欠落: " + str(len(no_hair)) + " 行")

print("")
print("--- サンプル ---")
for r in rows[:1] + rows[3:4]:
    print("id=" + r["id"])
    print("  pos: " + r["positive"])
    print("  neg: " + r["negative"][:260] + "...")
