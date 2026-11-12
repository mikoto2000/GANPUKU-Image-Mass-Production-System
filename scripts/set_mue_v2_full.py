"""Mue v2 を全行へ反映する。

- positive: スタイル固定部を v2 に置換（lineart / watercolor / detailed 系を撤去）
- positive: キャラ部（1girl 以降）は維持し、背景タグは固定部へ集約
- positive: 目の色に応じて "detailed {color} gradient iris" を付与
- negative: 既存の品質・解剖ネガティブに v2 の事故対策を統合
- negative: soft cel shading と衝突する flat color は除外
- 例外: wavy/curly hair の行だけ curly hair ネガティブを外す
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BAK_PATH = ROOT / "prompts" / "prompts.csv.bak8"

STYLE = [
    "masterpiece", "best quality", "(full color:1.2)",
    "finished anime illustration", "modern anime style", "light novel cover",
    "soft digital painting", "smooth gradient shading", "soft cel shading",
    "pastel color palette", "bright high-key lighting", "cool soft shadows",
    "translucent skin", "pink blush", "glossy eyes", "detailed gradient iris",
    "(simple background:1.2)", "plain white background",
    "clean composition", "negative space", "minimal decoration",
]

BASE_NEG = [
    "bad quality", "worst quality", "worst detail", "sketch", "censor",
    "jpeg artifacts", "signature", "watermark", "username", "blurry", "lowres",
    "bad anatomy", "bad hands", "extra digits", "fewer digits", "missing fingers",
    "expressionless", "blank stare", "dead eyes", "emotionless", "serious",
    "frown", "scowl", "straight-on", "arms at sides", "stiff pose", "t-pose",
    "photorealistic", "3d", "cgi", "render", "photo",
    "oil painting", "impasto", "gouache", "acrylic (medium)",
    "thick lineart", "bold outlines", "heavy shading",
    "dark colors", "saturated colors", "high contrast",
    "rough lineart", "messy lineart",
    "sepia", "monochrome", "greyscale", "ink", "pen (medium)",
    "hatching", "crosshatching", "ornament", "floral pattern", "decorative",
    "border", "frame", "detailed background", "scenery",
]

V2_NEG = [
    "lineart", "brown tint", "grayscale", "limited palette",
    "ink drawing", "pen drawing", "thick outlines", "heavy black shadows",
    "traditional media", "watercolor painting", "art nouveau", "ornate",
    "decorative frame", "floral ornament", "swirling patterns",
    "intricate background", "curly hair",
]

EXTRA_NEG = ["text", "logo", "title"]

DROP_CHAR = {
    "simple background", "white background", "plain white background",
    "no background", "abstract background", "depth of field",
}

EYE_RE = re.compile(r"^(?P<color>[a-z][a-z \-]*?) eyes$", re.IGNORECASE)


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def dedupe(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def build_positive(old: str) -> str:
    tags = split_tags(old)
    start = next((i for i, t in enumerate(tags) if t.lower().startswith("1girl")), None)
    char = tags[start:] if start is not None else tags
    char = [t for t in char if t.lower() not in DROP_CHAR and "background" not in t.lower()]
    char = dedupe(char)

    # 髪のシルエット指定（装飾的な渦巻き髪を抑える）
    if not any("hair silhouette" in t.lower() for t in char):
        idx = next((i for i, t in enumerate(char) if "hair" in t.lower()), None)
        if idx is None:
            char.append("simple hair silhouette")
        else:
            char.insert(idx + 1, "simple hair silhouette")

    # 目の色に追従したグラデーション虹彩
    eye_color = None
    for t in char:
        m = EYE_RE.match(t.strip())
        if m:
            eye_color = m.group("color").strip()
            break
    if eye_color:
        tag = "detailed " + eye_color + " gradient iris"
        if not any(tag.lower() == t.lower() for t in char):
            idx = next((i for i, t in enumerate(char) if t.lower().endswith(" eyes")), None)
            if idx is None:
                char.append(tag)
            else:
                char.insert(idx + 1, tag)

    return ", ".join(dedupe(STYLE + char))


def build_negative(old: str, positive: str) -> str:
    tags = list(BASE_NEG) + list(V2_NEG) + list(EXTRA_NEG)
    low = positive.lower()
    if "wavy hair" in low or "curly hair" in low:
        tags = [t for t in tags if t != "curly hair"]
    return ", ".join(dedupe(tags))


def main() -> int:
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields:
        print("ヘッダがない")
        return 2

    if not BAK_PATH.exists():
        shutil.copy2(CSV_PATH, BAK_PATH)

    changed = 0
    for r in rows:
        pos = build_positive(r["positive"])
        neg = build_negative(r["negative"], pos)
        if pos != r["positive"] or neg != r["negative"]:
            changed += 1
        r["positive"] = pos
        r["negative"] = neg

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print("行数=" + str(len(rows)) + " 更新=" + str(changed))
    print("backup=" + str(BAK_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
