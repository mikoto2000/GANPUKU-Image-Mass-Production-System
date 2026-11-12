"""Mue v2 反映: 線画タグ全撤去・具体色指定・装飾ネガティブ

- positive: スタイル固定部を v2 に置換（lineart / watercolor / detailed 系を全撤去）
- positive: 髪・目・服に具体的な色名を1つずつ付与
- negative: 今回の事故（セピア・装飾線画）を追加
- 冪等: 既に v2 なら実質変更なし
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BAK_PATH = ROOT / "prompts" / "prompts.csv.bak7"

STYLE = (
    "masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
    "illustration, anime style, soft shading, soft lighting, pale color, pastel colors"
)

OLD_STYLE = {
    "masterpiece", "best quality", "amazing quality", "very aesthetic", "absurdres",
    "illustration", "anime style", "watercolor (medium)", "traditional media",
    "pale color", "pastel colors", "soft lighting", "soft shading",
    "thin lineart", "detailed hair", "sharp lineart", "crisp lineart",
    "detailed lineart", "gradient shading", "detailed shading", "glossy hair",
    "vibrant colors", "painterly", "digital painting", "brush strokes",
    "refined lineart", "rich colors", "cel shading", "clean lineart", "anime coloring",
}

NEG = (
    "bad quality, worst quality, worst detail, sketch, censor, jpeg artifacts, "
    "signature, watermark, username, blurry, lowres, bad anatomy, bad hands, "
    "extra digits, fewer digits, missing fingers, expressionless, blank stare, "
    "dead eyes, emotionless, serious, frown, scowl, straight-on, arms at sides, "
    "stiff pose, t-pose, photorealistic, 3d, cgi, render, photo, oil painting, "
    "impasto, gouache, acrylic (medium), thick lineart, bold outlines, heavy shading, "
    "dark colors, saturated colors, high contrast, flat color, flat shading, "
    "simple shading, rough lineart, messy lineart, sepia, monochrome, greyscale, "
    "ink, pen (medium), hatching, crosshatching, ornament, floral pattern, "
    "decorative, border, frame, detailed background, scenery"
)

HAIR = ["black", "brown", "blonde", "silver", "pink", "blue", "red", "purple",
        "white", "green", "orange", "ash blonde", "light brown", "dark blue",
        "lavender", "mint green", "peach", "grey"]
EYES = ["blue", "red", "green", "purple", "brown", "amber", "pink", "grey",
        "teal", "gold", "violet", "hazel"]
CLOTH = ["white", "black", "navy", "beige", "grey", "brown", "pink", "blue",
         "red", "green", "cream", "lavender", "mint", "orange", "purple", "yellow"]

COLOR_WORDS = set(HAIR) | set(EYES) | set(CLOTH) | {
    "dark", "light", "pale", "deep", "bright", "pastel", "ash", "olive",
    "maroon", "turquoise", "coral", "ivory", "charcoal", "khaki", "burgundy",
}

GARMENTS = [
    "t-shirt", "shirt", "blouse", "hoodie", "parka", "sweater", "knit", "cardigan",
    "jacket", "coat", "blazer", "suit", "vest", "tank top", "camisole", "turtleneck",
    "dress", "skirt", "jeans", "pants", "slacks", "shorts", "cargo", "overalls",
    "jumpsuit", "coveralls", "sundress", "one-piece", "jersey", "uniform",
    "leotard", "wear", "outfit", "apron", "poncho", "cape", "trench", "duffle",
    "bomber", "varsity", "leather", "denim", "linen", "aloha", "polo",
]


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def is_color_tag(tag: str) -> bool:
    low = tag.lower()
    return any(w in low.split() for w in COLOR_WORDS)


def is_garment(tag: str) -> bool:
    low = tag.lower()
    return any(g in low for g in GARMENTS)


def build_positive(old: str, idx: int) -> str:
    tags = split_tags(old)
    # 旧スタイルタグを除去
    body = [t for t in tags if t.lower() not in OLD_STYLE]

    # キャラ部（1girl 以降）を抽出
    start = next((i for i, t in enumerate(body) if t.lower().startswith("1girl")), 0)
    char = body[start:]

    # 既存の髪・目・服の色指定を検出
    has_hair_color = any("hair" in t.lower() and is_color_tag(t) for t in char)
    has_eye_color = any("eyes" in t.lower() and is_color_tag(t) for t in char)
    has_cloth_color = any(is_garment(t) and is_color_tag(t) for t in char)

    extra: list[str] = []
    if not has_hair_color:
        extra.append(HAIR[idx % len(HAIR)] + " hair")
    if not has_eye_color:
        extra.append(EYES[idx % len(EYES)] + " eyes")
    if not has_cloth_color:
        # 服タグの直前に色を差し込む
        for i, t in enumerate(char):
            if is_garment(t) and not is_color_tag(t):
                char[i] = CLOTH[idx % len(CLOTH)] + " " + t
                has_cloth_color = True
                break
        if not has_cloth_color:
            extra.append(CLOTH[idx % len(CLOTH)] + " clothes")

    # 白背景を末尾に固定
    char = [t for t in char if t.lower() not in {"simple background", "white background"}]
    char = [t for t in char if t.lower() not in {"detailed background", "scenery"}]

    return ", ".join([STYLE] + char + extra + ["simple background", "white background"])


def main() -> int:
    if not CSV_PATH.exists():
        print("CSV が見つからない: " + str(CSV_PATH))
        return 2

    if not BAK_PATH.exists():
        shutil.copy2(CSV_PATH, BAK_PATH)
        print("バックアップ: " + str(BAK_PATH))

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames
        rows = list(reader)

    changed = 0
    for idx, row in enumerate(rows):
        new_pos = build_positive(row["positive"], idx)
        if new_pos != row["positive"]:
            row["positive"] = new_pos
            changed += 1
        if row["negative"] != NEG:
            row["negative"] = NEG

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print("更新 " + str(changed) + " / " + str(len(rows)) + " 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
