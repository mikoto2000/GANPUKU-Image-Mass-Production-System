"""set_modern_outfits.py — 背景を白に統一し、服装を現代的なパターンへ差し替える。

方針:
- 031-040 / 041-050 / 101-200 の positive を現代服のバリエーションで再構成する
- 001-030 / 051-100 はポーズ・髪型のテスト行なので触らない
- 背景は全行 simple background, white background に統一する
- バックアップは prompts.csv.bak3（初回のみ作成、以降は保持）

使い方:
    .\\venv\\Scripts\\python.exe scripts\\set_modern_outfits.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BACKUP_PATH = ROOT / "prompts" / "prompts.csv.bak3"

QUALITY = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
STYLE = (
    "illustration, painterly, digital painting, soft shading, detailed shading, "
    "brush strokes, refined lineart, rich colors"
)
BASE = "1girl, solo, full body"
BG = "simple background, white background"

# (outfit タグ, 日本語メモ)
OUTFITS = [
    ("t-shirt, jeans, sneakers, casual", "Tシャツ＋ジーンズ"),
    ("hoodie, jeans, sneakers, casual", "パーカー＋ジーンズ"),
    ("sweater, pleated skirt, casual", "ニット＋プリーツスカート"),
    ("cardigan, white shirt, skirt, casual", "カーディガン＋白シャツ＋スカート"),
    ("denim jacket, t-shirt, jeans, casual", "デニムジャケット＋Tシャツ"),
    ("one-piece dress, sleeveless, casual", "ワンピース（ノースリーブ）"),
    ("sundress, straw hat, summer, casual", "サンドレス＋麦わら帽"),
    ("blouse, long skirt, casual", "ブラウス＋ロングスカート"),
    ("tank top, shorts, casual", "タンクトップ＋ショートパンツ"),
    ("parka, leggings, sneakers, casual", "パーカ＋レギンス"),
    ("shirt, necktie, slacks, office lady", "シャツ＋ネクタイ＋スラックス"),
    ("business suit, pencil skirt, office lady", "ビジネススーツ＋ペンシルスカート"),
    ("trench coat, boots, autumn, casual", "トレンチコート＋ブーツ"),
    ("winter coat, scarf, gloves, winter", "冬コート＋マフラー＋手袋"),
    ("pajamas, nightwear, loungewear", "パジャマ"),
    ("overalls, t-shirt, casual", "オーバーオール＋Tシャツ"),
    ("crop top, high-waist skirt, casual", "クロップトップ＋ハイウエストスカート"),
    ("sailor collar, blouse, skirt, casual", "セーラーカラーブラウス"),
    ("hoodie, pleated skirt, casual", "パーカー＋プリーツスカート"),
    ("knit sweater, long skirt, winter, casual", "ニット＋ロングスカート"),
    ("t-shirt, denim shorts, summer, casual", "Tシャツ＋デニムショートパンツ"),
    ("striped shirt, jeans, casual", "ボーダーシャツ＋ジーンズ"),
    ("polo shirt, chino pants, casual", "ポロシャツ＋チノパン"),
    ("sweatshirt, jogger pants, casual", "スウェット＋ジョガーパンツ"),
    ("bomber jacket, t-shirt, jeans, streetwear", "ボンバージャケット＋ジーンズ"),
    ("leather jacket, t-shirt, skinny jeans, streetwear", "レザージャケット＋スキニー"),
    ("flannel shirt, t-shirt, jeans, casual", "ネルシャツ＋ジーンズ"),
    ("turtleneck, wide pants, casual", "タートルネック＋ワイドパンツ"),
    ("off-shoulder top, skirt, casual", "オフショルダートップ＋スカート"),
    ("camisole, cardigan, jeans, casual", "キャミソール＋カーディガン"),
    ("long cardigan, t-shirt, leggings, casual", "ロングカーディガン＋レギンス"),
    ("shirt dress, belt, casual", "シャツワンピース＋ベルト"),
    ("wrap dress, heels, elegant", "ラップワンピース＋ヒール"),
    ("knit dress, boots, winter", "ニットワンピース＋ブーツ"),
    ("slip dress, sheer cardigan, elegant", "スリップドレス＋シアーカーディガン"),
    ("a-line skirt, blouse, casual", "Aラインスカート＋ブラウス"),
    ("tiered skirt, t-shirt, casual", "ティアードスカート＋Tシャツ"),
    ("denim skirt, sweater, casual", "デニムスカート＋ニット"),
    ("corduroy skirt, blouse, autumn", "コーデュロイスカート＋ブラウス"),
    ("cargo pants, t-shirt, streetwear", "カーゴパンツ＋Tシャツ"),
    ("wide-leg pants, blouse, casual", "ワイドレッグパンツ＋ブラウス"),
    ("tapered pants, shirt, office lady", "テーパードパンツ＋シャツ"),
    ("shorts, t-shirt, summer", "ショートパンツ＋Tシャツ"),
    ("culottes, blouse, casual", "キュロット＋ブラウス"),
    ("leggings, long t-shirt, sporty", "レギンス＋ロングTシャツ"),
    ("track jacket, track pants, sportswear", "ジャージ上下"),
    ("jersey, shorts, sportswear", "ジャージ＋ショートパンツ"),
    ("tennis skirt, polo shirt, sportswear", "テニススカート＋ポロシャツ"),
    ("running wear, sportswear, sneakers", "ランニングウェア"),
    ("yoga pants, tank top, sportswear", "ヨガパンツ＋タンクトップ"),
    ("swim cover-up, sandals, resort", "水着の上に羽織り＋サンダル"),
    ("linen shirt, linen pants, summer", "リネンシャツ＋リネンパンツ"),
    ("aloha shirt, shorts, summer", "アロハシャツ＋ショートパンツ"),
    ("maxi dress, sandals, resort", "マキシワンピース＋サンダル"),
    ("halter top, wide pants, summer", "ホルタートップ＋ワイドパンツ"),
    ("romper, sandals, summer", "ロンパース＋サンダル"),
    ("jumpsuit, belt, casual", "ジャンプスーツ＋ベルト"),
    ("boiler suit, boots, workwear", "つなぎ＋ブーツ"),
    ("apron dress, casual", "エプロンワンピース"),
    ("pinafore dress, blouse, casual", "ジャンパースカート＋ブラウス"),
    ("suspenders, shirt, slacks, casual", "サスペンダー＋シャツ＋スラックス"),
    ("vest, shirt, slacks, casual", "ベスト＋シャツ＋スラックス"),
    ("blazer, t-shirt, jeans, casual", "ブレザー＋Tシャツ＋ジーンズ"),
    ("tailored jacket, skirt, office lady", "テーラードジャケット＋スカート"),
    ("double-breasted coat, slacks, elegant", "ダブルブレストコート＋スラックス"),
    ("chester coat, turtleneck, winter", "チェスターコート＋タートルネック"),
    ("duffle coat, scarf, winter", "ダッフルコート＋マフラー"),
    ("puffer jacket, jeans, winter", "ダウンジャケット＋ジーンズ"),
    ("down vest, sweater, winter", "ダウンベスト＋ニット"),
    ("raincoat, boots, rain", "レインコート＋ブーツ"),
    ("windbreaker, jogger pants, sporty", "ウィンドブレーカー＋ジョガーパンツ"),
    ("anorak, cargo pants, outdoor", "アノラック＋カーゴパンツ"),
    ("fleece jacket, leggings, outdoor", "フリースジャケット＋レギンス"),
    ("hooded coat, boots, winter", "フードコート＋ブーツ"),
    ("poncho, jeans, autumn", "ポンチョ＋ジーンズ"),
    ("cape coat, dress, elegant", "ケープコート＋ワンピース"),
    ("knit vest, shirt, skirt, casual", "ニットベスト＋シャツ＋スカート"),
    ("argyle sweater, slacks, casual", "アーガイルニット＋スラックス"),
    ("cable knit sweater, jeans, winter", "ケーブルニット＋ジーンズ"),
    ("mohair sweater, skirt, winter", "モヘアニット＋スカート"),
    ("oversized shirt, shorts, casual", "オーバーサイズシャツ＋ショートパンツ"),
    ("cropped jacket, high-waist pants, streetwear", "クロップドジャケット＋ハイウエストパンツ"),
    ("zip-up hoodie, t-shirt, jeans, casual", "ジップパーカー＋Tシャツ"),
    ("graphic t-shirt, denim jacket, streetwear", "プリントTシャツ＋デニムジャケット"),
    ("band t-shirt, leather pants, rock", "バンドTシャツ＋レザーパンツ"),
    ("mesh top, camisole, skirt, streetwear", "メッシュトップ＋キャミソール"),
    ("sheer blouse, tank top, slacks, casual", "シアーブラウス＋タンクトップ"),
    ("lace top, midi skirt, elegant", "レーストップ＋ミディスカート"),
    ("satin blouse, wide pants, elegant", "サテンブラウス＋ワイドパンツ"),
    ("ruffle blouse, skirt, girly", "フリルブラウス＋スカート"),
    ("puff sleeve top, skirt, girly", "パフスリーブトップ＋スカート"),
    ("ribbon blouse, pleated skirt, girly", "リボンブラウス＋プリーツスカート"),
    ("babydoll dress, frills, girly", "ベビードールワンピース"),
    ("tiered dress, floral print, girly", "花柄ティアードワンピース"),
    ("polka dot dress, belt, retro", "水玉ワンピース＋ベルト"),
    ("gingham dress, summer, retro", "ギンガムチェックワンピース"),
    ("checkered skirt, sweater, retro", "チェック柄スカート＋ニット"),
    ("houndstooth skirt, blazer, retro", "千鳥格子スカート＋ブレザー"),
    ("denim on denim, casual", "デニムオンデニム"),
    ("all white outfit, casual", "オールホワイトコーデ"),
    ("monochrome outfit, mode", "モノトーンコーデ"),
    ("oversized blazer, wide pants, mode", "オーバーサイズブレザー＋ワイドパンツ"),
    ("long shirt, leggings, mode", "ロングシャツ＋レギンス"),
    ("asymmetric skirt, top, mode", "アシンメトリーなスカート"),
    ("draped dress, heels, mode", "ドレープワンピース＋ヒール"),
    ("turtleneck dress, boots, mode", "タートルネックワンピース＋ブーツ"),
    ("leather skirt, knit top, mode", "レザースカート＋ニットトップ"),
    ("vinyl coat, boots, mode", "ビニールコート＋ブーツ"),
    ("cargo skirt, crop top, streetwear", "カーゴスカート＋クロップトップ"),
    ("utility jacket, cargo pants, streetwear", "ユーティリティジャケット＋カーゴパンツ"),
    ("track pants, hoodie, streetwear", "トラックパンツ＋パーカー"),
    ("bucket hat, t-shirt, shorts, streetwear", "バケットハット＋Tシャツ＋ショートパンツ"),
    ("varsity jacket, jeans, streetwear", "バーシティジャケット＋ジーンズ"),
    ("coach jacket, chino pants, streetwear", "コーチジャケット＋チノパン"),
    ("work shirt, jeans, workwear", "ワークシャツ＋ジーンズ"),
    ("painter pants, t-shirt, workwear", "ペインターパンツ＋Tシャツ"),
    ("overall shorts, t-shirt, summer", "オーバーオールショートパンツ"),
    ("knit cardigan, camisole, skirt, casual", "ニットカーディガン＋キャミソール"),
    ("long sleeve t-shirt, jeans, casual", "ロングTシャツ＋ジーンズ"),
    ("short sleeve shirt, slacks, casual", "半袖シャツ＋スラックス"),
]

# ポーズ・表情（行ごとに巡回させる）
POSES = [
    "standing, contrapposto, looking at viewer, smile",
    "standing, weight on one leg, hand on own hip, looking at viewer, light smile",
    "standing, arms behind back, looking at viewer, gentle smile",
    "standing, hand on own cheek, looking at viewer, happy, smile",
    "standing, crossed arms, looking at viewer, confident smile",
    "standing, one hand up, waving, looking at viewer, open mouth, smile",
    "standing, hand in hair, looking at viewer, closed eyes, smile",
    "standing, head tilt, looking at viewer, cheerful, smile",
    "standing, hands clasped together, looking at viewer, gentle smile",
    "standing, contrapposto, hand on own chest, looking at viewer, smile",
    "standing, from side, profile, looking at viewer, light smile",
    "standing, from behind, looking back, smile",
    "standing, leaning forward, hands on knees, looking at viewer, grin",
    "standing, leaning back, hands behind head, closed eyes, smile",
    "standing, legs apart, hands on own hips, looking at viewer, confident smile",
    "standing, on tiptoes, arms up, reaching, looking up, smile",
    "walking, looking at viewer, smile",
    "sitting, on chair, crossed legs, looking at viewer, smile",
    "sitting, on floor, knees up, arms around knees, looking at viewer, smile",
    "standing, dutch angle, hand on own hip, looking at viewer, gentle smile",
]

# 041-050 は照明バリエーションの検証行として使う
LIGHTING = [
    "soft lighting",
    "rim light",
    "natural lighting",
    "studio lighting",
    "warm lighting",
    "cool lighting",
    "day",
    "evening",
    "dramatic lighting",
    "soft lighting",
]

# 服装を差し替える範囲（031-050 と 101-200）
TARGET_RANGES = [(31, 50), (101, 200)]

# 非対象行から除去する背景・風景系タグ
STRIP_TAGS = {
    "abstract background", "pink background", "dark background",
    "blue background", "green background", "grey background",
    "warm lighting", "cool lighting", "day", "evening",
    "rain", "umbrella", "wet", "backlighting", "rim light",
    "cinematic lighting", "dramatic lighting", "winter",
    "sunflower", "depth of field",
}


def build_positive(pose: str, outfit: str, lighting: str = "") -> str:
    parts = [QUALITY, STYLE, BASE, pose, outfit]
    if lighting:
        parts.append(lighting)
    parts.append(BG)
    return ", ".join(parts)


def detect_newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def main() -> int:
    if not CSV_PATH.exists():
        print("CSV が見つからない: " + str(CSV_PATH))
        return 2

    raw = CSV_PATH.read_bytes()
    newline = detect_newline(raw)

    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print("バックアップ: " + str(BACKUP_PATH))
    else:
        print("バックアップは既存のものを保持: " + str(BACKUP_PATH))

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "positive" not in fieldnames:
        print("CSV のヘッダが不正")
        return 2

    changed = 0
    stripped = 0
    outfit_count = 0

    for row in rows:
        rid = (row.get("id") or "").strip()
        try:
            num = int(rid)
        except ValueError:
            continue

        in_target = any(lo <= num <= hi for lo, hi in TARGET_RANGES)

        if in_target:
            outfit, _ = OUTFITS[outfit_count % len(OUTFITS)]
            outfit_count += 1
            pose = POSES[outfit_count % len(POSES)]
            lighting = ""
            if 41 <= num <= 50:
                lighting = LIGHTING[num - 41]
            row["positive"] = build_positive(pose, outfit, lighting)
            changed += 1
        else:
            # 非対象行は背景タグだけ正規化する
            tags = [t.strip() for t in (row.get("positive") or "").split(",") if t.strip()]
            kept = [t for t in tags if t not in STRIP_TAGS]
            if kept != tags:
                stripped += 1
            # 背景タグを末尾へ寄せる
            kept = [t for t in kept if t not in ("simple background", "white background")]
            kept.extend(["simple background", "white background"])
            row["positive"] = ", ".join(kept)

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)

    print("服装差し替え: " + str(changed) + " 行")
    print("背景タグ除去: " + str(stripped) + " 行")
    print("服装パターン: " + str(len(OUTFITS)) + " 種")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
