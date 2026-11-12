"""fix_notes.py — 服装を差し替えた行の note 列を実際の内容に合わせる。"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"F:\ai-image")
CSV_PATH = ROOT / "prompts" / "prompts.csv"
BACKUP_PATH = ROOT / "prompts" / "prompts.csv.bak4"

# 服装タグ先頭 → 日本語メモ
NOTE_MAP = {
    "t-shirt": "Tシャツ",
    "hoodie": "パーカー",
    "sweater": "ニット",
    "cardigan": "カーディガン",
    "denim jacket": "デニムジャケット",
    "one-piece dress": "ワンピース",
    "sundress": "サンドレス",
    "blouse": "ブラウス",
    "tank top": "タンクトップ",
    "parka": "パーカ",
    "shirt": "シャツ",
    "business suit": "ビジネススーツ",
    "trench coat": "トレンチコート",
    "winter coat": "冬コート",
    "pajamas": "パジャマ",
    "overalls": "オーバーオール",
    "crop top": "クロップトップ",
    "sailor collar": "セーラーカラー",
    "knit sweater": "ニット",
    "striped shirt": "ボーダーシャツ",
    "polo shirt": "ポロシャツ",
    "sweatshirt": "スウェット",
    "bomber jacket": "ボンバージャケット",
    "leather jacket": "レザージャケット",
    "flannel shirt": "ネルシャツ",
    "turtleneck": "タートルネック",
    "off-shoulder top": "オフショルダー",
    "camisole": "キャミソール",
    "long cardigan": "ロングカーディガン",
    "shirt dress": "シャツワンピース",
    "wrap dress": "ラップワンピース",
    "knit dress": "ニットワンピース",
    "slip dress": "スリップドレス",
    "a-line skirt": "Aラインスカート",
    "tiered skirt": "ティアードスカート",
    "denim skirt": "デニムスカート",
    "corduroy skirt": "コーデュロイスカート",
    "cargo pants": "カーゴパンツ",
    "wide-leg pants": "ワイドパンツ",
    "tapered pants": "テーパードパンツ",
    "shorts": "ショートパンツ",
    "culottes": "キュロット",
    "leggings": "レギンス",
    "track jacket": "ジャージ",
    "jersey": "ジャージ",
    "tennis skirt": "テニスウェア",
    "running wear": "ランニングウェア",
    "yoga pants": "ヨガウェア",
    "swim cover-up": "リゾートウェア",
    "linen shirt": "リネンシャツ",
    "aloha shirt": "アロハシャツ",
    "maxi dress": "マキシワンピース",
    "halter top": "ホルタートップ",
    "romper": "ロンパース",
    "jumpsuit": "ジャンプスーツ",
    "boiler suit": "つなぎ",
    "apron dress": "エプロンワンピース",
    "pinafore dress": "ジャンパースカート",
    "suspenders": "サスペンダー",
    "vest": "ベスト",
    "blazer": "ブレザー",
    "tailored jacket": "テーラードジャケット",
    "double-breasted coat": "ダブルブレストコート",
    "chester coat": "チェスターコート",
    "duffle coat": "ダッフルコート",
    "puffer jacket": "ダウンジャケット",
    "down vest": "ダウンベスト",
    "raincoat": "レインコート",
    "windbreaker": "ウィンドブレーカー",
    "anorak": "アノラック",
    "fleece jacket": "フリース",
    "hooded coat": "フードコート",
    "poncho": "ポンチョ",
    "cape coat": "ケープコート",
    "knit vest": "ニットベスト",
    "argyle sweater": "アーガイルニット",
    "cable knit sweater": "ケーブルニット",
    "mohair sweater": "モヘアニット",
    "oversized shirt": "オーバーサイズシャツ",
    "cropped jacket": "クロップドジャケット",
    "zip-up hoodie": "ジップパーカー",
    "graphic t-shirt": "プリントTシャツ",
    "band t-shirt": "バンドTシャツ",
    "mesh top": "メッシュトップ",
    "sheer blouse": "シアーブラウス",
    "lace top": "レーストップ",
    "satin blouse": "サテンブラウス",
    "ruffle blouse": "フリルブラウス",
    "puff sleeve top": "パフスリーブ",
    "ribbon blouse": "リボンブラウス",
    "babydoll dress": "ベビードール",
    "tiered dress": "ティアードワンピース",
    "polka dot dress": "水玉ワンピース",
    "gingham dress": "ギンガムワンピース",
    "checkered skirt": "チェックスカート",
    "houndstooth skirt": "千鳥格子スカート",
    "denim on denim": "デニムオンデニム",
    "all white outfit": "オールホワイト",
    "monochrome outfit": "モノトーン",
    "oversized blazer": "オーバーサイズブレザー",
    "long shirt": "ロングシャツ",
    "asymmetric skirt": "アシンメトリー",
    "draped dress": "ドレープワンピース",
    "turtleneck dress": "タートルワンピース",
    "leather skirt": "レザースカート",
    "vinyl coat": "ビニールコート",
    "cargo skirt": "カーゴスカート",
    "utility jacket": "ユーティリティジャケット",
    "track pants": "トラックパンツ",
    "bucket hat": "バケットハット",
    "varsity jacket": "バーシティジャケット",
    "coach jacket": "コーチジャケット",
    "work shirt": "ワークシャツ",
    "painter pants": "ペインターパンツ",
    "overall shorts": "オーバーオールショート",
    "knit cardigan": "ニットカーディガン",
    "long sleeve t-shirt": "ロングTシャツ",
    "short sleeve shirt": "半袖シャツ",
}


def detect_newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def main() -> int:
    raw = CSV_PATH.read_bytes()
    newline = detect_newline(raw)

    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print("バックアップ: " + str(BACKUP_PATH))

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fieldnames = reader.fieldnames
        rows = list(reader)

    fixed = 0
    unknown = []

    for row in rows:
        rid = (row.get("id") or "").strip()
        try:
            num = int(rid)
        except ValueError:
            continue
        if not (31 <= num <= 50 or 101 <= num <= 200):
            continue

        tags = [t.strip() for t in (row.get("positive") or "").split(",") if t.strip()]
        try:
            i = tags.index("full body")
            j = tags.index("simple background")
        except ValueError:
            unknown.append(rid)
            continue

        outfit_tags = tags[i + 1 : j]
        # 照明タグを除いた服装部分
        outfit_tags = [t for t in outfit_tags if t not in {
            "soft lighting", "rim light", "natural lighting", "studio lighting",
            "warm lighting", "cool lighting", "day", "evening", "dramatic lighting",
        }]

        label = None
        label_idx = -1
        for idx, t in enumerate(outfit_tags):
            if t in NOTE_MAP:
                label = NOTE_MAP[t]
                label_idx = idx
                break

        if label is None:
            unknown.append(rid)
            continue

        # 主タグより後ろのタグだけを補足に使う（主タグの重複を防ぐ）
        extras = []
        for t in outfit_tags[label_idx + 1 :]:
            mapped = NOTE_MAP.get(t)
            if mapped and mapped != label and mapped not in extras:
                extras.append(mapped)
        note = label + ("＋" + "＋".join(extras[:2]) if extras else "")

        if row.get("note") != note:
            row["note"] = note
            fixed += 1

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)

    print("note 更新: " + str(fixed) + " 行")
    print("判定不能: " + str(len(unknown)) + (" -> " + ",".join(unknown) if unknown else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
