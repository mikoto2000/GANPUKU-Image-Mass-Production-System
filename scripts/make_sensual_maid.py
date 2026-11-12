# -*- coding: utf-8 -*-
"""prompts_maid_200.csv から「露骨でない範囲で色気を足した」派生CSVを作る。

方針 (make_sensual_v7.py と同一):
- 大人判定タグ (mature face / tall female / curvy / mature) がある行だけ対象
- 未成年判定タグ (petite / school uniform など) がある行は除外
- 追加するのは視線・表情・姿勢・素材・照明のタグのみ
- ヌード/下着/性行為/局部の語は一切追加しない
- 既存タグと矛盾する追加はスキップ
- 元ファイルは変更しない
"""
import copy
import csv
import random
import re

SRC = "prompts/prompts_maid_200.csv"
DST = "prompts/prompts_maid_200_sensual.csv"

MINOR_FLAGS = [
    "petite", "school uniform", "sailor uniform", "school bag",
    "elementary", "middle school", "child", "loli", "kid",
]
ADULT_FLAGS = ["mature face", "tall female", "curvy", "mature"]

GAZE = [
    "sultry gaze", "half-closed eyes", "upturned eyes",
    "soft gaze", "looking at viewer over shoulder",
]
EXPR = [
    "blush", "parted lips", "flushed cheeks",
    "gentle smile", "soft smile", "slightly parted lips",
]
POSE = [
    "contrapposto", "arched back", "hand on own hip",
    "hand on own chest", "hand in own hair", "leaning forward",
    "one knee up", "weight on one leg",
]
OUTFIT = [
    "off-shoulder", "sleeveless", "form-fitting", "silk fabric",
    "satin", "lace trim", "slit skirt", "thin strap", "choker",
]
LIGHT = [
    "rim lighting", "backlighting", "soft focus",
    "shallow depth of field", "dim lighting", "warm ambient light",
    "candlelight", "moody lighting", "golden hour",
]

# タグ完全一致で判定する禁止語（部分一致だと braid/glass/button を誤検知する）
BLOCKLIST_TAGS = {
    "nude", "naked", "topless", "bottomless", "lingerie", "underwear",
    "panties", "bra", "thong", "garter", "garter belt", "nipple", "nipples",
    "areola", "cleavage", "cameltoe", "spread legs", "legs apart",
    "sex", "cum", "penis", "vagina", "pussy", "ass", "butt", "crotch",
    "see-through", "see-through clothes", "sheer", "transparent",
    "transparent clothes", "wet", "wet skin", "wet clothes", "sweat",
    "ahegao", "tongue out", "saliva", "drool", "orgasm",
    "masturbation", "fellatio", "bondage", "bdsm", "nsfw", "explicit",
}


def tags_of(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def has_any(text, words):
    return any(w in text for w in words)


def pick(rng, pool, existing, added):
    cands = [t for t in pool if t not in existing and t not in added]
    if not cands:
        return None
    return rng.choice(cands)


def build_additions(rid, positive):
    rng = random.Random(int(rid))
    existing = set(tags_of(positive))
    added = []

    gaze_blocked = (
        "closed eyes" in existing
        or has_any(positive, ["looking down", "looking to the side",
                              "looking away", "looking back"])
    )
    if not gaze_blocked:
        t = pick(rng, GAZE, existing, added)
        if t:
            added.append(t)

    for pool in (EXPR, POSE, OUTFIT, LIGHT):
        t = pick(rng, pool, existing, added)
        if t:
            added.append(t)

    return added


def main():
    raw = open(SRC, "rb").read()
    lineterm = "\r\n" if b"\r\n" in raw else "\n"

    with open(SRC, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    originals = copy.deepcopy(rows)  # 比較用のスナップショット

    changed, skipped_minor, skipped_both, skipped_none = [], [], [], []
    added_counter = {}

    for r in rows:
        p = r["positive"]
        pl = p.lower()
        rid = r["id"]
        minor = has_any(pl, MINOR_FLAGS)
        adult = has_any(pl, ADULT_FLAGS)

        if minor and adult:
            skipped_both.append(rid)
            continue
        if minor:
            skipped_minor.append(rid)
            continue
        if not adult:
            skipped_none.append(rid)
            continue

        adds = build_additions(rid, p)
        if not adds:
            continue
        r["positive"] = p.rstrip() + ", " + ", ".join(adds)
        r["note"] = r["note"].rstrip() + "／色気強化"
        changed.append(rid)
        for t in adds:
            added_counter[t] = added_counter.get(t, 0) + 1

    with open(DST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                           lineterminator=lineterm)
        w.writeheader()
        w.writerows(rows)

    # --- 検証 ---
    with open(DST, encoding="utf-8-sig", newline="") as f:
        out = list(csv.DictReader(f))

    # 1) タグ完全一致での禁止語チェック
    hits = []
    for r in out:
        for col in ("positive", "negative"):
            for t in tags_of(r[col]):
                base = re.sub(r"^\(|\)$|:[0-9.]+", "", t).strip().lower()
                if base in BLOCKLIST_TAGS:
                    hits.append((r["id"], col, t))

    # 2) 追加タグが禁止語を含まないこと
    bad_adds = [t for t in added_counter if t.lower() in BLOCKLIST_TAGS]

    # 3) 行数・列の一致
    same_ids = [r["id"] for r in out] == [r["id"] for r in originals]
    same_cols = list(out[0].keys()) == list(originals[0].keys())

    # 4) 元のタグがすべて残っているか（削除していないこと）
    lost = []
    for o, n in zip(originals, out):
        ot = set(tags_of(o["positive"]))
        nt = set(tags_of(n["positive"]))
        if not ot.issubset(nt):
            lost.append((o["id"], sorted(ot - nt)))

    print("=== 生成結果 ===")
    print("出力:", DST)
    print("総行数:", len(out))
    print("変更行数:", len(changed))
    print("除外(未成年判定):", len(skipped_minor))
    print("除外(両方混在):", len(skipped_both))
    print("除外(判定なし):", len(skipped_none))
    print()
    print("=== 整合性チェック ===")
    print("ID順序一致:", same_ids)
    print("列構成一致:", same_cols)
    print("元タグの欠落:", "なし" if not lost else f"{len(lost)}件")
    print("追加タグに禁止語:", "なし" if not bad_adds else bad_adds)
    print()
    print("=== 露骨語スキャン（タグ完全一致） ===")
    if hits:
        print("検出:", len(hits))
        for rid, col, t in hits[:20]:
            print(f"  {rid} [{col}]: {t}")
    else:
        print("検出 0 件")
    print()
    print("=== 追加タグの内訳 ===")
    for t, c in sorted(added_counter.items(), key=lambda x: -x[1]):
        print(f"  {c:3d}  {t}")
    print()
    print("=== 変更例（追加分のみ） ===")
    for rid in changed[:5]:
        o = next(r for r in originals if r["id"] == rid)
        n = next(r for r in out if r["id"] == rid)
        diff = set(tags_of(n["positive"])) - set(tags_of(o["positive"]))
        print(f"  {rid}: + {', '.join(sorted(diff))}")


if __name__ == "__main__":
    main()
