# -*- coding: utf-8 -*-
"""prompts_maid_200_sensual.csv から「色気タグを追加した行だけ」を抜き出す。

判定は2系統で行い、両方一致した行だけを残す:
  1) note 列に「色気強化」マーカーがあるか
  2) 元CSV (prompts_maid_200.csv) と比べて positive に追加タグがあるか

元ファイルと sensual ファイルは変更しない。
"""
import csv

SRC_ORIG = "prompts/prompts_maid_200.csv"
SRC_SENSUAL = "prompts/prompts_maid_200_sensual.csv"
DST = "prompts/prompts_maid_200_sensual_only.csv"
MARKER = "色気強化"


def tags_of(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def load(path):
    raw = open(path, "rb").read()
    lineterm = "\r\n" if b"\r\n" in raw else "\n"
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader), lineterm


def main():
    fields_o, orig, _ = load(SRC_ORIG)
    fields_s, sensual, lineterm = load(SRC_SENSUAL)

    if fields_o != fields_s:
        raise SystemExit(f"列構成が違う: {fields_o} != {fields_s}")

    orig_by_id = {r["id"]: r for r in orig}

    kept, dropped = [], []
    mismatch = []

    for r in sensual:
        rid = r["id"]
        o = orig_by_id.get(rid)
        if o is None:
            mismatch.append((rid, "元CSVに存在しない"))
            continue

        by_note = MARKER in (r.get("note") or "")
        added = set(tags_of(r["positive"])) - set(tags_of(o["positive"]))
        by_diff = bool(added)

        if by_note != by_diff:
            mismatch.append((rid, f"note={by_note} diff={by_diff}"))

        if by_note and by_diff:
            kept.append(r)
        else:
            dropped.append(rid)

    with open(DST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_s, quoting=csv.QUOTE_ALL,
                           lineterminator=lineterm)
        w.writeheader()
        w.writerows(kept)

    # --- 検証 ---
    _, out, _ = load(DST)

    # 残した行がすべて「追加あり」であること
    bad = []
    for r in out:
        o = orig_by_id[r["id"]]
        if not (set(tags_of(r["positive"])) - set(tags_of(o["positive"]))):
            bad.append(r["id"])

    # 落とした行に「追加あり」が混ざっていないこと
    leaked = []
    for rid in dropped:
        r = next(x for x in sensual if x["id"] == rid)
        o = orig_by_id[rid]
        if set(tags_of(r["positive"])) - set(tags_of(o["positive"])):
            leaked.append(rid)

    # 元タグが欠落していないこと
    lost = []
    for r in out:
        o = orig_by_id[r["id"]]
        if not set(tags_of(o["positive"])).issubset(set(tags_of(r["positive"]))):
            lost.append(r["id"])

    print("=== 抽出結果 ===")
    print("入力:", SRC_SENSUAL, f"({len(sensual)}行)")
    print("出力:", DST)
    print("残した行:", len(out))
    print("削除した行:", len(dropped))
    print("合計:", len(out) + len(dropped), "(元:", len(sensual), ")")
    print()
    print("=== 整合性チェック ===")
    print("判定の不一致:", "なし" if not mismatch else mismatch)
    print("残した行に追加タグなし:", "なし" if not bad else bad)
    print("削除した行に追加タグあり:", "なし" if not leaked else leaked)
    print("元タグの欠落:", "なし" if not lost else lost)
    print("列構成一致:", fields_o == fields_s)
    print()
    print("=== 残したID（先頭10件） ===")
    print(", ".join(r["id"] for r in out[:10]))
    print()
    print("=== 残した行の例 ===")
    for r in out[:3]:
        o = orig_by_id[r["id"]]
        diff = sorted(set(tags_of(r["positive"])) - set(tags_of(o["positive"])))
        print(f"  {r['id']}: + {', '.join(diff)}")


if __name__ == "__main__":
    main()
