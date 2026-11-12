"""train_preference.py — favorites から好みモデルを学習する

favorites ディレクトリの画像を「好み」、それ以外の生成画像を「好みでない」
として、SigLIP 埋め込み上の好みモデルを作る。

重要な設計:
  - favorites と同一内容の画像は negatives から除外する（データ漏洩防止）。
    コピーして置いた場合、同じ画像が pos と neg の両方に入って AUC が
    不当に高くなるのを防ぐ。
  - 少数データでは centroid 方式が安定するため、LogReg は補助として保存し、
    枚数に応じて重みを移行する（pref_core 側で処理）。
  - LOO（leave-one-out）で「favorites の1枚を抜いて採点したとき何位になるか」
    を必ず報告する。これが実際の汎化の目安になる。

使い方:
    .\\venv\\Scripts\\python.exe scripts\\train_preference.py
    .\\venv\\Scripts\\python.exe scripts\\train_preference.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

ROOT = Path(r"F:\ai-image")
FAVORITES_DIR = ROOT / "favorites"
DEFAULT_NEGATIVE_DIRS = [ROOT / "output" / "raw", ROOT / "output" / "selected"]
MODEL_PATH = ROOT / "models" / "preference" / "pref_model.npz"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def content_hash(path: Path) -> str:
    """画素内容のハッシュ。コピーや再エンコードの差を吸収するため
    デコード後の RGB バイト列から計算する。"""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        return hashlib.sha256(im.tobytes()).hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="favorites から好みモデルを学習")
    ap.add_argument("--favorites", default=str(FAVORITES_DIR))
    ap.add_argument("--negatives", nargs="*", default=[str(p) for p in DEFAULT_NEGATIVE_DIRS],
                    help="負例として使うディレクトリ")
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="モデルを保存せず評価だけ行う")
    args = ap.parse_args(argv)

    fav_dir = Path(args.favorites)
    fav_paths = collect_images(fav_dir)
    if not fav_paths:
        print("favorites に画像がない: " + str(fav_dir))
        print("  " + str(fav_dir) + " に好みの画像を入れてから再実行してね")
        return 2

    print("favorites: " + str(len(fav_paths)) + " 枚")

    # --- 負例の収集 ---
    fav_hashes = {content_hash(p) for p in fav_paths}
    neg_paths: list[Path] = []
    excluded = 0
    for d in args.negatives:
        for p in collect_images(Path(d)):
            if content_hash(p) in fav_hashes:
                excluded += 1
                continue
            neg_paths.append(p)

    if excluded:
        print("favorites と同一内容のため負例から除外: " + str(excluded) + " 枚")
    print("negatives: " + str(len(neg_paths)) + " 枚")

    if len(neg_paths) < 5:
        print("負例が少なすぎる（5枚未満）。output\\raw を増やしてから再実行してね")
        return 2

    # --- 埋め込み ---
    from pref_embed import SiglipEmbedder

    print("SigLIP を読み込み中...")
    embedder = SiglipEmbedder(device=args.device)

    print("埋め込みを計算中...")
    fav_emb = []
    for i, p in enumerate(fav_paths, 1):
        fav_emb.append(embedder.embed(load_rgb(p)))
        print("  fav [" + str(i) + "/" + str(len(fav_paths)) + "] " + p.name)

    neg_emb = []
    for i, p in enumerate(neg_paths, 1):
        neg_emb.append(embedder.embed(load_rgb(p)))
        if i % 20 == 0 or i == len(neg_paths):
            print("  neg [" + str(i) + "/" + str(len(neg_paths)) + "]")

    X_pos = np.stack(fav_emb)
    X_neg = np.stack(neg_emb)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    mean = X.mean(axis=0)
    Xc = X - mean

    # --- centroid ---
    centroid = Xc[: len(X_pos)].mean(axis=0)

    # --- LogReg ---
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(Xc, y)
    coef = clf.coef_.ravel()
    intercept = clf.intercept_.ravel()

    # --- 評価 ---
    print("")
    print("=== 評価 ===")

    # centroid による分離
    def centroid_scores(idx_subset: list[int]) -> np.ndarray:
        c = Xc[idx_subset].mean(axis=0)
        n = float(np.linalg.norm(c))
        if n < 1e-9:
            return np.zeros(len(Xc))
        return Xc @ (c / n)

    all_idx = list(range(len(X_pos)))
    sims = centroid_scores(all_idx)
    pos_sims = sims[: len(X_pos)]
    neg_sims = sims[len(X_pos):]
    print("centroid  pos mean=" + str(round(float(pos_sims.mean()), 3))
          + "  neg mean=" + str(round(float(neg_sims.mean()), 3))
          + "  separation=" + str(round(float(pos_sims.mean() - neg_sims.mean()), 3)))

    # LOO: favorites の1枚を抜いて採点し、順位を見る
    ranks: list[int] = []
    for k in all_idx:
        others = [j for j in all_idx if j != k]
        s = centroid_scores(others)
        order = np.argsort(-s)
        ranks.append(int(np.where(order == k)[0][0]) + 1)
    n_total = len(X)
    print("LOO rank (of " + str(n_total) + "): median=" + str(int(np.median(ranks)))
          + "  mean=" + str(round(float(np.mean(ranks)), 1))
          + "  worst=" + str(max(ranks)))
    print("  → favorites の1枚を未知としても、中央値で上位 "
          + str(int(np.median(ranks))) + " 位に入る")

    # LogReg の交差検証（少数では不安定なので参考値）
    if len(X_pos) >= 5:
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        n_splits = min(5, len(X_pos))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        auc = cross_val_score(clf, Xc, y, cv=skf, scoring="roc_auc")
        print("LogReg " + str(n_splits) + "-fold AUC: " + str(np.round(auc, 3))
              + "  mean=" + str(round(float(auc.mean()), 3)))
        if float(auc.min()) < 0.6:
            print("  ※ fold によって大きく崩れる。少数データでは centroid 主体が安全")
        if float(auc.mean()) >= 0.99:
            print("  ※ AUC が 1.0 に張り付いている。好みではなく出所（画風・形式）を")
            print("    分離している疑いが強い。favorites と negatives を同じ生成元から")
            print("    揃えるか、negatives を『自分の生成物のうち好みでないもの』に限定してね")

    # --- ドメイン交絡の検査 ---
    # favorites と negatives が別の生成元だと、埋め込みは好みではなく
    # 出所（形式・解像度・画風）を分離してしまう。実際に使う前に必ず確認する。
    from PIL import Image

    def _domain(paths: list[Path]) -> tuple[dict, dict]:
        fmts: dict[str, int] = {}
        sizes: dict[tuple, int] = {}
        for p in paths:
            try:
                with Image.open(p) as im:
                    fmts[im.format] = fmts.get(im.format, 0) + 1
                    sizes[im.size] = sizes.get(im.size, 0) + 1
            except Exception:
                pass
        return fmts, sizes

    ff, fs = _domain(fav_paths)
    nf, ns = _domain(neg_paths)
    print("")
    print("=== ドメイン検査 ===")
    print("favorites 形式: " + str(ff)
          + "  主なサイズ: " + str(sorted(fs.items(), key=lambda x: -x[1])[:2]))
    print("negatives 形式: " + str(nf)
          + "  主なサイズ: " + str(sorted(ns.items(), key=lambda x: -x[1])[:2]))
    if set(ff) != set(nf) or set(fs) != set(ns):
        print("  ※ 形式またはサイズが一致しない。埋め込みが好みではなく出所を")
        print("    分離している可能性がある。スコアの絶対値は信用せず、順位だけを見てね")

    # --- 保存 ---
    meta = {
        "n_pos": int(len(X_pos)),
        "n_neg": int(len(X_neg)),
        "loo_median_rank": int(np.median(ranks)),
        "loo_mean_rank": float(np.mean(ranks)),
        "centroid_separation": float(pos_sims.mean() - neg_sims.mean()),
        "embedding_dim": int(X.shape[1]),
        "favorites_dir": str(fav_dir),
        "negatives": [str(d) for d in args.negatives],
    }

    if args.dry_run:
        print("")
        print("dry-run のため保存しない")
        return 0

    from pref_core import save_model

    save_model(Path(args.out), mean, centroid, coef, intercept, meta)
    print("")
    print("モデルを保存: " + str(args.out))
    print("  favorites " + str(len(X_pos)) + " 枚 / negatives " + str(len(X_neg)) + " 枚")
    return 0


if __name__ == "__main__":
    sys.exit(main())
