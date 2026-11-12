"""pref_core.py — 好みモデルの保存・読込・採点（numpy のみ）

train_preference.py が学習して保存し、select_images.py が読み込んで使う。
重い依存を持たないので、どちらからでも安全に import できる。

スコアの設計:
  - centroid 方式: 中心化した埋め込みと favorites 重心のコサイン類似度。
    少数データ（10枚程度）でも安定する。
  - LogReg 方式: 中心化埋め込みに対するロジスティック回帰。
    データが増えるほど強いが、少数では fold によって崩れる。
  - ハイブリッド: favorites 数に応じて重みを連続的に移行する。
    10枚で centroid 100%、30枚で LogReg 100%。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_DIR = Path(r"F:\ai-image\models\preference")
MODEL_PATH = MODEL_DIR / "pref_model.npz"

# この枚数を超えると LogReg の重みが立ち上がり始める
LR_RAMP_START = 10
LR_RAMP_END = 30


def save_model(
    path: Path,
    mean: np.ndarray,
    centroid: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=np.asarray(mean, dtype=np.float64),
        centroid=np.asarray(centroid, dtype=np.float64),
        coef=np.asarray(coef, dtype=np.float64),
        intercept=np.asarray(intercept, dtype=np.float64),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )


def load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    return {
        "mean": d["mean"],
        "centroid": d["centroid"],
        "coef": d["coef"],
        "intercept": d["intercept"],
        "meta": json.loads(str(d["meta"])),
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def score_embeddings(emb: np.ndarray, model: dict) -> np.ndarray:
    """生の埋め込み (N, D) から好みスコア (N,) を 0..1 で返す。"""
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    centered = emb - model["mean"]

    # --- centroid 方式 ---
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    unit = centered / norms
    cen = model["centroid"]
    cen_norm = float(np.linalg.norm(cen))
    if cen_norm < 1e-9:
        cos = np.zeros(len(unit))
    else:
        cos = unit @ (cen / cen_norm)
    centroid_score = (cos + 1.0) / 2.0

    # --- LogReg 方式 ---
    coef = model["coef"]
    if coef.size == 0:
        return centroid_score

    z = centered @ coef + float(model["intercept"][0])
    lr_score = _sigmoid(z)

    n_pos = int(model["meta"].get("n_pos", 0))
    w = (n_pos - LR_RAMP_START) / float(LR_RAMP_END - LR_RAMP_START)
    w = float(min(1.0, max(0.0, w)))
    return w * lr_score + (1.0 - w) * centroid_score
