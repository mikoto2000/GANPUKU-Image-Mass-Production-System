"""
select_images.py — 生成画像の自動選別

output\\raw 配下の画像を走査し、以下の指標で採点して output\\selected へコピーする。

  1. aesthetic  : SigLIP ベースの美的スコア (aesthetic-predictor-v2-5)
  2. anatomy    : DWPose による人体構造の健全性
  3. duplicate  : pHash による重複排除

結果は logs\\select_YYYYMMDD_HHMMSS.csv に全件記録する。
閾値を後から調整できるよう、スコアは必ず残す。

使い方:
    .\\venv\\Scripts\\python.exe scripts\\select_images.py
    .\\venv\\Scripts\\python.exe scripts\\select_images.py --top 0.25 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"F:\ai-image")
RAW_DIR = ROOT / "output" / "raw"
SELECTED_DIR = ROOT / "output" / "selected"
LOG_DIR = ROOT / "logs"
DWPOSE_DIR = ROOT / "models" / "dwpose"

DWPOSE_DET = DWPOSE_DIR / "yolox_l.onnx"
DWPOSE_POSE = DWPOSE_DIR / "dw-ll_ucoco_384.onnx"

PREF_MODEL_PATH = ROOT / "models" / "preference" / "pref_model.npz"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    """合成スコアの重み。合計が 1.0 になる必要はない（正規化して使う）。"""
    aesthetic: float = 0.75
    anatomy: float = 0.15
    detail: float = 0.10
    preference: float = 0.0


@dataclass
class AnatomyConfig:
    """人体構造スコアの調整値。"""
    # キーポイント信頼度の閾値。これ未満は「見えていない」とみなす
    keypoint_conf: float = 0.30
    penalize_symmetry: bool = False
    # 必須キーポイント（OpenPose 18点）
    # 0=鼻 1=首 2=右肩 3=右肘 4=右手首 5=左肩 6=左肘 7=左手首
    # 8=右腰 9=右膝 10=右足首 11=左腰 12=左膝 13=左足首 14=右目 15=左目 16=右耳 17=左耳
    #
    # 首と両肩だけを必須にする。腰・脚はバストアップや膝上構図で
    # 正当に写らないため、欠落を破綻とみなしてはいけない。
    core_keypoints: tuple[int, ...] = (1, 2, 5)
    # この点数以上が見えていれば「構図として十分」とみなす。
    # 全身を要求しないことで、バストアップも公平に評価する。
    sufficient_keypoints: int = 6
    # 左右の対称性を評価するペア
    symmetry_pairs: tuple[tuple[int, int], ...] = (
        (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13),
    )
    # 骨格長の比率がこの範囲を外れたら破綻とみなす
    limb_ratio_min: float = 0.55
    limb_ratio_max: float = 1.80


@dataclass
class ImageScore:
    path: str
    filename: str
    width: int = 0
    height: int = 0
    aesthetic_raw: float = 0.0
    aesthetic_norm: float = 0.0
    anatomy_raw: float = 0.0
    anatomy_norm: float = 0.0
    detail_raw: float = 0.0
    detail_norm: float = 0.0
    preference_raw: float = 0.0
    preference_norm: float = 0.0
    total: float = 0.0
    phash: str = ""
    is_duplicate: bool = False
    duplicate_of: str = ""
    person_detected: bool = False
    keypoints_visible: int = 0
    anatomy_notes: list[str] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# 画像読み込み・メタデータ
# ---------------------------------------------------------------------------

def load_image(path: Path):
    """PIL で画像を開き、RGB の numpy 配列とサイズを返す。"""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im)
        return arr, im.size


def read_png_metadata(path: Path) -> dict[str, Any]:
    """PNG の tEXt/iTXt チャンクから生成パラメータを取り出す。

    ComfyUI は 'prompt' と 'workflow' に JSON を埋める。
    壊れていても選別は続行できるよう、失敗は空 dict で返す。
    """
    from PIL import Image

    out: dict[str, Any] = {}
    try:
        with Image.open(path) as im:
            info = getattr(im, "info", {}) or {}
            for key in ("prompt", "workflow", "parameters", "Comment"):
                raw = info.get(key)
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    out[key] = json.loads(raw)
                except Exception:
                    out[key] = raw
    except Exception as e:
        out["__error__"] = repr(e)
    return out


# ---------------------------------------------------------------------------
# 指標 1: 美的スコア
# ---------------------------------------------------------------------------

class AestheticScorer:
    """aesthetic-predictor-v2-5 (SigLIP) による美的スコア。

    出力はおおむね 1〜10 程度。学習データの分布に依存するため、
    絶対値ではなくバッチ内の相対順位として使うのが安全。
    """

    def __init__(self, device: str = "cuda"):
        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        self.torch = torch
        self.device = device
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).to(device).eval()

    def score(self, rgb: np.ndarray) -> float:
        from PIL import Image

        torch = self.torch
        img = Image.fromarray(rgb)
        pv = self.preprocessor(images=img, return_tensors="pt").pixel_values
        pv = pv.to(torch.bfloat16).to(self.device)
        with torch.inference_mode():
            out = self.model(pv).logits.squeeze().float().cpu().numpy()
        return float(out)

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        """好みモデル用の SigLIP 埋め込み (1152,) を返す。

        美的スコアと同じモデルを使うため、追加のロードは発生しない。
        """
        from PIL import Image

        torch = self.torch
        img = Image.fromarray(rgb)
        pv = self.preprocessor(images=img, return_tensors="pt").pixel_values
        pv = pv.to(torch.bfloat16).to(self.device)
        with torch.inference_mode():
            emb = self.model.embeddings(pixel_values=pv)
            enc = self.model.encoder(inputs_embeds=emb)
            h = self.model.post_layernorm(enc.last_hidden_state)
            v = h.mean(dim=1)
        return v.squeeze(0).float().cpu().numpy()


# ---------------------------------------------------------------------------
# 指標 2: 人体構造
# ---------------------------------------------------------------------------

class AnatomyScorer:
    """DWPose のキーポイントから人体構造の破綻度を推定する。

    評価する観点:
      - 人物が検出できたか
      - 体幹の必須キーポイントが信頼できるか
      - 左右の対応キーポイントがバランスしているか
      - 上腕/前腕、大腿/下腿の長さ比が不自然でないか

    これは「破綻の検出」であって「正しさの保証」ではない。
    ポーズが極端な場合（大振りな動作、強い遠近）は誤検出しうる。
    """

    def __init__(self, cfg: AnatomyConfig | None = None, device: str = "cuda"):
        from controlnet_dwpose import DWposeDetector

        self.cfg = cfg or AnatomyConfig()
        if not DWPOSE_DET.exists() or not DWPOSE_POSE.exists():
            raise FileNotFoundError(
                "DWPose の ONNX モデルが見つからない: "
                + str(DWPOSE_DET) + " / " + str(DWPOSE_POSE)
            )

        # onnxruntime が CUDA プロバイダを実際に掴めたかを確認する。
        # 掴めていなければ CPU に落とす（CUDA 版の DLL 不足で起きる）。
        self.device = device
        if device != "cpu":
            try:
                import onnxruntime as ort

                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
                probe = ort.InferenceSession(
                    str(DWPOSE_POSE), providers=["CUDAExecutionProvider"]
                )
                if "CUDAExecutionProvider" not in probe.get_providers():
                    print("CUDA プロバイダを利用できないため CPU で実行する")
                    self.device = "cpu"
            except Exception as e:
                print("CUDA 初期化に失敗したため CPU で実行する: " + repr(e)[:200])
                self.device = "cpu"

        self.detector = DWposeDetector(
            str(DWPOSE_DET), str(DWPOSE_POSE), device=self.device
        )

    def score(self, rgb: np.ndarray) -> tuple[float, bool, int, list[str]]:
        """戻り値: (スコア 0..1, 人物検出有無, 可視キーポイント数, 注記)"""
        cfg = self.cfg
        notes: list[str] = []

        try:
            pose = self.detector(rgb)
        except Exception as e:
            return 0.0, False, 0, ["pose_error: " + repr(e)]

        bodies = pose.get("bodies", {})
        candidate = bodies.get("candidate")
        scores = bodies.get("score")

        if candidate is None or scores is None:
            return 0.0, False, 0, ["no_pose_output"]

        # DWposeDetector の戻り値は candidate=(N*18, 2), score=(N, 18)。
        # candidate を (N, 18, 2) に戻してから人物単位で扱う。
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        n_person = int(scores.shape[0])
        if n_person == 0:
            return 0.0, False, 0, ["no_person"]

        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.shape[0] != n_person * 18:
            return 0.0, False, 0, [
                "unexpected_shape:" + str(candidate.shape) + "/" + str(scores.shape)
            ]
        kp_all = candidate.reshape(n_person, 18, 2)

        # 最も信頼度の高い人物を採用
        person_idx = int(np.argmax(np.mean(scores, axis=1)))
        kp = kp_all[person_idx]        # (18, 2)
        conf = scores[person_idx]      # (18,)

        # DWPose は人物が写っていなくても 18 点を返すため、
        # 信頼度そのもので「人物がいるか」を判定する。
        person_detected = bool(float(np.max(conf)) >= cfg.keypoint_conf)

        visible = conf >= cfg.keypoint_conf
        n_visible = int(visible.sum())

        if n_visible == 0:
            return 0.0, person_detected, 0, ["all_keypoints_low_conf"]

        penalties: list[float] = []

        # --- 体幹の必須キーポイント ---
        core_missing = [i for i in cfg.core_keypoints if not visible[i]]
        if core_missing:
            notes.append("core_missing:" + ",".join(str(i) for i in core_missing))
            penalties.append(0.30 * len(core_missing) / len(cfg.core_keypoints))

        # --- 左右対称性 ---
        # 肩幅を基準スケールにする
        if visible[2] and visible[5]:
            shoulder_w = float(np.linalg.norm(kp[2] - kp[5]))
        else:
            shoulder_w = 0.0

        if cfg.penalize_symmetry and shoulder_w > 1e-6:
            asym: list[float] = []
            for a, b in cfg.symmetry_pairs:
                if not (visible[a] and visible[b]):
                    continue
                # 体幹中心線（首と腰中点）からの距離の左右差
                if visible[1] and visible[8] and visible[11]:
                    hip_mid = (kp[8] + kp[11]) / 2.0
                    axis = kp[1] - hip_mid
                    norm = float(np.linalg.norm(axis))
                    if norm < 1e-6:
                        continue
                    axis = axis / norm
                    # 中心線に直交する向きへの射影
                    perp = np.array([-axis[1], axis[0]])
                    da = abs(float(np.dot(kp[a] - hip_mid, perp)))
                    db = abs(float(np.dot(kp[b] - hip_mid, perp)))
                    if max(da, db) > 1e-6:
                        asym.append(abs(da - db) / max(da, db))
            if asym:
                mean_asym = float(np.mean(asym))
                if mean_asym > 0.35:
                    notes.append("asymmetry:" + str(round(mean_asym, 2)))
                penalties.append(min(0.25, mean_asym * 0.5))

        # --- 四肢の長さ比 ---
        def seg_len(a: int, b: int) -> float | None:
            if visible[a] and visible[b]:
                return float(np.linalg.norm(kp[a] - kp[b]))
            return None

        for upper, lower, label in (
            ((2, 3), (3, 4), "right_arm"),
            ((5, 6), (6, 7), "left_arm"),
            ((8, 9), (9, 10), "right_leg"),
            ((11, 12), (12, 13), "left_leg"),
        ):
            up = seg_len(*upper)
            lo = seg_len(*lower)
            if up is None or lo is None or up < 1e-6:
                continue
            ratio = lo / up
            if ratio < cfg.limb_ratio_min or ratio > cfg.limb_ratio_max:
                notes.append(label + "_ratio:" + str(round(ratio, 2)))
                penalties.append(0.15)

        # --- 信頼度の平均 ---
        conf_score = float(np.mean(conf[visible]))

        # 構図として十分な点数が取れているか。
        # 全身を要求するとバストアップが不当に落ちるため、
        # sufficient_keypoints までは満点として扱う。
        coverage = min(1.0, n_visible / float(cfg.sufficient_keypoints))

        raw = 0.60 * conf_score + 0.40 * coverage
        raw = max(0.0, raw - float(np.sum(penalties)))
        return float(min(1.0, raw)), person_detected, n_visible, notes


# ---------------------------------------------------------------------------
# 指標 3: ディテール（ラプラシアン分散）
# ---------------------------------------------------------------------------

def detail_score(rgb: np.ndarray) -> float:
    """エッジ密度によるディテール指標 (0..1)。

    ラプラシアン分散はノイズ画像で極端に高くなるため使わない。
    Canny のエッジ密度が「適正帯」に入っているかで評価する。

      - ほぼ平坦（密度 < 2%）      : 0 〜 0.5 で立ち上げ
      - 適正（2%〜20%）            : 0.5 〜 1.0 で連続評価
      - 過剰（ノイズ・ざらつき）    : 減点

    適正帯を一律 1.0 にすると、実際の生成画像はほぼ全部が帯内に入るため
    指標が全件同点になり識別力が失われる。帯内でも密度に応じて差を付ける。

    戻り値は 0..1 に収まるので、バッチ内正規化は不要。
    """
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    density = float(np.count_nonzero(edges)) / float(edges.size)

    low, high = 0.02, 0.20
    if density < low:
        return float(max(0.0, density / low * 0.5))
    if density <= high:
        # 帯内を 0.5..1.0 に線形写像する。
        return float(0.5 + 0.5 * (density - low) / (high - low))
    return float(max(0.0, 1.0 - (density - high) / 0.30))


# ---------------------------------------------------------------------------
# 指標 4: 重複排除
# ---------------------------------------------------------------------------

def compute_phash(rgb: np.ndarray) -> str:
    import imagehash
    from PIL import Image

    return str(imagehash.phash(Image.fromarray(rgb)))


# pHash がこの値になる画像は「ほぼ一様な画像」で、内容の区別ができない。
# 平坦画像どうしを重複扱いすると別画像まで落ちるため、重複判定から除外する。
DEGENERATE_PHASHES = {"0000000000000000", "8000000000000000", "ffffffffffffffff"}


def is_degenerate_phash(h: str) -> bool:
    return h in DEGENERATE_PHASHES


def hamming(a: str, b: str) -> int:
    import imagehash

    return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------

def minmax_normalize(values: list[float]) -> list[float]:
    """外れ値に強い 5-95 パーセンタイル正規化。"""
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    out = (arr - lo) / (hi - lo)
    return [float(min(1.0, max(0.0, v))) for v in out]


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成画像の自動選別")
    ap.add_argument("--raw", default=str(RAW_DIR), help="入力ディレクトリ")
    ap.add_argument("--out", default=str(SELECTED_DIR), help="出力ディレクトリ")
    ap.add_argument("--top", type=float, default=0.30,
                    help="上位何割を残すか (0..1)")
    ap.add_argument("--dup-threshold", type=int, default=6,
                    help="pHash のハミング距離がこれ以下なら重複とみなす")
    ap.add_argument("--w-aesthetic", type=float, default=0.75)
    ap.add_argument("--w-anatomy", type=float, default=0.15)
    ap.add_argument("--w-detail", type=float, default=0.10)
    ap.add_argument("--w-preference", type=float, default=0.0,
                    help="好みモデルの重み。0 なら好みモデルを使わない")
    ap.add_argument("--pref-model", default=str(PREF_MODEL_PATH),
                    help="好みモデルのパス")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-person", action="store_true",
                    help="人物が検出できない画像を選別対象から除外する")
    ap.add_argument("--dry-run", action="store_true",
                    help="コピーせずスコアだけ出す")
    ap.add_argument("--limit", type=int, default=0,
                    help="先頭 N 枚だけ処理 (0=全部)")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    images = collect_images(raw_dir)
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        print("対象画像なし: " + str(raw_dir))
        return 0

    print("対象 " + str(len(images)) + " 枚")

    weights = Weights(args.w_aesthetic, args.w_anatomy, args.w_detail, args.w_preference)
    wsum = weights.aesthetic + weights.anatomy + weights.detail + weights.preference
    if wsum <= 0:
        print("重みの合計が 0")
        return 2

    # --- 好みモデル ---
    pref_model = None
    if weights.preference > 0:
        from pref_core import load_model

        pref_model = load_model(Path(args.pref_model))
        if pref_model is None:
            print("好みモデルが見つからない: " + str(args.pref_model))
            print("  先に scripts\\train_preference.py を実行してね")
            return 2
        meta = pref_model["meta"]
        print("好みモデル: favorites " + str(meta.get("n_pos")) + " 枚 / negatives "
              + str(meta.get("n_neg")) + " 枚"
              + "  (LOO 中央値 " + str(meta.get("loo_median_rank")) + " 位)")

    print("美的スコアラを読み込み中...")
    aesthetic = AestheticScorer(device=args.device)

    print("DWPose を読み込み中...")
    anatomy = AnatomyScorer(device=args.device)

    results: list[ImageScore] = []
    embeddings: list[np.ndarray | None] = []

    for i, path in enumerate(images, 1):
        rec = ImageScore(path=str(path), filename=path.name)
        emb = None
        try:
            rgb, (w, h) = load_image(path)
            rec.width, rec.height = w, h
            rec.aesthetic_raw = aesthetic.score(rgb)
            if pref_model is not None:
                emb = aesthetic.embed(rgb)
            a_raw, person, n_kp, notes = anatomy.score(rgb)
            rec.anatomy_raw = a_raw
            rec.person_detected = person
            rec.keypoints_visible = n_kp
            rec.anatomy_notes = notes
            rec.detail_raw = detail_score(rgb)
            rec.phash = compute_phash(rgb)
        except Exception as e:
            rec.error = repr(e)
        results.append(rec)
        embeddings.append(emb)
        print("[" + str(i) + "/" + str(len(images)) + "] " + path.name
              + "  aes=" + str(round(rec.aesthetic_raw, 2))
              + " ana=" + str(round(rec.anatomy_raw, 2))
              + ("  ERR" if rec.error else ""))

    # --- 正規化 ---
    # aesthetic と anatomy はバッチ内の相対順位で正規化する。
    # detail は定義上 0..1 なのでそのまま使う。
    ok = [r for r in results if not r.error]
    aes_norm = minmax_normalize([r.aesthetic_raw for r in ok])
    ana_norm = minmax_normalize([r.anatomy_raw for r in ok])

    # 好みスコアは絶対値（0..1）なのでバッチ内正規化しない。
    # バッチ依存にすると「同じ画像でも一緒に処理する画像次第でスコアが変わる」。
    pref_scores: dict[str, float] = {}
    if pref_model is not None:
        from pref_core import score_embeddings

        ok_embs = [e for r, e in zip(results, embeddings) if not r.error and e is not None]
        if ok_embs:
            arr = score_embeddings(np.stack(ok_embs), pref_model)
            # 埋め込みが取れた画像だけ対応付ける
            idx = 0
            for r, e in zip(results, embeddings):
                if r.error or e is None:
                    continue
                pref_scores[r.path] = float(arr[idx])
                idx += 1

    for r, a, b in zip(ok, aes_norm, ana_norm):
        r.aesthetic_norm = a
        r.anatomy_norm = b
        r.detail_norm = r.detail_raw
        p = pref_scores.get(r.path, 0.0)
        r.preference_raw = p
        r.preference_norm = p
        r.total = (
            weights.aesthetic * a
            + weights.anatomy * b
            + weights.detail * r.detail_raw
            + weights.preference * p
        ) / wsum

    # --- 重複排除 ---
    # スコアの高い順に見て、既に採用した画像と近ければ重複扱い。
    # 退化ハッシュ（一様な画像）は比較に使わない。
    ranked = sorted(ok, key=lambda r: r.total, reverse=True)
    kept: list[ImageScore] = []
    for r in ranked:
        if is_degenerate_phash(r.phash):
            kept.append(r)
            continue
        dup_of = ""
        for k in kept:
            if is_degenerate_phash(k.phash):
                continue
            if hamming(r.phash, k.phash) <= args.dup_threshold:
                dup_of = k.filename
                break
        if dup_of:
            r.is_duplicate = True
            r.duplicate_of = dup_of
        else:
            kept.append(r)

    # --- 選別 ---
    # --require-person 指定時は、人物が検出できた画像だけを候補にする。
    pool = kept
    if args.require_person:
        pool = [r for r in kept if r.person_detected]
        dropped = len(kept) - len(pool)
        if dropped:
            print("人物未検出のため除外: " + str(dropped) + " 枚")

    n_keep = max(1, int(round(len(pool) * args.top))) if pool else 0
    selected = pool[:n_keep]

    # --- 出力 ---
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / ("select_" + stamp + ".csv")
    fields = list(asdict(ImageScore("", "")).keys())
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.total, reverse=True):
            row = asdict(r)
            row["anatomy_notes"] = "|".join(r.anatomy_notes)
            writer.writerow(row)

    print("")
    print("スコア記録: " + str(csv_path))
    print("重複除外: " + str(sum(1 for r in ok if r.is_duplicate)) + " 枚")
    print("選別: " + str(len(selected)) + " / " + str(len(ok)) + " 枚")

    if args.dry_run:
        print("dry-run のためコピーなし")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for r in selected:
        src = Path(r.path)
        dst = out_dir / src.name
        if dst.exists():
            dst = out_dir / (src.stem + "_" + stamp + src.suffix)
        shutil.copy2(src, dst)
    print("コピー先: " + str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
