"""CSV の各行 × pose_reference 直下の全画像を生成する専用コマンド。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import time
from pathlib import Path

from PIL import Image, ImageOps

from generate_images import (
    ROOT, INPUT_DIR, LOG_DIR, STYLE_REFERENCE_DIR, DEFAULT_SERVER, SEED_MAX,
    ComfyClient, build_workflow, configure_style_reference, effective_values,
    load_common_prompt, load_prompts, load_workflow,
)


def pose_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"ポーズ画像フォルダがありません: {directory}")
    extensions = Image.registered_extensions()
    images = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: p.name.casefold(),
    )
    if not images:
        raise ValueError(f"ポーズ画像がありません: {directory}")
    for path in images:
        with Image.open(path) as image:
            image.verify()
    return images


def add_pose(workflow: dict, image: str, model: str, strength: float,
             start: float, end: float) -> None:
    """各サンプラーの元の条件を保持し、ControlNet を接続する。"""
    workflow["pose_load"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    workflow["pose_model"] = {
        "class_type": "ControlNetLoader", "inputs": {"control_net_name": model},
    }
    for node_id, node in list(workflow.items()):
        if node.get("class_type") != "KSampler":
            continue
        inputs = node["inputs"]
        apply_id = f"pose_apply_{node_id}"
        workflow[apply_id] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": inputs["positive"], "negative": inputs["negative"],
                "control_net": ["pose_model", 0], "image": ["pose_load", 0],
                "strength": strength, "start_percent": start, "end_percent": end,
            },
        }
        inputs["positive"] = [apply_id, 0]
        inputs["negative"] = [apply_id, 1]


class PoseMaps:
    def __init__(self, directory: Path, mode: str):
        self.directory = directory
        self.mode = mode
        self.detector = None
        self.images = {}
        self.maps = {}

    def get(self, source: Path, width: int, height: int) -> str:
        key = (source, width, height)
        if key in self.maps:
            return self.maps[key]
        if source not in self.images:
            with Image.open(source) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1536, 1536))
            if self.mode == "image":
                import numpy as np
                from controlnet_dwpose import DWposeDetector, draw_pose

                if self.detector is None:
                    models = ROOT / "models" / "dwpose"
                    self.detector = DWposeDetector(
                        str(models / "yolox_l.onnx"),
                        str(models / "dw-ll_ucoco_384.onnx"), device="cpu",
                    )
                pose = self.detector(np.array(image))
                scores = pose["bodies"]["score"]
                if not scores.size or not ((scores > 0.3).sum(axis=1) >= 4).any():
                    raise ValueError(f"骨格を検出できません: {source}")
                # draw_pose は RGB の CHW 配列を返す。
                pixels = draw_pose(pose, image.height, image.width, ref_w=1024)
                image = Image.fromarray(pixels.transpose(1, 2, 0))
            self.images[source] = image
        image = ImageOps.pad(
            self.images[source], (width, height),
            method=Image.Resampling.LANCZOS, color="black",
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"pose_{len(self.maps):04d}_{width}x{height}.png"
        image.save(destination)
        name = destination.relative_to(INPUT_DIR).as_posix()
        self.maps[key] = name
        return name

    def release(self):
        if self.detector is not None:
            self.detector.release_memory()
            self.detector = None


def safe_name(value: str) -> str:
    return re.sub(r'[^\w.-]+', "_", value).strip(". ")[:80] or "image"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, nargs="+", required=True)
    ap.add_argument("--pose-dir", type=Path, default=INPUT_DIR / "pose_reference")
    ap.add_argument("--pose-mode", choices=["image", "openpose"], default="image",
                    help="image: 人物画像から骨格抽出 / openpose: 作成済み骨格画像")
    ap.add_argument("--workflow", type=Path,
                    default=ROOT / "workflows" / "sdxl_dreamwash_anmi_api.json")
    ap.add_argument("--controlnet", default="xinsir-openpose-sdxl-1.0.safetensors")
    ap.add_argument("--pose-strength", type=float, default=0.8)
    ap.add_argument("--pose-start", type=float, default=0.0)
    ap.add_argument("--pose-end", type=float, default=1.0)
    ap.add_argument("--ref-dir", default=str(STYLE_REFERENCE_DIR))
    ap.add_argument("--no-style-reference", action="store_true")
    ap.add_argument("--ipadapter-weight", type=float, default=-1.0)
    ap.add_argument("--common-positive", type=Path)
    ap.add_argument("--common-negative", type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="CSV 行数の上限（各行に全ポーズを適用）")
    ap.add_argument("--batch", type=int, default=0, help="各組み合わせの枚数。省略時は CSV / ワークフロー値")
    ap.add_argument("--seed", type=int, default=-1)
    for name in ("width", "height", "steps"):
        ap.add_argument("--" + name, type=int, default=0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="組み合わせ確認のみ。抽出・書き込み・API 通信なし")
    args = ap.parse_args(argv)
    if not (0 <= args.pose_start < args.pose_end <= 1) or not (0 < args.pose_strength <= 10):
        ap.error("pose は 0 <= start < end <= 1、0 < strength <= 10 を指定してください")
    if min(args.start, args.limit, args.batch, args.width, args.height, args.steps) < 0:
        ap.error("start/limit/batch/width/height/steps は 0 以上にしてください")
    if args.seed < -1 or args.seed > SEED_MAX:
        ap.error("seed は -1 または 0〜2**64-1 を指定してください")
    if args.timeout <= 0 or args.poll <= 0:
        ap.error("timeout/poll は正数にしてください")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        poses = pose_images(args.pose_dir)
        rows = [(p, row) for p in args.prompts for row in load_prompts(p)][args.start:]
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            raise ValueError("プロンプトが 0 件です")
        base = load_workflow(args.workflow)
        if base.get("4", {}).get("class_type") != "EmptyLatentImage":
            raise ValueError("EmptyLatentImage（ノード4）を持つ txt2img ワークフローを指定してください")
        overrides = {name: getattr(args, name) for name in
                     ("width", "height", "steps", "cfg", "batch", "ipadapter_weight")}
        for name in ("common_positive", "common_negative"):
            path = getattr(args, name)
            overrides[name] = load_common_prompt(path) if path else ""
        references = configure_style_reference(
            base, image="", directory=args.ref_dir, disabled=args.no_style_reference,
            dry_run=args.dry_run,
        )
        jobs = []
        total_images = 0
        for row_index, (csv_path, row) in enumerate(rows, 1):
            # 同一行では同じシードを使い、ポーズ間で比較しやすくする。
            seed = args.seed if args.seed >= 0 else random.randint(0, SEED_MAX)
            for pose_index, pose in enumerate(poses, 1):
                wf = build_workflow(base, row, seed, overrides)
                size = wf["4"]["inputs"]
                if min(size["width"], size["height"], size["batch_size"]) <= 0:
                    raise ValueError("解像度とバッチ枚数は正数にしてください")
                prefix = (f"{safe_name(csv_path.stem)}/poses/{stamp}/"
                          f"{row_index:04d}_{safe_name(row.id)}_{pose_index:04d}_{safe_name(pose.stem)}")
                for node in wf.values():
                    if node.get("class_type") == "SaveImage":
                        node["inputs"]["filename_prefix"] = prefix
                jobs.append((csv_path, row, pose, seed, wf))
                total_images += size["batch_size"]
        print(f"プロンプト {len(rows)} 行 × ポーズ {len(poses)} 枚 = {len(jobs)} 組、生成 {total_images} 枚")
        if args.dry_run:
            for csv_path, row, pose, seed, wf in jobs:
                add_pose(wf, "<prepared pose map>", args.controlnet,
                         args.pose_strength, args.pose_start, args.pose_end)
                print(json.dumps({"csv": str(csv_path), "id": row.id, "pose": str(pose),
                                  "seed": seed, "workflow": wf}, ensure_ascii=False))
            print("dry-run: 骨格抽出・ファイル書き込み・生成なし")
            return 0

        client = ComfyClient(args.server)
        ready, message = client.check_ready()
        print(message)
        if not ready:
            return 3
        maps = PoseMaps(INPUT_DIR / "pose_maps" / stamp, args.pose_mode)
        try:
            # 全画像の抽出を先に行い、検出失敗時は生成を開始しない。
            for _, _, pose, _, wf in jobs:
                size = wf["4"]["inputs"]
                image = maps.get(pose, size["width"], size["height"])
                add_pose(wf, image, args.controlnet, args.pose_strength, args.pose_start, args.pose_end)
        finally:
            maps.release()
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        print(f"準備に失敗: {exc}")
        return 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"poses_{stamp}.csv"
    failures = 0
    fields = ["source_csv", "row_id", "pose_image", "pose_map", "seed", "style_images",
              "pose_strength", "pose_start", "pose_end", "prompt_id", "files", "seconds",
              "status", "error", "positive", "negative", "settings"]
    with log_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (csv_path, row, pose, seed, wf) in enumerate(jobs, 1):
            record = dict(source_csv=str(csv_path.resolve()), row_id=row.id,
                          pose_image=str(pose.resolve()), pose_map=wf["pose_load"]["inputs"]["image"],
                          seed=seed, style_images="|".join(references),
                          pose_strength=args.pose_strength, pose_start=args.pose_start, pose_end=args.pose_end,
                          positive=wf["2"]["inputs"]["text"], negative=wf["3"]["inputs"]["text"],
                          settings=json.dumps(effective_values(wf, references), ensure_ascii=False))
            started = time.monotonic()
            try:
                record["prompt_id"] = client.queue_prompt(wf)
                entry = client.wait(record["prompt_id"], poll=args.poll, timeout=args.timeout)
                record.update(files="|".join(client.output_files(entry)), status="OK")
            except Exception as exc:
                record.update(status="FAIL", error=repr(exc))
                failures += 1
            record["seconds"] = round(time.monotonic() - started, 2)
            writer.writerow(record)
            stream.flush()
            print(f"[{index}/{len(jobs)}] id={row.id} pose={pose.name} {record['status']} "
                  f"{record.get('error', '')}", flush=True)
            if record["status"] == "FAIL" and args.stop_on_error:
                break
    print(f"ログ: {log_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
