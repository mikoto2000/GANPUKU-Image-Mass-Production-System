"""Generate localized retouch candidates for a ComfyUI PNG with visible bare toes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from controlnet_dwpose import Wholebody
from PIL import Image, ImageDraw, ImageFilter

from generate_images import DEFAULT_SERVER, INPUT_DIR, RAW_DIR, ROOT, ComfyClient, load_workflow
from refine_inpaint import build_inpaint, run_inpaint, stage_crop


@dataclass
class ToeRegion:
    box: tuple[int, int, int, int]
    noise: Image.Image
    blend: Image.Image


def toe_region(points, scores, size):
    if not np.all(scores >= 0.45) or not np.isfinite(points).all():
        return None
    if np.any(points < 0) or np.any(points >= size):
        return None
    big_toe, small_toe, heel = points
    toes = (big_toe + small_toe) / 2
    axis = toes - heel
    length = np.linalg.norm(axis)
    if length < 38:
        return None
    axis /= length
    across = np.array([-axis[1], axis[0]])
    span = abs((big_toe - small_toe) @ across)
    if span < max(15, length * 0.18):
        return None

    # Leave space outside the silhouette for new toe contours.
    center = toes - axis * 0.08 * length
    radii = np.array([0.36 * length, max(0.31 * length, 0.76 * span)])
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    outline = (center + np.cos(theta)[:, None] * radii[0] * axis
               + np.sin(theta)[:, None] * radii[1] * across)
    feather = max(2, round(length * 0.026))
    if np.any(outline.min(axis=0) < 3 * feather) or np.any(outline.max(axis=0) >= np.array(size) - 3 * feather):
        return None
    context = round(max(length * 0.36, 5 * feather))
    left, top = np.maximum(0, np.floor(outline.min(axis=0) - context)).astype(int)
    right, bottom = np.minimum(size, np.ceil(outline.max(axis=0) + context)).astype(int)
    mask = Image.new("L", (right - left, bottom - top))
    polygon = np.rint(outline - (left, top)).astype(int).tolist()
    ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon], fill=255)
    return ToeRegion(
        (int(left), int(top), int(right), int(bottom)),
        mask.filter(ImageFilter.MaxFilter(2 * feather + 1)),
        mask.filter(ImageFilter.GaussianBlur(feather)),
    )


def detect_toes(image, detector):
    points, scores = detector(np.asarray(image)[:, :, ::-1].copy())
    # DWPose Wholebody inserts a neck point before the COCO-WholeBody feet.
    regions = [
        toe_region(person[start:start + 3], confidence[start:start + 3], image.size)
        for person, confidence in zip(points, scores)
        for start in (18, 21)
    ]
    return [region for region in regions if region is not None]


def build_candidate(workflow, source, staged, seed, denoise):
    positive = (
        "finished anime illustration, delicate light-colored outlines, soft blended shading, "
        "one bare foot, (five distinct natural toes:1.4), anatomically correct foot, "
        "natural toe spacing, coherent foot silhouette, soft even skin tone"
    )
    negative = (
        "four toes, six toes, extra toes, missing toes, fused toes, malformed foot, "
        "extra foot, shoe, sock, nail polish, colored nails, saturated colors, "
        "heavy shading, thick outline, photorealistic, 3d, blurry, lowres"
    )
    graph = {}
    previous = None
    for index, (staging, region) in enumerate(staged, 1):
        folder = f"foot_{index}"
        part = build_inpaint(
            workflow, source, staging, region.box[:2], region.blend.size,
            folder=folder, positive=positive, negative=negative, denoise=denoise,
        )
        part[f"{folder}_sampled"]["inputs"]["seed"] = (seed + index - 1) % (2**64)
        if previous is not None:
            del graph[f"{previous}_save"]
            part[f"{folder}_composite"]["inputs"]["destination"] = [f"{previous}_composite", 0]
        graph.update(part)
        previous = folder
    graph[f"{previous}_save"]["inputs"]["filename_prefix"] = (
        source.parent / "feet" / source.stem
    ).as_posix()
    return graph


def main():
    parser = argparse.ArgumentParser(description="裸足のつま先を検出して補修候補を生成する")
    parser.add_argument("image", type=Path, help="output/raw 配下の ComfyUI 生成 PNG")
    parser.add_argument("--workflow", type=Path, help="PNG に生成情報がない場合の API ワークフロー")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--denoise", type=float, default=0.53)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.candidates < 1 or not 0 < args.denoise <= 1:
        parser.error("--candidates は1以上、--denoise は0より大きく1以下で指定してください")
    path = args.image.resolve()
    if not path.is_relative_to(RAW_DIR.resolve()):
        parser.error("output/raw 配下の画像を指定してください")
    source = path.relative_to(RAW_DIR.resolve())
    with Image.open(path) as original:
        metadata = original.info.get("prompt")
        image = original.convert("RGB")
    if args.workflow:
        workflow = load_workflow(args.workflow)
    elif metadata:
        workflow = json.loads(metadata)
    else:
        parser.error("PNG に生成情報がありません。--workflow を指定してください")

    models = ROOT / "models" / "dwpose"
    detector = Wholebody(str(models / "yolox_l.onnx"), str(models / "dw-ll_ucoco_384.onnx"))
    regions = detect_toes(image, detector)
    print(f"足先の検出数: {len(regions)}", flush=True)
    if not regions:
        return 0

    client = ComfyClient(args.server)
    ready, message = client.check_ready()
    if not ready:
        parser.exit(1, message + "\n")
    staged = [
        (stage_crop(INPUT_DIR, "foot_refinement", image.crop(region.box),
                    region.noise, region.blend, 896), region)
        for region in regions
    ]
    for index in range(args.candidates):
        seed = (args.seed + index * len(regions)) % (2**64)
        graph = build_candidate(workflow, source, staged, seed, args.denoise)
        filename = run_inpaint(client, graph, poll=1.0, timeout=900.0)
        print(RAW_DIR / filename, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
