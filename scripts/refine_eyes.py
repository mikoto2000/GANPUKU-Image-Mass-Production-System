"""Apply the larger-eyes inpaint preset to an image saved by ComfyUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from refine_inpaint import build_inpaint, run_inpaint, skin_tags_from, stage_crop


EYE_PROMPT = (
    "(large almond-shaped anime eyes:1.45), (large irises:1.3), "
    "gently arched upper eyelids, rounded lower eyelids, soft relaxed gaze, "
    "short tapered upper lashes clustered at the outer corners, "
    "fine short lower eyelashes, thin defined iris rim, smooth iris shading, "
    "oval pupils, soft crescent-shaped catchlights along the lower iris"
)
EYE_NEGATIVE = (
    "(small eyes:1.3), narrow eyes, squinting, long spiky eyelashes, surprised expression, "
    "white patches on eyelids, white eye makeup"
)
DENOISE = 0.60
MIN_EYE_WIDTH = 18


def eye_positive(source: str) -> str:
    return ", ".join([*skin_tags_from(source), "even skin tone around the eyes", EYE_PROMPT])


def skin_safe_blend(crop, blend, eyes, regions, offset):
    """Protect source-colored skin outside the padded eyelid contours."""
    pixels = np.asarray(crop, dtype=np.float32)
    protected = np.zeros(pixels.shape[:2], dtype=np.float32)
    offset_x, offset_y = offset
    for eye, region in zip(eyes, regions):
        x0, y0, x1, y1 = region
        # Include the soft boundary of the rectangular mask when protecting skin.
        left, top = max(0, x0 - offset_x - 8), max(0, y0 - offset_y - 8)
        right = min(crop.width, x1 - offset_x + 9)
        bottom = min(crop.height, y1 - offset_y + 9)
        patch = pixels[top:bottom, left:right]
        contour = eye - (offset_x + left, offset_y + top)
        eye_width = float(np.ptp(contour[:, 0]))
        outline = Image.new("L", (right - left, bottom - top))
        ImageDraw.Draw(outline).polygon([tuple(point) for point in contour], fill=255)
        grow = max(2, round(eye_width * 0.11))
        core = outline.filter(ImageFilter.MaxFilter(2 * grow + 1))
        core = np.asarray(core.filter(ImageFilter.GaussianBlur(max(0.8, grow / 2))), dtype=np.float32) / 255

        # Sample separately above and below so the original eyelid shadow is represented.
        yy = np.arange(top, bottom)[:, None]
        xx = np.arange(left, right)[None, :]
        eye_left = contour[:, 0].min() + left
        eye_right = contour[:, 0].max() + left
        in_eye_width = (xx >= eye_left) & (xx <= eye_right)
        # Allow vertical enlargement in the center while keeping skin at the corners.
        corner_distance = np.minimum(xx - eye_left, eye_right - xx) / eye_width
        core *= np.clip((corner_distance - 0.08) / 0.20, 0, 1)
        above = (yy < contour[:, 1].min() + top) & (yy >= y0 - offset_y)
        below = (yy > contour[:, 1].max() + top) & (yy <= y1 - offset_y)
        skin_distance = np.full(patch.shape[:2], np.inf, dtype=np.float32)
        for band in (above, below):
            samples = patch[band & in_eye_width]
            if len(samples):
                for color in np.percentile(samples, (25, 50, 75), axis=0):
                    distance = np.max(np.abs(patch - color), axis=2)
                    np.minimum(skin_distance, distance, out=skin_distance)
        skin = np.clip((42 - skin_distance) / 24, 0, 1)
        area = protected[top:bottom, left:right]
        np.maximum(area, skin * (1 - core), out=area)

    alpha = np.asarray(blend, dtype=np.float32) * (1 - protected)
    return Image.fromarray(np.rint(alpha).astype(np.uint8))


def align_visible_irises(image, eyes):
    """Correct shifted landmarks when two nearby, matching cool irises are visible."""
    import cv2

    centers = np.array([eye.mean(axis=0) for eye in eyes])
    axis = centers[1] - centers[0]
    separation = np.linalg.norm(axis)
    if separation < 18:
        return eyes
    radius = separation * 0.68
    padding = radius + separation * 0.30
    left, top = np.maximum(0, np.floor(centers.min(axis=0) - padding)).astype(int)
    right, bottom = np.minimum(image.size, np.ceil(centers.max(axis=0) + padding)).astype(int)
    rgb = np.asarray(image.crop((left, top, right, bottom)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    # Cool, saturated colors distinguish these irises from nearby skin and shadows.
    visible = (hue >= 35) & (hue <= 145) & (saturation >= 115) & (value >= 65)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(visible.astype(np.uint8), connectivity=8)
    candidates = []
    for index, (x, y, width, height, area) in enumerate(stats[1:], 1):
        if (not 0.025 * separation**2 <= area <= 0.38 * separation**2
                or min(width, height) < 0.48 * max(width, height)
                or area < 0.42 * width * height
                or x == 0 or y == 0 or x + width == rgb.shape[1] or y + height == rgb.shape[0]):
            continue
        center = centroids[index] + (left, top)
        distances = np.linalg.norm(centers - center, axis=1)
        candidates.append((center, np.array([width, height]), float(np.median(hue[labels == index])), distances))

    pairs = []
    for first in candidates:
        for second in candidates:
            gap = second[0] - first[0]
            length = np.linalg.norm(gap)
            if (not 0.68 * separation <= length <= 1.95 * separation
                    or np.dot(gap, axis) < 0.86 * length * separation
                    or abs(first[2] - second[2]) > 12
                    or max(first[3][0], second[3][1]) > radius):
                continue
            pairs.append((first[3][0] + second[3][1], first, second))
    if not pairs:
        return eyes

    _, first, second = min(pairs, key=lambda pair: pair[0])
    aligned = []
    for index, (eye, iris) in enumerate(zip(eyes, (first, second))):
        center, size, _, distances = iris
        extent = np.ptp(eye, axis=0)
        if distances[index] < 0.20 * extent[0]:
            aligned.append(eye)
            continue
        scale = np.clip(size * (1.20, 1.05) / np.maximum(extent, 1), 1, 2.8)
        aligned.append((eye - centers[index]) * scale + center)
    return aligned


class EyeRefiner:
    def __init__(self, root: Path, input_dir: Path, output_dir: Path):
        from controlnet_dwpose import DWposeDetector

        model_dir = root / "models" / "dwpose"
        detector = model_dir / "yolox_l.onnx"
        pose = model_dir / "dw-ll_ucoco_384.onnx"
        for model in (detector, pose):
            if not model.is_file():
                raise FileNotFoundError(model)
        self.detector = DWposeDetector(str(detector), str(pose), device="cpu")
        self.input_dir = input_dir
        self.output_dir = output_dir

    def _eye_regions(self, image):
        bgr = np.asarray(image)[:, :, ::-1].copy()
        pose = self.detector(bgr)
        # COCO-WholeBody's 68-point face layout uses landmarks 36–47 for the eyes.
        faces = [
            face[36:48] * image.size
            for face, confidence in zip(pose["faces"], pose["faces_score"])
            if np.all(confidence[36:48] >= 0.4)
        ]
        if len(faces) != 1:
            return None

        eyes = sorted(np.split(faces[0], 2), key=lambda eye: eye[:, 0].mean())
        eyes = align_visible_irises(image, eyes)
        regions = []
        widths = []
        for index, eye in enumerate(eyes):
            left, top = eye.min(axis=0)
            right, bottom = eye.max(axis=0)
            width = right - left
            if width < MIN_EYE_WIDTH or bottom - top < 3:
                return None
            widths.append(width)
            outer, inner = width * 0.55, width * 0.20
            pad_left, pad_right = (outer, inner) if index == 0 else (inner, outer)
            regions.append((
                round(left - pad_left), round(top - width * 0.34),
                round(right + pad_right), round(bottom + width * 0.30),
            ))

        a, b = regions
        if a[2] >= b[0] or any(
            x0 < 0 or y0 < 0 or x1 >= image.width or y1 >= image.height
            for x0, y0, x1, y1 in regions
        ):
            return None
        return eyes, regions, min(widths)

    def _stage_crop(self, image, eyes, regions):
        pad = max(16, round(max(right - left for left, _, right, _ in regions) * 0.24))
        left = max(0, min(region[0] for region in regions) - pad)
        top = max(0, min(region[1] for region in regions) - pad)
        right = min(image.width, max(region[2] for region in regions) + pad)
        bottom = min(image.height, max(region[3] for region in regions) + pad)
        crop = image.crop((left, top, right, bottom))
        mask = Image.new("L", crop.size)
        draw = ImageDraw.Draw(mask)
        for x0, y0, x1, y1 in regions:
            draw.rounded_rectangle((x0 - left, y0 - top, x1 - left, y1 - top),
                                   radius=round((y1 - y0) * 0.32), fill=255)

        feather = max(1, pad / 14)
        blend = mask.filter(ImageFilter.GaussianBlur(feather))
        blend = skin_safe_blend(crop, blend, eyes, regions, (left, top))
        noise = mask.filter(ImageFilter.MaxFilter(2 * round(feather) + 1))
        staging = stage_crop(self.input_dir, "eye_refinement", crop, noise, blend, 928)
        return staging, (left, top), crop.size

    def refine(self, client, workflow, filename: str, poll: float, timeout: float) -> str | None:
        relative = Path(filename)
        path = (self.output_dir / relative).resolve()
        if not path.is_relative_to(self.output_dir.resolve()):
            raise ValueError(f"出力ディレクトリ外の画像: {filename}")
        with Image.open(path) as image:
            original = image.convert("RGB")
        detected = self._eye_regions(original)
        if detected is None:
            return None
        eyes, regions, eye_width = detected

        staging, offset, size = self._stage_crop(original, eyes, regions)
        positive = eye_positive(workflow["2"]["inputs"]["text"])
        negative = ", ".join(filter(None, (workflow["3"]["inputs"]["text"], EYE_NEGATIVE)))
        # Full-body shots leave fewer source pixels to anchor the skin and eye details.
        denoise = min(DENOISE, 0.45 + max(0, eye_width - 32) * 0.15 / 16)
        graph = build_inpaint(
            workflow, relative, staging, offset, size, folder="eyes",
            positive=positive, negative=negative, denoise=round(float(denoise), 3),
        )
        return run_inpaint(client, graph, poll, timeout,
                           output_dir=self.output_dir, source=relative)
