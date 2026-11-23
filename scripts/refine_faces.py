"""Conservative full-face inpainting for generated anime portraits."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from controlnet_dwpose import DWposeDetector
from PIL import Image

from detect_anime_face import detect_image, load_session
from refine_eyes import align_visible_irises
from refine_inpaint import build_inpaint, run_inpaint, skin_tags_from, stage_crop


FACE_DENOISE = 0.45
FACE_DETAILS = (
    "finished anime illustration, clean tapered linework, smooth gradient shading, "
    "even skin tone"
)
EYE_DETAILS = (
    "(large expressive almond-shaped eyes:1.35), large irises, "
    "thin defined iris rim, smooth iris shading, (round pupils:1.25), short tapered upper lashes, "
    "fine lower lashes, soft reflections along the lower iris"
)
FACE_NEGATIVE = (
    "long spiky eyelashes, surprised expression, white patches on eyelids, "
    "white eye makeup, heart-shaped pupils, strong cheek blush"
)
OPEN_EYES_NEGATIVE = "(small eyes:1.3), narrow eyes, squinting"
CLOSED_EYES = re.compile(r"\bclosed eyes\b", re.IGNORECASE)
EXPRESSION = re.compile(
    r"\b(smile|grin|smiling|grinning|laughing|frown|frowning|smirk|smirking|"
    r"closed eyes|wink|winking|open mouth|closed mouth|looking (?:at|away|up|down|to|back)|head tilt)\b",
    re.IGNORECASE,
)


def face_positive(source):
    expression = [tag.strip() for tag in source.split(",") if EXPRESSION.search(tag)]
    eyes = [] if CLOSED_EYES.search(source) else [EYE_DETAILS]
    return ", ".join([FACE_DETAILS, *skin_tags_from(source), *expression, *eyes])


def face_masks(face, jaw):
    """Find the connected source skin, filling internal holes for facial features."""
    width, height = face.size
    lab = cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2LAB).astype(np.float32)
    y0, y1 = round(height * 0.43), round(height * 0.76)
    x0, x1 = round(width * 0.29), round(width * 0.71)
    anchor = lab[y0:y1, x0:x1]
    if not anchor.size:
        return None
    color = np.median(anchor.reshape(-1, 3), axis=0)
    delta = lab - color
    chroma = color[1:] - 128
    # Keep pale neutral backgrounds out of a warm face even when brightness overlaps.
    chroma_projection = (lab[:, :, 1:] - 128) @ chroma
    # Blond hair can be as bright as skin, but often has a more yellow cast.
    yellow_limit = np.percentile(anchor[:, :, 2] - anchor[:, :, 1], 90) + 3
    yellow_cast = lab[:, :, 2] - lab[:, :, 1]
    skin = ((np.abs(delta[:, :, 1]) <= 17) & (np.abs(delta[:, :, 2]) <= 20)
            & (delta[:, :, 0] >= -53) & (delta[:, :, 0] <= 35)
            & (chroma_projection >= 0.24 * (chroma @ chroma))
            & (yellow_cast <= yellow_limit))
    radius = max(1, round(min(width, height) * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    skin_pixels = skin.astype(np.uint8)
    edge_radius = max(1, round(min(width, height) * 0.015))
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_radius + 1,) * 2)
    joined = cv2.morphologyEx(skin_pixels, cv2.MORPH_CLOSE, edge_kernel)
    # Bridge the dark outlines around eyes without joining the jaw to the neck.
    eye_band = slice(round(height * 0.23), round(height * 0.63))
    joined[eye_band] = cv2.morphologyEx(skin_pixels, cv2.MORPH_CLOSE, kernel)[eye_band]
    _, components = cv2.connectedComponents(joined)
    counts = np.bincount(components[y0:y1, x0:x1].ravel())
    if len(counts) < 2:
        return None
    label = int(np.argmax(counts[1:])) + 1
    selected = (components == label).astype(np.uint8)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros_like(selected)
    cv2.drawContours(silhouette, contours, -1, 1, cv2.FILLED)
    # A skin-colored neck or ear can join the cheeks in color space.
    # The landmark contour bounds the cheeks and jaw before the blend is made.
    face_limit = np.zeros_like(selected)
    boundary = np.vstack((jaw, (jaw[-1, 0], 0), (jaw[0, 0], 0)))
    cv2.fillPoly(face_limit, [np.rint(boundary).astype(np.int32)], 1)
    silhouette &= cv2.dilate(face_limit, kernel)
    coverage = float(silhouette.mean())
    if not 0.22 <= coverage <= 0.90:
        return None

    # Feather inward, so fringe and face edges outside the detected skin stay exact.
    padded = np.pad(silhouette, 1)
    border_distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    feather = max(2, min(width, height) * 0.025)
    edge = np.clip((border_distance - 1) / feather, 0, 1)

    noise = cv2.GaussianBlur(silhouette * 255, (0, 0), sigmaX=max(0.8, feather / 3))
    return Image.fromarray(noise), Image.fromarray(np.rint(edge * 255).astype(np.uint8))


def feature_blend(blend, points, scores, source):
    """Keep the source skin around the nose, eyes, brows and mouth."""
    width, height = blend.size
    weights = np.zeros((height, width), dtype=np.float32)
    lightness = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2LAB)[:, :, 0]
    radius = max(2, round(height * 0.022))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    local_background = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, kernel)
    line_weight = np.clip((local_background.astype(np.float32) - lightness - 15) / 20, 0, 1)
    for start, end, margin in (
        (17, 22, 0.009), (22, 27, 0.009),
        (36, 42, 0.025), (42, 48, 0.025),
        (48, 60, 0.010),
    ):
        if np.median(scores[start:end]) < 0.4:
            continue
        contour = cv2.convexHull(np.rint(points[start:end]).astype(np.int32))
        outside = np.ones((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(outside, contour, 0)
        if outside.all():
            continue
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        # A narrow soft edge follows the tilted feature instead of painting an oval of skin.
        weight = np.clip(1 - distance / max(2, height * margin), 0, 1)
        weight = weight * weight * (3 - 2 * weight)
        # Brow and lip landmarks can surround plain skin, especially on a closed smile.
        if start not in (36, 42):
            weight *= line_weight
        np.maximum(weights, weight, out=weights)
    return Image.fromarray(np.rint(np.asarray(blend) * weights).astype(np.uint8))


def enlarge_eye_guide(crop, eyes):
    """Give low-denoise inpainting a larger eye shape using source pixels."""
    original = np.asarray(crop)
    guide = original.copy()
    centers = np.array([eye.mean(axis=0) for eye in eyes])
    separation = np.linalg.norm(centers[1] - centers[0])
    for eye, center in zip(eyes, centers):
        axis = eye[3] - eye[0]
        width = np.linalg.norm(axis)
        if not 18 <= width <= separation * 0.78:
            continue
        axis = axis / width
        basis = np.array([axis, [-axis[1], axis[0]]])
        height = np.ptp((eye - center) @ basis[1])
        if height < 4:
            continue
        radii = np.array([width * 0.67, max(height * 0.85, width * 0.30)])
        reach = min(separation * 0.49, max(radii) * 1.55)
        left, top = np.maximum(0, np.floor(center - reach)).astype(int)
        right, bottom = np.minimum(crop.size, np.ceil(center + reach + 1)).astype(int)
        y, x = np.mgrid[top:bottom, left:right]
        grid = np.stack((x, y), axis=-1)
        local = (grid - center) @ basis.T
        distance = np.maximum(
            np.linalg.norm(local / radii, axis=-1),
            1.55 * np.linalg.norm(grid - center, axis=-1) / reach,
        )
        falloff = np.clip((1.55 - distance) / 0.68, 0, 1)
        falloff = falloff**2 * (3 - 2 * falloff)
        shift = (local * (1 - 1 / np.array([1.12, 1.25]))) @ basis
        sample = (grid - shift * falloff[:, :, None]).astype(np.float32)
        warped = cv2.remap(
            original, sample[:, :, 0], sample[:, :, 1], cv2.INTER_LINEAR
        )
        active = falloff > 0
        guide[top:bottom, left:right][active] = warped[active]
    return Image.fromarray(guide)


class FaceRefiner:
    def __init__(self, root: Path, input_dir: Path, output_dir: Path):
        load_session()
        pose_dir = root / "models" / "dwpose"
        detector = pose_dir / "yolox_l.onnx"
        pose = pose_dir / "dw-ll_ucoco_384.onnx"
        for model in (detector, pose):
            if not model.is_file():
                raise FileNotFoundError(model)
        self.landmarks = DWposeDetector(str(detector), str(pose), device="cpu")
        self.input_dir = input_dir
        self.output_dir = output_dir

    def _face_points(self, image, box):
        x, y, width, height = box
        pose = self.landmarks(np.asarray(image)[:, :, ::-1].copy())
        candidates = []
        for face, scores in zip(pose["faces"], pose["faces_score"]):
            points = face * image.size - (x, y)
            nose_x, nose_y = points[30]
            jaw = points[:17]
            if (scores[30] >= 0.4 and np.median(scores[:17]) >= 0.4
                    and 0 <= nose_x < width and 0 <= nose_y < height
                    and np.ptp(jaw[:, 0]) >= width * 0.35):
                candidates.append((points, scores))
        return candidates[0] if len(candidates) == 1 else None

    def refine(self, client, workflow, filename: str, poll: float, timeout: float) -> str | None:
        relative = Path(filename)
        path = (self.output_dir / relative).resolve()
        if not path.is_relative_to(self.output_dir.resolve()):
            raise ValueError(f"出力ディレクトリ外の画像: {filename}")
        with Image.open(path) as image:
            original = image.convert("RGB")
        faces = detect_image(original)
        if len(faces) != 1:
            return None
        x, y, width, height = faces[0]
        if min(width, height) < 48:
            return None
        landmarks = self._face_points(original, faces[0])
        if landmarks is None:
            return None
        points, scores = landmarks
        face = original.crop((x, y, x + width, y + height))
        positive = workflow["2"]["inputs"]["text"]
        can_enlarge_eyes = not CLOSED_EYES.search(positive) and np.all(scores[36:48] >= 0.5)
        if can_enlarge_eyes:
            eyes = sorted(np.split(points[36:48], 2), key=lambda eye: eye[:, 0].mean())
            eyes = align_visible_irises(face, eyes)
            points = points.copy()
            points[36:48] = np.concatenate(eyes)
        masks = face_masks(face, points[:17])
        if masks is None:
            return None

        padding = round(max(width, height) * 0.15)
        left, top = max(0, x - padding), max(0, y - padding)
        right = min(original.width, x + width + padding)
        bottom = min(original.height, y + height + padding)
        crop = original.crop((left, top, right, bottom))
        noise, blend = Image.new("L", crop.size), Image.new("L", crop.size)
        noise.paste(masks[0], (x - left, y - top))
        features = feature_blend(masks[1], points, scores, face)
        if features.getbbox() is None:
            return None
        blend.paste(features, (x - left, y - top))
        if can_enlarge_eyes:
            crop = enlarge_eye_guide(crop, [eye + (x - left, y - top) for eye in eyes])
        staging = stage_crop(self.input_dir, "face_refinement", crop, noise, blend, 896)
        negative = [workflow["3"]["inputs"]["text"]]
        if not CLOSED_EYES.search(positive):
            negative.append(OPEN_EYES_NEGATIVE)
        negative.append(FACE_NEGATIVE)
        graph = build_inpaint(
            workflow, relative, staging, (left, top), crop.size, folder="faces",
            positive=face_positive(positive),
            negative=", ".join(filter(None, negative)),
            denoise=FACE_DENOISE,
        )
        return run_inpaint(client, graph, poll, timeout,
                           output_dir=self.output_dir, source=relative)

