"""Build and execute a local crop inpaint using the generation workflow's model."""

from __future__ import annotations

import copy
import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps


def skin_tags_from(source: str) -> list[str]:
    skin_tags = []
    for tag in map(str.strip, source.split(",")):
        if not re.search(r"\b(?:skin|skinned)\b", tag, re.IGNORECASE):
            continue
        if tag.startswith("(") and tag.endswith(")"):
            skin_tags.append(tag)
        else:
            skin_tags.append(f"({tag}:1.35)")
    return skin_tags


def stage_crop(input_dir, folder, crop, noise, blend, target_size):
    scale = target_size / max(crop.size)
    sample_size = tuple(max(64, round(size * scale / 8) * 8) for size in crop.size)
    staging = Path(folder) / uuid.uuid4().hex
    (input_dir / staging).mkdir(parents=True)
    sample = crop.resize(sample_size, Image.Resampling.LANCZOS)
    sample.putalpha(ImageOps.invert(noise.resize(sample_size, Image.Resampling.BILINEAR)))
    sample.save(input_dir / staging / "sample.png")
    blend_image = Image.new("RGBA", crop.size)
    blend_image.putalpha(ImageOps.invert(blend))
    blend_image.save(input_dir / staging / "blend.png")
    return staging


def build_inpaint(workflow, source, staging, offset, size, *, folder, positive, negative, denoise):
    graph = copy.deepcopy({key: node for key, node in workflow.items()
                           if node["class_type"] != "SaveImage"})
    sampler_node = graph.get("25", {})
    if sampler_node.get("class_type") != "KSampler":
        sampler_node = graph["5"]
    sampler = dict(sampler_node["inputs"])
    vae = graph["6"]["inputs"]["vae"]
    graph["2"]["inputs"]["text"] = positive
    graph["3"]["inputs"]["text"] = negative
    sampler.update(latent_image=[f"{folder}_mask", 0], denoise=denoise)

    def output(name):
        return [f"{folder}_{name}", 0]

    def node(class_type, **inputs):
        return {"class_type": class_type, "inputs": inputs}

    prefix = (source.parent / folder / source.stem).as_posix()
    left, top = offset
    width, height = size
    nodes = {
        "source": node("LoadImage", image=f"{source.as_posix()} [output]"),
        "sample": node("LoadImage", image=(staging / "sample.png").as_posix()),
        "blend": node("LoadImage", image=(staging / "blend.png").as_posix()),
        "encode": node("VAEEncode", pixels=output("sample"), vae=vae),
        "mask": node("SetLatentNoiseMask", samples=output("encode"), mask=[f"{folder}_sample", 1]),
        "sampled": node("KSampler", **sampler),
        "decode": node("VAEDecode", samples=output("sampled"), vae=vae),
        "resize": node("ImageScale", image=output("decode"), upscale_method="lanczos",
                       width=width, height=height, crop="disabled"),
        "composite": node("ImageCompositeMasked", destination=output("source"), source=output("resize"),
                          x=left, y=top, resize_source=False, mask=[f"{folder}_blend", 1]),
        "save": node("SaveImage", images=output("composite"), filename_prefix=prefix),
    }
    graph.update({f"{folder}_{key}": value for key, value in nodes.items()})
    return graph


def archive_source(output_dir, source, refined):
    """Move a successfully refined input under <CSV name>_origin."""
    root = Path(output_dir).resolve()
    original = (root / source).resolve()
    result = (root / refined).resolve()
    if not original.is_relative_to(root) or not result.is_relative_to(root):
        raise ValueError("補修の入出力が出力ディレクトリ外です")
    relative = original.relative_to(root)
    if len(relative.parts) < 2 or original == result:
        raise ValueError(f"補修元のCSV名ディレクトリを特定できません: {source}")
    if not result.is_file():
        raise FileNotFoundError(f"補修結果が見つかりません: {result}")
    destination = (root / f"{relative.parts[0]}_origin" / Path(*relative.parts[1:])).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"移動先が出力ディレクトリ外です: {destination}")
    if destination.exists():
        raise FileExistsError(f"元画像の移動先が既に存在します: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original.rename(destination)


def run_inpaint(client, graph, poll, timeout, *, output_dir=None, source=None):
    prompt_id = client.queue_prompt(graph)
    files = client.output_files(client.wait(prompt_id, poll=poll, timeout=timeout))
    if len(files) != 1:
        raise RuntimeError(f"補修の出力枚数が不正: {len(files)}")
    if source is not None:
        archive_source(output_dir, source, files[0])
    return files[0]
