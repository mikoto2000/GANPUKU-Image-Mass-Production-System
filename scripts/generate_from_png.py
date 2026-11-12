"""ComfyUI の生成設定が埋め込まれた PNG から指定枚数を再生成する。"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid
from pathlib import Path

import requests
from PIL import Image

from generate_images import (
    DEFAULT_SERVER,
    NODE_LORA,
    NODE_SAMPLER,
    SEED_MAX,
    ComfyClient,
    load_workflow,
)


def validate_graph(graph):
    if not isinstance(graph, dict) or not graph or not all(
        isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs"), dict)
        for node in graph.values()
    ):
        raise ValueError("ComfyUI の API 形式ワークフローが必要です")
    return graph


def read_png_prompt(path):
    with Image.open(path, formats=["PNG"]) as image:
        prompt = image.text.get("prompt")
    if not prompt:
        raise ValueError(f"PNG に ComfyUI の prompt メタデータがありません: {path}")
    return validate_graph(json.loads(prompt))


def samplers(graph):
    return {
        node_id: node["inputs"]
        for node_id, node in graph.items()
        if node["class_type"] in {"KSampler", "KSamplerAdvanced"}
    }


def prompt_input(graph, sampler, kind):
    link = sampler.get(kind)
    if isinstance(link, list) and len(link) == 2:
        inputs = graph.get(link[0], {}).get("inputs", {})
        if isinstance(inputs.get("text"), str):
            return inputs
    raise ValueError(f"{kind} の入力にテキストを持つノードが必要です")


def seed_key(sampler):
    return "seed" if "seed" in sampler else "noise_seed"


def transfer_prompt(source, graph):
    source_samplers = samplers(source)
    target_samplers = samplers(graph)
    if not source_samplers or not target_samplers:
        raise ValueError("--workflow には PNG と指定先の両方に KSampler 系ノードが必要です")
    primary = source_samplers.get(NODE_SAMPLER) or next(iter(source_samplers.values()))
    for node_id, target in target_samplers.items():
        original = source_samplers.get(node_id, primary)
        for kind in ("positive", "negative"):
            prompt_input(graph, target, kind)["text"] = prompt_input(source, original, kind)["text"]
        target[seed_key(target)] = original[seed_key(original)]


def apply_overrides(graph, prepend, lora_strength):
    if prepend:
        targets = samplers(graph)
        if not targets:
            raise ValueError("--prepend には KSampler 系ノードが必要です")
        # hires の複数サンプラーが同じ positive ノードを使うことがある。
        text_inputs = [prompt_input(graph, sampler, "positive") for sampler in targets.values()]
        for inputs in {id(inputs): inputs for inputs in text_inputs}.values():
            inputs["text"] = ", ".join(filter(None, (prepend, inputs["text"])))
    if lora_strength is not None:
        loaders = {
            node_id: node["inputs"]
            for node_id, node in graph.items()
            if node["class_type"] in {"LoraLoader", "LoraLoaderModelOnly"}
        }
        if NODE_LORA in loaders:
            inputs = loaders[NODE_LORA]
        elif len(loaders) == 1:
            inputs = next(iter(loaders.values()))
        else:
            raise ValueError("--lora-strength にはノード 8 の LoRA、または単一の LoRA が必要です")
        for key in ("strength_model", "strength_clip"):
            if key in inputs:
                inputs[key] = lora_strength


def single_image_output(graph):
    save_nodes = [node_id for node_id, node in graph.items() if node["class_type"] == "SaveImage"]
    if len(save_nodes) != 1:
        raise ValueError("ワークフローには SaveImage ノードが 1 つ必要です")
    save_id = save_nodes[0]
    save = graph[save_id]["inputs"]
    image = save["images"]
    producer = graph[image[0]]
    if producer["class_type"] != "ImageFromBatch" or producer["inputs"].get("length") != 1:
        # バッチ設定を保持したまま、1 回の投入につき 1 枚だけ保存する。
        numeric_ids = (int(node_id) for node_id in graph if node_id.isdecimal())
        selector_id = str(max(numeric_ids, default=0) + 1)
        graph[selector_id] = {
            "class_type": "ImageFromBatch",
            "inputs": {"image": image, "batch_index": 0, "length": 1},
        }
        save["images"] = [selector_id, 0]
    return save_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png", type=Path, help="生成設定を抽出する PNG")
    parser.add_argument("--count", type=int, default=1, help="生成枚数（既定: 1）")
    parser.add_argument("--seed", type=int, help="固定シード（省略: PNG の値、-1: 各回ランダム）")
    parser.add_argument("--workflow", type=Path, help="置き換え先の API 形式ワークフロー JSON")
    parser.add_argument("--prepend", default="", help="positive の先頭に付ける文字列")
    parser.add_argument("--lora-strength", "--lola-strength", type=float, help="LoRA の強度を上書き")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI の URL")
    parser.add_argument("--dry-run", action="store_true", help="投入せず 1 件目の API JSON を表示")
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count は 1 以上を指定してください")
    if args.seed is not None and not -1 <= args.seed <= SEED_MAX:
        parser.error(f"--seed は -1 または 0～{SEED_MAX} を指定してください")
    if args.lora_strength is not None and not math.isfinite(args.lora_strength):
        parser.error("--lora-strength には有限の数値を指定してください")

    try:
        graph = read_png_prompt(args.png)
        if args.workflow is not None:
            source = graph
            graph = validate_graph(load_workflow(args.workflow))
            transfer_prompt(source, graph)
        apply_overrides(graph, args.prepend.rstrip(", "), args.lora_strength)
        save_id = single_image_output(graph)
        targets = samplers(graph)
        if args.seed is not None and not targets:
            raise ValueError("--seed には KSampler 系ノードが必要です")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        parser.error(str(error))

    run_dir = f"regenerated/{args.png.stem}/{uuid.uuid4().hex[:12]}"
    try:
        if not args.dry_run:
            client = ComfyClient(args.server)
            ready, message = client.check_ready()
            if not ready:
                raise RuntimeError(message)
        for index in range(1, args.count + 1):
            if args.seed is not None:
                seed = random.randint(0, SEED_MAX) if args.seed == -1 else args.seed
                for target in targets.values():
                    target[seed_key(target)] = seed
            # 固定シードでも SaveImage が毎回実行されるよう、保存先を変える。
            graph[save_id]["inputs"]["filename_prefix"] = f"{run_dir}/{index:05d}"
            if args.dry_run:
                print(json.dumps(graph, ensure_ascii=False, indent=2))
                return 0
            prompt_id = client.queue_prompt(graph)
            entry = client.wait(prompt_id)
            output = entry.get("outputs", {}).get(save_id, {})
            files = [
                Path(image.get("subfolder", "")) / image["filename"]
                for image in output.get("images", [])
                if image.get("type") == "output" and image.get("filename")
            ]
            if len(files) != 1 or files[0].suffix.lower() != ".png":
                raise RuntimeError(f"{index} 枚目の PNG を確認できません: {prompt_id}")
            print(f"[{index}/{args.count}] {files[0]}", flush=True)
    except (RuntimeError, TimeoutError, requests.RequestException) as error:
        print(f"生成に失敗しました: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
