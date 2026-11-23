"""
generate_images.py — ComfyUI /prompt API による量産生成

指定した CSV の各行をワークフローへ流し込み、ComfyUI に順次投入する。
生成結果は output\\raw\\<プロンプト CSV の拡張子なしファイル名> 配下へ保存され、
実行内容は logs\\gen_YYYYMMDD_HHMMSS.csv に記録される。

使い方:
    .\\venv\\Scripts\\python.exe scripts\\generate_images.py --dry-run --limit 1
    .\\venv\\Scripts\\python.exe scripts\\generate_images.py --limit 4
    .\\venv\\Scripts\\python.exe scripts\\generate_images.py --batch 4 --limit 30

共通プロンプトを使う場合（CSV には行ごとの差分だけを書く）:
    .\\venv\\Scripts\\python.exe scripts\\generate_images.py ^
        --common-positive prompts\\common_positive.txt ^
        --common-negative prompts\\common_negative.txt ^
        --limit 4

前提:
    ComfyUI が起動していること（start-comfyui.ps1）。
    ワークフローは API 形式 JSON（workflows\\sdxl_base_api.json）。
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import random
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# Windows のコンソールは既定で CP932 のため、UTF-8 で出力しないと
# PowerShell やリダイレクト先で日本語が化ける。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(r"F:\ai-image")
WORKFLOW_PATH = ROOT / "workflows" / "sdxl_base_api.json"
PROMPTS_PATH = ROOT / "prompts" / "prompts.csv"
LOG_DIR = ROOT / "logs"
RAW_DIR = ROOT / "output" / "raw"
INPUT_DIR = ROOT / "input"
STYLE_REFERENCE_DIR = INPUT_DIR / "style_reference"

DEFAULT_SERVER = "http://127.0.0.1:8188"

# ワークフロー内のノード ID。雛形と一致させること。
NODE_CKPT = "1"
NODE_POS = "2"
NODE_NEG = "3"
NODE_LATENT = "4"
NODE_SAMPLER = "5"
NODE_DECODE = "6"
NODE_SAVE = "7"
NODE_LORA = "8"
NODE_LOAD_IMAGE = "11"
NODE_IMAGE_SCALE = "12"
NODE_VAE_ENCODE = "13"
NODE_IPADAPTER_LOADER = "20"
NODE_IPADAPTER_IMAGE = "21"
NODE_IPADAPTER_APPLY = "22"
NODE_HIRES_SAMPLER = "25"

SEED_MAX = 18446744073709551615


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class PromptRow:
    """prompts.csv の 1 行。空欄はワークフロー既定値を使う。"""
    id: str
    positive: str
    negative: str = ""
    width: int = 0
    height: int = 0
    steps: int = 0
    cfg: float = 0.0
    sampler: str = ""
    scheduler: str = ""
    batch: int = 0
    note: str = ""


@dataclass
class GenRecord:
    """1 生成単位の記録。"""
    prompt_id: str
    row_id: str
    positive: str
    negative: str
    seed: int
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    batch: int
    denoise: float = 1.0
    image: str = ""
    files: str = ""
    seconds: float = 0.0
    status: str = ""
    error: str = ""
    ipadapter_weight: float = 0.0
    ipadapter_weight_type: str = ""
    ref_image: str = ""
    source_csv: str = ""


# ---------------------------------------------------------------------------
# 入力読み込み
# ---------------------------------------------------------------------------

def load_workflow(path: Path) -> dict[str, Any]:
    """API 形式ワークフロー JSON を読む。"""
    if not path.exists():
        raise FileNotFoundError("ワークフローが見つからない: " + str(path))
    with open(path, "r", encoding="utf-8-sig") as f:
        wf = json.load(f)
    if not isinstance(wf, dict) or not wf:
        raise ValueError("ワークフロー JSON が空、または不正: " + str(path))
    return wf


def _to_int(value: str, default: int) -> int:
    value = (value or "").strip()
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _to_float(value: str, default: float) -> float:
    value = (value or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_prompts(path: Path) -> list[PromptRow]:
    """prompts.csv を読む。ヘッダ必須。空行と # コメント行は無視する。"""
    if not path.exists():
        raise FileNotFoundError("プロンプト CSV が見つからない: " + str(path))

    rows: list[PromptRow] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        if not reader.fieldnames:
            raise ValueError("CSV にヘッダ行がない: " + str(path))
        if "positive" not in reader.fieldnames:
            raise ValueError("CSV のヘッダに positive 列がない: " + str(path))
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV のヘッダに重複した列名がある: " + str(path))
        for raw in reader:
            if (raw.get("id") or "").lstrip().startswith("#"):
                continue
            if None in raw or any(value is None for value in raw.values()):
                actual = sum(value is not None for key, value in raw.items() if key is not None)
                actual += len(raw.get(None) or [])
                raise ValueError(
                    str(path) + ": " + str(reader.line_num) + " 行目の列数が不正"
                    + "（ヘッダ " + str(len(reader.fieldnames)) + " 列、データ " + str(actual) + " 列）。"
                    + "空欄のカンマ抜け、またはプロンプトを囲む引用符を確認してください。"
                )
            positive = (raw.get("positive") or "").strip()
            if not positive:
                continue
            rows.append(
                PromptRow(
                    id=(raw.get("id") or "").strip(),
                    positive=positive,
                    negative=(raw.get("negative") or "").strip(),
                    width=_to_int(raw.get("width", ""), 0),
                    height=_to_int(raw.get("height", ""), 0),
                    steps=_to_int(raw.get("steps", ""), 0),
                    cfg=_to_float(raw.get("cfg", ""), 0.0),
                    sampler=(raw.get("sampler") or "").strip(),
                    scheduler=(raw.get("scheduler") or "").strip(),
                    batch=_to_int(raw.get("batch", ""), 0),
                    note=(raw.get("note") or "").strip(),
                )
            )
    return rows


def load_common_prompt(path: Path) -> str:
    """共通プロンプトファイルを読む。

    1 行 1 タグ（カンマ区切りも可）。空行と # で始まる行は無視する。
    戻り値は ", " で連結した 1 本の文字列。
    """
    if not path.exists():
        raise FileNotFoundError("共通プロンプトが見つからない: " + str(path))

    tags: list[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tag in line.split(","):
                tag = tag.strip()
                if tag:
                    tags.append(tag)
    return ", ".join(tags)


# ---------------------------------------------------------------------------
# ワークフローへの流し込み
# ---------------------------------------------------------------------------

def build_workflow(
    base: dict[str, Any],
    row: PromptRow,
    seed: int,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """雛形をコピーし、1 行分の値を流し込んだワークフローを返す。

    優先順位は 引数 > CSV > 雛形 の順。
    """
    wf = copy.deepcopy(base)

    def node(nid: str) -> dict[str, Any]:
        if nid not in wf:
            raise KeyError("ワークフローにノード " + nid + " がない")
        return wf[nid]["inputs"]

    # プロンプト（共通 → prepend → 行個別 の順に連結）
    pos_parts: list[str] = []
    if overrides.get("common_positive"):
        pos_parts.append(overrides["common_positive"])
    if overrides.get("prepend"):
        pos_parts.append(overrides["prepend"].rstrip(", "))
    if row.positive:
        pos_parts.append(row.positive)
    node(NODE_POS)["text"] = ", ".join(p for p in pos_parts if p)

    neg_parts: list[str] = []
    if overrides.get("common_negative"):
        neg_parts.append(overrides["common_negative"])
    if row.negative:
        neg_parts.append(row.negative)
    if neg_parts:
        node(NODE_NEG)["text"] = ", ".join(p for p in neg_parts if p)

    # img2img かどうか（LoadImage ノードの有無で判定）
    is_img2img = NODE_LOAD_IMAGE in wf

    # 解像度・バッチ
    if is_img2img:
        # VAEEncode は解像度を持たないので ImageScale 側で合わせる
        latent = node(NODE_IMAGE_SCALE)
        if row.width > 0:
            latent["width"] = row.width
        if row.height > 0:
            latent["height"] = row.height
    else:
        latent = node(NODE_LATENT)
        if row.width > 0:
            latent["width"] = row.width
        if row.height > 0:
            latent["height"] = row.height
        if row.batch > 0:
            latent["batch_size"] = row.batch

    # サンプラー
    sampler = node(NODE_SAMPLER)
    sampler["seed"] = seed
    if row.steps > 0:
        sampler["steps"] = row.steps
    if row.cfg > 0:
        sampler["cfg"] = row.cfg
    if row.sampler:
        sampler["sampler_name"] = row.sampler
    if row.scheduler:
        sampler["scheduler"] = row.scheduler

    # 引数による上書き（全行に適用）
    if overrides.get("steps"):
        sampler["steps"] = overrides["steps"]
    if overrides.get("cfg"):
        sampler["cfg"] = overrides["cfg"]
    if overrides.get("sampler"):
        sampler["sampler_name"] = overrides["sampler"]
    if overrides.get("scheduler"):
        sampler["scheduler"] = overrides["scheduler"]
    if overrides.get("width"):
        latent["width"] = overrides["width"]
    if overrides.get("height"):
        latent["height"] = overrides["height"]
    if overrides.get("batch") and not is_img2img:
        latent["batch_size"] = overrides["batch"]
    if overrides.get("ckpt"):
        model_key = "unet_name" if wf[NODE_CKPT]["class_type"] == "UNETLoader" else "ckpt_name"
        node(NODE_CKPT)[model_key] = overrides["ckpt"]
    if overrides.get("prefix"):
        node(NODE_SAVE)["filename_prefix"] = overrides["prefix"]
    if overrides.get("denoise", -1.0) >= 0.0:
        sampler["denoise"] = overrides["denoise"]

    # img2img の元画像
    if overrides.get("image"):
        if not is_img2img:
            raise KeyError(
                "ワークフローに LoadImage ノード " + NODE_LOAD_IMAGE + " がない: "
                + "img2img を使うには workflows\\sdxl_img2img_api.json を指定して"
            )
        node(NODE_LOAD_IMAGE)["image"] = overrides["image"]

    # LoRA（LoRA ノードを持つワークフローでのみ有効）
    if overrides.get("lora"):
        if NODE_LORA not in wf:
            raise KeyError(
                "ワークフローに LoRA ノード " + NODE_LORA + " がない: "
                + "LoRA を使うには workflows\\sdxl_lora_api.json を指定して"
            )
        node(NODE_LORA)["lora_name"] = overrides["lora"]
    if overrides.get("lora_strength", -1.0) >= 0.0:
        if NODE_LORA not in wf:
            raise KeyError(
                "ワークフローに LoRA ノード " + NODE_LORA + " がない: "
                + "LoRA を使うには workflows\\sdxl_lora_api.json を指定して"
            )
        node(NODE_LORA)["strength_model"] = overrides["lora_strength"]
        node(NODE_LORA)["strength_clip"] = overrides["lora_strength"]

    # IP-Adapter（参照画像によるスタイル転写）
    if NODE_IPADAPTER_APPLY in wf:
        apply_node = node(NODE_IPADAPTER_APPLY)
        if overrides.get("ipadapter_weight", -1.0) >= 0.0:
            apply_node["weight"] = overrides["ipadapter_weight"]
        if overrides.get("ipadapter_weight_type"):
            apply_node["weight_type"] = overrides["ipadapter_weight_type"]
        if overrides.get("ipadapter_start", -1.0) >= 0.0:
            apply_node["start_at"] = overrides["ipadapter_start"]
        if overrides.get("ipadapter_end", -1.0) >= 0.0:
            apply_node["end_at"] = overrides["ipadapter_end"]

    if is_img2img and overrides.get("resize_mode", "preserve") == "preserve":
        from PIL import Image, ImageOps
        source = Path(overrides.get("source_image") or INPUT_DIR / node(NODE_LOAD_IMAGE)["image"])
        with Image.open(source) as im:
            w, h = ImageOps.exif_transpose(im).size
        # Keep the requested pixel area while matching the input aspect ratio.
        area = (latent["width"] or 832) * (latent["height"] or 1216)
        latent["width"] = max(64, round((area * w / h) ** 0.5 / 8) * 8)
        latent["height"] = max(64, round((area * h / w) ** 0.5 / 8) * 8)
        latent["crop"] = "disabled"
    elif is_img2img:
        latent["crop"] = "center"

    if wf.get(NODE_HIRES_SAMPLER, {}).get("class_type") == "KSampler":
        hires_sampler = node(NODE_HIRES_SAMPLER)
        for key in ("seed", "steps", "cfg", "sampler_name", "scheduler"):
            hires_sampler[key] = sampler[key]

    return wf


def effective_values(wf: dict[str, Any], reference_names: list[str]) -> dict[str, Any]:
    """記録用に、実際に使われた値をワークフローから取り出す。"""
    sampler = wf[NODE_SAMPLER]["inputs"]
    if NODE_LOAD_IMAGE in wf:
        # img2img: 解像度は ImageScale、バッチは常に 1
        scale = wf[NODE_IMAGE_SCALE]["inputs"]
        values = {
            "width": scale.get("width"),
            "height": scale.get("height"),
            "batch": 1,
            "steps": sampler.get("steps"),
            "cfg": sampler.get("cfg"),
            "sampler": sampler.get("sampler_name"),
            "scheduler": sampler.get("scheduler"),
            "denoise": sampler.get("denoise"),
            "image": wf[NODE_LOAD_IMAGE]["inputs"].get("image"),
        }
    else:
        latent = wf[NODE_LATENT]["inputs"]
        values = {
            "width": latent.get("width"),
            "height": latent.get("height"),
            "batch": latent.get("batch_size"),
            "steps": sampler.get("steps"),
            "cfg": sampler.get("cfg"),
            "sampler": sampler.get("sampler_name"),
            "scheduler": sampler.get("scheduler"),
            "denoise": sampler.get("denoise"),
            "image": "",
        }
    if NODE_IPADAPTER_APPLY in wf:
        apply_node = wf[NODE_IPADAPTER_APPLY]["inputs"]
        values["ipadapter_weight"] = apply_node.get("weight")
        values["ipadapter_weight_type"] = apply_node.get("weight_type")
        values["ref_image"] = "|".join(reference_names)
    else:
        values["ipadapter_weight"] = 0.0
        values["ipadapter_weight_type"] = ""
        values["ref_image"] = ""
    return values


def configure_style_reference(
    workflow: dict[str, Any],
    *,
    image: str,
    directory: str,
    disabled: bool,
    dry_run: bool,
) -> list[str]:
    if NODE_IPADAPTER_APPLY not in workflow:
        return []

    image_node = workflow[NODE_IPADAPTER_IMAGE]
    default_image = image_node["inputs"].get("image")
    names: list[str] = []
    if not disabled:
        if image:
            source = Path(image)
            if source.is_file():
                destination = INPUT_DIR / source.name
                if not dry_run and source.resolve() != destination.resolve():
                    INPUT_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                names = [source.name]
        elif directory or not default_image:
            from PIL import Image

            folder = Path(directory) if directory else STYLE_REFERENCE_DIR
            if folder.is_dir():
                extensions = Image.registered_extensions()
                input_root = INPUT_DIR.resolve()
                staging_dir = input_root / "style_reference_copies" / uuid.uuid4().hex
                for source in sorted(folder.iterdir(), key=lambda path: path.name.casefold()):
                    if not source.is_file() or source.suffix.lower() not in extensions:
                        continue
                    try:
                        with Image.open(source) as reference:
                            reference.verify()
                    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
                        print(f"参照画像を読み込めないためスキップ: {source} ({exc})")
                        continue

                    resolved = source.resolve()
                    if not resolved.is_relative_to(input_root):
                        resolved = staging_dir / source.name
                        if not dry_run:
                            staging_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, resolved)
                    names.append(resolved.relative_to(input_root).as_posix())
        else:
            names = [default_image]

    if not names:
        sampler = workflow[NODE_SAMPLER]["inputs"]
        loader = workflow[NODE_IPADAPTER_LOADER]["inputs"]
        if sampler["model"] != [NODE_IPADAPTER_APPLY, 0]:
            raise ValueError("サンプラーに接続された IP-Adapter を外せません")
        hires = workflow.get(NODE_HIRES_SAMPLER, {})
        if hires.get("class_type") == "KSampler" and hires["inputs"]["model"] == sampler["model"]:
            hires["inputs"]["model"] = loader["model"]
        sampler["model"] = loader["model"]
        for node_id in (NODE_IPADAPTER_APPLY, NODE_IPADAPTER_IMAGE, NODE_IPADAPTER_LOADER):
            del workflow[node_id]
        print("スタイル参照なし" if disabled else "参照画像がないため、スタイル参照なしで生成します")
        return []

    image_node["inputs"]["image"] = names[0]
    if len(names) > 1:
        apply_node = workflow[NODE_IPADAPTER_APPLY]["inputs"]
        batch_output = None
        for index, name in enumerate(names, start=1):
            load_id = NODE_IPADAPTER_IMAGE
            if index > 1:
                load_id = f"style_reference_load_{index}"
                workflow[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            prep_id = f"style_reference_prep_{index}"
            workflow[prep_id] = {
                "class_type": "PrepImageForClipVision",
                "inputs": {
                    "image": [load_id, 0],
                    "interpolation": "BICUBIC",
                    "crop_position": "center",
                    "sharpening": 0.0,
                },
            }
            if batch_output is None:
                batch_output = [prep_id, 0]
                continue
            batch_id = f"style_reference_batch_{index}"
            workflow[batch_id] = {
                "class_type": "ImageBatch",
                "inputs": {"image1": batch_output, "image2": [prep_id, 0]},
            }
            batch_output = [batch_id, 0]
        apply_node["image"] = batch_output
        apply_node["combine_embeds"] = "average"

    print(f"スタイル参照画像 {len(names)} 件: " + ", ".join(names))
    return names


# ---------------------------------------------------------------------------
# ComfyUI API
# ---------------------------------------------------------------------------

def _explain_error(response: requests.Response) -> str:
    """ComfyUI の 400 応答から、人が読める要点だけを組み立てる。"""
    try:
        body = response.json()
    except Exception:
        return response.text[:500]

    parts: list[str] = []
    err = body.get("error") or {}
    if err.get("message"):
        parts.append(str(err["message"]))

    node_errors = body.get("node_errors") or {}
    for nid, info in node_errors.items():
        ctype = info.get("class_type", "?")
        for e in info.get("errors") or []:
            msg = e.get("message", "")
            details = e.get("details", "")
            parts.append("node " + str(nid) + " (" + str(ctype) + "): " + str(msg) + " " + str(details))

    if not parts:
        return json.dumps(body, ensure_ascii=False)[:500]
    return " / ".join(parts)[:800]


class ComfyClient:
    """ComfyUI の /prompt, /history, /queue, /system_stats を扱う薄いクライアント。"""

    def __init__(self, server: str, timeout: float = 30.0):
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.client_id = str(uuid.uuid4())

    def _url(self, path: str) -> str:
        return self.server + path

    def system_stats(self) -> dict[str, Any]:
        r = self.session.get(self._url("/system_stats"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def check_ready(self) -> tuple[bool, str]:
        """起動確認。落ちていれば分かりやすいメッセージを返す。"""
        try:
            stats = self.system_stats()
        except requests.exceptions.ConnectionError:
            return False, (
                "ComfyUI に接続できない: " + self.server
                + "\n起動するには: powershell -NoProfile -ExecutionPolicy Bypass "
                + "-File F:\\ai-image\\start-comfyui.ps1"
            )
        except Exception as e:
            return False, "ComfyUI の状態取得に失敗: " + repr(e)

        devices = stats.get("devices") or []
        if not devices:
            return False, "ComfyUI は応答するが GPU デバイスが見えない"
        dev = devices[0]
        free_gb = dev.get("vram_free", 0) / (1024 ** 3)
        if free_gb < 1.0:
            return False, (
                "VRAM 空きが不足: " + str(round(free_gb, 2)) + " GB"
                + "\n他のプロセスが GPU を使っていないか確認して"
            )
        return True, (
            "ComfyUI " + str(stats.get("system", {}).get("comfyui_version", "?"))
            + " / " + str(dev.get("name", "?"))
            + " / VRAM 空き " + str(round(free_gb, 1)) + " GB"
        )

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """ワークフローを投入し prompt_id を返す。"""
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = self.session.post(self._url("/prompt"), json=payload, timeout=self.timeout)
        if r.status_code != 200:
            # ComfyUI はノード検証エラーを 400 で返す。
            # 生 JSON は長いので、ノードごとの message だけ抜き出して伝える。
            raise RuntimeError("投入失敗 HTTP " + str(r.status_code) + ": " + _explain_error(r))
        body = r.json()
        pid = body.get("prompt_id")
        if not pid:
            raise RuntimeError("prompt_id が返らない: " + json.dumps(body, ensure_ascii=False))
        return str(pid)

    def history(self, prompt_id: str) -> dict[str, Any]:
        r = self.session.get(self._url("/history/" + prompt_id), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def wait(self, prompt_id: str, poll: float = 1.0, timeout: float = 900.0) -> dict[str, Any]:
        """完了まで /history をポーリングする。"""
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(
                    "生成が " + str(int(timeout)) + " 秒以内に終わらない: " + prompt_id
                )
            h = self.history(prompt_id)
            entry = h.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("completed") or status.get("status_str") == "success":
                    return entry
                if status.get("status_str") == "error":
                    msgs = status.get("messages") or []
                    raise RuntimeError("ComfyUI 側でエラー: " + json.dumps(msgs, ensure_ascii=False)[:2000])
            time.sleep(poll)

    def output_files(self, entry: dict[str, Any]) -> list[str]:
        """history エントリから保存ファイル名を集める。"""
        files: list[str] = []
        outputs = entry.get("outputs") or {}
        for node_out in outputs.values():
            for img in node_out.get("images") or []:
                name = img.get("filename")
                if name:
                    files.append(str(Path(img.get("subfolder", "")) / name))
        return files


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="ComfyUI による量産生成")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI の URL")
    ap.add_argument("--workflow", default=str(WORKFLOW_PATH), help="API 形式ワークフロー JSON")
    ap.add_argument("--prompts", type=Path, nargs="+", action="extend",
                    help="プロンプト CSV（複数指定可。省略時は prompts/prompts.csv）")
    ap.add_argument("--start", type=int, default=0, help="全 CSV を指定順につないだ先頭 N 行を飛ばす")
    ap.add_argument("--limit", type=int, default=0, help="--start の位置から N 行だけ処理 (0=全部)")
    ap.add_argument("--batch", type=int, default=0, help="1 プロンプトあたりの枚数（CSV より優先）")
    ap.add_argument("--steps", type=int, default=0, help="steps を上書き")
    ap.add_argument("--cfg", type=float, default=0.0, help="CFG を上書き")
    ap.add_argument("--sampler", default="", help="サンプラー名を上書き")
    ap.add_argument("--scheduler", default="", help="スケジューラを上書き")
    ap.add_argument("--width", type=int, default=0, help="幅を上書き")
    ap.add_argument("--height", type=int, default=0, help="高さを上書き")
    ap.add_argument("--ckpt", default="", help="モデル名を上書き（CheckpointLoaderSimple / UNETLoader）")
    ap.add_argument("--prefix", default="", help="CSV 名のフォルダ内で使う保存先とファイル名の接頭辞（末尾に実行日時を追加）")
    ap.add_argument("--lora", default="", help="LoRA 名を上書き（例: rei_style_v2.safetensors）")
    ap.add_argument("--lora-strength", type=float, default=-1.0, help="LoRA の strength_model/clip を上書き")
    ap.add_argument("--prepend", default="", help="positive の先頭に付ける文字列（トリガーワード等）")
    ap.add_argument("--common-positive", default="",
                    help="全行共通のポジティブプロンプトファイル（1 行 1 タグ）")
    ap.add_argument("--common-negative", default="",
                    help="全行共通のネガティブプロンプトファイル（1 行 1 タグ）")
    ap.add_argument("--image", default="", help="img2img の元画像パス（ComfyUI の input へ自動コピー）")
    ap.add_argument("--resize-mode", choices=["preserve", "crop"], default="preserve",
                    help="preserve: 元画像の比率と指定面積を保持 / crop: 指定サイズへ中央切抜き")
    ap.add_argument("--denoise", type=float, default=-1.0, help="denoise を上書き（img2img は 0.3〜0.6 推奨）")
    reference_args = ap.add_mutually_exclusive_group()
    reference_args.add_argument("--ref-image", default="", help="IP-Adapter の参照画像ファイル")
    reference_args.add_argument("--ref-dir", default="", help="IP-Adapter の参照画像フォルダ")
    ap.add_argument("--no-style-reference", action="store_true", help="IP-Adapter を外して、同じワークフローで参照なし生成")
    ap.add_argument("--ipadapter-weight", type=float, default=-1.0, help="IP-Adapter の weight（0.5〜0.7 推奨）")
    ap.add_argument("--ipadapter-weight-type", default="", help="IP-Adapter の weight_type（例: style transfer）")
    ap.add_argument("--ipadapter-start", type=float, default=-1.0, help="IP-Adapter の start_at")
    ap.add_argument("--ipadapter-end", type=float, default=-1.0, help="IP-Adapter の end_at")
    ap.add_argument("--seed", type=int, default=-1, help="固定シード (-1=毎回ランダム)")
    ap.add_argument("--larger-eyes", action="store_true",
                    help="生成後に両目を検出し、肌色を引き継いで目元を補修（ローカル ComfyUI）")
    ap.add_argument("--face-refine", "--refine-face", action="store_true",
                    help="生成後に顔を検出し、髪と肌色を保護して denoise 0.45 で補修（ローカル ComfyUI）")
    ap.add_argument("--dry-run", action="store_true", help="POST せず組み立てた JSON を表示")
    ap.add_argument("--timeout", type=float, default=900.0, help="1 生成あたりの待ち時間（秒）")
    ap.add_argument("--poll", type=float, default=0.2, help="ポーリング間隔（秒）")
    ap.add_argument("--stop-on-error", action="store_true", help="1 件失敗したら中断する")
    args = ap.parse_args(argv)
    prompts_paths = args.prompts or [PROMPTS_PATH]

    try:
        base = load_workflow(Path(args.workflow))
        rows = [(path, row) for path in prompts_paths for row in load_prompts(path)]
    except Exception as e:
        print("入力の読み込みに失敗: " + repr(e))
        return 2

    common_positive = ""
    common_negative = ""
    try:
        if args.common_positive:
            common_positive = load_common_prompt(Path(args.common_positive))
        if args.common_negative:
            common_negative = load_common_prompt(Path(args.common_negative))
    except Exception as e:
        print("共通プロンプトの読み込みに失敗: " + repr(e))
        return 2

    if args.start > 0:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("プロンプトが 0 件: " + ", ".join(map(str, prompts_paths)))
        return 0

    try:
        reference_names = configure_style_reference(
            base,
            image=args.ref_image,
            directory=args.ref_dir,
            disabled=args.no_style_reference,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(f"スタイル参照の設定に失敗: {exc}")
        return 2

    output_prefixes = {}
    for node_id, save_node in base.items():
        if save_node.get("class_type") != "SaveImage":
            continue
        inputs = save_node["inputs"]
        prefix = Path(inputs.get("filename_prefix", "ComfyUI")).name
        if node_id == NODE_SAVE and args.prefix:
            prefix = args.prefix.rstrip("/\\")
        output_prefixes[node_id] = f"{prefix}_{stamp}"

    overrides = {
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "width": args.width,
        "height": args.height,
        "batch": args.batch,
        "ckpt": args.ckpt,
        "lora": args.lora,
        "lora_strength": args.lora_strength,
        "resize_mode": args.resize_mode,
        "prepend": args.prepend,
        "common_positive": common_positive,
        "common_negative": common_negative,
        "image": "",
        "denoise": args.denoise,
        "ipadapter_weight": args.ipadapter_weight,
        "ipadapter_weight_type": args.ipadapter_weight_type,
        "ipadapter_start": args.ipadapter_start,
        "ipadapter_end": args.ipadapter_end,
    }

    # img2img: 元画像を ComfyUI の input ディレクトリへコピーする。
    # LoadImage は input 配下のファイル名しか受け付けないため。
    if args.image:
        src = Path(args.image)
        if not src.exists():
            print("元画像が見つからない: " + str(src))
            return 2
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        dst = INPUT_DIR / src.name
        if not args.dry_run and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        overrides["image"] = src.name
        overrides["source_image"] = str(src)
        print("元画像: " + str(src) + (" -> " + str(dst) if not args.dry_run else " (dry-run のためコピーなし)"))

    print("プロンプト " + str(len(rows)) + " 件")
    if common_positive:
        print("共通ポジティブ: " + str(args.common_positive)
              + " (" + str(len(common_positive.split(","))) + " タグ)")
    if common_negative:
        print("共通ネガティブ: " + str(args.common_negative)
              + " (" + str(len(common_negative.split(","))) + " タグ)")

    def iter_workflows():
        for path, row in rows:
            seed = args.seed if args.seed >= 0 else random.randint(0, SEED_MAX)
            wf = build_workflow(base, row, seed, overrides)
            for node_id, prefix in output_prefixes.items():
                wf[node_id]["inputs"]["filename_prefix"] = f"{path.stem}/{prefix}"
            yield path, row, seed, wf

    # --- dry-run: 組み立てだけ ---
    if args.dry_run:
        if args.face_refine:
            print("顔補修: 顔のマスクを自動作成（denoise 0.45）")
        if args.larger_eyes:
            print("目元補修: 両目を検出してマスクを作成（denoise 0.45–0.60）")
        for stem in dict.fromkeys(path.stem for path, _ in rows):
            for prefix in output_prefixes.values():
                print(f"出力ファイル名の接頭辞: {stem}/{prefix}")
        for path, row, seed, wf in iter_workflows():
            eff = effective_values(wf, reference_names)
            print("")
            source = f" csv={path}" if len(prompts_paths) > 1 else ""
            print("--- id=" + row.id + " seed=" + str(seed) + source + " ---")
            print("positive: " + str(wf[NODE_POS]["inputs"]["text"]))
            print("negative: " + str(wf[NODE_NEG]["inputs"]["text"]))
            if NODE_LORA in wf:
                li = wf[NODE_LORA]["inputs"]
                print(
                    "lora: " + str(li.get("lora_name"))
                    + " strength=" + str(li.get("strength_model"))
                )
            print(
                "size=" + str(eff["width"]) + "x" + str(eff["height"])
                + " batch=" + str(eff["batch"])
                + " steps=" + str(eff["steps"])
                + " cfg=" + str(eff["cfg"])
                + " sampler=" + str(eff["sampler"])
                + " scheduler=" + str(eff["scheduler"])
                + " denoise=" + str(eff["denoise"])
            )
            if eff.get("image"):
                print("image: " + str(eff["image"]))
            if eff.get("ref_image"):
                print(
                    "ipadapter: ref=" + str(eff["ref_image"])
                    + " weight=" + str(eff["ipadapter_weight"])
                    + " type=" + str(eff["ipadapter_weight_type"])
                )
        print("")
        print("dry-run のため投入なし")
        return 0

    # --- 起動確認 ---
    client = ComfyClient(args.server)
    ok, msg = client.check_ready()
    print(msg)
    if not ok:
        return 3

    refiners = []
    try:
        if args.face_refine:
            from refine_faces import FaceRefiner

            refiners.append((FaceRefiner(ROOT, INPUT_DIR, RAW_DIR), "顔補修"))
        if args.larger_eyes:
            from refine_eyes import EyeRefiner

            refiners.append((EyeRefiner(ROOT, INPUT_DIR, RAW_DIR), "目元補修"))
    except (ImportError, OSError) as exc:
        print(f"補修の初期化に失敗: {exc}")
        return 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / ("gen_" + stamp + ".csv")

    records: list[GenRecord] = []
    failures = 0

    for i, (path, row, seed, wf) in enumerate(iter_workflows(), 1):
        eff = effective_values(wf, reference_names)

        rec = GenRecord(
            prompt_id="",
            row_id=row.id,
            positive=wf[NODE_POS]["inputs"]["text"],
            negative=wf[NODE_NEG]["inputs"]["text"],
            seed=seed,
            width=int(eff["width"]),
            height=int(eff["height"]),
            steps=int(eff["steps"]),
            cfg=float(eff["cfg"]),
            sampler=str(eff["sampler"]),
            scheduler=str(eff["scheduler"]),
            batch=int(eff["batch"]),
            denoise=float(eff["denoise"]),
            image=str(eff.get("image") or ""),
            ipadapter_weight=float(eff.get("ipadapter_weight") or 0.0),
            ipadapter_weight_type=str(eff.get("ipadapter_weight_type") or ""),
            ref_image=str(eff.get("ref_image") or ""),
            source_csv=str(path),
        )

        t0 = time.time()
        try:
            pid = client.queue_prompt(wf)
            rec.prompt_id = pid
            entry = client.wait(pid, poll=args.poll, timeout=args.timeout)
            files = client.output_files(entry)
            rec.files = "|".join(files)
            for refiner, label in refiners:
                for index, filename in enumerate(files):
                    refined = refiner.refine(client, wf, filename, args.poll, args.timeout)
                    if refined is None:
                        print(f"  {label}スキップ（対象が小さいか、領域を特定できない）: {filename}")
                        continue
                    files[index] = refined
                    rec.files = "|".join(files)
            rec.status = "OK"
        except Exception as e:
            rec.status = "FAIL"
            rec.error = repr(e)[:1000]
            failures += 1
        rec.seconds = round(time.time() - t0, 2)
        records.append(rec)

        print(
            "[" + str(i) + "/" + str(len(rows)) + "] id=" + row.id
            + " seed=" + str(seed)
            + " " + rec.status
            + " " + str(rec.seconds) + "s"
            + ("  " + rec.files if rec.files else "")
            + ("  " + rec.error if rec.error else "")
        )

        if rec.status == "FAIL" and args.stop_on_error:
            print("--stop-on-error のため中断")
            break

    # --- ログ出力 ---
    fields = [
        "prompt_id", "row_id", "positive", "negative", "seed",
        "width", "height", "steps", "cfg", "sampler", "scheduler",
        "batch", "denoise", "image", "files", "seconds", "status", "error",
        "ipadapter_weight", "ipadapter_weight_type", "ref_image", "source_csv",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in fields})

    ok_count = sum(1 for r in records if r.status == "OK")
    total_imgs = sum(r.batch for r in records if r.status == "OK")
    print("")
    print("成功 " + str(ok_count) + " / " + str(len(records)) + " 件、想定 " + str(total_imgs) + " 枚")
    print("ログ: " + str(csv_path))
    print("出力: " + str(RAW_DIR))

    if failures and failures == len(records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
