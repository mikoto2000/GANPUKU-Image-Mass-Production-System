"""公式Anima Base / Aestheticと補助モデルを、SHA256検証付きで配置する。"""

import argparse
import hashlib
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
REVISION = "6e0b2da2112573380abfb6a1435a60a994b254f2"
DIFFUSION_MODELS = {
    "base": ("diffusion_models/anima-base-v1.0.safetensors",
             "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"),
    "aesthetic": ("diffusion_models/anima-aesthetic-v1.1.safetensors",
                  "3c1868387a3a1ff504bbb87c33678321965ead381fcf87afbd0264daa600c082"),
}
MODELS = {
    "text_encoders/qwen_3_06b_base.safetensors":
        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    "vae/qwen_image_vae.safetensors":
        "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=DIFFUSION_MODELS, default="base",
                        help="導入するAnimaの種類（既定: base）")
    args = parser.parse_args()
    for name, expected in [DIFFUSION_MODELS[args.variant], *MODELS.items()]:
        target = ROOT / "ComfyUI" / "models" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != expected:
                raise ValueError(f"SHA256 mismatch: {target}")
            print(f"Verified: {target}", flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".part")
        url = f"https://huggingface.co/circlestone-labs/Anima/resolve/{REVISION}/split_files/{name}"
        print(f"Downloading: {name}", flush=True)
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            downloaded = 0
            report_at = 256 * 1024 * 1024
            with partial.open("wb") as stream:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= report_at:
                        print(f"  {downloaded / 1024**2:.0f} MiB", flush=True)
                        report_at += 256 * 1024 * 1024
        if sha256(partial) != expected:
            raise ValueError(f"SHA256 mismatch: {partial}")
        partial.replace(target)
        print(f"Verified and saved: {target}", flush=True)


if __name__ == "__main__":
    main()
