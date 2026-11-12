"""train_lora.py — 個別キャプション付きデータから SDXL LoRA を学習する

favorites の画像を「この見た目が好き」として LoRA に焼き込む。
生成時に LoraLoader で適用すると、プロンプト次第で自由にバリエーションを
出せる（img2img のように構図が縛られない）。

設計:
  - VAE 潜在と text 埋め込みは事前に計算してキャッシュする（学習ループを軽くする）。
  - UNet の attention 層だけを LoRA 化する（rank 8 で約 11.6M パラメータ）。
  - gradient checkpointing で VRAM 16GB に収める（実測ピーク 5.56GB）。
  - 保存は kohya 形式（ComfyUI の LoraLoader がそのまま読める）。

使い方:
    .\\venv\\Scripts\\python.exe scripts\\train_lora.py --epochs 1
    .\\venv\\Scripts\\python.exe scripts\\train_lora.py --epochs 10 --rank 16
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import math
import random
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"F:\ai-image")
CKPT = ROOT / "models" / "checkpoints" / "waiIllustriousSDXL_v170.safetensors"
LORA_DIR = ROOT / "models" / "loras"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def mem(tag: str) -> None:
    free, _ = torch.cuda.mem_get_info()
    print("[mem] %-18s free %.2fGB alloc %.2fGB" % (
        tag, free / 1024**3, torch.cuda.memory_allocated() / 1024**3))


def load_and_fit(path: Path, tw: int, th: int) -> np.ndarray:
    """アスペクト比を保って resize し、中央 crop する。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max(tw / w, th / h)
        nw, nh = max(tw, int(round(w * scale))), max(th, int(round(h * scale)))
        im = im.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        im = im.crop((left, top, left + tw, top + th))
        return np.array(im, copy=True)


def kohya_key(module_path: str, kind: str) -> str:
    """diffusers のモジュールパスを kohya 形式のキーに変換する。"""
    flat = module_path.replace(".", "_")
    return "lora_unet_%s.%s" % (flat, kind)


def build_lora_state(unet, cfg: dict) -> dict:
    """PEFT の LoRA 重みを kohya 形式の state_dict に変換する。"""
    from peft import get_peft_model_state_dict

    raw = get_peft_model_state_dict(unet)
    out = {}
    for key, tensor in raw.items():
        # 例: base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight
        if ".lora_A." in key:
            kind = "lora_down.weight"
            mod = key.split(".lora_A.")[0]
        elif ".lora_B." in key:
            kind = "lora_up.weight"
            mod = key.split(".lora_B.")[0]
        else:
            continue
        mod = mod.replace("base_model.model.", "")
        out[kohya_key(mod, kind)] = tensor.detach().to(torch.float16).cpu()
        if kind == "lora_down.weight":
            out[kohya_key(mod, "alpha")] = torch.tensor(float(cfg["alpha"]), dtype=torch.float16)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="favorites から SDXL LoRA を学習")
    ap.add_argument("--out", default=str(LORA_DIR / "rei_style_v3.safetensors"))
    ap.add_argument("--dataset", type=Path, default=ROOT / "input/rei_style_v3_dataset/manifest.json")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None, help="既定は rank と同じ")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=0, help="0 なら最後だけ保存")
    ap.add_argument("--dry-run", action="store_true", help="学習せず枚数と設定だけ表示")
    args = ap.parse_args(argv)

    alpha = args.alpha if args.alpha is not None else float(args.rank)
    if min(args.batch, args.accumulate, args.epochs, args.rank) <= 0:
        raise ValueError("batch, accumulate, epochs, rank must be positive")
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    data = [r for r in json.loads(args.dataset.read_text(encoding='utf-8')) if r['split'] == 'train']
    paths = [Path(r['image']) for r in data]
    groups = defaultdict(list)
    for i, r in enumerate(data): groups[(r['width'], r['height'])].append(i)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not paths:
        print("学習画像がない: " + str(args.dataset))
        return 1

    print("favorites: %d 枚" % len(paths))
    print("captions: per-image; bucket counts: " + str({str(k):len(v) for k,v in groups.items()}))
    print("rank=%d alpha=%.1f lr=%g epochs=%d batch=%d accumulate=%d" % (
        args.rank, alpha, args.lr, args.epochs, args.batch, args.accumulate))
    if args.dry_run:
        steps = math.ceil(sum(math.ceil(len(v)/args.batch) for v in groups.values()) / args.accumulate) * args.epochs
        print("予定 optimizer steps: %d; gradient accumulation: %d" % (steps, args.accumulate))
        return 0

    from diffusers import StableDiffusionXLPipeline

    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(CKPT), torch_dtype=torch.bfloat16, use_safetensors=True
    )
    print("モデルロード: %.1f 秒" % (time.time() - t0))

    # --- 潜在と text 埋め込みを事前計算 ---
    pipe.vae.to("cuda", dtype=torch.float32)
    latents, embs, pooleds = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        rgb = load_and_fit(p, data[i]['width'], data[i]['height'])
        px = torch.from_numpy(rgb).float() / 127.5 - 1.0
        px = px.permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float32)
        with torch.no_grad():
            lat = pipe.vae.encode(px).latent_dist.sample() * pipe.vae.config.scaling_factor
        if not torch.isfinite(lat).all(): raise FloatingPointError('invalid latent')
        latents.append(lat.squeeze(0).to(torch.bfloat16).cpu())
        del px
        if i % 10 == 0:
            print("  latent %d/%d" % (i + 1, len(paths)))
    pipe.vae.to("cpu")
    gc.collect(); torch.cuda.empty_cache()
    print("潜在計算: %.1f 秒" % (time.time() - t0))

    pipe.text_encoder.to('cuda'); pipe.text_encoder_2.to('cuda')
    with torch.no_grad():
        for r in data:
            tags = r['caption'].split(', ')
            while len(tags)>1 and any(len(tok(', '.join(tags)).input_ids)>tok.model_max_length
                                      for tok in (pipe.tokenizer,pipe.tokenizer_2)):
                tags.pop()
            r['caption'] = ', '.join(tags)
            emb, _, pooled, _ = pipe.encode_prompt(
                r['caption'], device="cuda", num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            embs.append(emb[0].to(dtype=torch.bfloat16).cpu())
            pooleds.append(pooled[0].to(dtype=torch.bfloat16).cpu())
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).with_suffix('.captions.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    gc.collect(); torch.cuda.empty_cache()
    mem("after precompute")

    # --- LoRA を付けて学習 ---
    from peft import LoraConfig, get_peft_model

    unet = pipe.unet.to("cuda")
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    lcfg = LoraConfig(
        r=args.rank, lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0, bias="none",
    )
    unet = get_peft_model(unet, lcfg)
    unet.train()
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print("trainable: %.2f M" % (trainable / 1e6))
    mem("after lora")

    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=args.lr)
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    if scheduler.config.prediction_type not in ('epsilon', 'v_prediction'):
        raise ValueError(scheduler.config.prediction_type)

    per_epoch = math.ceil(sum(math.ceil(len(v)/args.batch) for v in groups.values()) / args.accumulate)
    total = per_epoch * args.epochs
    step = 0
    losses = []
    t_start = time.time()
    for epoch in range(args.epochs):
        batches = []
        for ids in groups.values():
            ids = ids.copy(); random.shuffle(ids)
            batches.extend(ids[i:i+args.batch] for i in range(0,len(ids),args.batch))
        random.shuffle(batches)
        for offset in range(0,len(batches),args.accumulate):
            block = batches[offset:offset+args.accumulate]
            count = sum(map(len,block))
            opt.zero_grad(set_to_none=True)
            for ids in block:
                h,w = data[ids[0]]['height'],data[ids[0]]['width']
                lat = torch.stack([latents[i] for i in ids]).to('cuda')
                noise = torch.randn_like(lat)
                ts = torch.randint(0,scheduler.config.num_train_timesteps,(len(ids),),device='cuda')
                noisy = scheduler.add_noise(lat,noise,ts)
                add = {'text_embeds':torch.stack([pooleds[i] for i in ids]).to('cuda'),
                       'time_ids':torch.tensor([[h,w,0,0,h,w]],device='cuda',dtype=torch.bfloat16).repeat(len(ids),1)}
                pred = unet(noisy,ts,encoder_hidden_states=torch.stack([embs[i] for i in ids]).to('cuda'),added_cond_kwargs=add).sample
                target = noise if scheduler.config.prediction_type=='epsilon' else scheduler.get_velocity(lat,noise,ts)
                loss = torch.nn.functional.mse_loss(pred.float(),target.float())
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                (loss*len(ids)/count).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad],1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 2 == 0 or step == total:
                el = time.time() - t_start
                eta = el / step * (total - step)
                print("step %d/%d loss %.4f  %.1fs/step  ETA %.1f 分" % (
                    step, total, np.mean(losses[-10:]), el / step, eta / 60))
            if args.save_every and step % args.save_every == 0:
                save_lora(unet, lcfg, args, alpha, step, suffix="_step%05d" % step)
        save_lora(unet,lcfg,args,alpha,step,suffix='_epoch%02d' % (epoch+1))

    save_lora(unet, lcfg, args, alpha, total)
    print("完了: %s" % args.out)
    print("loss 平均 %.4f / 最終10件 %.4f" % (np.mean(losses), np.mean(losses[-10:])))
    Path(args.out).with_suffix('.training.json').write_text(json.dumps(dict(
        images=len(paths),epochs=args.epochs,steps=step,losses=losses,seconds=time.time()-t_start,
        peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3),indent=2))
    return 0


def save_lora(unet, lcfg, args, alpha: float, step: int, suffix: str = "") -> None:
    from safetensors.torch import save_file

    out = Path(args.out)
    if suffix:
        out = out.with_name(out.stem + suffix + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = build_lora_state(unet, {"alpha": alpha})
    meta = {
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(args.rank),
        "ss_network_alpha": str(alpha),
        "ss_num_train_images": str(sum(r['split']=='train' for r in json.loads(args.dataset.read_text(encoding='utf-8')))),
        "ss_epoch": str(args.epochs),
        "ss_steps": str(step),
        "ss_learning_rate": str(args.lr),
        "ss_output_name": out.stem,
        "ss_training_comment": 'rei_style; per-image WD-v3 captions; aspect buckets; held-out validation',
        "dataset_manifest": str(args.dataset),
        "seed": str(args.seed),
    }
    save_file(state, str(out), metadata=meta)
    print("保存: %s (%d keys, %.1f MB)" % (out, len(state), out.stat().st_size / 1024**2))


if __name__ == "__main__":
    raise SystemExit(main())
