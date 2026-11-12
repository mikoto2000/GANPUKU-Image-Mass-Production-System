"""pref_embed.py — SigLIP 埋め込みの抽出

aesthetic-predictor-v2-5 の内部 SigLIP vision tower をそのまま使う。
追加パッケージは不要。1152 次元の埋め込みを返す。

train_preference.py と select_images.py の両方から使う。
"""

from __future__ import annotations

import numpy as np


class SiglipEmbedder:
    """SigLIP vision tower による画像埋め込み。"""

    def __init__(self, device: str = "cuda"):
        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        self.torch = torch
        self.device = device
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).to(device).eval()

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        """RGB の numpy 配列 (H, W, 3) から 1152 次元の埋め込みを返す。"""
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
