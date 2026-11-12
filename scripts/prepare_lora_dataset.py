"""Create a separate captioned dataset; source favorites are never modified."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = [(768,1344),(832,1216),(896,1152),(1024,1024),(1152,896),(1216,832),(1344,768)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=ROOT/'output/favorites')
    ap.add_argument('--out', type=Path, default=ROOT/'input/rei_style_v3_dataset')
    ap.add_argument('--trigger', default='rei_style')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
    cache = ROOT/'models/tagger'
    model = hf_hub_download('SmilingWolf/wd-vit-tagger-v3', 'model.onnx', local_dir=cache)
    tags = hf_hub_download('SmilingWolf/wd-vit-tagger-v3', 'selected_tags.csv', local_dir=cache)
    with open(tags, encoding='utf-8') as f:
        names = list(csv.DictReader(f))
    # CPU tagger can run alongside ComfyUI without competing for VRAM.
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 6
    session = ort.InferenceSession(model, sess_options=opts, providers=['CPUExecutionProvider'])
    inp = session.get_inputs()[0]
    size = inp.shape[1]
    rows, seen = [], set()
    for p in sorted(a.source.rglob('*')):
        if p.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}: continue
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            digest = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            if digest in seen: continue
            seen.add(digest)
            w,h = im.size
            bw,bh = min(BUCKETS, key=lambda b: abs(np.log((w/h)/(b[0]/b[1]))))
            side = max(w,h)
            canvas = Image.new('RGB',(side,side),'white'); canvas.paste(im,((side-w)//2,(side-h)//2))
            rgb = np.asarray(canvas.resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)
            probs = session.run(None,{inp.name:rgb[:,:,::-1][None]})[0][0]
            general = [(float(prob),tag['name'].replace('_',' ')) for tag,prob in zip(names,probs)
                       if tag['category']=='0' and prob >= 0.35]
            general.sort(reverse=True)
            caption = a.trigger+', '+', '.join(t for _,t in general[:45])
            dest = a.out/(digest[:16]+'.png')
            if not dest.exists(): im.save(dest)
        dest.with_suffix('.txt').write_text(caption,encoding='utf-8')
        rows.append(dict(image=str(dest),source=str(p),caption=caption,width=bw,height=bh,sha256=digest))
        print('caption',len(rows),p.name,flush=True)
    # Deterministic holdout: never trained on, available for visual evaluation.
    for i,row in enumerate(rows): row['split']='validation' if i%10==0 else 'train'
    (a.out/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('dataset',len(rows),'train',sum(r['split']=='train' for r in rows),flush=True)


if __name__=='__main__': main()
