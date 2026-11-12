"""Reproducible ComfyUI comparisons and selected-image finishing.

Every result stores the complete API graph, seed and output paths in results.json.
Re-running skips successful cases. No original images or models are replaced.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import requests
from filelock import FileLock
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from generate_images import ComfyClient, load_workflow, ROOT

SHORT_NEG = 'worst quality, bad quality, lowres, blurry, bad anatomy, bad hands, extra digits, watermark, signature'
RUN = ROOT/'output/quality_20261031'


def execute(wf, case, directory=RUN):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'results.json'
    records=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    signature=hashlib.sha256(json.dumps(wf,sort_keys=True).encode()).hexdigest()
    for r in records:
        if r['case']==case and r.get('signature')==signature and all(Path(p).exists() for p in r['files']):return r
    client=ComfyClient('http://127.0.0.1:8188')
    start=time.time();pid=client.queue_prompt(wf);entry=client.wait(pid,timeout=1200,poll=0.5)
    files=[]
    for out in entry.get('outputs',{}).values():
        for im in out.get('images',[]):
            if im.get('type')=='output':files.append(str(ROOT/'output/raw'/im.get('subfolder','')/im['filename']))
    if not files:raise RuntimeError('No saved images: '+case)
    rec=dict(case=case,signature=signature,prompt_id=pid,seconds=round(time.time()-start,2),files=files,workflow=wf)
    # Merge again after generation so simultaneous producers cannot drop records.
    with FileLock(str(path)+'.lock'):
        records=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        records=[r for r in records if r['case']!=case]+[rec]
        temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)
    print(case,rec['seconds'],files,flush=True)
    return rec


def comparison(model, lora=''):
    import csv
    with open(ROOT/'prompts/prompts.csv',encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
    positive=row['positive']
    prompts=[positive,positive.replace('cowboy shot','upper body').replace('open mouth','closed mouth')+', hand in hair, head tilt']
    for pi,pos in enumerate(prompts):
        for seed in (12345,54321,20261031):
            variants=[('base',5.5,832,1216,row['negative'])]
            if model=='wai' and not lora:
                variants += [('cfg5',5,832,1216,row['negative']),('cfg6',6,832,1216,row['negative']),
                             ('size896',5.5,896,1152,row['negative']),('shortneg',5.5,832,1216,SHORT_NEG)]
            if model=='noob': variants=[('shortneg',5.5,832,1216,SHORT_NEG)]
            for name,cfg,w,h,neg in variants:
                wf=load_workflow(ROOT/'workflows'/('sdxl_lora_api.json' if lora else 'sdxl_base_api.json'))
                wf['1']['inputs']['ckpt_name']='NoobAI-XL-v1.1.safetensors' if model=='noob' else 'waiIllustriousSDXL_v170.safetensors'
                wf['2']['inputs']['text']=('rei_style, ' if lora or model=='wai_trigger' else '')+pos
                wf['3']['inputs']['text']=neg
                wf['4']['inputs'].update(width=w,height=h,batch_size=1)
                wf['5']['inputs'].update(seed=seed,cfg=cfg)
                label=Path(lora).stem if lora else model
                if lora:wf['8']['inputs'].update(lora_name=lora,strength_model=0.5,strength_clip=0)
                case=f'{label}_p{pi}_s{seed}_{name}'
                wf['7']['inputs']['filename_prefix']='quality_20261031/comparison/'+case
                execute(wf,case)
    contact_sheet()


def contact_sheet():
    records=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
    records=[r for r in records if '_p' in r['case']]
    for start in range(0,len(records),15):
        subset=records[start:start+15];sheet=Image.new('RGB',(5*256,((len(subset)+4)//5)*408),'#20232b')
        draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(subset):
            with Image.open(r['files'][-1]) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(250,366))
            x=(i%5)*256;y=(i//5)*408;sheet.paste(thumb,(x+(256-thumb.width)//2,y))
            draw.text((x+3,y+369),r['case'].replace('_s','\ns'),fill='white')
        sheet.save(RUN/f'comparison_{start//15+1:02d}.jpg')


def finish(source, scale=1.5,denoise=0.25,face=False):
    with Image.open(source) as im:
        info=im.info.copy();w,h=ImageOps.exif_transpose(im).size
    source_wf=json.loads(info['prompt']) if isinstance(info.get('prompt'),str) else None
    if not source_wf:raise ValueError('Source requires ComfyUI prompt metadata')
    # Preserve the original checkpoint, prompts and LoRA model connections.
    wf=copy.deepcopy(source_wf)
    sampler=next(n for n in wf.values() if n.get('class_type')=='KSampler')['inputs']
    decode=next(k for k,n in wf.items() if n.get('class_type')=='VAEDecode')
    decode_vae=wf[decode]['inputs']['vae']
    name='quality_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.png'
    shutil.copy2(source,ROOT/'input'/name)
    wf['901']={'class_type':'LoadImage','inputs':{'image':name}}
    wf['902']={'class_type':'ImageScale','inputs':{'image':['901',0],'upscale_method':'lanczos',
                    'width':round(w*scale/8)*8,'height':round(h*scale/8)*8,'crop':'disabled'}}
    wf['903']={'class_type':'VAEEncode','inputs':{'pixels':['902',0],'vae':decode_vae}}
    sampler.update(latent_image=['903',0],denoise=denoise,steps=28)
    # Only save the final decode. Other SaveImage nodes could reference unrelated latents.
    for k in list(wf):
        if wf[k].get('class_type')=='SaveImage':del wf[k]
    case=source.stem+f'_finish_d{denoise:.2f}'
    wf['904']={'class_type':'SaveImage','inputs':{'images':[decode,0],'filename_prefix':'quality_20261031/finished/'+case}}
    rec=execute(wf,case,RUN/'finishing')
    target=ROOT/'output/upscaled'/Path(rec['files'][0]).name
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(rec['files'][0],target)
    if face:face_refine(target,source_wf)
    return target


def face_refine(source, source_wf):
    """Anime face detection, cropped diffusion, feathered paste; skip ambiguous detections."""
    import subprocess
    import sys
    with Image.open(source) as im: original=im.convert('RGB');metadata=im.info.copy()
    result=subprocess.run([sys.executable,str(ROOT/'scripts/detect_anime_face.py'),str(source)],
                          check=True,capture_output=True,text=True)
    boxes=json.loads(result.stdout)
    if len(boxes)!=1:
        print('face pass skipped; detections',len(boxes),source.name,flush=True);return
    x,y,w,h=map(int,boxes[0]);side=int(max(w,h)*1.6)
    left=max(0,x+w//2-side//2);top=max(0,y+h//2-side//2)
    right=min(original.width,left+side);bottom=min(original.height,top+side)
    crop=original.crop((left,top,right,bottom))
    name=source.stem+'_face_input.png'
    # Feed the same source graph metadata to the finishing helper.
    from PIL.PngImagePlugin import PngInfo
    meta=PngInfo();meta.add_text('prompt',json.dumps(source_wf))
    crop_path=ROOT/'input'/name;crop.save(crop_path,pnginfo=meta)
    factor=768/max(crop.size)
    refined=finish(crop_path,scale=factor,denoise=0.20,face=False)
    with Image.open(refined) as im: patch=im.convert('RGB').resize(crop.size,Image.Resampling.LANCZOS)
    mask=Image.new('L',crop.size,0);d=ImageDraw.Draw(mask)
    border=max(4,int(min(crop.size)*.10));d.rectangle((border,border,crop.width-border,crop.height-border),fill=255)
    mask=mask.filter(ImageFilter.GaussianBlur(border/2))
    original.paste(patch,(left,top),mask)
    out=source.with_stem(source.stem+'_face')
    meta=PngInfo()
    for k in ('prompt','workflow'):
        if k in metadata:meta.add_text(k,metadata[k])
    meta.add_text('face_refinement',json.dumps(dict(box=[left,top,right,bottom],denoise=.20,patch=str(refined))))
    original.save(out,pnginfo=meta)
    print('face refined',out,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['compare','finish','sheet'])
    ap.add_argument('--model',choices=['wai','noob','wai_trigger'],default='wai')
    ap.add_argument('--lora',default='')
    ap.add_argument('--image',type=Path)
    ap.add_argument('--scale',type=float,default=1.5)
    ap.add_argument('--denoise',type=float,default=.25)
    ap.add_argument('--face',action='store_true')
    a=ap.parse_args()
    if a.stage=='compare':comparison(a.model,a.lora)
    elif a.stage=='finish':
        if not a.image:ap.error('--image required')
        finish(a.image,a.scale,a.denoise,a.face)
    else:contact_sheet()


if __name__=='__main__':main()
