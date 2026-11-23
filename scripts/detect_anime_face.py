"""Anime-specific YOLO face detection using ONNX Runtime (no OpenCV dependency)."""
import json
from functools import lru_cache
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from PIL import Image
import onnxruntime as ort


@lru_cache(maxsize=1)
def load_session():
    model=ROOT/'models/face/face_detect_v1.4_n/model.onnx'
    if not model.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download('deepghs/anime_face_detection','face_detect_v1.4_n/model.onnx',local_dir=ROOT/'models/face')
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    return ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])


def detect_image(im):
    session=load_session()
    im=im.convert('RGB');w,h=im.size
    scale=640/max(w,h);nw,nh=round(w*scale),round(h*scale)
    dx,dy=(640-nw)//2,(640-nh)//2
    canvas=Image.new('RGB',(640,640),(114,114,114))
    canvas.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),(dx,dy))
    x=np.asarray(canvas,dtype=np.float32).transpose(2,0,1)[None]/255
    pred=session.run(None,{'images':x})[0][0].T
    if pred.shape[1]!=5:raise ValueError('Unexpected YOLO output shape')
    pred=pred[pred[:,4]>=.278]
    boxes=[]
    for cx,cy,bw,bh,score in sorted(pred,key=lambda r:-r[4]):
        box=np.array([max(0,(cx-bw/2-dx)/scale),max(0,(cy-bh/2-dy)/scale),
                      min(w,(cx+bw/2-dx)/scale),min(h,(cy+bh/2-dy)/scale)])
        if box[2]<=box[0] or box[3]<=box[1]:continue
        area=(box[2]-box[0])*(box[3]-box[1])
        duplicate=False
        for other in boxes:
            inter=np.maximum(0,np.minimum(box[2:],other[2:])-np.maximum(box[:2],other[:2])).prod()
            other_area=(other[2]-other[0])*(other[3]-other[1])
            if inter/(area+other_area-inter)>0.5:duplicate=True;break
        if not duplicate:boxes.append(box)
    return [[round(b[0]),round(b[1]),round(b[2]-b[0]),round(b[3]-b[1])] for b in boxes]


def detect(path):
    with Image.open(path) as image:
        return detect_image(image)

if __name__=='__main__':
    print(json.dumps(detect(sys.argv[1])))
