"""Score the final comparison set; metrics assist, never replace visual review."""
import csv
import json
import math
from pathlib import Path
import shutil
from collections import defaultdict
import numpy as np
import requests
import torch
from select_images import AestheticScorer,AnatomyScorer,detail_score,load_image,compute_phash,is_degenerate_phash,hamming
from quality_pipeline import RUN


def main():
    torch.set_num_threads(6)
    q=requests.get('http://127.0.0.1:8188/queue',timeout=10).json()
    if q['queue_running'] or q['queue_pending']:raise RuntimeError('Wait for generation queue to finish')
    requests.post('http://127.0.0.1:8188/free',json={'unload_models':True,'free_memory':True},timeout=30).raise_for_status()
    aesthetic=AestheticScorer('cuda');anatomy=AnatomyScorer(device='cpu')
    rows=[]
    for r in json.loads((RUN/'results.json').read_text(encoding='utf-8')):
        rgb,_=load_image(Path(r['files'][-1]))
        aes=aesthetic.score(rgb);ana,person,visible,notes=anatomy.score(rgb)
        group=r['case'].split('_p')[0]+'_'+r['case'].rsplit('_',1)[-1]
        rows.append(dict(case=r['case'],group=group,file=r['files'][-1],aesthetic=aes,anatomy=ana,
                         detail=detail_score(rgb),person=person,notes=';'.join(notes),seconds=r['seconds'],
                         phash=compute_phash(rgb),duplicate_of=''))
        print(r['case'],round(aes,3),round(ana,3),flush=True)
    # Same normalization across all compared images.
    for key in ('aesthetic','anatomy','detail'):
        values=np.array([r[key] for r in rows]);lo,hi=np.percentile(values,[5,95])
        for r in rows:r[key+'_norm']=float(np.clip((r[key]-lo)/(hi-lo),0,1)) if hi>lo else .5
    for r in rows:r['total']=.75*r['aesthetic_norm']+.15*r['anatomy_norm']+.10*r['detail_norm']
    kept=[]
    for r in sorted(rows,key=lambda r:-r['total']):
        duplicate=next((k for k in kept if not is_degenerate_phash(r['phash'])
                        and not is_degenerate_phash(k['phash']) and hamming(r['phash'],k['phash'])<=6),None)
        if duplicate:r['duplicate_of']=duplicate['case']
        else:kept.append(r)
    with open(RUN/'scores.csv','w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(r)
    summary={g:dict(count=len(rs),aesthetic_mean=float(np.mean([r['aesthetic'] for r in rs])),
                   anatomy_mean=float(np.mean([r['anatomy'] for r in rs])),
                   seconds_mean=float(np.mean([r['seconds'] for r in rs]))) for g,rs in groups.items()}
    (RUN/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    selected=RUN/'selected_final';selected.mkdir(exist_ok=True)
    chosen=kept[:max(1,round(len(kept)*.3))]
    previous_path=RUN/'selection.json'
    previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    chosen_names={Path(r['file']).name for r in chosen}
    for old in previous.get('selected',[]):
        name=Path(old['file']).name
        if name not in chosen_names:
            stale=selected/name
            if stale.exists():stale.unlink()
    (RUN/'selection.json').write_text(json.dumps(dict(total=len(rows),unique=len(kept),selected=chosen),indent=2),encoding='utf-8')
    for r in chosen:
        shutil.copy2(r['file'],selected/Path(r['file']).name)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
