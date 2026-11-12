"""Finish dependent GPU stages after training; keep logs and fail on any error."""
import json
from pathlib import Path
import subprocess
import sys
import time
from quality_pipeline import comparison, execute, ROOT, RUN
from generate_images import load_workflow, build_workflow, PromptRow

if __name__=='__main__':
    deadline=time.time()+2400
    while not (ROOT/'models/loras/rei_style_v3.training.json').exists():
        if time.time()>deadline:raise TimeoutError('LoRA training did not finish')
        log=ROOT/'logs/quality_lora_v3.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf-8',errors='replace'):
            raise RuntimeError('Training failed; inspect quality_lora_v3.log')
        time.sleep(3)
    time.sleep(3)
    comparison('wai','rei_style_v2_step00074.safetensors')
    comparison('wai','rei_style_v3.safetensors')
    comparison('wai_trigger')
    # Exercise the reusable high-resolution workflow through the public builder.
    import csv
    with open(ROOT/'prompts/quality.csv',encoding='utf-8-sig') as f:row=next(csv.DictReader(f))
    wf=build_workflow(load_workflow(ROOT/'workflows/sdxl_hires_api.json'),
        PromptRow('hires_smoke',row['positive'],row['negative']),12345,{'prefix':'quality_20261031/smoke/hires'})
    execute(wf,'hires_smoke',RUN/'smoke')
    # Full final dataset, including the new LoRA and all six NoobAI cases.
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/evaluate_quality.py')],check=True)
    (RUN/'COMPLETED.json').write_text(json.dumps({'status':'complete','stages':[
        'aspect_preservation','setting_comparison','model_comparison','hires','face_refinement',
        'caption_dataset','lora_training','lora_comparison','selection','regression_tests']}),encoding='utf-8')
