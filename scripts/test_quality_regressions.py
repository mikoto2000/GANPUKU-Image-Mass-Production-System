"""Regression checks for image distortion and dynamic-pose selection."""
import unittest
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from generate_images import build_workflow, PromptRow, load_workflow, load_prompts, ROOT
from select_images import AnatomyScorer, AnatomyConfig
import quality_pipeline


class QualityRegressions(unittest.TestCase):
    def test_csv_column_shift_is_rejected_before_generation(self):
        with TemporaryDirectory(dir=ROOT/'input') as td:
            p=Path(td)/'bad.csv'
            p.write_text('id,positive,negative,width,height,steps,cfg,sampler,scheduler,batch,note\n'
                         '001,portrait,832,1216,28,5.5,euler_ancestral,normal,1,note\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'2 行目の列数が不正'):
                load_prompts(p)

    def test_csv_explicit_empty_negative_preserves_sampler(self):
        with TemporaryDirectory(dir=ROOT/'input') as td:
            p=Path(td)/'good.csv'
            p.write_text('id,positive,negative,width,height,steps,cfg,sampler,scheduler,batch,note\n'
                         '001,portrait,,832,1216,28,5.5,euler_ancestral,normal,1,note\n',encoding='utf-8')
            row=load_prompts(p)[0]
            self.assertEqual((row.negative,row.width,row.height,row.sampler,row.scheduler),
                             ('',832,1216,'euler_ancestral','normal'))

    def test_concurrent_generation_keeps_both_records(self):
        barrier=threading.Barrier(2)
        class Client:
            def __init__(self,*args):pass
            def queue_prompt(self,wf):return wf['name']
            def wait(self,pid,**kwargs):
                barrier.wait(timeout=5)
                return {'outputs':{'7':{'images':[{'type':'output','filename':pid+'.png'}]}}}
        with TemporaryDirectory(dir=ROOT/'input') as td, patch.object(quality_pipeline,'ComfyClient',Client):
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(quality_pipeline.execute,{'name':case},case,Path(td)) for case in ('a','b')]
                for f in futures:f.result()
            records=json.loads((Path(td)/'results.json').read_text(encoding='utf-8'))
            self.assertEqual({r['case'] for r in records},{'a','b'})

    def test_img2img_preserves_landscape_and_portrait(self):
        with TemporaryDirectory(dir=ROOT/'input') as td:
            for size in ((1024,512),(512,1024)):
                p=Path(td)/'source.png';Image.new('RGB',size).save(p)
                wf=build_workflow(load_workflow(ROOT/'workflows/sdxl_img2img_api.json'),
                    PromptRow('test','portrait',width=832,height=1216),42,
                    {'source_image':str(p),'image':p.name})
                scale=wf['12']['inputs']
                self.assertAlmostEqual(scale['width']/scale['height'],size[0]/size[1],delta=.02)
                self.assertLess(abs(scale['width']*scale['height']/(832*1216)-1),.025)
                self.assertEqual(scale['width']%8,0)

    def test_crop_uses_requested_dimensions(self):
        wf=build_workflow(load_workflow(ROOT/'workflows/sdxl_img2img_api.json'),
            PromptRow('test','portrait',width=896,height=1152),42,{'resize_mode':'crop'})
        self.assertEqual(wf['12']['inputs']['crop'],'center')
        self.assertEqual(wf['12']['inputs']['width'],896)

    def test_hires_fixed_seed_and_cfg_reach_second_pass(self):
        wf=build_workflow(load_workflow(ROOT/'workflows/sdxl_hires_api.json'),
            PromptRow('test','portrait',cfg=6),54321,{})
        self.assertEqual(wf['25']['inputs']['seed'],54321)
        self.assertEqual(wf['25']['inputs']['cfg'],6)
        self.assertEqual(wf['25']['inputs']['denoise'],.25)

    def test_dynamic_pose_not_penalized_for_asymmetry(self):
        scorer=AnatomyScorer.__new__(AnatomyScorer)
        # Raise one wrist: valid asymmetry must not itself lower the score.
        kp=np.array([[.5,.1],[.5,.2],[.35,.25],[.2,.35],[.1,.2],
            [.65,.25],[.7,.45],[.65,.65],[.4,.6],[.4,.8],[.4,1],
            [.6,.6],[.6,.8],[.6,1],[.47,.1],[.53,.1],[.44,.12],[.56,.12]])
        scorer.detector=lambda rgb:dict(bodies=dict(candidate=kp,score=np.ones((1,18))))
        scorer.cfg=AnatomyConfig()
        score,detected,_,notes=scorer.score(np.zeros((64,64,3),dtype=np.uint8))
        self.assertTrue(detected)
        self.assertFalse(any('asymmetry' in n for n in notes))
        self.assertGreater(score,.7)


if __name__=='__main__':unittest.main()
