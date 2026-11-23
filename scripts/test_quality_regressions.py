"""Regression checks for image distortion and dynamic-pose selection."""
import csv
import unittest
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image, ImageDraw
import generate_images
from generate_images import build_workflow, PromptRow, load_workflow, load_prompts, ROOT
from refine_eyes import align_visible_irises, eye_positive, skin_safe_blend
from refine_faces import enlarge_eye_guide, face_masks, face_positive, feature_blend
from select_images import AnatomyScorer, AnatomyConfig
import quality_pipeline


class QualityRegressions(unittest.TestCase):
    def test_face_guide_enlarges_both_eyes_and_leaves_distant_pixels_exact(self):
        source = Image.new("RGB", (180, 110), (198, 144, 112))
        draw = ImageDraw.Draw(source)
        centers = ((52, 45), (126, 48))
        outline = np.array([[-16, 0], [-8, -8], [8, -8], [16, 0], [8, 8], [-8, 8]])
        for x, y in centers:
            draw.ellipse((x - 11, y - 7, x + 11, y + 7), fill=(12, 35, 42))
        guide = enlarge_eye_guide(source, [outline + center for center in centers])
        for x, y in centers:
            box = (x - 22, y - 18, x + 22, y + 18)
            count = lambda image: np.count_nonzero(np.asarray(image.crop(box))[:, :, 0] < 100)
            self.assertGreater(count(guide), count(source))
        np.testing.assert_array_equal(np.asarray(guide)[90:], np.asarray(source)[90:])

    def test_face_mask_keeps_hair_and_neck_outside_the_blend(self):
        source = Image.new("RGB", (160, 180), (253, 253, 250))
        draw = ImageDraw.Draw(source)
        skin = (179, 119, 84)
        draw.rectangle((58, 138, 102, 179), fill=skin)
        draw.ellipse((20, 10, 140, 155), fill=skin)
        draw.polygon([(18, 6), (92, 6), (52, 61), (25, 77)], fill=(38, 39, 46))
        draw.ellipse((47, 71, 66, 82), fill=(45, 26, 22))
        draw.ellipse((94, 71, 113, 82), fill=(45, 26, 22))
        jaw = np.array([[20, 64], [24, 103], [38, 129], [58, 148], [80, 155],
                        [102, 148], [122, 129], [136, 103], [140, 64]])

        _, silhouette = face_masks(source, jaw)
        points = np.zeros((68, 2))
        scores = np.zeros(68)
        eye = np.array([[-8, 0], [-4, -4], [4, -4], [8, 0], [4, 4], [-4, 4]])
        points[36:42] = eye + (55, 77)
        points[42:48] = eye + (103, 77)
        scores[36:48] = 1
        points[48:60] = [(65, 120), (68, 116), (74, 115), (80, 115), (86, 115),
                         (92, 116), (95, 120), (92, 124), (86, 125), (80, 126),
                         (74, 125), (68, 124)]
        scores[48:60] = 1
        draw.line((65, 120, 95, 120), fill=(72, 30, 25), width=1)
        blend = feature_blend(silhouette, points, scores, source)
        self.assertEqual(blend.getpixel((36, 29)), 0)
        self.assertEqual(blend.getpixel((80, 176)), 0)
        self.assertEqual(blend.getpixel((6, 105)), 0)
        self.assertGreater(blend.getpixel((55, 77)), 200)
        self.assertGreater(blend.getpixel((103, 77)), 200)
        self.assertEqual(blend.getpixel((78, 105)), 0)
        self.assertEqual(blend.getpixel((55, 94)), 0)
        self.assertEqual(blend.getpixel((80, 124)), 0)
        self.assertGreater(blend.getpixel((80, 120)), 200)

    def test_face_prompt_carries_over_skin_and_expression(self):
        prompt = face_positive("jade eyes, medium brown skin, relaxed smile, looking down, winter jacket")
        self.assertIn("(medium brown skin:1.35)", prompt)
        self.assertIn("relaxed smile", prompt)
        self.assertIn("looking down", prompt)
        self.assertNotIn("jade eyes", prompt)

    def test_eye_alignment_corrects_a_shifted_landmark_with_matching_irises(self):
        image = Image.new("RGB", (210, 160), (238, 173, 142))
        draw = ImageDraw.Draw(image)
        draw.ellipse((49, 79, 73, 105), fill=(6, 180, 153))
        draw.ellipse((122, 50, 150, 80), fill=(6, 180, 153))
        contour = np.array([[-10, 0], [-4, -4], [4, -4], [10, 0], [4, 4], [-4, 4]])
        eyes = [contour + (64, 92), contour + (113, 82)]

        aligned = align_visible_irises(image, eyes)
        np.testing.assert_array_equal(aligned[0], eyes[0])
        np.testing.assert_allclose(aligned[1].mean(axis=0), [136, 65], atol=1)
        self.assertGreater(np.ptp(aligned[1][:, 1]), np.ptp(eyes[1][:, 1]))

    def test_eye_blend_protects_skin_but_keeps_the_eyelid_editable(self):
        eye = np.array([[12, 19], [18, 16], [25, 16], [31, 19], [25, 23], [18, 23]])
        for skin in ((103, 64, 48), (241, 204, 181)):
            with self.subTest(skin=skin):
                source = Image.new("RGB", (44, 36), skin)
                blend = Image.new("L", source.size, 255)
                protected = skin_safe_blend(source, blend, [eye], [(6, 8, 37, 30)], (0, 0))
                self.assertEqual(protected.getpixel((22, 9)), 0)
                self.assertEqual(protected.getpixel((22, 30)), 0)
                self.assertEqual(protected.getpixel((31, 19)), 0)
                self.assertEqual(protected.getpixel((22, 19)), 255)
                self.assertGreater(protected.getpixel((22, 16)), 240)

    def test_eye_refinement_preserves_skin_tags_without_setting_iris_color(self):
        prompt = eye_positive("amber eyes, medium brown skin, (freckled skin:1.2), winter jacket")
        self.assertIn("(medium brown skin:1.35)", prompt)
        self.assertIn("(freckled skin:1.2)", prompt)
        self.assertNotIn("amber eyes", prompt)
        self.assertNotIn("winter jacket", prompt)

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

    def test_multiple_prompt_csvs_keep_source_and_order_across_slice(self):
        with TemporaryDirectory(dir=ROOT / "input") as td:
            directory = Path(td)
            first = directory / "first prompts.csv"
            second = directory / "second prompts.csv"
            first.write_text("id,positive\n00,skip\n01,forest\n", encoding="utf-8")
            second.write_text("id,positive\n01,coast\n02,skip\n", encoding="utf-8")
            with (
                patch.object(generate_images, "LOG_DIR", directory),
                patch.object(generate_images, "RAW_DIR", directory),
                patch.object(generate_images, "ComfyClient") as client_type,
            ):
                client = client_type.return_value
                client.check_ready.return_value = (True, "ready")
                client.output_files.return_value = []
                result = generate_images.main([
                    "--prompts", str(first), str(second),
                    "--start", "1", "--limit", "2", "--prefix", "test", "--seed", "42",
                ])
                self.assertEqual(result, 0)
                prefixes = [
                    call.args[0][generate_images.NODE_SAVE]["inputs"]["filename_prefix"]
                    for call in client.queue_prompt.call_args_list
                ]
                self.assertEqual([prefix.split("/test_")[0] for prefix in prefixes],
                                 [first.stem, second.stem])

            with next(directory.glob("gen_*.csv")).open(encoding="utf-8-sig", newline="") as log:
                rows = list(csv.DictReader(log))
            self.assertEqual(
                [(row["source_csv"], row["row_id"], row["positive"]) for row in rows],
                [(str(first), "01", "forest"), (str(second), "01", "coast")],
            )

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
