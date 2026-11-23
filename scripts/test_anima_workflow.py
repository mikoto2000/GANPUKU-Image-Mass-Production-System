import unittest

from generate_images import ROOT, PromptRow, build_workflow, effective_values, load_workflow


class AnimaWorkflowTests(unittest.TestCase):
    def test_existing_csv_fields_and_cli_overrides(self):
        base = load_workflow(ROOT / "workflows/anima_base_api.json")
        row = PromptRow("001", "winter coat, blue eyes", "extra limbs",
                        width=896, height=1152, steps=28, cfg=5.5,
                        sampler="dpmpp_2m", scheduler="karras", batch=2)
        wf = build_workflow(base, row, 42, {
            "steps": 40, "cfg": 4.0, "sampler": "er_sde", "scheduler": "simple",
            "batch": 1, "common_positive": "masterpiece, safe",
            "common_negative": "worst quality",
        })
        self.assertEqual(wf["2"]["inputs"]["text"], "masterpiece, safe, winter coat, blue eyes")
        self.assertEqual(wf["3"]["inputs"]["text"], "worst quality, extra limbs")
        self.assertEqual(wf["4"]["inputs"]["length"], 1)
        self.assertEqual(wf["6"]["inputs"]["vae"], ["14", 0])
        values = effective_values(wf, [])
        for key, expected in {"width": 896, "height": 1152, "batch": 1,
                              "steps": 40, "cfg": 4.0, "sampler": "er_sde",
                              "scheduler": "simple", "ref_image": ""}.items():
            self.assertEqual(values[key], expected)
        self.assertEqual(base["2"]["inputs"]["text"],
                         "masterpiece, best quality, score_7, safe, mountain landscape, morning light")

    def test_model_override_uses_loader_input(self):
        for filename, key in [("anima_base_api.json", "unet_name"),
                              ("sdxl_base_api.json", "ckpt_name")]:
            with self.subTest(filename=filename):
                wf = build_workflow(load_workflow(ROOT / "workflows" / filename),
                                    PromptRow("001", "landscape"), 1,
                                    {"ckpt": "alternative.safetensors"})
                self.assertEqual(wf["1"]["inputs"][key], "alternative.safetensors")
                self.assertNotIn("ckpt_name" if key == "unet_name" else "unet_name",
                                 wf["1"]["inputs"])


if __name__ == "__main__":
    unittest.main()
