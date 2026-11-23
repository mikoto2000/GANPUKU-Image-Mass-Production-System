import contextlib
import csv
import io
import json
from unittest.mock import Mock
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate_three_styles as three
import generate_images as gen


class ThreeStyleTests(unittest.TestCase):
    def test_same_seed_per_row_and_original_profile_settings(self):
        with contextlib.redirect_stdout(io.StringIO()):
            profiles = three.prepare_profiles(True)
        rows = [(Path("example.csv"), i, gen.PromptRow("same-id", "blue coat", "extra hands",
                  width=832, height=1216, batch=4, steps=17, cfg=3.2,
                  sampler="dpmpp_2m", scheduler="karras")) for i in [1, 2]]
        with patch.object(three.random, "randint", side_effect=[11, 22]) as rng:
            jobs = list(three.iter_jobs(rows, profiles, -1, "test"))
        self.assertEqual(rng.call_count, 2)
        self.assertEqual([job[5] for job in jobs], [11, 22, 11, 22, 11, 22])
        self.assertEqual([job[1] for job in jobs], [1, 2, 1, 2, 1, 2])
        self.assertEqual([job[3] for job in jobs], [p[0] for p in profiles for _ in rows])
        self.assertEqual(len({job[6]["7"]["inputs"]["filename_prefix"] for job in jobs}), 6)
        for job in jobs:
            profile = job[3]
            w = job[6]
            self.assertEqual(w["4"]["inputs"]["batch_size"], 1)
            self.assertEqual([k for k, v in w.items() if v["class_type"] == "SaveImage"], ["7"])
            self.assertEqual(w["25"]["inputs"]["seed"], job[5])
            self.assertEqual(w["25"]["inputs"]["model"], w["5"]["inputs"]["model"])
            self.assertEqual(w["25"]["inputs"]["denoise"], .25)
            if profile == "dw060_chenbin080_ip030_hires":
                self.assertEqual(w["8"]["inputs"]["lora_name"], "il_chenbin_try_3-000006.safetensors")
                self.assertEqual(w["22"]["inputs"]["weight"], .3)
                self.assertIn("dreamwash, artist: chen bin", w["2"]["inputs"]["text"])
            else:
                self.assertNotIn("22", w)
            if profile == "dw060_anmi030_original":
                self.assertEqual(w["8"]["inputs"]["strength_model"], .3)
                self.assertEqual(w["5"]["inputs"]["steps"], 17)
                self.assertEqual(w["5"]["inputs"]["cfg"], 3.2)
                self.assertEqual(w["5"]["inputs"]["sampler_name"], "dpmpp_2m")
                self.assertTrue(w["2"]["inputs"]["text"].startswith(gen.load_common_prompt(three.ROOT / "prompts/common_positive_dreamwash_anmi.txt")))
            else:
                self.assertEqual(w["8"]["inputs"]["strength_model"], .8)
                self.assertEqual(w["5"]["inputs"]["steps"], 28)
                self.assertEqual(w["5"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual([j[5] for j in three.iter_jobs(rows, profiles, 42, "test")], [42] * 6)
        with patch.object(three.random, "randint", side_effect=[11, 22]):
            row_jobs = list(three.iter_jobs(rows, profiles, -1, "test", order="row"))
        self.assertEqual([j[5] for j in row_jobs], [11, 11, 11, 22, 22, 22])
        self.assertEqual([j[1] for j in row_jobs], [1, 1, 1, 2, 2, 2])
        self.assertEqual({(j[1], j[3]): j[6] for j in jobs},
                         {(j[1], j[3]): j[6] for j in row_jobs})
        def switches(sequence):
            return sum(a[3] != b[3] for a, b in zip(sequence, sequence[1:]))
        self.assertEqual(switches(jobs), 2)
        self.assertEqual(switches(row_jobs), 5)

    def test_dry_run_has_no_server_or_log_side_effects(self):
        with tempfile.TemporaryDirectory(dir=three.ROOT / "input") as tmp:
            path = Path(tmp) / "test.csv"
            path.write_text("id,positive\n1,red coat\n2,blue coat\n", encoding="utf-8")
            output = io.StringIO()
            with patch.object(three, "ComfyClient") as client, patch.object(three, "LOG_DIR", Path(tmp) / "logs"), contextlib.redirect_stdout(output):
                result = three.main(["--prompts", str(path), "--start", "1", "--limit", "1", "--seed", "42", "--dry-run"])
            self.assertEqual(result, 0)
            client.assert_not_called()
            self.assertFalse((Path(tmp) / "logs").exists())
            jobs = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith("{")]
            self.assertEqual(len(jobs), 3)
            self.assertEqual({job["row_index"] for job in jobs}, {2})
            self.assertEqual({job["seed"] for job in jobs}, {42})

    def test_one_output_each_and_failure_logs_preserve_seed(self):
        with tempfile.TemporaryDirectory(dir=three.ROOT / "input") as tmp:
            path = Path(tmp) / "test.csv"
            path.write_text("id,positive\n1,blue coat\n", encoding="utf-8")
            with patch.object(three, "LOG_DIR", Path(tmp) / "logs"), patch.object(three, "ComfyClient") as client, contextlib.redirect_stdout(io.StringIO()):
                fake = client.return_value
                fake.check_ready.return_value = (True, "ready")
                fake.queue_prompt.side_effect = ["first", "second", "third"]
                entry = {"outputs": {"7": {"images": [{"filename": "test.png"}]}}}
                fake.wait.side_effect = [entry, RuntimeError("test failure"), entry]
                fake.output_files.return_value = ["test.png"]
                self.assertEqual(three.main(["--prompts", str(path), "--seed", "42"]), 1)
                self.assertEqual(fake.queue_prompt.call_count, 3)
                for call in fake.queue_prompt.call_args_list:
                    self.assertEqual(call.args[0]["5"]["inputs"]["seed"], 42)
            with next((Path(tmp) / "logs").glob("*.csv")).open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([r["status"] for r in records], ["OK", "ERROR", "OK"])
            self.assertEqual({r["seed"] for r in records}, {"42"})

    def test_refinement_order_skip_and_final_paths_for_all_profiles(self):
        with tempfile.TemporaryDirectory(dir=three.ROOT / "input") as tmp:
            path = Path(tmp) / "test.csv"
            path.write_text("id,positive\n1,blue coat\n", encoding="utf-8")
            face, eyes = Mock(), Mock()
            face.refine.side_effect = ["faces/test.png", None, "faces/test.png"]
            eyes.refine.return_value = "faces/eyes/test.png"
            with patch.object(three, "LOG_DIR", Path(tmp) / "logs"), patch.object(three, "ComfyClient") as client, \
                    patch.object(three, "make_refiners", return_value=[(face, "顔補修"), (eyes, "目元補修")]) as prepare, \
                    contextlib.redirect_stdout(io.StringIO()):
                fake = client.return_value
                fake.check_ready.return_value = (True, "ready")
                fake.queue_prompt.return_value = "pid"
                fake.wait.return_value = {"outputs": {"7": {"images": [{"filename": "test.png"}]}}}
                fake.output_files.return_value = ["test.png"]
                self.assertEqual(three.main(["--prompts", str(path), "--seed", "42",
                                             "--face-refine", "--larger-eyes"]), 0)
                prepare.assert_called_once_with(True, True)
            self.assertEqual(face.refine.call_count, 3)
            self.assertEqual([call.args[2] for call in eyes.refine.call_args_list],
                             ["faces/test.png", "test.png", "faces/test.png"])
            for call in face.refine.call_args_list + eyes.refine.call_args_list:
                self.assertEqual(call.args[1]["25"]["inputs"]["seed"], 42)
            with next((Path(tmp) / "logs").glob("*.csv")).open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual({r["files"] for r in records}, {"faces/eyes/test.png"})
            self.assertEqual({r["original_files"] for r in records}, {"test.png"})
            self.assertEqual({r["face_refine"] for r in records}, {"True"})
            self.assertEqual([s["status"] for s in json.loads(records[1]["refinements"])], ["SKIPPED", "OK"])

    def test_refinement_dry_run_does_not_initialize_detectors(self):
        output = io.StringIO()
        with patch.object(three, "make_refiners") as prepare, contextlib.redirect_stdout(output):
            self.assertEqual(three.main(["--limit", "1", "--seed", "42", "--face-refine", "--larger-eyes", "--dry-run"]), 0)
        prepare.assert_not_called()
        jobs = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith("{")]
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(job["face_refine"] and job["larger_eyes"] for job in jobs))


if __name__ == "__main__":
    unittest.main()
