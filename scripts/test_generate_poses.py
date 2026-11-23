"""専用コマンドの組み合わせ・ControlNet 接続・骨格画像処理を検証。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import generate_poses as poses


class PoseTests(unittest.TestCase):
    def test_controlnet_keeps_original_conditions_and_style_model(self):
        graph = poses.load_workflow(poses.ROOT / "workflows/sdxl_dreamwash_anmi_api.json")
        original = copy.deepcopy(graph)
        poses.add_pose(graph, "pose.png", "control.safetensors", 0.8, 0.1, 0.9)
        for node_id in ("5", "25"):
            apply = graph[f"pose_apply_{node_id}"]["inputs"]
            for field in ("positive", "negative"):
                self.assertEqual(apply[field], original[node_id]["inputs"][field])
            self.assertEqual(graph[node_id]["inputs"]["model"], ["22", 0])
            self.assertEqual(graph[node_id]["inputs"]["positive"], [f"pose_apply_{node_id}", 0])

    def test_all_combinations_dry_run_and_actual_submission(self):
        with tempfile.TemporaryDirectory(dir=poses.ROOT / "input") as temp:
            root = Path(temp)
            folder = root / "poses"
            folder.mkdir()
            for name in ("b.png", "a.png"):
                Image.new("RGB", (64, 32), "red").save(folder / name)
            csv_path = root / "prompts.csv"
            csv_path.write_text("id,positive\n1,anime character\n2,blue hair\n", encoding="utf-8")
            arguments = ["--prompts", str(csv_path), "--pose-dir", str(folder),
                         "--pose-mode", "openpose", "--seed", "123", "--no-style-reference",
                         "--width", "64", "--height", "64", "--batch", "1"]
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                self.assertEqual(poses.main(arguments + ["--dry-run"]), 0)
            jobs = [json.loads(line) for line in capture.getvalue().splitlines() if line.startswith("{")]
            self.assertEqual([(j["id"], Path(j["pose"]).name) for j in jobs],
                             [("1", "a.png"), ("1", "b.png"), ("2", "a.png"), ("2", "b.png")])
            self.assertEqual({j["seed"] for j in jobs}, {123})
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["poses", "prompts.csv"])
            with patch.object(poses, "INPUT_DIR", root), patch.object(poses, "LOG_DIR", root / "logs"), \
                    patch.object(poses, "ComfyClient") as client, contextlib.redirect_stdout(io.StringIO()):
                client.return_value.check_ready.return_value = (True, "ready")
                client.return_value.queue_prompt.return_value = "test"
                client.return_value.output_files.return_value = ["test.png"]
                self.assertEqual(poses.main(arguments), 0)
                self.assertEqual(client.return_value.queue_prompt.call_count, 4)
            maps = list((root / "pose_maps").rglob("*.png"))
            self.assertEqual(len(maps), 2)
            with Image.open(maps[0]) as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(image.getpixel((32, 32)), (255, 0, 0))

    def test_chw_render_conversion_and_reject_missing_pose(self):
        with tempfile.TemporaryDirectory(dir=poses.ROOT / "input") as temp:
            root = Path(temp)
            source = root / "person.png"
            Image.new("RGB", (64, 64)).save(source)
            maps = poses.PoseMaps(root / "maps", "image")
            detector = maps.detector = unittest.mock.Mock()
            detector.return_value = {"bodies": {"score": np.ones((1, 18))}}
            rgb = np.zeros((3, 64, 64), dtype=np.uint8)
            rgb[0] = 255
            with patch.object(poses, "INPUT_DIR", root), \
                    patch("controlnet_dwpose.draw_pose", return_value=rgb):
                name = maps.get(source, 64, 64)
                with Image.open(root / name) as image:
                    self.assertEqual(image.getpixel((32, 32)), (255, 0, 0))
                maps.images.clear()
                maps.maps.clear()
                detector.return_value = {"bodies": {"score": np.zeros((1, 18))}}
                with self.assertRaisesRegex(ValueError, "骨格を検出できません"):
                    maps.get(source, 64, 64)
            maps.release()
            detector.release_memory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
