"""Check that only successful face/eye inputs are archived without data loss."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from refine_inpaint import run_inpaint


class RefineArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "input")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = Path("example/three_styles/run/image.png")
        self.face = self.source.parent / "faces/face.png"
        self.eyes = self.face.parent / "eyes/eyes.png"
        self.write(self.source, b"original")
        self.client = Mock()

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def run_stage(self, source, outputs):
        self.client.output_files.return_value = [str(path) for path in outputs]
        return run_inpaint(self.client, {}, 0.2, 900,
                           output_dir=self.root, source=source)

    def test_face_then_eyes_preserve_structure_and_final_image(self):
        self.write(self.face, b"face")
        self.assertEqual(self.run_stage(self.source, [self.face]), str(self.face))
        self.write(self.eyes, b"eyes")
        self.assertEqual(self.run_stage(self.face, [self.eyes]), str(self.eyes))
        self.assertFalse((self.root / self.source).exists())
        self.assertFalse((self.root / self.face).exists())
        self.assertEqual((self.root / "example_origin/three_styles/run/image.png").read_bytes(), b"original")
        self.assertEqual((self.root / "example_origin/three_styles/run/faces/face.png").read_bytes(), b"face")
        self.assertEqual((self.root / self.eyes).read_bytes(), b"eyes")

    def test_eyes_alone_archive_original(self):
        self.write(self.eyes, b"eyes")
        self.run_stage(self.source, [self.eyes])
        self.assertFalse((self.root / self.source).exists())
        self.assertTrue((self.root / "example_origin/three_styles/run/image.png").is_file())

    def test_failed_generation_keeps_source(self):
        self.client.wait.side_effect = TimeoutError("not finished")
        with self.assertRaises(TimeoutError):
            self.run_stage(self.source, [self.face])
        self.assertTrue((self.root / self.source).is_file())
        self.assertFalse((self.root / "example_origin").exists())

    def test_invalid_output_count_keeps_source(self):
        for outputs in ([], [self.face, self.eyes]):
            with self.subTest(outputs=outputs), self.assertRaises(RuntimeError):
                self.run_stage(self.source, outputs)
        self.assertTrue((self.root / self.source).is_file())

    def test_missing_output_keeps_source(self):
        with self.assertRaises(FileNotFoundError):
            self.run_stage(self.source, [self.face])
        self.assertTrue((self.root / self.source).is_file())

    def test_existing_archive_is_not_overwritten(self):
        archived = Path("example_origin/three_styles/run/image.png")
        self.write(archived, b"previous original")
        self.write(self.face, b"face")
        with self.assertRaises(FileExistsError):
            self.run_stage(self.source, [self.face])
        self.assertEqual((self.root / archived).read_bytes(), b"previous original")
        self.assertEqual((self.root / self.source).read_bytes(), b"original")
        self.assertEqual((self.root / self.face).read_bytes(), b"face")

    def test_paths_outside_output_and_same_output_are_rejected(self):
        for source, result in (("../outside.png", self.face),
                               (self.source, "../outside.png"),
                               (self.source, self.source)):
            with self.subTest(source=source, result=result), self.assertRaises(ValueError):
                self.run_stage(source, [result])
        self.assertTrue((self.root / self.source).is_file())

    def test_other_inpaint_users_do_not_archive(self):
        self.client.output_files.return_value = [str(self.face)]
        self.assertEqual(run_inpaint(self.client, {}, 0.2, 900), str(self.face))
        self.assertTrue((self.root / self.source).is_file())


if __name__ == "__main__":
    unittest.main()
