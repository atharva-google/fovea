"""Where Fovea writes its files: working files outside the code folder, weights with the code.

    python -m unittest discover -s tests -v
"""

import os
import unittest
from pathlib import Path
from unittest import mock

from fovea.config import PipelineConfig, fovea_home

CODE = Path(__file__).resolve().parent.parent
PATH_VARS = ("FOVEA_HOME", "FOVEA_WEIGHTS_DIR", "FOVEA_WORK_DIR", "FOVEA_OUTPUT_DIR")


def env(**values):
    clean = {k: v for k, v in os.environ.items() if k not in PATH_VARS}
    return mock.patch.dict(os.environ, {**clean, **values}, clear=True)


class DataFolderTest(unittest.TestCase):
    def test_default_is_next_to_the_code_not_inside_it(self):
        with env():
            self.assertEqual(fovea_home(), CODE.parent / "fovea_data")
            cfg = PipelineConfig()
            self.assertEqual(cfg.work_dir, CODE.parent / "fovea_data")   # <data folder>/<video name>/ per video
            self.assertNotIn(CODE, [cfg.work_dir, *cfg.work_dir.parents])
            self.assertIsNone(cfg.output_dir)                             # final videos stay in the video's folder

    def test_weights_stay_with_the_code(self):
        with env():
            self.assertEqual(PipelineConfig().weights_dir, CODE / "weights")
        with env(FOVEA_HOME="/somewhere/else"):
            self.assertEqual(PipelineConfig().weights_dir, CODE / "weights")

    def test_fovea_home_moves_the_working_files(self):
        with env(FOVEA_HOME="/somewhere/else"):
            cfg = PipelineConfig()
            self.assertEqual(cfg.work_dir, Path("/somewhere/else"))
            self.assertIsNone(cfg.output_dir)

    def test_an_output_folder_is_only_used_when_asked_for(self):
        with env(FOVEA_OUTPUT_DIR="/my/exports"):
            self.assertEqual(PipelineConfig().output_dir, Path("/my/exports"))
        with env():
            self.assertEqual(PipelineConfig(output_dir="/my/exports").output_dir, Path("/my/exports"))
            self.assertEqual(PipelineConfig().to_dict()["output_dir"], None)

    def test_one_folder_can_be_moved_on_its_own(self):
        with env(FOVEA_WEIGHTS_DIR="/drive/weights"):
            cfg = PipelineConfig()
            self.assertEqual(cfg.weights_dir, Path("/drive/weights"))
            self.assertEqual(cfg.work_dir, CODE.parent / "fovea_data")

    def test_building_a_config_creates_no_folders(self):
        with env():
            before = set(CODE.iterdir())
            PipelineConfig()
            self.assertEqual(set(CODE.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
