"""Where Fovea writes its files: never inside the code folder.

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
            for folder in (cfg.weights_dir, cfg.work_dir, cfg.output_dir):
                self.assertEqual(folder.parent, CODE.parent / "fovea_data")
                self.assertNotIn(CODE, folder.parents)

    def test_fovea_home_moves_everything(self):
        with env(FOVEA_HOME="/somewhere/else"):
            cfg = PipelineConfig()
            self.assertEqual((cfg.weights_dir, cfg.work_dir, cfg.output_dir),
                             (Path("/somewhere/else/weights"), Path("/somewhere/else/working"),
                              Path("/somewhere/else/outputs")))

    def test_one_folder_can_be_moved_on_its_own(self):
        with env(FOVEA_WEIGHTS_DIR="/drive/weights"):
            cfg = PipelineConfig()
            self.assertEqual(cfg.weights_dir, Path("/drive/weights"))
            self.assertEqual(cfg.work_dir, CODE.parent / "fovea_data" / "working")

    def test_building_a_config_creates_no_folders(self):
        with env():
            before = set(CODE.iterdir())
            PipelineConfig()
            self.assertEqual(set(CODE.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
