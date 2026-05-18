#!/usr/bin/env python3
"""One-command 0203 stereo VO evaluation and IMU-GT generation."""

from pathlib import Path
import importlib.util
import sys


COMMON_PATH = Path(__file__).resolve().with_name("0203_pipeline_common.py")
spec = importlib.util.spec_from_file_location("plato_0203_pipeline_common", COMMON_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.main(default_stage="all")

