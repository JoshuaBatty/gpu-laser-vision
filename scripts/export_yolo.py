#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.11.0",
#   "torchvision==0.26.0",
#   "ultralytics==8.4.51",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Download and export YOLO11n segmentation weights for tch-rs."""

import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    os.chdir(ASSETS)

    model = YOLO("yolo11n-seg.pt")
    exported = model.export(
        format="torchscript",
        imgsz=640,
        batch=1,
        dynamic=False,
        optimize=False,
    )
    print(f"Exported {Path(exported).resolve()}")


if __name__ == "__main__":
    main()
