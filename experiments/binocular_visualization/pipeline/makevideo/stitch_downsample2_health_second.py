#!/usr/bin/env python3
"""
Run the tuned health stitching pipeline on the second health sequence.

This is a small wrapper around stitch_downsample2_health.py so the same
partial-chessboard, green-marker, torsion, and flip checks are reused.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "stitch_downsample2_health.py"

DEFAULT_ARGS = [
    "--left-dir",
    r"D:\visualization2026\visualization\left_downsample2_health_second",
    "--right-dir",
    r"D:\visualization2026\visualization\right_downsample2_health_second",
    "--output-dir",
    r"D:\visualization2026\dowsample2_health_second",
]


def main() -> None:
    sys.argv = [str(BASE_SCRIPT), *DEFAULT_ARGS, *sys.argv[1:]]
    runpy.run_path(str(BASE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
