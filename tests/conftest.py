"""Shared import bootstrap for the V01–V08 regression tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_modules"
VIDEO = ROOT / "video_restoration_pipeline"
PHOTO_STAGES = ROOT / "photo_restoration_pipeline" / "stages"

for _p in (str(SHARED), str(VIDEO), str(PHOTO_STAGES), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
