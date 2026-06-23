#!/usr/bin/env python3
"""Quick syntax verification for modified stage3_ai_upscale.py"""
import ast
import sys

files = [
    "photo_restoration_pipeline/stages/stage3_ai_upscale.py",
    "photo_restoration_pipeline/stages/stage4_ai_face.py",
    "photo_restoration_pipeline/stages/stage5_postprocess.py",
    "photo_restoration_pipeline/stages/stage6_encode.py",
    "photo_restoration_pipeline/main.py",
]

all_ok = True
for f in files:
    try:
        with open(f, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  OK: {f}")
    except SyntaxError as e:
        print(f"  FAIL: {f} - {e}")
        all_ok = False
    except FileNotFoundError:
        print(f"  MISSING: {f}")
        all_ok = False

if all_ok:
    print("\nAll files pass syntax check.")
    sys.exit(0)
else:
    print("\nSome files failed syntax check.")
    sys.exit(1)
