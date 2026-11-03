"""Build a reference profile (mean/std of contact-frame features) from a folder of pro clips.

Usage:  python -m kickcoach.reference reference/instep_pass_clips/ reference/instep_pass.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from .features import FEATURES, detect_contact, frame_features
from .pose import extract_landmarks

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def build_reference(folder: str, out_json: str):
    vals = {k: [] for k in FEATURES}
    used = 0
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in VIDEO_EXT:
            continue
        try:
            lm, _, _ = extract_landmarks(str(path))
            frame, side, _ = detect_contact(lm)
            feats = frame_features(lm[frame], side)
        except Exception as e:  # skip clips where pose fails
            print(f"skip {path.name}: {e}")
            continue
        for k in FEATURES:
            vals[k].append(feats[k])
        used += 1
        print(f"ok   {path.name} (contact frame {frame}, {side}-foot)")
    if used == 0:
        raise SystemExit("No usable clips found.")
    ref = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in vals.items()}
    ref["_meta"] = {"n_clips": used, "source": str(folder)}
    Path(out_json).write_text(json.dumps(ref, indent=2))
    print(f"Saved reference from {used} clips -> {out_json}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build_reference(sys.argv[1], sys.argv[2])
