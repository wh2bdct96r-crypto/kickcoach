"""End-to-end analysis: video -> contact frame, features, feedback, annotated image.

Usage:  python -m kickcoach.analyze my_kick.mp4 --ref reference/placeholder_reference.json --out out/
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .compare import compare
from .features import detect_contact, frame_features
from .pose import extract_landmarks

# MediaPipe skeleton connections (body only)
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32)]


def draw_skeleton(frame, lm, rows=None):
    img = frame.copy()
    for a, b in BONES:
        cv2.line(img, tuple(int(v) for v in lm[a, :2]), tuple(int(v) for v in lm[b, :2]), (0, 255, 0), 3)
    for pt in lm[11:33, :2]:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    y = 30
    for r in rows or []:
        col = (0, 200, 0) if r["status"] == "ok" else (0, 0, 255)
        cv2.putText(img, f'{r["feature"]}: {r["value"]} (ref {r["ref_mean"]})', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y += 26
    return img


def read_frame(video_path, index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def analyze(video_path: str, ref_path: str, out_dir: str = "out"):
    lm, fps, _ = extract_landmarks(video_path)
    frame, side, speed = detect_contact(lm)
    feats = frame_features(lm[frame], side)
    ref = {k: v for k, v in json.loads(Path(ref_path).read_text()).items() if not k.startswith("_")}
    score, rows = compare(feats, ref)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_path = None
    img = read_frame(video_path, frame)
    if img is not None:
        img_path = str(Path(out_dir) / "contact_frame.png")
        cv2.imwrite(img_path, draw_skeleton(img, lm[frame], rows))
    report = {"contact_frame": frame, "contact_time_s": round(frame / fps, 2), "kicking_foot": side,
              "score": score, "rows": rows, "speed": speed.tolist(), "contact_image": img_path}
    (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    rep = analyze(a.video, a.ref, a.out)
    print(f"Kicking foot: {rep['kicking_foot']}  contact at {rep['contact_time_s']}s  score {rep['score']}/100")
    for r in rep["rows"]:
        if r["advice"]:
            print(" -", r["advice"])
