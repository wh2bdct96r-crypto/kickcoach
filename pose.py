"""Pose extraction: video -> (T, 33, 3) array of [x_px, y_px, visibility] using MediaPipe BlazePose."""
import cv2
import numpy as np


def _fill_nans(lm: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames where no person was detected."""
    lm = lm.copy()
    t = np.arange(len(lm))
    for j in range(lm.shape[1]):
        for c in range(2):
            col = lm[:, j, c]
            bad = np.isnan(col)
            if bad.all():
                continue
            if bad.any():
                col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return lm


def extract_landmarks(video_path: str, model_complexity: int = 1):
    """Run BlazePose on every frame. Returns (landmarks, fps, (width, height))."""
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose = mp.solutions.pose.Pose(static_image_mode=False,
                                  model_complexity=model_complexity,
                                  smooth_landmarks=True)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            arr = np.array([[l.x * w, l.y * h, l.visibility] for l in res.pose_landmarks.landmark])
        else:
            arr = np.full((33, 3), np.nan)
        frames.append(arr)
    cap.release()
    pose.close()
    if not frames:
        raise ValueError("Video has no frames")
    lm = np.stack(frames)
    if np.isnan(lm[:, :, 0]).all():
        raise ValueError("No person detected in the video")
    return _fill_nans(lm), fps, (w, h)
