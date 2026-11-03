"""Biomechanical features from pose landmarks + kick-contact detection."""
import numpy as np

NOSE = 0
LS, RS, LH, RH = 11, 12, 23, 24
LK, RK, LA, RA = 25, 26, 27, 28
LF, RF = 31, 32  # foot index (toe)

FEATURES = ["kick_knee", "plant_knee", "kick_hip", "plant_hip", "kick_ankle",
            "trunk_lean", "shoulder_hip_twist", "foot_gap", "head_drop"]

# Features measured in degrees vs. unitless ratios (used for minimum std in comparison)
RATIO_FEATURES = {"foot_gap", "head_drop"}


def angle(a, b, c) -> float:
    """Angle at point b (degrees) formed by a-b-c."""
    v1, v2 = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _mid(lm, i, j):
    return (lm[i, :2] + lm[j, :2]) / 2


def torso_length(lm) -> float:
    """Shoulder-mid to hip-mid distance; used to normalise for body size / camera distance."""
    return float(np.linalg.norm(_mid(lm, LS, RS) - _mid(lm, LH, RH))) + 1e-9


def _smooth(x, k=3):
    if len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def ankle_speed(lm_seq, side: str) -> np.ndarray:
    """Ankle speed per frame, in torso-lengths per frame (size/camera invariant)."""
    idx = RA if side == "R" else LA
    pos = lm_seq[:, idx, :2]
    scale = np.median([torso_length(f) for f in lm_seq])
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)] / scale
    return _smooth(speed)


def detect_contact(lm_seq):
    """Estimate kicking side and ball-contact frame.

    Heuristic: the kicking foot is the ankle with the fastest peak speed, and contact is
    at (or right after) that peak. Returns (frame_index, side, speed_array).
    """
    sp = {s: ankle_speed(lm_seq, s) for s in ("L", "R")}
    side = "R" if sp["R"].max() >= sp["L"].max() else "L"
    return int(np.argmax(sp[side])), side, sp[side]


def frame_features(lm, side: str) -> dict:
    """Feature dict for one frame. 'kick'/'plant' naming makes left- and right-footers comparable."""
    if side == "R":
        kS, kH, kK, kA, kF = RS, RH, RK, RA, RF
        pS, pH, pK, pA = LS, LH, LK, LA
    else:
        kS, kH, kK, kA, kF = LS, LH, LK, LA, LF
        pS, pH, pK, pA = RS, RH, RK, RA
    p = lm[:, :2]
    tl = torso_length(lm)
    trunk = _mid(lm, LS, RS) - _mid(lm, LH, RH)  # hip -> shoulder vector (image y is down)
    lean = float(np.degrees(np.arctan2(abs(trunk[0]), -trunk[1])))
    sh, hp = p[RS] - p[LS], p[RH] - p[LH]
    twist = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(sh, hp)) / (np.linalg.norm(sh) * np.linalg.norm(hp) + 1e-9), -1, 1))))
    return {
        "kick_knee": angle(p[kH], p[kK], p[kA]),
        "plant_knee": angle(p[pH], p[pK], p[pA]),
        "kick_hip": angle(p[kS], p[kH], p[kK]),
        "plant_hip": angle(p[pS], p[pH], p[pK]),
        "kick_ankle": angle(p[kK], p[kA], p[kF]),
        "trunk_lean": lean,
        "shoulder_hip_twist": twist,
        "foot_gap": float(abs(p[kA][0] - p[pA][0]) / tl),
        "head_drop": float((p[NOSE][1] - _mid(lm, LS, RS)[1]) / tl),
    }


def sequence_features(lm_seq, side: str) -> np.ndarray:
    """(T, len(FEATURES)) matrix, for DTW alignment."""
    return np.array([[frame_features(f, side)[k] for k in FEATURES] for f in lm_seq])
