"""Compare a player's contact-frame features with a reference and produce coaching feedback."""
from .features import RATIO_FEATURES

MIN_STD = {"angle": 5.0, "ratio": 0.1}

# feature -> (message when value is below reference, message when above)
ADVICE = {
    "kick_knee": ("Kicking knee is too bent at contact - extend the leg through the ball.",
                  "Kicking leg is over-straightened at contact - keep a slight bend to stay in control."),
    "plant_knee": ("Plant-leg knee is too bent - firm up the standing leg for stability.",
                   "Plant leg is too stiff - soften the knee to absorb the movement."),
    "kick_hip": ("Hip is too closed at contact - drive the hip through for more power.",
                 "Hip is too open at contact - bring the knee over the ball."),
    "plant_hip": ("Plant-side hip is too flexed - stand taller over the standing leg.",
                  "Plant-side hip is too extended - lean slightly into the shot."),
    "kick_ankle": ("Foot is not extended enough - lock the ankle and point the toe down.",
                   "Ankle is over-extended - keep the foot firm, not floppy."),
    "trunk_lean": ("Trunk is too upright - lean slightly over the ball to keep the shot down.",
                   "Trunk leans too far - stay more upright to keep balance."),
    "shoulder_hip_twist": ("Little shoulder-hip separation - rotate the upper body for more power.",
                           "Too much shoulder-hip twist - square the shoulders for accuracy."),
    "foot_gap": ("Plant foot is too close to the kicking foot - give yourself more room.",
                 "Plant foot is too far from the kicking foot - plant closer for balance."),
    "head_drop": ("Head is too high - look down at the ball through contact.",
                  "Head is dropped too far - keep eyes on the ball but don't hunch."),
}


def compare(features: dict, reference: dict, z_thresh: float = 1.5):
    """reference: {feature: {"mean": m, "std": s}}. Returns (score_0_100, rows)."""
    rows = []
    for name, val in features.items():
        if name not in reference:
            continue
        mean, std = reference[name]["mean"], reference[name]["std"]
        std = max(std, MIN_STD["ratio"] if name in RATIO_FEATURES else MIN_STD["angle"])
        z = (val - mean) / std
        status = "ok" if abs(z) <= z_thresh else ("low" if z < 0 else "high")
        rows.append({"feature": name, "value": round(val, 2), "ref_mean": round(mean, 2),
                     "z": round(z, 2), "status": status,
                     "advice": "" if status == "ok" else ADVICE[name][0 if status == "low" else 1]})
    if not rows:
        return 0.0, rows
    score = 100 * (1 - sum(min(abs(r["z"]), 3) for r in rows) / (3 * len(rows)))
    return round(score, 1), rows
