# KickCoach - Pose-Based Biomechanical Feedback for Football Kicks

Upload a side-on video of a kick. KickCoach extracts the player's skeleton (MediaPipe BlazePose),
detects the ball-contact frame, measures body angles at contact and compares them to a reference
profile built from pro-player clips, then gives plain-English coaching feedback.

## Pipeline
1. `pose.py` - BlazePose landmarks per frame (gaps interpolated)
2. `features.py` - contact detection (peak kicking-ankle speed, torso-normalised) + 9 biomechanical features
   (kick/plant knee, hip, ankle angle, trunk lean, shoulder-hip twist, foot gap, head drop); left/right-footers handled
3. `reference.py` - builds mean/std per feature from a folder of pro clips
4. `compare.py` - z-score vs reference -> technique score (0-100) + coaching text
5. `dtw.py` - Dynamic Time Warping to align a player's motion to a reference motion
6. `analyze.py` (CLI) and `app.py` (Streamlit UI)

## Run
    pip install -r requirements.txt
    PYTHONPATH=. python tests/test_core.py                       # logic tests (synthetic skeletons)
    PYTHONPATH=. python -m kickcoach.analyze my_kick.mp4 --ref reference/placeholder_reference.json
    streamlit run app.py

## Build the real pro reference
Collect 30-50 clean, side-on clips of one skill (e.g. instep pass) into a folder, then:

    PYTHONPATH=. python -m kickcoach.reference clips/instep_pass/ reference/instep_pass.json

`reference/placeholder_reference.json` contains generic ranges only so the app runs. It is NOT pro data.

## Limitations
- 2D pose: use a fixed side-on camera. Contact is estimated from foot speed (ball is not tracked yet).
- Trunk lean is unsigned (direction of play is not inferred).
- Next steps: YOLO ball tracking to refine contact, receiving/first-touch skill, 3D pose lifting, LLM feedback.
