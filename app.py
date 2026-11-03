"""Streamlit front end:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from kickcoach.analyze import analyze

st.set_page_config(page_title="KickCoach", layout="wide")
st.title("KickCoach - AI biomechanical feedback for football kicks")

refs = sorted(Path("reference").glob("*.json"))
ref_name = st.selectbox("Reference profile", [r.name for r in refs])
video = st.file_uploader("Upload a side-on video of one kick", type=["mp4", "mov", "avi"])

if video and st.button("Analyze"):
    with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as f:
        f.write(video.read())
    with st.spinner("Running pose estimation..."):
        rep = analyze(f.name, str(Path("reference") / ref_name), "out")
    c1, c2 = st.columns(2)
    with c1:
        if rep["contact_image"]:
            st.image(rep["contact_image"], caption=f"Detected contact at {rep['contact_time_s']}s")
        st.line_chart(rep["speed"])
        st.caption("Kicking-foot speed (peak = detected contact)")
    with c2:
        st.metric("Technique score vs reference", f"{rep['score']}/100")
        st.write(f"Kicking foot: **{rep['kicking_foot']}**")
        tips = [r["advice"] for r in rep["rows"] if r["advice"]]
        for t in tips or ["Close to the reference on every measured feature."]:
            st.write("- " + t)
        st.dataframe([{k: r[k] for k in ("feature", "value", "ref_mean", "z", "status")} for r in rep["rows"]])
