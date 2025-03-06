import streamlit as st

st.title("🕹️ RL Model Deployment")

st.markdown("### Select Mode:")
mode = st.radio("Choose:", ["Live Stream (10K Steps)", "Live Stream (100K Steps)"])

url_map = {
    "Live Stream (10K Steps)": "https://your-app.onrender.com/stream/10k",
    "Live Stream (100K Steps)": "https://your-app.onrender.com/stream/100k"
}

if "Pre-recorded" in mode:
    st.video(url_map[mode])
else:
    st.image(url_map[mode], use_column_width=True)
