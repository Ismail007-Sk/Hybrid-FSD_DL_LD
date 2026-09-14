import streamlit as st

st.set_page_config(page_title="Hybrid Face Spoofing Detection", layout="centered")

# RESET stop flag so detect.py works again
if "stop" not in st.session_state:
    st.session_state.stop = False
else:
    st.session_state.stop = False

# ------------ CUSTOM CSS FOR BEAUTIFUL UI -----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0d0d0d, #1a1a40);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.title-box {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 80px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}
.start-btn {
    margin-top: 40px;
    padding: 12px 30px;
    font-size: 20px;
    border-radius: 12px;
    background: #5a5af7;
    color: white;
    border: none;
    cursor: pointer;
    transition: 0.3s;
}
.start-btn:hover {
    background: #7b7bfa;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ------------ UI CONTENT -----------
st.markdown("""
<div class="title-box">
    <h1 style="font-size: 38px; font-weight: 700;">Hybrid Face Spoofing Detection System</h1>
    <p style="opacity: 0.8; font-size: 18px;">
        Deep Learning (YOLO) + Local Descriptors (LBP, HOG, LPQ, LTP, BSIF, WLD)
    </p>
</div>
""", unsafe_allow_html=True)

# ------------ START BUTTON -----------
if st.button("▶ Start Detection", use_container_width=True):
    st.switch_page("Pages/detect.py")
