import streamlit as st
import requests
import os

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def build_url(endpoint: str):
    return f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"


st.set_page_config(page_title="eKYC Liveness Portal", page_icon="🛡️", layout="wide")

# --- Initialize Session State ---
if "api_result" not in st.session_state:
    st.session_state.api_result = None
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0

def reset_ui():
    """Resets the camera and clears previous results"""
    st.session_state.api_result = None
    st.session_state.camera_key += 1
    # Note: st.rerun() is called automatically when state changes in most versions

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    enable_checks = st.checkbox("Enable Image Quality Gate", value=True)
    if st.button("🔄 Reset / Clear All", on_click=reset_ui, use_container_width=True):
        st.toast("UI Reset Successful")

st.title("🛡️ eKYC Face Liveness Detector")

# --- Main Layout ---
col_cam, col_res = st.columns([1.2, 1], gap="large")

with col_cam:
    st.subheader("Capture")
    source = st.radio("Select Image Source:", ("Camera Input", "Upload Image"), horizontal=True)
    
    img_file = None
    if source == "Upload Image":
        img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    else:
        # The key changes when reset_ui is called, forcing the camera to refresh
        img_file = st.camera_input("Take a photo", key=f"cam_{st.session_state.camera_key}")

with col_res:
    st.subheader("Results")
    
    if img_file:
        # Trigger inference only if we haven't already or if it's a new file
        with st.spinner("Analyzing..."):
            try:
                url = build_url("v1/liveness") 

                files = {"file": (img_file.name, img_file.getvalue(), img_file.type)}
                params = {"enable_checks": enable_checks}

                response = requests.post(url, files=files, params=params)
                response.raise_for_status()

                st.session_state.api_result = response.json()
            except Exception as e:
                st.error(f"Backend Error: {e}")

    # --- Display Results from State ---
    if st.session_state.api_result:
        res = st.session_state.api_result
        
        if res["status"] == "QUALITY_FAILED":
            st.warning("⚠️ Quality Check Failed")
            for reason in res["reasons"]:
                st.write(f"- {reason}")
            
            with st.expander("View Quality Scores"):
                st.json(res["scores"])
                
        elif res["status"] == "SUCCESS":
            label = "LIVE ✅" if res["is_live"] else "SPOOF ❌"
            color = "green" if res["is_live"] else "red"
            
            st.markdown(f"### Result: :{color}[{label}]")
            st.progress(res["confidence"])
            st.write(f"Confidence Score: **{res['confidence']:.2%}**")
            
            # Additional metadata
            st.info(f"Quality Check: {'Enabled' if res.get('quality_checked') else 'Bypassed'}")
    else:
        st.info("Upload or capture an image to see the analysis here.")