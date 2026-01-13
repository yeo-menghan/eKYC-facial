import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Setup Session
@st.cache_resource
def load_model():
    return ort.InferenceSession("./model/ekyc_liveness_mobilenetv3.onnx")

session = load_model()

# Preprocessing (Must match your training transforms)
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).numpy()
    return img_tensor

st.title("🛡️ eKYC Face Liveness Detector")
st.write("Upload a photo or use the camera to check for spoofing.")

source = st.radio("Select Image Source:", ("Upload Image", "Camera Input"))

if source == "Upload Image":
    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    img_file = st.camera_input("Take a photo")

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Processed Image", use_container_width=True)
    
    # Inference
    input_tensor = preprocess(image)
    outputs = session.run(None, {'input': input_tensor})
    logit = outputs[0][0][0]
    
    # Sigmoid for probability
    prob = 1 / (1 + np.exp(-logit))
    
    # Class mapping (Check your training labels, usually 0: Live, 1: Spoof)
    # If using BCEWithLogits, higher values often mean the positive class (Spoof)
    label = "SPOOF ❌" if prob > 0.5 else "LIVE ✅"
    confidence = prob if prob > 0.5 else (1 - prob)

    st.subheader(f"Result: {label}")
    st.progress(float(confidence))
    st.write(f"Confidence Score: {confidence:.2%}")