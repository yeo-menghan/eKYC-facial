import cv2
import numpy as np
import io
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Assuming your quality gate and constants are in the root or accessible via path
from quality.checks import ImageQualityGate

app = FastAPI(title="eKYC Liveness API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX Session once at startup
# Ensure path points to your model relative to api/app.py
session = ort.InferenceSession("./model/ekyc_liveness_mobilenetv3.onnx")
gate = ImageQualityGate()

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = image.resize((224, 224))
    
    # CRITICAL FIX: Explicitly cast to float32
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # HWC to CHW and add Batch Dimension
    # Ensure the final array is float32
    img_np = img_np.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    return img_np

@app.get("/")
def health(enable_checks: bool = False):
    return {
        "status": "ok",
        "checks_enabled": enable_checks
    }

@app.post("/v1/liveness")
async def check_liveness(
    file: UploadFile = File(...),
    enable_checks: bool = Query(True)
):
    # 1. Read file
    data = await file.read()
    
    # 2. Quality Checks (MediaPipe/OpenCV)
    if enable_checks:
        # Decode for OpenCV checks
        nparr = np.frombuffer(data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        quality_res = gate.analyze(img_cv)
        if not quality_res.is_valid:
            return {
                "status": "QUALITY_FAILED",
                "reasons": quality_res.reasons,
                "scores": quality_res.scores
            }

    # 3. Inference
    input_tensor = preprocess(data)
    outputs = session.run(None, {'input': input_tensor})
    logit = outputs[0][0][0]
    
    # Sigmoid for probability
    prob = float(1 / (1 + np.exp(-logit)))
    
    return {
        "status": "SUCCESS",
        "is_live": prob < 0.5,
        "liveness_score": prob,
        "confidence": prob if prob > 0.5 else (1 - prob)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)