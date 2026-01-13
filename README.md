# eKYC-facial

## Introduction
Design and deploy a facial identity verification pipeline that:
- Rejects low-quality inputs
- Detects spoofing vs live faces
- Serves predictions via a production-style API on AWS

This mirrors real eKYC systems used in fintech and e-commerce.


## Architecture
```
Client (image upload)
        |
        v
Image Quality Gate
(blur / brightness / face size)
        |
        v
Face Liveness Model (PyTorch)
        |
        v
Inference API (FastAPI)
        |
        v
AWS Deployment (EC2 or ECS) + Model export (TorchScript / ONNX) + CloudWatch logging
```

## Getting Started

### Repo Structure
```
facial-ekyc-system/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│       └── cd.yml
├── data/
│   └── face/
│
├── quality/
│   └── checks.py
│
├── model/
│   ├── liveness_net.py
│   └── inference.py
│
├── training/
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── api/
│   └── app.py
│
├── deploy/
│   ├── Dockerfile
│   └── run.sh
│
├── utils/
│   ├── image_processing.py  # Resizing, Normalization
│   └── aws_s3.py            # Logic to push "failed" attempts for manual review
├── tests/
│   ├── test_api.py          # Pytest for FastAPI endpoints
│   └── test_quality.py      # Test cases for blur/brightness
│
├── README.md
└── requirements.txt
```

### Running individual components

Run calibration for the quality checks:
```bash
python -m quality.calibration
```
Utilise the calibration results to suit your camera's needs

Run application
```bash
# First terminal - backend
python -m api.app
# Second terminal - streamlit frontend
streamlit run frontend/app.py
```

## Image Quality Assessment (Input Gating)

Purpose: Reject bad inputs before ML inference (industry best practice).

Checks
- Blur detection (Laplacian variance)
- Brightness / contrast
- Face size ratio (face detector bounding box)
- Resolution threshold

Additionally check for:
- Head Pose Estimation: Reject profiles. The user must face the camera.
- Eye Occlusion: Detect if the user is wearing heavy sunglasses (standard eKYC requirement).
- Library Suggestion: Use Mediapipe for the quality gate. It’s incredibly fast on CPU and gives you 3D landmarks to calculate head tilt and eye opening ratios easily.

Run file:
```bash
python -m quality.checks
```

Results:
```bash
{
  "quality_pass": true,
  "reasons": []
}
```

## Face Liveness / Anti-Spoofing Model

Binary classification: live vs spoof

Spoof types:
- Printed image
- Screen replay

Model
- Backbone: MobileNetV3-Small or EfficientNet-B0 as they are optimized for the "mobile-first" nature of eKYC.
- Input: cropped face
- Output: liveness probability

Feature,MobileNetV3 (Small/Large),YOLO-n (Nano variants)
Primary Task,Image Classification (Live vs. Spoof),Object Detection (Finding & naming boxes)
Architecture,"Optimized for 1:1 classification. Efficient ""Depthwise Separable Convolutions.""","Optimized for ""Dense Prediction"" (detecting many objects at once)."
Overhead,Extremely Low. Just a backbone and a tiny head.,"High. Includes ""Neck"" (FPN/PAN) and ""Heads"" for box regression and masking."
Deployment,Perfect for AWS Lambda or basic EC2 CPUs.,Better on GPUs; higher memory/compute footprint for the same input size.
Input Focus,Looks at the entire cropped face for global texture/noise cues.,"Focused on localizing edges to draw boxes, which can miss subtle global moiré patterns."

Using YOLO for liveness is "overkill"—you are paying for box-drawing logic you don't need. MobileNet gives you higher classification precision with a fraction of the compute.

Data Augmentation:
- Blur
- Compression artifacts
- Color jitter
- On top of RGB, use YCbCr or HSV color spaces because spoofing artifacts (like screen moiré patterns) are often more visible in the Chrominance channels than in RGB.

Metrics
- ROC-AUC
- False Accept Rate

### Dataset utilised
[CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof?tab=readme-ov-file). It contains 600,000+ images with 43 rich attributes, covering various spoof types (print, replay, paper cutouts).

Mini version: https://www.kaggle.com/datasets/phatntse192617/celeba-spoof-mini/data


## Training & Evaluation Pipeline

Training
- Config-driven training script
- GPU / CPU compatible
- Checkpointing
- Experiment logging (simple CSV / TensorBoard)

Evaluation
- Confusion matrix
- Failure case analysis (examples of spoof false negatives)

data/dataset.py: Implement a custom Sampler. Liveness datasets are often imbalanced (more live than spoof). Use a WeightedRandomSampler to ensure the model sees enough spoof examples in every batch.

training/evaluate.py: Add BPCER (Bona Fide Presentation Classification Error Rate) and APCER (Attack Presentation Classification Error Rate). These are the ISO/IEC 30107-3 standards for biometric testing

## Inference API 
Endpoint
```bash
POST /verify/face
```

Request: Image upload (JPEG/PNG)

Response:
```bash
{
  "liveness": "live",
  "confidence": 0.94,
  "quality": {
    "blur": "ok",
    "brightness": "ok"
  }
}
```

Additional notes:
- Asynchronous Processing: Use FastAPI's BackgroundTasks if you decide to add logging of images to S3, so the user doesn't wait for the upload to finish before getting a result.
- Security: Add a simple API Key header requirement. In eKYC, you never leave an endpoint public

## AWS Deployment
Choice
- EC2 + Docker (simplest)
- ECS (bonus)
- Lambda only if model is tiny

Include
- Dockerfile
- Environment-based config
- Health check endpoint

Model Export & Optimization
- Quantization: Since you are deploying to AWS, use OpenVINO (if using Intel EC2) or TensorRT (if using G-type GPU instances).
- ONNX is a must: Don't serve raw PyTorch. Converting to ONNX and using onnxruntime will significantly reduce latency and memory footprint.


## Future TODOs


- [ ] Deepfake attacks
- [ ] Temporal liveness
- [ ] Model distillation
- [ ] VLM-based cues

