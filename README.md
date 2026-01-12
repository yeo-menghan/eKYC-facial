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
├── README.md
└── requirements.txt
```

## Image Quality Assessment (Input Gating)

Purpose: Reject bad inputs before ML inference (industry best practice).

Checks
- Blur detection (Laplacian variance)
- Brightness / contrast
- Face size ratio (face detector bounding box)
- Resolution threshold

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
- Backbone: ResNet18 or MobileNetV3 or YOLOv11-n for fast inference (can train multiple models to benchmark)
- Input: cropped face
- Output: liveness probability

Data Augmentation:
- Blur
- Compression artifacts
- Color jitter
- ...

Metrics
- ROC-AUC
- False Accept Rate

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

## AWS Deployment
Choice
- EC2 + Docker (simplest)
- ECS (bonus)
- Lambda only if model is tiny

Include
- Dockerfile
- Environment-based config
- Health check endpoint


## Future TODOs


- [ ] Deepfake attacks
- [ ] Temporal liveness
- [ ] Model distillation
- [ ] VLM-based cues