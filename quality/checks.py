import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

@dataclass
class QualityResult:
    is_valid: bool
    reasons: List[str]
    # Detailed scores for observability/logging
    scores: Dict[str, float] = field(default_factory=dict)
    face_bbox: Optional[Tuple[int, int, int, int]] = None

class ImageQualityGate:
    def __init__(self, model_path: str = "model/weights/blaze_face_short_range.tflite"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        
        # Configurable Thresholds
        self.blur_threshold = 100.0
        self.brightness_range = (40, 230)
        self.min_face_ratio = 0.15 

    def analyze(self, numpy_image: np.ndarray) -> QualityResult:
        reasons = []
        scores = {}
        h, w, _ = numpy_image.shape

        # 1. Image Pre-processing for checks
        gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Brightness Check
        avg_brightness = float(np.mean(gray))
        scores["brightness"] = round(avg_brightness, 2)
        if not (self.brightness_range[0] < avg_brightness < self.brightness_range[1]):
            reasons.append("Bad lighting")

        # 3. Blur Check (Laplacian Variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scores["blur_variance"] = round(blur_score, 2)
        if blur_score > self.blur_threshold:
            reasons.append("Image blurry")

        # 4. Face Detection & Size Check
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.detections:
            scores["face_coverage"] = 0.0
            return QualityResult(False, ["No face detected"], scores=scores)

        # Calculate coverage of the primary face
        bbox = detection_result.detections[0].bounding_box
        face_area_ratio = (bbox.width * bbox.height) / (w * h)
        scores["face_coverage"] = round(float(face_area_ratio), 4)

        if face_area_ratio < self.min_face_ratio:
            reasons.append("Face too small/far")

        # 5. Result Aggregation
        is_valid = len(reasons) == 0
        return QualityResult(
            is_valid=is_valid, 
            reasons=reasons, 
            scores=scores,
            face_bbox=(bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        )

if __name__ == "__main__":
    gate = ImageQualityGate()
    img = cv2.imread("data/checks/blur-face.jpeg")
    if img is not None:
        res = gate.analyze(img)
        print(f"Valid: {res.is_valid}")
        print(f"Scores: {res.scores}")
        print(f"Failed Reasons: {res.reasons}")