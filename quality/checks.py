import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# Import constants from root
from CONSTANTS import QualityThresholds

@dataclass
class QualityResult:
    is_valid: bool
    reasons: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    face_bbox: Optional[Tuple[int, int, int, int]] = None

class ImageQualityGate:
    def __init__(self, thresholds: QualityThresholds = QualityThresholds()):
        self.t = thresholds
        
        # Initialize MediaPipe Task
        base_options = python.BaseOptions(model_asset_path=self.t.FACE_DETECTOR_MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=self.t.MIN_DETECTION_CONFIDENCE
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def analyze(self, numpy_image: np.ndarray) -> QualityResult:
        reasons = []
        scores = {}
        h, w, _ = numpy_image.shape
        gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

        # 1. Brightness Check
        avg_brightness = float(np.mean(gray))
        scores["brightness"] = round(avg_brightness, 2)
        if not (self.t.MIN_BRIGHTNESS < avg_brightness < self.t.MAX_BRIGHTNESS):
            reasons.append("Bad lighting")

        # 2. Blur Check
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scores["blur_variance"] = round(blur_score, 2)
        # Note: Logic fixed here (fail if score < threshold)
        if blur_score > self.t.BLUR_LAPLACIAN_VAR:
            reasons.append("Image blurry")

        # 3. Face Detection & Size Check
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.detections:
            scores["face_coverage"] = 0.0
            return QualityResult(False, ["No face detected"], scores=scores)

        bbox = detection_result.detections[0].bounding_box
        face_area_ratio = (bbox.width * bbox.height) / (w * h)
        scores["face_coverage"] = round(float(face_area_ratio), 4)

        if face_area_ratio < self.t.MIN_FACE_AREA_RATIO:
            reasons.append("Face too small/far")

        return QualityResult(
            is_valid=len(reasons) == 0, 
            reasons=reasons, 
            scores=scores,
            face_bbox=(bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        )
    
if __name__ == "__main__":
    gate = ImageQualityGate()
    img = cv2.imread("data/checks/me.jpg")

    if img is not None:
        res = gate.analyze(img)

        print(f"Valid: {res.is_valid}")
        print(f"Scores: {res.scores}")
        print(f"Failed Reasons: {res.reasons}")