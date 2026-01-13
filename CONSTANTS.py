# CONSTANTS.py
from dataclasses import dataclass

@dataclass(frozen=True)
class QualityThresholds:
    # Image Quality
    BLUR_LAPLACIAN_VAR: float = 100.0
    MIN_BRIGHTNESS: float = 40.0
    MAX_BRIGHTNESS: float = 230.0
    
    # Face Geometry
    MIN_FACE_AREA_RATIO: float = 0.1
    MIN_DETECTION_CONFIDENCE: float = 0.5
    
    # Paths
    FACE_DETECTOR_MODEL_PATH: str = "model/weights/blaze_face_short_range.tflite"

@dataclass(frozen=True)
class ModelConfigs:
    LIVENESS_INPUT_SIZE: int = 224
    BATCH_SIZE: int = 32