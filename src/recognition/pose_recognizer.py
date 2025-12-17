"""
Pose Recognizer - Wrapper for ML classifier.

Provides unified interface for pose recognition.
"""

from typing import Tuple, List
from .ml_classifier import MLPoseClassifier


class PoseRecognizer:
    """
    Unified interface for pose recognition.
    
    Currently uses ML classifier, can be extended with rule-based in future.
    
    Example:
        >>> recognizer = PoseRecognizer('./models/pose_classification.pth')
        >>> pose, conf = recognizer.recognize(keypoints)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize recognizer.
        
        Args:
            model_path: Path to ML model weights
        """
        self.classifier = MLPoseClassifier(model_path)
    
    def recognize(self, keypoints: List[float]) -> Tuple[str, float]:
        """
        Recognize pose from keypoints.
        
        Args:
            keypoints: List of 34 normalized values
        
        Returns:
            (pose_name, confidence)
        """
        return self.classifier.predict(keypoints)
    
    def __call__(self, keypoints: List[float]) -> Tuple[str, float]:
        """Shortcut for recognize()."""
        return self.recognize(keypoints)
