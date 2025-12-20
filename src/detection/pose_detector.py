"""
Pose Detector - YOLOv8 wrapper for human pose detection.

Detects person in image and extracts 17 COCO keypoints.
"""

import sys
import cv2
import numpy as np
from typing import Optional, List, Tuple

from ultralytics import YOLO
from ultralytics.engine.results import Results

from .keypoint_constants import KEYPOINTS


class PoseDetector:
    """
    YOLOv8 Pose Detection wrapper.
    
    Detects person in image/frame and returns normalized keypoints.
    
    Example:
        >>> detector = PoseDetector('yolov8m-pose.pt')
        >>> keypoints = detector.detect(image)
        >>> print(f"Found {len(keypoints)} keypoints")
    """
    
    def __init__(self, model_path: str = "yolov8m-pose.pt", confidence_threshold: float = 0.5, use_gpu: bool = True):
        """
        Initialize pose detector.
        
        Args:
            model_path: Path to YOLOv8 pose model (.pt file)
            confidence_threshold: Minimum confidence for keypoint detection (0-1)
            use_gpu: Use GPU if available (default: True)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.keypoints_const = KEYPOINTS
        self.use_gpu = use_gpu
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 pose model with GPU support."""
        if "pose" not in self.model_path:
            sys.exit("Error: Model must be a YOLOv8 Pose model (e.g., yolov8m-pose.pt)")
        
        try:
            import torch
            
            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 GPU detected: {gpu_name}")
            else:
                self.device = 'cpu'
                if self.use_gpu:
                    print("⚠️ GPU requested but not available, using CPU")
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Move model to device (YOLOv8 handles this internally)
            print(f"✅ Loaded YOLOv8 Pose model: {self.model_path}")
            print(f"   Device: {self.device.upper()}")
            
        except Exception as e:
            sys.exit(f"Error loading model: {e}")
    
    def extract_keypoints(self, keypoint_array: np.ndarray) -> List[float]:
        """
        Extract keypoints as flat list.
        
        Args:
            keypoint_array: Numpy array of shape (17, 2) or (17, 3)
        
        Returns:
            List of 34 values: [x1, y1, x2, y2, ..., x17, y17]
        """
        features = []
        for idx in range(17):
            x, y = keypoint_array[idx][0], keypoint_array[idx][1]
            features.extend([x, y])
        return features  # 34 values total
    
    def get_keypoints_normalized(self, result: Results) -> Optional[List[float]]:
        """
        Get normalized keypoints (0-1 range) from YOLOv8 result.
        
        Args:
            result: YOLOv8 Results object
        
        Returns:
            List of 34 normalized values, or None if no person detected
        """
        if result.keypoints is None:
            return None
        
        # Get normalized keypoints (xyn format: 0-1 range)
        keypoints = result.keypoints.xyn.cpu().numpy()
        
        if len(keypoints) == 0:
            return None
        
        # Return first person detected
        return self.extract_keypoints(keypoints[0])
    
    def get_keypoints_absolute(self, result: Results) -> Optional[List[Tuple[int, int]]]:
        """
        Get absolute pixel keypoints from YOLOv8 result.
        
        Args:
            result: YOLOv8 Results object
        
        Returns:
            List of 17 (x, y) tuples in pixel coordinates, or None if no person detected
        """
        if result.keypoints is None:
            return None
        
        # Get absolute keypoints (xy format: pixel coordinates)
        keypoints = result.keypoints.xy.cpu().numpy()
        
        if len(keypoints) == 0:
            return None
        
        # Convert to list of tuples
        keypoints_list = []
        for kp in keypoints[0]:
            keypoints_list.append((int(kp[0]), int(kp[1])))
        
        return keypoints_list
    
    def detect(self, image: np.ndarray, return_absolute: bool = False) -> Optional[List]:
        """
        Detect person and extract keypoints.
        
        Args:
            image: Input image (BGR format)
            return_absolute: If True, return absolute pixel coords. If False, return normalized (0-1)
        
        Returns:
            - If return_absolute=False: List of 34 normalized values
            - If return_absolute=True: List of 17 (x, y) pixel tuples
            - None if no person detected
        """
        # Run YOLOv8 prediction
        result = self.predict(image)
        
        # Extract keypoints
        if return_absolute:
            return self.get_keypoints_absolute(result)
        else:
            return self.get_keypoints_normalized(result)
    
    def predict(self, image: np.ndarray) -> Results:
        """
        Run YOLOv8 prediction on image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            YOLOv8 Results object
        """
        # Pass device to YOLOv8 for GPU acceleration
        results = self.model.predict(
            image, 
            save=False, 
            verbose=False,
            device=self.device  # Use GPU if available
        )
        return results[0]
    
    def __call__(self, image: np.ndarray) -> Results:
        """Shortcut for predict()."""
        return self.predict(image)
