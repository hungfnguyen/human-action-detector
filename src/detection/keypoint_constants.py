"""
COCO Keypoint Constants.

Defines the 17 COCO keypoint indices used by YOLOv8 Pose model.
"""

from pydantic import BaseModel


class COCOKeypoints(BaseModel):
    """
    COCO keypoint indices (17 points total).
    
    Used for pose detection and skeleton visualization.
    Reference: https://cocodataset.org/#keypoints-2020
    """
    
    # Head (5 points)
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    
    # Upper body (6 points)
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    
    # Lower body (6 points)
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16


# Singleton instance for easy access
KEYPOINTS = COCOKeypoints()


# Keypoint names for visualization
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]
