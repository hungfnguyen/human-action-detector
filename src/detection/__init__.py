"""
Detection module - YOLOv8 Pose Detection wrapper.
"""

from .pose_detector import PoseDetector
from .keypoint_constants import COCOKeypoints, KEYPOINTS, KEYPOINT_NAMES

__all__ = ['PoseDetector', 'COCOKeypoints', 'KEYPOINTS', 'KEYPOINT_NAMES']
