"""
Skeleton Drawer - Draw pose skeleton on image.

Draws COCO skeleton with color-coding based on score.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.app_config import COCO_SKELETON, VISUALIZATION_CONFIG


class SkeletonDrawer:
    """
    Draw human pose skeleton on image.
    
    Features:
    - COCO skeleton connections (17 keypoints)
    - Color-coding based on score (Green/Orange/Red)
    - Keypoint markers
    
    Example:
        >>> drawer = SkeletonDrawer()
        >>> result = drawer.draw(frame, keypoints, score=85)
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize skeleton drawer.
        
        Args:
            config: Optional custom visualization config
        """
        self.config = config or VISUALIZATION_CONFIG
        self.skeleton = COCO_SKELETON
        self.thickness = self.config.get('skeleton_thickness', 3)
        self.keypoint_radius = self.config.get('keypoint_radius', 5)
    
    def get_color(self, score: int) -> Tuple[int, int, int]:
        """
        Get color based on score.
        
        Args:
            score: Score from 0-100
        
        Returns:
            BGR color tuple
        """
        if score >= 80:
            return (0, 255, 0)  # Green - Excellent
        elif score >= 60:
            return (0, 165, 255)  # Orange - Good
        else:
            return (0, 0, 255)  # Red - Needs work
    
    def draw_skeleton(self, 
                     frame: np.ndarray, 
                     keypoints: List[Tuple[int, int]], 
                     color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw skeleton connections on frame.
        
        Args:
            frame: Image to draw on
            keypoints: List of 17 (x, y) tuples
            color: BGR color
        
        Returns:
            Frame with skeleton drawn
        """
        if len(keypoints) != 17:
            raise ValueError(f"Expected 17 keypoints, got {len(keypoints)}")
        
        # Draw connections
        for (i, j) in self.skeleton:
            pt1 = keypoints[i]
            pt2 = keypoints[j]
            
            # Skip if either point is invalid (0, 0)
            if pt1 == (0, 0) or pt2 == (0, 0):
                continue
            
            # Draw line
            cv2.line(frame, pt1, pt2, color, self.thickness, cv2.LINE_AA)
        
        return frame
    
    def draw_keypoints(self,
                      frame: np.ndarray,
                      keypoints: List[Tuple[int, int]],
                      color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw keypoint markers on frame.
        
        Args:
            frame: Image to draw on
            keypoints: List of 17 (x, y) tuples
            color: BGR color
        
        Returns:
            Frame with keypoints drawn
        """
        for pt in keypoints:
            # Skip invalid points
            if pt == (0, 0):
                continue
            
            # Draw filled circle
            cv2.circle(frame, pt, self.keypoint_radius, color, -1, cv2.LINE_AA)
            # Draw border
            cv2.circle(frame, pt, self.keypoint_radius, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def draw(self,
            frame: np.ndarray,
            keypoints: List[Tuple[int, int]],
            score: int = 75) -> np.ndarray:
        """
        Draw complete skeleton with keypoints.
        
        Args:
            frame: Image to draw on (will be modified in-place)
            keypoints: List of 17 (x, y) pixel coordinates
            score: Score 0-100 (affects color)
        
        Returns:
            Frame with skeleton and keypoints drawn
        """
        # Get color based on score
        color = self.get_color(score)
        
        # Draw skeleton connections
        frame = self.draw_skeleton(frame, keypoints, color)
        
        # Draw keypoint markers
        frame = self.draw_keypoints(frame, keypoints, color)
        
        return frame
