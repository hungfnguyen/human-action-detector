"""
Geometry Utilities - Minimal implementation.

Chỉ implement các functions cơ bản cần thiết cho scoring.
"""

import numpy as np


class GeometryUtils:
    """Utility class for basic geometric calculations."""
    
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Tính góc tại điểm p2 giữa 3 điểm p1, p2, p3.
        
        Args:
            p1: Điểm 1 (x, y) hoặc [x, y]
            p2: Điểm 2 (x, y) - đỉnh góc
            p3: Điểm 3 (x, y) hoặc [x, y]
        
        Returns:
            float: Góc tại p2 (đơn vị: độ, 0-180)
        
        Example:
            >>> angle = GeometryUtils.calculate_angle([0, 0], [0, 1], [1, 1])
            >>> print(f"{angle:.1f}°")  # 90.0°
        """
        # Convert to numpy arrays
        p1 = np.array(p1[:2])  # Only take x, y
        p2 = np.array(p2[:2])
        p3 = np.array(p3[:2])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        
        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    @staticmethod
    def calculate_distance(p1, p2):
        """
        Tính khoảng cách Euclidean giữa 2 điểm.
        
        Args:
            p1: Điểm 1 (x, y)
            p2: Điểm 2 (x, y)
        
        Returns:
            float: Khoảng cách
        """
        p1 = np.array(p1[:2])
        p2 = np.array(p2[:2])
        return np.linalg.norm(p1 - p2)
