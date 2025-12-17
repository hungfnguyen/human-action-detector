"""
Overlay UI - Draw score and feedback overlay on frame.

Displays pose name, score, and feedback message on image.
"""

import cv2
import numpy as np
from typing import Tuple


class OverlayUI:
    """
    Draw score and feedback overlay on frame.
    
    Features:
    - Semi-transparent panel
    - Pose name display
    - Score with progress bar
    - Feedback message
    
    Example:
        >>> ui = OverlayUI()
        >>> result = ui.draw_scoreboard(frame, "Plank", 85, "Good! Hip slightly high")
    """
    
    def __init__(self, 
                 panel_width: int = 400,
                 panel_height: int = 180,
                 alpha: float = 0.7):
        """
        Initialize overlay UI.
        
        Args:
            panel_width: Width of info panel
            panel_height: Height of info panel
            alpha: Transparency (0=transparent, 1=opaque)
        """
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.alpha = alpha
        
        # Colors (BGR)
        self.bg_color = (0, 0, 0)  # Black
        self.text_color = (255, 255, 255)  # White
        self.excellent_color = (0, 255, 0)  # Green
        self.good_color = (0, 165, 255)  # Orange
        self.poor_color = (0, 0, 255)  # Red
    
    def get_score_color(self, score: int) -> Tuple[int, int, int]:
        """Get color based on score."""
        if score >= 80:
            return self.excellent_color
        elif score >= 60:
            return self.good_color
        else:
            return self.poor_color
    
    def draw_semi_transparent_panel(self,
                                   frame: np.ndarray,
                                   x: int, y: int,
                                   width: int, height: int) -> np.ndarray:
        """
        Draw semi-transparent background panel.
        
        Args:
            frame: Image to draw on
            x, y: Top-left corner
            width, height: Panel dimensions
        
        Returns:
            Frame with panel drawn
        """
        # Create overlay
        overlay = frame.copy()
        
        # Draw rectangle on overlay
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     self.bg_color, -1)
        
        # Blend with original
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        
        return frame
    
    def draw_score_bar(self,
                      frame: np.ndarray,
                      score: int,
                      x: int, y: int,
                      width: int = 300,
                      height: int = 20) -> np.ndarray:
        """
        Draw score progress bar.
        
        Args:
            frame: Image to draw on
            score: Score 0-100
            x, y: Position
            width, height: Bar dimensions
        
        Returns:
            Frame with score bar
        """
        # Draw background (empty bar)
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     (100, 100, 100), 2)
        
        # Calculate filled width
        filled_width = int(width * (score / 100.0))
        
        # Draw filled portion
        color = self.get_score_color(score)
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height),
                     color, -1)
        
        return frame
    
    def draw_scoreboard(self,
                       frame: np.ndarray,
                       pose_name: str,
                       score: int,
                       feedback: str = "") -> np.ndarray:
        """
        Draw complete scoreboard overlay.
        
        Args:
            frame: Image to draw on
            pose_name: Name of detected pose
            score: Score 0-100
            feedback: Feedback message
        
        Returns:
            Frame with scoreboard overlay
        """
        h, w = frame.shape[:2]
        
        # Panel position (top-left with margin)
        panel_x = 15
        panel_y = 15
        
        # Draw semi-transparent panel
        frame = self.draw_semi_transparent_panel(
            frame, panel_x, panel_y, 
            self.panel_width, self.panel_height
        )
        
        # Text starting position
        text_x = panel_x + 20
        text_y = panel_y + 40
        
        # 1. Pose name (large)
        pose_text = pose_name.upper()
        cv2.putText(frame, pose_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.text_color, 2, cv2.LINE_AA)
        
        # 2. Score label
        text_y += 50
        score_text = f"Score:"
        cv2.putText(frame, score_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2, cv2.LINE_AA)
        
        # 3. Score value
        score_value = f"{score}/100"
        score_color = self.get_score_color(score)
        cv2.putText(frame, score_value, (text_x + 160, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)
        
        # 4. Score progress bar
        text_y += 10
        self.draw_score_bar(frame, score, text_x, text_y, width=350, height=15)
        
        # 5. Feedback message (smaller, wrapped if needed)
        if feedback:
            text_y += 35
            # Split long feedback into multiple lines
            max_width = 50  # characters
            if len(feedback) > max_width:
                # Simple word wrap
                words = feedback.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_width:
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                lines.append(current_line.strip())
            else:
                lines = [feedback]
            
            # Draw feedback lines
            for line in lines:
                cv2.putText(frame, line, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                text_y += 20
        
        return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter (bottom-right corner).
        
        Args:
            frame: Image to draw on
            fps: Frames per second
        
        Returns:
            Frame with FPS display
        """
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        
        # Position (bottom-right)
        text_x = w - 150
        text_y = h - 20
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame
