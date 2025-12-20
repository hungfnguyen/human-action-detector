"""
Overlay UI - Draw score and feedback overlay on frame.

Displays pose name, score, and feedback message on image with adaptive scaling.
Supports Vietnamese text using PIL.
"""

import cv2
import numpy as np
from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFont


class OverlayUI:
    """
    Draw score and feedback overlay on frame.
    
    Features:
    - Semi-transparent panel
    - Pose name display
    - Score with progress bar
    - Feedback message
    - Adaptive scaling (Responsive Design) based on image resolution
    
    Example:
        >>> ui = OverlayUI()
        >>> result = ui.draw_scoreboard(frame, "Plank", 85, "Good! Hip slightly high")
    """
    
    def __init__(self, 
                 base_panel_width: int = 350,
                 base_panel_height: int = 120,
                 alpha: float = 0.7):
        """
        Initialize overlay UI with base dimensions (for ~720p resolution).
        
        Args:
            base_panel_width: Width of info panel at base resolution
            base_panel_height: Height of info panel at base resolution
            alpha: Transparency (0=transparent, 1=opaque)
        """
        self.base_panel_width = base_panel_width
        self.base_panel_height = base_panel_height
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
        """Draw semi-transparent background panel."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), self.bg_color, -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        return frame
    
    def draw_score_bar(self,
                      frame: np.ndarray,
                      score: int,
                      x: int, y: int,
                      width: int,
                      height: int) -> np.ndarray:
        """Draw score progress bar."""
        # Draw background (gray)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), -1)
        
        # Calculate filled width
        filled_width = int(width * (score / 100.0))
        
        # Draw filled portion
        color = self.get_score_color(score)
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)
        
        return frame
    
    def put_vietnamese_text(self,
                           frame: np.ndarray,
                           text: str,
                           position: Tuple[int, int],
                           font_size: int,
                           color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw Vietnamese text on frame using PIL (supports Unicode).
        
        Args:
            frame: OpenCV image (BGR)
            text: Text to draw (supports Vietnamese)
            position: (x, y) position
            font_size: Font size in pixels
            color: BGR color tuple
        
        Returns:
            Frame with text drawn
        """
        # Convert BGR to RGB for PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a Unicode font (DejaVu supports Vietnamese)
        try:
            # Try system fonts that support Vietnamese
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                # Fallback to default
                font = ImageFont.load_default()
        
        # Convert BGR to RGB for PIL
        color_rgb = (color[2], color[1], color[0])
        
        # Draw text
        draw.text(position, text, font=font, fill=color_rgb)
        
        # Convert back to BGR for OpenCV
        frame_rgb = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def draw_scoreboard(self,
                       frame: np.ndarray,
                       pose_name: str,
                       score: int,
                       feedback: str = "") -> np.ndarray:
        """
        Draw complete scoreboard overlay with adaptive scaling.
        
        The overlay size and font size will adjust based on the image resolution.
        
        Args:
            frame: Image to draw on
            pose_name: Name of detected pose
            score: Score 0-100
            feedback: Feedback message
        
        Returns:
            Frame with scoreboard overlay
        """
        h, w = frame.shape[:2]
        
        # --- 1. TÍNH TOÁN TỶ LỆ CO GIÃN (ADAPTIVE SCALE) ---
        # Lấy chuẩn HD (1280px) làm gốc.
        # Nếu ảnh 4K (3840px) -> Scale = 3.0. Nếu ảnh nhỏ (640px) -> Scale = 0.5.
        target_width = 1280.0
        scale = w / target_width
        
        # Giới hạn scale để tránh bảng quá bé hoặc quá khổng lồ
        # Min 0.6 (để đọc được trên ảnh nhỏ), Max 2.5 (để không choán hết ảnh lớn)
        scale = max(0.6, min(scale, 2.5))
        
        # --- 2. CẬP NHẬT KÍCH THƯỚC THEO SCALE ---
        margin = int(20 * scale)
        panel_w = int(self.base_panel_width * scale)
        
        # Font settings (Cỡ chữ cũng phải scale theo)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_header = 1.0 * scale
        font_scale_normal = 0.6 * scale
        thickness_header = max(2, int(2 * scale))
        thickness_normal = max(1, int(1 * scale))
        line_spacing = int(30 * scale)
        
        # --- 3. XỬ LÝ TEXT & LAYOUT ---
        
        # Header Text
        header_text = pose_name.upper()
        score_text = f"{score}%"
        
        # Xử lý xuống dòng cho Feedback (Word Wrap)
        feedback_lines = []
        if feedback:
            # Ước lượng số ký tự tối đa trên 1 dòng dựa trên độ rộng bảng
            # Khoảng 35 ký tự cho chiều rộng cơ bản
            max_chars = int(35) 
            
            words = feedback.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += word + " "
                else:
                    feedback_lines.append(current_line.strip())
                    current_line = word + " "
            feedback_lines.append(current_line.strip())
        
        # Tính chiều cao bảng (Panel Height) dựa trên nội dung
        # Header + Thanh điểm + Các dòng feedback + Padding
        content_height = 0
        content_height += int(40 * scale) # Khoảng header
        content_height += int(20 * scale) # Khoảng thanh bar
        content_height += int(20 * scale) # Khoảng cách
        content_height += len(feedback_lines) * line_spacing # Chiều cao feedback
        content_height += int(10 * scale) # Padding dưới cùng
        
        panel_h = max(int(self.base_panel_height * scale), content_height)
        
        # --- 4. VẼ LÊN ẢNH ---
        
        # Vị trí góc trái trên
        # Dời sang phải thêm 50px (đã scale) theo yêu cầu
        extra_shift_x = int(100 * scale)
        x1 = margin + extra_shift_x
        y1 = margin
        
        # Vẽ nền mờ
        frame = self.draw_semi_transparent_panel(frame, x1, y1, panel_w, panel_h)
        
        # Điểm bắt đầu vẽ chữ
        curr_x = x1 + int(20 * scale)
        curr_y = y1 + int(40 * scale)
        
        # 1. Tên tư thế
        cv2.putText(frame, header_text, (curr_x, curr_y),
                   font_face, font_scale_header, self.text_color, thickness_header, cv2.LINE_AA)
        
        # 2. Điểm số (Căn phải)
        score_color = self.get_score_color(score)
        (sw, sh), _ = cv2.getTextSize(score_text, font_face, font_scale_header, thickness_header)
        score_x = x1 + panel_w - sw - int(20 * scale)
        cv2.putText(frame, score_text, (score_x, curr_y),
                   font_face, font_scale_header, score_color, thickness_header, cv2.LINE_AA)
        
        # 3. Thanh tiến trình (Progress Bar)
        curr_y += int(15 * scale)
        bar_h = int(10 * scale)
        bar_w = panel_w - int(40 * scale)
        self.draw_score_bar(frame, score, curr_x, curr_y, bar_w, bar_h)
        
        # 4. Lời nhận xét (Feedback lines) - Use PIL for Vietnamese support
        curr_y += int(35 * scale)
        font_size = int(16 * scale)  # Convert font_scale to pixel size
        
        for line in feedback_lines:
            frame = self.put_vietnamese_text(
                frame, 
                line, 
                (curr_x, curr_y - font_size),  # Adjust Y for PIL rendering
                font_size, 
                (220, 220, 220)
            )
            curr_y += line_spacing
            
        return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter."""
        h, w = frame.shape[:2]
        # Scale FPS text too
        scale = max(0.6, min(w / 1280.0, 2.0))
        
        text = f"FPS: {fps:.1f}"
        font_scale = 0.6 * scale
        thickness = max(1, int(1 * scale))
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Bottom-Right
        x = w - tw - int(20 * scale)
        y = h - int(20 * scale)
        
        cv2.putText(frame, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        return frame