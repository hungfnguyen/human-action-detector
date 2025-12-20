"""
Pose Recognizer - Hybrid Approach.

Kết hợp AI Classification và Geometric Rules để sửa lỗi nhận diện sai.
Đặc biệt xử lý trường hợp Tree vs Goddess và Warrior vs Goddess.
"""

from typing import Tuple, List
import numpy as np
import math
from .ml_classifier import MLPoseClassifier

class PoseRecognizer:
    def __init__(self, model_path: str):
        self.classifier = MLPoseClassifier(model_path)
    
    def _calculate_angle(self, a, b, c):
        """Tính góc giữa 3 điểm (a, b, c) với b là đỉnh."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _normalize_keypoints(self, keypoints: List[float]) -> List[float]:
        """Chuẩn hóa giữ nguyên tỷ lệ (Aspect Ratio)."""
        data = np.array(keypoints)
        if len(data) != 34: return keypoints

        data = data.reshape(-1, 2)
        min_x, min_y = np.min(data, axis=0)
        max_x, max_y = np.max(data, axis=0)
        
        scale = max(max_x - min_x, max_y - min_y)
        if scale == 0: scale = 1

        normalized_data = []
        for x, y in data:
            normalized_data.extend([(x - min_x) / scale, (y - min_y) / scale])
        return normalized_data
    
    def _override_prediction(self, ai_pose: str, keypoints: List[float]) -> str:
        """
        Dùng hình học để kiểm tra lại kết quả của AI.
        Nếu AI sai logic cơ bản, ép về kết quả đúng.
        """
        # Chuyển đổi keypoints về dạng dễ truy cập
        # Index COCO: Hip(11,12), Knee(13,14), Ankle(15,16)
        # Vì keypoints ở đây là list phẳng [x0, y0, ...], ta cần ánh xạ lại index
        # Index trong list phẳng = coco_index * 2
        
        def get_pt(idx): 
            return (keypoints[idx*2], keypoints[idx*2+1])
        
        try:
            # ✅ CASE 0: Kiểm tra cúi xuống (Bent Over) - STRICT!
            # Nếu AI bảo Goddess nhưng người đang cúi → KHÔNG PHẢI Goddess!
            if ai_pose == 'Goddess':
                shoulder_y_left = keypoints[5*2 + 1]   # Y của vai trái
                shoulder_y_right = keypoints[6*2 + 1]  # Y của vai phải
                hip_y_left = keypoints[11*2 + 1]       # Y của hông trái
                hip_y_right = keypoints[12*2 + 1]      # Y của hông phải
                wrist_y_left = keypoints[9*2 + 1]      # Y của cổ tay trái
                wrist_y_right = keypoints[10*2 + 1]    # Y của cổ tay phải
                
                avg_shoulder_y = (shoulder_y_left + shoulder_y_right) / 2
                avg_hip_y = (hip_y_left + hip_y_right) / 2
                avg_wrist_y = (wrist_y_left + wrist_y_right) / 2
                
                # CHECK 1: Goddess PHẢI có vai TRÊN hông (y nhỏ hơn trong coordinate)
                # Nếu vai DƯỚI hông rõ ràng (y lớn hơn nhiều) = đang cúi
                # RELAX: Cho phép vai hơi thấp hơn hông một chút (khi squat sâu)
                body_height = abs(shoulder_y_left - hip_y_left)
                if avg_shoulder_y > avg_hip_y + (body_height * 0.15):  # Vai thấp hơn 15%
                    return 'Unknown'  # Cúi rõ ràng!
                
                # CHECK 2: Cổ tay phải cao hơn hông đáng kể (tay giơ lên)
                # Goddess thật: tay giơ cao (wrist_y << hip_y)
                # Cúi xuống: tay chạm đất (wrist_y ≈ hip_y)
                # RELAX: Threshold lớn hơn để cho phép tay không giơ quá cao
                if avg_wrist_y >= avg_hip_y:  # Cổ tay thấp hơn hoặc ngang hông
                    return 'Unknown'  # Tay không giơ lên = CÚI!
            
            # ✅ CASE 0.5: Kiểm tra Downdog vs Plank (Hip angle check)
            # Downdog và Plank đều có tay chống + chân thẳng → dễ nhầm!
            if ai_pose == 'Downdog' or ai_pose == 'Plank':
                shoulder_left = get_pt(5)
                shoulder_right = get_pt(6)
                hip_left = get_pt(11)
                hip_right = get_pt(12)
                knee_left = get_pt(13)
                knee_right = get_pt(14)
                
                # Tính góc vai-hông-gối (hip angle)
                avg_shoulder = ((shoulder_left[0] + shoulder_right[0]) / 2, 
                               (shoulder_left[1] + shoulder_right[1]) / 2)
                avg_hip = ((hip_left[0] + hip_right[0]) / 2,
                          (hip_left[1] + hip_right[1]) / 2)
                avg_knee = ((knee_left[0] + knee_right[0]) / 2,
                           (knee_left[1] + knee_right[1]) / 2)
                
                hip_angle = self._calculate_angle(avg_shoulder, avg_hip, avg_knee)
                
                # Downdog: Góc nhỏ (70-120°) - hông nâng cao, hình chữ V
                # Plank: Góc lớn (>140°) - thân ngang, thẳng
                if ai_pose == 'Downdog' and hip_angle > 140:
                    return 'Plank'  # Thân ngang → Plank!
                elif ai_pose == 'Plank' and hip_angle < 120:
                    return 'Downdog'  # Hông nâng cao → Downdog!
            
            # Tính góc đầu gối trái và phải
            angle_l = self._calculate_angle(get_pt(11), get_pt(13), get_pt(15))
            angle_r = self._calculate_angle(get_pt(12), get_pt(14), get_pt(16))
            
            # Logic phân biệt Tree vs Goddess
            # Tree: 1 chân thẳng (>160), 1 chân gập
            # Goddess: 2 chân đều gập (<140)
            
            is_one_leg_straight = (angle_l > 160) or (angle_r > 160)
            is_both_legs_bent = (angle_l < 150) and (angle_r < 150)
            
            # CASE 1: AI bảo là Goddess, nhưng có 1 chân thẳng tắp -> Chắc chắn là Tree hoặc Warrior
            if ai_pose == 'Goddess' and is_one_leg_straight:
                # Kiểm tra thêm độ rộng chân để phân biệt Tree và Warrior
                # Tree: Chân khép (khoảng cách 2 mắt cá nhỏ)
                # Warrior: Chân mở rộng
                ankle_dist = abs(keypoints[15*2] - keypoints[16*2]) # Khoảng cách x
                hip_width = abs(keypoints[11*2] - keypoints[12*2])
                
                # Nếu khoảng cách 2 chân nhỏ hơn 2 lần vai -> Khả năng cao là Tree
                if ankle_dist < (hip_width * 2.5):
                    return 'Tree'
                else:
                    return 'Warrior2'

            # CASE 2: AI bảo là Tree, nhưng cả 2 chân đều cong -> Chắc chắn là Goddess
            if ai_pose == 'Tree' and is_both_legs_bent:
                return 'Goddess'
                
        except Exception:
            # Nếu có lỗi tính toán (do keypoints rác), giữ nguyên kết quả AI
            pass
            
        return ai_pose

    def recognize(self, keypoints: List[float]) -> Tuple[str, float]:
        # 1. Chuẩn hóa
        normalized_kps = self._normalize_keypoints(keypoints)
        
        # 2. AI Dự đoán
        pose_name, confidence = self.classifier.predict(normalized_kps)
        
        # 3. Kiểm tra logic hình học (Hybrid Check)
        # Chỉ chạy check nếu input là raw keypoints (chưa normalize để tính góc cho chuẩn)
        final_pose = self._override_prediction(pose_name, keypoints)
        
        # Nếu logic hình học thay đổi kết quả, ta giảm confidence xuống chút (để cảnh báo)
        # hoặc giữ nguyên. Ở đây tôi giữ nguyên để UI hiển thị.
        
        return final_pose, confidence
    
    def __call__(self, keypoints: List[float]) -> Tuple[str, float]:
        return self.recognize(keypoints)