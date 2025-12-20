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