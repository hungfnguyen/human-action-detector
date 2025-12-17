import math
import numpy as np

class PoseCorrector:
    def __init__(self):
        # Định nghĩa các ngưỡng (threshold) góc chuẩn cho từng tư thế
        self.criteria = {
            'Plank': {'hip_angle': (170, 190)}, # Lưng thẳng
            'Downdog': {'elbow_angle': (160, 190), 'knee_angle': (160, 190)}, # Tay thẳng, chân thẳng
            'Warrior2': {'front_knee': (90, 120), 'back_knee': (160, 190)}, 
            'Tree': {'standing_leg': (165, 190)},
            'Goddess': {'knee_angle': (80, 140)} # Đầu gối mở
        }

    def calculate_angle(self, a, b, c):
        """Tính góc giữa 3 điểm a, b, c (b là đỉnh)"""
        # a, b, c là tuple (x, y)
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

    def get_point(self, keypoints, idx):
        """Lấy toạ độ (x, y) từ list phẳng 34 phần tử"""
        # idx là chỉ số keypoint của COCO (0-16)
        # keypoints list: [x0, y0, x1, y1, ...]
        return (keypoints[idx*2], keypoints[idx*2+1])

    def evaluate(self, pose_name, keypoints):
        """
        Input: Tên tư thế, List 34 keypoints
        Output: Score (0-100), List các lỗi (feedback)
        """
        score = 100
        feedback = []

        # Lấy các điểm quan trọng
        # COCO Format: 
        # 5: L-Shoulder, 6: R-Shoulder, 11: L-Hip, 12: R-Hip
        # 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
        
        # Helper lấy điểm nhanh
        p = lambda i: self.get_point(keypoints, i)

        if pose_name == 'Plank':
            # Kiểm tra bên trái hay phải (dựa vào vai nào rõ hơn hoặc trung bình)
            # Tính góc: Vai - Hông - Mắt cá (Lưng có thẳng không?)
            angle_body = self.calculate_angle(p(5), p(11), p(15)) # Left side
            
            if angle_body < 160:
                score -= 30
                feedback.append("Hông quá cao (Hip too high)")
            elif angle_body > 195: # Logic bù trừ
                score -= 30
                feedback.append("Hông quá thấp/võng lưng (Hip too low)")
            else:
                feedback.append("Thân người thẳng (Good body alignment)")

        elif pose_name == 'Downdog':
            # Kiểm tra chân thẳng (Hông - Gối - Mắt cá)
            knee_ang = self.calculate_angle(p(11), p(13), p(15))
            if knee_ang < 150:
                score -= 20
                feedback.append("Thẳng chân ra (Straighten legs)")
            
            # Kiểm tra lưng (Vai - Hông - Gối) không dùng vì Downdog là gập hông
            # Kiểm tra tay thẳng (Vai - Khuỷu - Cổ tay)
            elbow_ang = self.calculate_angle(p(5), p(7), p(9))
            if elbow_ang < 150:
                score -= 20
                feedback.append("Thẳng tay ra (Straighten arms)")

        elif pose_name == 'Warrior2':
            # Cần xác định chân nào trước. Giả sử chân có góc gối nhỏ hơn là chân trước
            l_knee = self.calculate_angle(p(11), p(13), p(15))
            r_knee = self.calculate_angle(p(12), p(14), p(16))
            
            front_knee = l_knee if l_knee < r_knee else r_knee
            back_knee = r_knee if l_knee < r_knee else l_knee
            
            if front_knee > 120:
                score -= 20
                feedback.append("Hạ thấp trọng tâm/Gập gối thêm (Bend front knee more)")
            elif front_knee < 80:
                score -= 20
                feedback.append("Gập gối quá sâu (Knee bent too much)")
                
            if back_knee < 160:
                score -= 20
                feedback.append("Thẳng chân sau (Straighten back leg)")

        elif pose_name == 'Tree':
            # Chân trụ phải thẳng. Chân co phải cao.
            l_knee = self.calculate_angle(p(11), p(13), p(15))
            r_knee = self.calculate_angle(p(12), p(14), p(16))
            
            # Chân nào thẳng hơn là chân trụ
            standing_leg_ang = l_knee if l_knee > r_knee else r_knee
            
            if standing_leg_ang < 165:
                score -= 30
                feedback.append("Chân trụ bị khuỵu (Straighten standing leg)")
            else:
                feedback.append("Chân trụ tốt")

        elif pose_name == 'Goddess':
            l_knee = self.calculate_angle(p(11), p(13), p(15))
            r_knee = self.calculate_angle(p(12), p(14), p(16))
            
            avg_knee = (l_knee + r_knee) / 2
            if avg_knee > 140:
                score -= 30
                feedback.append("Hạ thấp hông xuống (Squat deeper)")
            elif avg_knee < 80:
                score -= 20
                feedback.append("Gối gập quá sâu (Knees bent too much)")

        # Giới hạn score
        return max(0, score), feedback