"""
Pose Evaluator - Improved scoring for yoga poses.

Đánh giá dựa trên nhiều tiêu chí cho độ chính xác cao hơn.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.geometry_utils import GeometryUtils


class PoseEvaluator:
    """
    Evaluates yoga poses with multi-criteria geometric rules.
    
    Returns score (0-100) and feedback message in Vietnamese.
    """
    
    def __init__(self):
        self.geo = GeometryUtils()
    
    def evaluate(self, pose_name, keypoints):
        """
        Evaluate a detected pose.
        
        Args:
            pose_name: Name of the pose ("Plank", "Tree", etc.)
            keypoints: List of 17 COCO keypoints [(x, y, conf), ...]
        
        Returns:
            tuple: (score, feedback_message)
                score: 0-100 integer
                feedback_message: String describing the evaluation (Vietnamese)
        """
        # Map pose names to evaluation methods
        evaluators = {
            'Plank': self._evaluate_plank,
            'Tree': self._evaluate_tree,
            'Warrior2': self._evaluate_warrior,
            'Goddess': self._evaluate_goddess,
            'Downdog': self._evaluate_downdog,
        }
        
        evaluator = evaluators.get(pose_name)
        if evaluator:
            return evaluator(keypoints)
        else:
            return 75, "Đã nhận dạng tư thế"
    
    def _evaluate_plank(self, kp):
        """
        Plank: Đánh giá 2 tiêu chí
        1. Thân thẳng (vai-hông-gối ≈ 180°)
        2. Tay chống thẳng (vai-khuỷu-cổ tay ≈ 180°)
        """
        scores = []
        issues = []
        
        # 1. TIÊU CHÍ 1: Thân thẳng (60%)
        body_angle = self.geo.calculate_angle(kp[5], kp[11], kp[13])  # Vai-Hông-Gối
        
        if body_angle >= 170:
            scores.append(100)
        elif body_angle >= 160:
            scores.append(80)
            issues.append("hông hơi cao")
        elif body_angle >= 150:
            scores.append(60)
            issues.append("hông quá cao")
        else:
            scores.append(40)
            issues.append("cần giữ thân thẳng")
        
        # 2. TIÊU CHÍ 2: Tay chống thẳng (40%)
        # Góc vai-khuỷu-cổ tay
        left_arm = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        right_arm = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_arm = (left_arm + right_arm) / 2
        
        if avg_arm >= 160:
            scores.append(100)
        elif avg_arm >= 140:
            scores.append(70)
            issues.append("tay hơi cong")
        else:
            scores.append(40)
            issues.append("duỗi tay thẳng hơn")
        
        # Tính điểm: 60% thân + 40% tay
        final_score = int(scores[0] * 0.6 + scores[1] * 0.4)
        
        if final_score >= 90:
            feedback = "Tuyệt vời! Thân thẳng hoàn hảo ✅"
        elif final_score >= 75:
            feedback = f"Tốt! Cần điều chỉnh: {', '.join(issues)} ⚠️"
        elif final_score >= 60:
            feedback = f"Cần cải thiện: {', '.join(issues)} ⚠️"
        else:
            feedback = f"Chưa đúng tư thế. Kiểm tra: {', '.join(issues)} ❌"
        
        return final_score, feedback
    
    def _evaluate_tree(self, kp):
        """
        Tree: Đánh giá 3 tiêu chí
        1. Chân đứng thẳng (góc gối ≥ 165°)
        2. Chân kia gập vào (góc gối < 120°)  
        3. Tay chắp lên (góc khuỷu tay < 100°)
        """
        scores = []
        issues = []
        
        # 1. TIÊU CHÍ 1: Chân đứng phải thẳng
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        standing_angle = max(angle_left, angle_right)
        bent_angle = min(angle_left, angle_right)
        
        if standing_angle >= 165:
            scores.append(100)
        elif standing_angle >= 155:
            scores.append(80)
            issues.append("chân đứng hơi cong")
        else:
            scores.append(60)
            issues.append("chân đứng cong quá")
        
        # 2. TIÊU CHÍ 2: Chân kia phải gập vào
        if bent_angle < 120:  # Gập tốt
            scores.append(100)
        elif bent_angle < 140:  # Gập nhẹ
            scores.append(70)
            issues.append("chân gập chưa đủ")
        else:  # Không gập - SAI!
            scores.append(30)
            issues.append("chân kia chưa gập vào")
        
        # 3. TIÊU CHÍ 3: Tay phải chắp lên/giơ cao
        elbow_angle_l = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        elbow_angle_r = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_elbow = (elbow_angle_l + elbow_angle_r) / 2
        
        if avg_elbow < 100:  # Tay giơ cao/chắp
            scores.append(100)
        elif avg_elbow < 130:
            scores.append(70)
            issues.append("tay chưa giơ cao đủ")
        else:  # Tay thả xuống
            scores.append(30)
            issues.append("tay cần chắp/giơ cao")
        
        # Tính điểm tổng (trung bình 3 tiêu chí)
        final_score = int(sum(scores) / len(scores))
        
        if final_score >= 90:
            feedback = "Hoàn hảo! Tư thế Tree chuẩn ✅"
        elif final_score >= 75:
            feedback = f"Tốt! Cần điều chỉnh: {', '.join(issues)} ⚠️"
        elif final_score >= 60:
            feedback = f"Cần cải thiện: {', '.join(issues)} ⚠️"
        else:
            feedback = f"Chưa đúng tư thế. Kiểm tra: {', '.join(issues)} ❌"
        
        return final_score, feedback
    
    def _evaluate_warrior(self, kp):
        """
        Warrior II: Đánh giá 3 tiêu chí
        1. Chân trước gập 80-125° (40%)
        2. Chân sau thẳng ≥155° (30%)
        3. Tay dang ngang ≥150° (30%)
        """
        scores = []
        issues = []
        
        # Tính góc chân
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        front_angle = min(angle_left, angle_right)
        back_angle = max(angle_left, angle_right)
        
        # 1. TIÊU CHÍ 1: Chân trước gập (40%)
        if 80 <= front_angle <= 125:
            scores.append(100)
        elif 70 <= front_angle <= 135:
            scores.append(80)
            issues.append("góc chân trước điều chỉnh nhẹ")
        elif 60 <= front_angle <= 145:
            scores.append(60)
            issues.append("chân trước cần gập hơn")
        else:
            scores.append(40)
            issues.append("chân trước chưa đúng")
        
        # 2. TIÊU CHÍ 2: Chân sau thẳng (30%)
        if back_angle >= 155:
            scores.append(100)
        elif back_angle >= 140:
            scores.append(70)
            issues.append("chân sau hơi cong")
        else:
            scores.append(40)
            issues.append("duỗi chân sau thẳng")
        
        # 3. TIÊU CHÍ 3: Tay dang ngang (30%)
        # Góc giữa 2 vai và khuỷu tay (phải gần thẳng hàng = góc lớn)
        left_shoulder_arm = self.geo.calculate_angle(kp[6], kp[5], kp[7])   # Vai phải - Vai trái - Khuỷu trái
        right_shoulder_arm = self.geo.calculate_angle(kp[5], kp[6], kp[8])  # Vai trái - Vai phải - Khuỷu phải
        avg_arm_angle = (left_shoulder_arm + right_shoulder_arm) / 2
        
        if avg_arm_angle >= 150:  # Tay dang thẳng ngang
            scores.append(100)
        elif avg_arm_angle >= 130:
            scores.append(70)
            issues.append("tay chưa dang thẳng ngang")
        else:
            scores.append(40)
            issues.append("cần dang tay ra 2 bên")
        
        # Tính điểm: 40% chân trước + 30% chân sau + 30% tay
        final_score = int(scores[0] * 0.4 + scores[1] * 0.3 + scores[2] * 0.3)
        
        if final_score >= 90:
            feedback = "Tuyệt vời! Tư thế Warrior2 hoàn hảo ✅"
        elif final_score >= 75:
            feedback = f"Tốt! Cần điều chỉnh: {', '.join(issues)} ⚠️"
        elif final_score >= 60:
            feedback = f"Cần cải thiện: {', '.join(issues)} ⚠️"
        else:
            feedback = f"Chưa đúng tư thế. Kiểm tra: {', '.join(issues)} ❌"
        
        return final_score, feedback
    
    def _evaluate_goddess(self, kp):
        """
        Goddess: Đánh giá 2 tiêu chí
        1. Cả 2 chân squat 80-100° (70%)
        2. Tay trước ngực/giơ (30%)
        """
        scores = []
        issues = []
        
        # 1. TIÊU CHÍ 1: Cả 2 chân squat (70%)
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        avg_angle = (angle_left + angle_right) / 2
        
        if 80 <= avg_angle <= 100:
            scores.append(100)
        elif 70 <= avg_angle <= 110:
            scores.append(80)
            issues.append("độ squat điều chỉnh nhẹ")
        elif 60 <= avg_angle <= 120:
            scores.append(60)
            issues.append("squat sâu hơn")
        else:
            scores.append(40)
            issues.append("gối cần ở 90°")
        
        # 2. TIÊU CHÍ 2: Tay (30%)
        # Kiểm tra khuỷu tay (gập = trước ngực, < 120°)
        left_elbow = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        right_elbow = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_elbow = (left_elbow + right_elbow) / 2
        
        if avg_elbow < 120:  # Tay gập/trước ngực
            scores.append(100)
        elif avg_elbow < 140:
            scores.append(70)
            issues.append("tay chưa đúng vị trí")
        else:
            scores.append(50)
            issues.append("đặt tay trước ngực")
        
        # Tính điểm: 70% chân + 30% tay
        final_score = int(scores[0] * 0.7 + scores[1] * 0.3)
        
        if final_score >= 90:
            feedback = "Hoàn hảo! Độ sâu squat chuẩn ✅"
        elif final_score >= 75:
            feedback = f"Tốt! Cần điều chỉnh: {', '.join(issues)} ⚠️"
        elif final_score >= 60:
            feedback = f"Cần cải thiện: {', '.join(issues)} ⚠️"
        else:
            feedback = f"Chưa đúng tư thế. Kiểm tra: {', '.join(issues)} ❌"
        
        return final_score, feedback
    
    def _evaluate_downdog(self, kp):
        """
        Downdog: Đánh giá 3 tiêu chí
        1. Chân thẳng ≥165° (40%)
        2. Tay thẳng ≥160° (30%)
        3. Hông gập (chữ V) 70-110° (30%)
        """
        scores = []
        issues = []
        
        # 1. TIÊU CHÍ 1: Chân thẳng (40%)
        angle_left_leg = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right_leg = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        avg_leg = (angle_left_leg + angle_right_leg) / 2
        
        if avg_leg >= 165:
            scores.append(100)
        elif avg_leg >= 155:
            scores.append(80)
            issues.append("chân hơi cong")
        elif avg_leg >= 145:
            scores.append(60)
            issues.append("duỗi chân thẳng hơn")
        else:
            scores.append(40)
            issues.append("chân cần thẳng")
        
        # 2. TIÊU CHÍ 2: Tay thẳng (30%)
        left_arm = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        right_arm = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_arm = (left_arm + right_arm) / 2
        
        if avg_arm >= 160:
            scores.append(100)
        elif avg_arm >= 145:
            scores.append(75)
            issues.append("tay hơi cong")
        else:
            scores.append(50)
            issues.append("duỗi tay thẳng")
        
        # 3. TIÊU CHÍ 3: Hông gập (chữ V ngược) (30%)
        # Góc vai-hông-gối (phải nhỏ = hông gập lên)
        hip_angle = self.geo.calculate_angle(kp[5], kp[11], kp[13])
        
        if 70 <= hip_angle <= 110:  # Gập tốt (chữ V)
            scores.append(100)
        elif 60 <= hip_angle <= 120:
            scores.append(75)
            issues.append("hông điều chỉnh nhẹ")
        else:
            scores.append(50)
            issues.append("nâng hông lên cao hơn")
        
        # Tính điểm: 40% chân + 30% tay + 30% hông
        final_score = int(scores[0] * 0.4 + scores[1] * 0.3 + scores[2] * 0.3)
        
        if final_score >= 90:
            feedback = "Tuyệt vời! Tư thế Downdog hoàn hảo ✅"
        elif final_score >= 75:
            feedback = f"Tốt! Cần điều chỉnh: {', '.join(issues)} ⚠️"
        elif final_score >= 60:
            feedback = f"Cần cải thiện: {', '.join(issues)} ⚠️"
        else:
            feedback = f"Chưa đúng tư thế. Kiểm tra: {', '.join(issues)} ❌"
        
        return final_score, feedback

