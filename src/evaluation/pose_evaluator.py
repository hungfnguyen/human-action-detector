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
            # Unknown pose - không nhận dạng được
            return 0, "Không xác định được tư thế chuẩn"
    
    def _evaluate_plank(self, kp):
        """
        Plank: Đánh giá 2 tiêu chí
        1. Thân thẳng (60%)
        2. Tay chống thẳng (40%)
        """
        scores = []
        issues = []
        
        # 1. TIÊU CHÍ 1: Thân thẳng (60%)
        body_angle = self.geo.calculate_angle(kp[5], kp[11], kp[13])
        
        # Plank CHUẨN: 160-175° (hông hơi cao hơn thẳng tuyệt đối)
        if 160 <= body_angle <= 175:
            scores.append(100)
        elif 175 < body_angle <= 180:  # Quá thẳng (như nằm sấp)
            scores.append(85)
            issues.append("hông cần cao hơn chút")
        elif 150 <= body_angle < 160:
            scores.append(85)
            issues.append("hông hơi cao")
        else:
            scores.append(70)
            issues.append("hông quá cao")
        
        # 2. TIÊU CHÍ 2: Tay chống (40%)
        # Support BOTH: Straight-arm plank AND elbow plank
        left_arm = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        right_arm = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_arm = (left_arm + right_arm) / 2
        
        # Check if elbows OR wrists are supporting
        avg_elbow_y = (kp[7][1] + kp[8][1]) / 2
        avg_wrist_y = (kp[9][1] + kp[10][1]) / 2
        avg_shoulder_y = (kp[5][1] + kp[6][1]) / 2
        
        # Elbow plank: elbows near ground (close to wrist level)
        # Straight-arm plank: wrists near ground, elbows raised
        is_elbow_plank = abs(avg_elbow_y - avg_wrist_y) < 30  # Elbow & wrist same level
        is_straight_arm = avg_elbow_y < avg_wrist_y - 30  # Elbow raised
        
        # Check if actually supporting (not lying flat)
        is_supporting = (avg_elbow_y > avg_shoulder_y + 50) or (avg_wrist_y > avg_shoulder_y + 50)
        
        if (avg_arm >= 160 or is_elbow_plank) and is_supporting:  # Good form + supporting
            scores.append(100)
        elif (avg_arm >= 145 or is_elbow_plank) and is_supporting:
            scores.append(85)
            issues.append("tay hơi cong")
        elif is_supporting:
            scores.append(75)
            issues.append("duỗi tay thẳng hơn")
        else:  # Not supporting = lying flat
            scores.append(30)
            issues.append("cần nâng người lên bằng tay")
        
        # Tính điểm: 60% thân + 40% tay
        final_score = int(scores[0] * 0.6 + scores[1] * 0.4)
        
        if final_score >= 90:
            feedback = "Tuyệt vời! Tư thế Plank hoàn hảo ✅"
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
        1. Chân đứng thẳng (33%)
        2. Chân kia gập vào (33%)
        3. Tay chắp lên (33%)
        """
        scores = []
        issues = []
        
        # 1. Chân đứng thẳng
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        standing_angle = max(angle_left, angle_right)
        bent_angle = min(angle_left, angle_right)
        
        if standing_angle >= 155:  # Relaxed from 165° due to ankle detection variance
            scores.append(100)
        elif standing_angle >= 145:
            scores.append(85)
            issues.append("chân đứng hơi cong")
        else:
            scores.append(70)
            issues.append("chân đứng cong quá")
        
        # 2. Chân kia gập vào
        if bent_angle < 120:
            scores.append(100)
        elif bent_angle < 140:
            scores.append(70)
            issues.append("chân gập chưa đủ")
        else:
            scores.append(30)
            issues.append("chân kia chưa gập vào")
        
        # 3. Tay chắp/giơ cao - CHECK BOTH elbow angle AND wrist distance
        elbow_angle_l = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        elbow_angle_r = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_elbow = (elbow_angle_l + elbow_angle_r) / 2
        
        # Check wrist distance (hands should be together/clasped)
        wrist_left = kp[9]
        wrist_right = kp[10]
        wrist_distance = ((wrist_left[0] - wrist_right[0])**2 + 
                         (wrist_left[1] - wrist_right[1])**2)**0.5
        
        # Shoulder width for reference
        shoulder_width = ((kp[5][0] - kp[6][0])**2 + 
                         (kp[5][1] - kp[6][1])**2)**0.5
        
        # Wrists should be close (within 50% of shoulder width)
        wrists_together = wrist_distance < (shoulder_width * 0.5)
        
        if avg_elbow < 100 and wrists_together:  # Tay gập + chắp
            scores.append(100)
        elif avg_elbow < 130 and wrists_together:  # Tay hơi cao + chắp
            scores.append(80)
            issues.append("tay chưa giơ cao đủ")
        elif avg_elbow < 100:  # Tay gập nhưng KHÔNG chắp
            scores.append(50)
            issues.append("tay cần chắp lại")
        else:  # Tay không gập hoặc không chắp
            scores.append(30)
            issues.append("tay cần chắp/giơ cao")
        
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
        1. Chân trước gập (40%)
        2. Chân sau thẳng (30%)
        3. Tay dang ngang (30%)
        """
        scores = []
        issues = []
        
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        front_angle = min(angle_left, angle_right)
        back_angle = max(angle_left, angle_right)
        
        # 1. Chân trước gập (40%)
        if 80 <= front_angle <= 125:
            scores.append(100)
        elif 70 <= front_angle <= 135:
            scores.append(85)
            issues.append("góc chân trước điều chỉnh nhẹ")
        elif 60 <= front_angle <= 145:
            scores.append(70)
            issues.append("chân trước cần gập hơn")
        else:
            scores.append(50)
            issues.append("chân trước chưa đúng")
        
        # 2. Chân sau thẳng (30%)
        if back_angle >= 155:
            scores.append(100)
        elif back_angle >= 145:
            scores.append(80)
            issues.append("chân sau hơi cong")
        else:
            scores.append(60)
            issues.append("duỗi chân sau thẳng")
        
        # 3. Tay dang ngang (30%)
        left_shoulder_arm = self.geo.calculate_angle(kp[6], kp[5], kp[7])
        right_shoulder_arm = self.geo.calculate_angle(kp[5], kp[6], kp[8])
        avg_arm_angle = (left_shoulder_arm + right_shoulder_arm) / 2
        
        if avg_arm_angle >= 150:
            scores.append(100)
        elif avg_arm_angle >= 130:
            scores.append(75)
            issues.append("tay chưa dang thẳng ngang")
        else:
            scores.append(50)
            issues.append("cần dang tay ra 2 bên")
        
        # Tính điểm: 40% + 30% + 30%
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
        1. Chân squat (70%)
        2. Tay giơ vuông góc (30%)
        """
        scores = []
        issues = []
        
        # 1. Chân squat (70%)
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        avg_angle = (angle_left + angle_right) / 2
        
        if 70 <= avg_angle <= 140:
            scores.append(100)
        elif 60 <= avg_angle <= 150:
            scores.append(85)
            issues.append("độ squat điều chỉnh nhẹ")
        elif 50 <= avg_angle <= 160:
            scores.append(70)
            issues.append("squat sâu hơn")
        else:
            scores.append(50)
            issues.append("gối cần ở 90°")
        
        
        # 2. Tay giơ vuông góc (30%) - CHECK Y POSITION!
        # Kiểm tra cổ tay có CAO HƠN vai không (tay giơ lên)
        shoulder_y_left = kp[5][1]
        shoulder_y_right = kp[6][1]
        wrist_y_left = kp[9][1]
        wrist_y_right = kp[10][1]
        
        avg_shoulder_y = (shoulder_y_left + shoulder_y_right) / 2
        avg_wrist_y = (wrist_y_left + wrist_y_right) / 2
        
        # Y axis: smaller = higher. Wrist should be ABOVE shoulder
        wrist_lift = avg_shoulder_y - avg_wrist_y  # Positive = wrist above shoulder
        
        # Also check elbow straightness (arms should be straight when raised)
        left_elbow_angle = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        right_elbow_angle = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_elbow = (left_elbow_angle + right_elbow_angle) / 2
        
        if wrist_lift > 20 and avg_elbow >= 140:  # Tay giơ cao + thẳng (ngang vai trở lên)
            scores.append(100)
        elif wrist_lift > 10 and avg_elbow >= 120:  # Tay giơ gần ngang vai
            scores.append(85)
            issues.append("tay chưa giơ đủ cao")
        elif wrist_lift > 0:  # Tay hơi cao hơn vai
            scores.append(60)
            issues.append("cần giơ tay cao hơn")
        else:  # Tay KHÔNG cao hơn vai = không giơ!
            scores.append(0)  # 0 điểm!
            issues.append("tay phải giơ lên vuông góc")
        
        # Tính điểm: 70% chân + 30% tay
        final_score = int(scores[0] * 0.7 + scores[1] * 0.3)
        
        if final_score >= 90:
            feedback = "Hoàn hảo! Tư thế Goddess chuẩn ✅"
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
        1. Chân thẳng (40%)
        2. Tay thẳng (30%)
        3. Hông gập chữ V (30%)
        """
        scores = []
        issues = []
        
        # 1. Chân thẳng (40%)
        angle_left_leg = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right_leg = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        avg_leg = (angle_left_leg + angle_right_leg) / 2
        
        if avg_leg >= 165:
            scores.append(100)
        elif avg_leg >= 155:
            scores.append(85)
            issues.append("chân hơi cong")
        elif avg_leg >= 145:
            scores.append(70)
            issues.append("duỗi chân thẳng hơn")
        else:
            scores.append(50)
            issues.append("chân cần thẳng")
        
        # 2. Tay thẳng (30%)
        left_arm = self.geo.calculate_angle(kp[5], kp[7], kp[9])
        right_arm = self.geo.calculate_angle(kp[6], kp[8], kp[10])
        avg_arm = (left_arm + right_arm) / 2
        
        if avg_arm >= 160:
            scores.append(100)
        elif avg_arm >= 145:
            scores.append(80)
            issues.append("tay hơi cong")
        else:
            scores.append(60)
            issues.append("duỗi tay thẳng")
        
        # 3. Hông gập (chữ V) (30%)
        hip_angle = self.geo.calculate_angle(kp[5], kp[11], kp[13])
        
        if 70 <= hip_angle <= 110:
            scores.append(100)
        elif 60 <= hip_angle <= 120:
            scores.append(80)
            issues.append("hông điều chỉnh nhẹ")
        else:
            scores.append(60)
            issues.append("nâng hông lên cao hơn")
        
        # Tính điểm: 40% + 30% + 30%
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

