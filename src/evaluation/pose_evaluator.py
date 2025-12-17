"""
Pose Evaluator - Minimal scoring for yoga poses.

Đánh giá đơn giản dựa trên 1-2 góc quan trọng nhất cho mỗi tư thế.
"""

from ..geometry.geometry_utils import GeometryUtils


class PoseEvaluator:
    """
    Evaluates yoga poses with simple geometric rules.
    
    Returns score (0-100) and feedback message.
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
                feedback_message: String describing the evaluation
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
            return 75, "Pose detected"
    
    def _evaluate_plank(self, kp):
        """
        Plank: Kiểm tra thân thẳng (Shoulder-Hip-Knee angle ≈ 180°)
        """
        # Keypoints: 5=left_shoulder, 11=left_hip, 13=left_knee
        angle = self.geo.calculate_angle(kp[5], kp[11], kp[13])
        
        if angle >= 170:
            return 95, "Excellent! Body alignment perfect ✅"
        elif angle >= 160:
            return 80, "Good! Hip slightly high, lower a bit ⚠️"
        elif angle >= 150:
            return 65, "Hip too high, align body straighter ⚠️"
        else:
            return 50, "Poor alignment, keep body straight ❌"
    
    def _evaluate_tree(self, kp):
        """
        Tree: Kiểm tra chân đứng thẳng (Standing knee angle > 160°)
        """
        # Keypoints: 11=left_hip, 13=left_knee, 15=left_ankle
        # Hoặc right leg tùy pose
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        
        # Chọn chân đứng (góc lớn hơn = chân thẳng)
        standing_angle = max(angle_left, angle_right)
        
        if standing_angle >= 165:
            return 95, "Perfect! Standing leg very straight ✅"
        elif standing_angle >= 155:
            return 80, "Good balance, straighten leg more ⚠️"
        else:
            return 65, "Bend in standing leg, try to straighten ⚠️"
    
    def _evaluate_warrior(self, kp):
        """
        Warrior II: Kiểm tra front knee bend (≈ 90°)
        """
        # Check both legs, front leg should be ~90°
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        
        # Front leg có góc nhỏ hơn
        front_angle = min(angle_left, angle_right)
        
        if 80 <= front_angle <= 100:
            return 95, "Excellent lunge position! ✅"
        elif 70 <= front_angle <= 110:
            return 80, "Good, adjust knee angle slightly ⚠️"
        else:
            return 65, "Bend front knee more towards 90° ⚠️"
    
    def _evaluate_goddess(self, kp):
        """
        Goddess: Kiểm tra squat depth (knee angles ~90°)
        """
        # Both knees should be bent ~90°
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        
        avg_angle = (angle_left + angle_right) / 2
        
        if 80 <= avg_angle <= 100:
            return 95, "Perfect squat depth! ✅"
        elif 70 <= avg_angle <= 110:
            return 80, "Good squat, adjust depth slightly ⚠️"
        else:
            return 65, "Squat deeper, knees at 90° ⚠️"
    
    def _evaluate_downdog(self, kp):
        """
        Downdog: Kiểm tra chân thẳng (knee angles > 160°)
        """
        # Both legs should be straight
        angle_left = self.geo.calculate_angle(kp[11], kp[13], kp[15])
        angle_right = self.geo.calculate_angle(kp[12], kp[14], kp[16])
        
        avg_angle = (angle_left + angle_right) / 2
        
        if avg_angle >= 165:
            return 95, "Excellent! Legs very straight ✅"
        elif avg_angle >= 155:
            return 80, "Good, straighten legs more ⚠️"
        else:
            return 65, "Bend in legs, try to straighten ⚠️"
