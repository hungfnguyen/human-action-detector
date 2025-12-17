"""
Test Visualization module.

Tests skeleton drawing and UI overlay.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detection import PoseDetector
from recognition import MLPoseClassifier
from evaluation import PoseEvaluator
from visualization import SkeletonDrawer, OverlayUI


def test_visualization():
    """Test visualization components."""
    
    print("=" * 60)
    print("🧪 TESTING VISUALIZATION MODULE")
    print("=" * 60)
    
    # Find test image
    test_images = list(Path('images').glob('*.jpeg'))
    if not test_images:
        print("❌ No test images found")
        return False
    
    image_path = test_images[0]
    print(f"\n📸 Test image: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print("❌ Failed to load image")
        return False
    
    print(f"   Image shape: {image.shape}")
    
    # Initialize modules
    print("\n📦 Loading modules...")
    detector = PoseDetector('yolov8m-pose.pt')
    classifier = MLPoseClassifier('./models/pose_classification.pth')
    evaluator = PoseEvaluator()
    
    # Detect and recognize
    print("\n🔍 Processing...")
    keypoints_norm = detector.detect(image, return_absolute=False)
    keypoints_abs = detector.detect(image, return_absolute=True)
    
    if keypoints_norm is None:
        print("❌ No person detected")
        return False
    
    pose_name, confidence = classifier.predict(keypoints_norm)
    print(f"   Pose: {pose_name} ({confidence:.1%})")
    
    # Evaluate pose
    score, feedback = evaluator.evaluate(pose_name, keypoints_abs)
    print(f"   Score: {score}/100")
    print(f"   Feedback: {feedback}")
    
    # === TEST 1: SKELETON DRAWER ===
    print("\n" + "─" * 60)
    print("TEST 1: SKELETON DRAWER")
    print("─" * 60)
    
    try:
        drawer = SkeletonDrawer()
        
        # Draw skeleton
        result1 = image.copy()
        result1 = drawer.draw(result1, keypoints_abs, score)
        
        # Save
        output_path1 = 'tests/output/test_skeleton.jpg'
        cv2.imwrite(output_path1, result1)
        print(f"✅ Saved skeleton visualization: {output_path1}")
        
    except Exception as e:
        print(f"❌ Skeleton drawer failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === TEST 2: OVERLAY UI ===
    print("\n" + "─" * 60)
    print("TEST 2: OVERLAY UI")
    print("─" * 60)
    
    try:
        ui = OverlayUI()
        
        # Draw scoreboard
        result2 = image.copy()
        result2 = ui.draw_scoreboard(result2, pose_name, score, feedback)
        
        # Save
        output_path2 = 'tests/output/test_overlay.jpg'
        cv2.imwrite(output_path2, result2)
        print(f"✅ Saved overlay UI: {output_path2}")
        
    except Exception as e:
        print(f"❌ Overlay UI failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === TEST 3: COMBINED ===
    print("\n" + "─" * 60)
    print("TEST 3: COMBINED VISUALIZATION")
    print("─" * 60)
    
    try:
        # Draw both skeleton and UI
        result3 = image.copy()
        result3 = drawer.draw(result3, keypoints_abs, score)
        result3 = ui.draw_scoreboard(result3, pose_name, score, feedback)
        result3 = ui.draw_fps(result3, 25.5)  # Mock FPS
        
        # Save
        output_path3 = 'tests/output/test_viz_combined.jpg'
        cv2.imwrite(output_path3, result3)
        print(f"✅ Saved combined visualization: {output_path3}")
        
    except Exception as e:
        print(f"❌ Combined visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === SUCCESS ===
    print("\n" + "=" * 60)
    print("🎉 ALL VISUALIZATION TESTS PASSED!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   ✓ Skeleton drawer working")
    print(f"   ✓ Overlay UI working")
    print(f"   ✓ Combined visualization working")
    print(f"\n📁 Output files:")
    print(f"   - {output_path1}")
    print(f"   - {output_path2}")
    print(f"   - {output_path3}")
    print("\n✅ Visualization module ready!")
    
    return True


if __name__ == "__main__":
    success = test_visualization()
    sys.exit(0 if success else 1)
