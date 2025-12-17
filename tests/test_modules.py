"""
Test Detection and Recognition modules.

Quick test to verify both modules are working correctly.
"""

import sys
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detection import PoseDetector
from recognition import MLPoseClassifier


def test_modules():
    """Test both detection and recognition modules."""
    
    print("=" * 60)
    print("🧪 TESTING DETECTION & RECOGNITION MODULES")
    print("=" * 60)
    
    # Find test image
    test_images = list(Path('images').glob('*.jpeg'))
    if not test_images:
        print("❌ No test images found in images/")
        return False
    
    image_path = test_images[0]
    print(f"\n📸 Test image: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image")
        return False
    
    print(f"   Image shape: {image.shape}")
    
    # ===== TEST 1: DETECTION =====
    print("\n" + "─" * 60)
    print("TEST 1: POSE DETECTION")
    print("─" * 60)
    
    try:
        detector = PoseDetector('yolov8m-pose.pt')
        
        # Detect keypoints (normalized)
        keypoints_norm = detector.detect(image, return_absolute=False)
        
        if keypoints_norm is None:
            print("❌ No person detected in image!")
            return False
        
        print(f"✅ Detected keypoints (normalized): {len(keypoints_norm)} values")
        print(f"   Range: [{min(keypoints_norm):.3f}, {max(keypoints_norm):.3f}]")
        print(f"   Sample (first 10): {[f'{x:.3f}' for x in keypoints_norm[:10]]}")
        
        # Also get absolute coordinates for visualization
        keypoints_abs = detector.detect(image, return_absolute=True)
        print(f"✅ Detected keypoints (absolute): {len(keypoints_abs)} points")
        print(f"   Sample (first 3): {keypoints_abs[:3]}")
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== TEST 2: RECOGNITION =====
    print("\n" + "─" * 60)
    print("TEST 2: POSE RECOGNITION")
    print("─" * 60)
    
    try:
        classifier = MLPoseClassifier('./models/pose_classification.pth')
        
        # Predict pose
        pose_name, confidence = classifier.predict(keypoints_norm)
        
        print(f"✅ Recognized pose: {pose_name}")
        print(f"✅ Confidence: {confidence:.1%}")
        
        # Test all classes
        print(f"   Available classes: {classifier.classes}")
        
    except Exception as e:
        print(f"❌ Recognition failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== TEST 3: INTEGRATION =====
    print("\n" + "─" * 60)
    print("TEST 3: FULL PIPELINE")
    print("─" * 60)
    
    try:
        # Full pipeline
        result = detector.predict(image)
        keypoints = detector.get_keypoints_normalized(result)
        pose, conf = classifier.predict(keypoints)
        
        print(f"✅ Pipeline result:")
        print(f"   Pose: {pose}")
        print(f"   Confidence: {conf:.1%}")
        
        # Visualize
        result_img = result.plot()
        
        # Add text overlay
        h, w = result_img.shape[:2]
        text = f"{pose.upper()} ({conf:.0%})"
        cv2.putText(result_img, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Save result
        output_path = 'tests/output/test_modules_output.jpg'
        cv2.imwrite(output_path, result_img)
        print(f"✅ Saved result to: {output_path}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== SUCCESS =====
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   ✓ Detection module working")
    print(f"   ✓ Recognition module working")
    print(f"   ✓ Full pipeline working")
    print(f"   ✓ Result: {pose} ({conf:.1%})")
    print("\n✅ Modules ready for next phase!")
    
    return True


if __name__ == "__main__":
    success = test_modules()
    sys.exit(0 if success else 1)
