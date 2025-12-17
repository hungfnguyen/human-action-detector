"""
Test script for Detection module.

Tests PoseDetector with sample image.
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detection import PoseDetector


def test_detection():
    """Test pose detection on sample image."""
    print("=" * 50)
    print("TESTING DETECTION MODULE")
    print("=" * 50)
    
    # 1. Initialize detector
    print("\n1. Loading model...")
    detector = PoseDetector('yolov8m-pose.pt')
    
    # 2. Load test image
    print("\n2. Loading test image...")
    test_images = list(Path('images').glob('*.jpeg'))
    
    if not test_images:
        print("❌ No test images found in images/")
        return
    
    image_path = test_images[0]
    print(f"   Using: {image_path}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"   Image shape: {image.shape}")
    
    # 3. Detect keypoints (normalized)
    print("\n3. Detecting keypoints (normalized)...")
    keypoints_norm = detector.detect(image, return_absolute=False)
    
    if keypoints_norm is None:
        print("   ❌ No person detected!")
        return
    
    print(f"   ✅ Detected keypoints: {len(keypoints_norm)} values")
    print(f"   First 10 values: {keypoints_norm[:10]}")
    
    # 4. Detect keypoints (absolute pixels)
    print("\n4. Detecting keypoints (absolute)...")
    keypoints_abs = detector.detect(image, return_absolute=True)
    
    if keypoints_abs:
        print(f"   ✅ Detected keypoints: {len(keypoints_abs)} points")
        print(f"   First 5 points: {keypoints_abs[:5]}")
    
    # 5. Visualize result
    print("\n5. Visualizing...")
    result = detector.predict(image)
    result_img = result.plot()
    
    output_path = 'tests/output/test_detection_output.jpg'
    cv2.imwrite(output_path, result_img)
    print(f"   ✅ Saved to: {output_path}")
    
    print("\n" + "=" * 50)
    print("✅ DETECTION MODULE TEST PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_detection()
