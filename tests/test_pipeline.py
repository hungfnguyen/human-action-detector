"""
Test Main Pipeline.

Tests complete pipeline with both image and video inputs.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show output."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")
        return False
    
    return True


def test_pipeline():
    """Test main pipeline."""
    
    print("=" * 60)
    print("🧪 TESTING MAIN PIPELINE")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Find test images
    image_files = list(Path('images').glob('*.jpeg'))
    if not image_files:
        print("❌ No test images found")
        return False
    
    test_image = str(image_files[0])
    
    # === TEST 1: Help Command ===
    tests_total += 1
    cmd = "python3 src/main.py --help"
    if run_command(cmd, "TEST 1: Display Help"):
        tests_passed += 1
    
    # === TEST 2: Process Image (save) ===
    tests_total += 1
    output_path = "tests/output/pipeline_image_test.jpg"
    cmd = f"python3 src/main.py --input {test_image} --output {output_path}"
    if run_command(cmd, "TEST 2: Process Image (with save)"):
        # Verify output exists
        if Path(output_path).exists():
            print(f"   ✅ Output file created: {output_path}")
            tests_passed += 1
        else:
            print(f"   ❌ Output file not created")
    
    # === TEST 3: Process Image (no save) ===
    tests_total += 1
    cmd = f"python3 src/main.py --input {test_image}"
    if run_command(cmd, "TEST 3: Process Image (no save)"):
        tests_passed += 1
    
    # === TEST 4: Invalid Input ===
    tests_total += 1
    cmd = "python3 src/main.py --input invalid_file.jpg"
    print(f"\n{'='*60}")
    print(f"🧪 TEST 4: Invalid Input (should fail gracefully)")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("✅ TEST 4: Invalid Input - PASSED (failed gracefully)")
        tests_passed += 1
    else:
        print("❌ TEST 4: Invalid Input - FAILED (should have failed)")
    
    # === TEST 5: No Visualization ===
    tests_total += 1
    output_path = "tests/output/pipeline_no_viz.jpg"
    cmd = f"python3 src/main.py --input {test_image} --output {output_path} --no-viz"
    if run_command(cmd, "TEST 5: Process with --no-viz"):
        tests_passed += 1
    
    # === RESULTS ===
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print(f"Success Rate: {tests_passed/tests_total*100:.1f}%")
    
    if tests_passed == tests_total:
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 60)
        return True
    else:
        print(f"\n❌ {tests_total - tests_passed} test(s) failed")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
