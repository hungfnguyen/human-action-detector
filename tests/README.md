# Tests Directory

This directory contains test scripts and test outputs.

## Structure

```
tests/
├── test_modules.py          # Main test script
├── test_detection.py         # Detection module test
└── output/                   # Test output images
    └── test_modules_output.jpg
```

## Running Tests

### Full module test:
```bash
cd /home/hungfnguyen/project/human-action-detector
python3 tests/test_modules.py
```

### Detection only:
```bash
python3 tests/test_detection.py
```

## Test Output

Test result images are saved to `tests/output/`.

## Notes

- Tests require dependencies from `requirements.txt`
- Models must be downloaded: `yolov8m-pose.pt`, `models/pose_classification.pth`
- Test images should be in `images/` directory
