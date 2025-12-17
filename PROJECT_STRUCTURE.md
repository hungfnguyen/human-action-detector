# 📁 Project Structure - Yoga Pose Recognition

## 🏗️ Architecture: Hybrid (ML Recognition + Minimal Geometric Scoring)

```
human-action-detector/
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
│
├── 📂 src/                          # Source code
│   ├── __init__.py
│   ├── main.py                      # CLI entry point (Image + Video)
│   │
│   ├── 📂 detection/                # YOLOv8 Pose Detection
│   │   ├── __init__.py
│   │   ├── pose_detector.py         # YOLOv8 wrapper
│   │   └── keypoint_constants.py    # COCO keypoints
│   │
│   ├── 📂 recognition/              # ML Classification
│   │   ├── __init__.py
│   │   ├── ml_classifier.py         # Neural Network (CORE)
│   │   └── pose_recognizer.py       # Wrapper
│   │
│   ├── 📂 geometry/                 # Minimal Geometric Analysis
│   │   ├── __init__.py
│   │   └── geometry_utils.py        # calculate_angle(), distance
│   │
│   ├── 📂 evaluation/               # Pose Scoring
│   │   ├── __init__.py
│   │   └── pose_evaluator.py        # Simple scoring (0-100)
│   │
│   ├── 📂 visualization/            # Drawing & UI
│   │   ├── __init__.py
│   │   ├── skeleton_drawer.py       # Skeleton drawing
│   │   └── overlay_ui.py            # Score overlay
│   │
│   └── 📂 config/                   # Configuration
│       ├── __init__.py
│       └── app_config.py            # Settings
│
├── 📂 models/                       # Trained models
│   └── pose_classification.pth      # ML weights
│
├── 📂 datasets/                     # Training data
│   └── yoga_pose_keypoint.csv       # 1000+ samples
│
└── app_gui.py                       # GUI Application
```

---

## 🔄 Data Flow

```
┌─────────────────┐
│  Image / Video  │
└────────┬────────┘
         │
  ┌──────▼──────┐
  │   YOLOv8    │ (Pose Detection)
  │ 17 Keypoints│
  └──────┬──────┘
         │
         ├─────────────┬──────────────┐
         │             │              │
  ┌──────▼──────┐ ┌───▼────────┐    │
  │ML Classifier│ │  Geometry  │    │
  │  "Plank"    │ │ Calc angles│    │
  └──────┬──────┘ └─────┬──────┘    │
         │              │            │
         └──────┬───────┘            │
                │                    │
         ┌──────▼────────┐           │
         │  Evaluator    │           │
         │  Score: 85    │           │
         │  "Good! ⚠️"   │           │
         └──────┬────────┘           │
                │                    │
         ┌──────▼────────────────────▼┐
         │      Visualization          │
         │  Skeleton + Score + Feedback│
         └─────────────────────────────┘
```

---

## 📋 Module Responsibilities

### 1. **detection/** - YOLOv8 Integration
- Extract 17 COCO keypoints from image/video

### 2. **recognition/** - ML Classification (MAIN)
- Classify pose using trained Neural Network
- Output: pose_name, confidence

### 3. **geometry/** - Minimal Geometric Analysis
- **calculate_angle()**: Tính góc giữa 3 điểm
- **calculate_distance()**: Khoảng cách Euclidean
- **Lightweight**: Chỉ 2 functions cơ bản

### 4. **evaluation/** - Pose Scoring
- Simple rule-based scoring per pose
- 1-2 angle checks per pose
- Output: score (0-100), feedback message

### 5. **visualization/** - Display
- Draw skeleton with score
- Color-coded feedback
- Support both image and video output

### 6. **config/** - Settings
- App configuration
- Visualization parameters

---

## 🎯 Implementation Priority

### Phase 1: Refactor Existing Code
1. Move `detection_keypoint.py` → `detection/pose_detector.py`
2. Move `classification_keypoint.py` → `recognition/ml_classifier.py`
3. Test basic pipeline

### Phase 2: Add Scoring
4. Implement `geometry/geometry_utils.py` ✅
5. Implement `evaluation/pose_evaluator.py` ✅
6. Test scoring accuracy

### Phase 3: Visualization
7. Implement `visualization/skeleton_drawer.py`
8. Implement `visualization/overlay_ui.py`

### Phase 4: Integration
9. Complete `main.py` (image + video support)
10. Update `app_gui.py`
11. Testing & demo

---

## 🔑 Design Decisions

### 1. Why Hybrid (ML + Geometric)?
- ✅ **ML for Recognition**: Fast, accurate classification
- ✅ **Geometric for Scoring**: Meaningful score based on angles
- ✅ **Best of both**: Speed + Interpretability

### 2. Why Minimal Geometric?
- ✅ Simple to implement (1-2 hours)
- ✅ Sufficient for project needs
- ✅ Provides real feedback (not fake scores)

### 3. Why Image + Video (no webcam)?
- ✅ Easier to demo and test
- ✅ Can save results for presentation
- ✅ Less complexity than real-time streaming

---

## 📝 Next Steps

1. ✅ Create minimal geometry module
2. ✅ Create evaluation module
3. **TODO**: Refactor detection code
4. **TODO**: Refactor recognition code
5. **TODO**: Implement visualization
6. **TODO**: Integrate in main.py

---

**Status**: Minimal geometric analysis added, ready for refactoring ✅
