# 📁 Project Structure - Yoga Pose Recognition

## 🏗️ Architecture: Hybrid (ML Recognition + Geometric Overrides + Multi-Criteria Scoring)

```
human-action-detector/
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── logo.png                          # HCMUTE logo
│
├── 📂 src/                          # Source code
│   ├── __init__.py
│   ├── main.py                      # CLI entry point (Image + Video)
│   │
│   ├── 📂 detection/                # YOLOv8 Pose Detection
│   │   ├── __init__.py
│   │   ├── pose_detector.py         # YOLOv8 wrapper (GPU-accelerated)
│   │   └── keypoint_constants.py    # COCO keypoints
│   │
│   ├── 📂 recognition/              # ML Classification + Geometric Overrides
│   │   ├── __init__.py
│   │   ├── ml_classifier.py         # Neural Network (CORE)
│   │   └── pose_recognizer.py       # Hybrid: AI + Geometry rules
│   │
│   ├── 📂 geometry/                 # Geometric Analysis
│   │   ├── __init__.py
│   │   └── geometry_utils.py        # calculate_angle(), distance
│   │
│   ├── 📂 evaluation/               # Multi-Criteria Pose Scoring
│   │   ├── __init__.py
│   │   └── pose_evaluator.py        # Detailed scoring (0-100) + feedback
│   │
│   ├── 📂 visualization/            # Drawing & UI
│   │   ├── __init__.py
│   │   ├── skeleton_drawer.py       # Skeleton drawing
│   │   └── overlay_ui.py            # Score overlay + feedback
│   │
│   └── 📂 config/                   # Configuration
│       ├── __init__.py
│       └── app_config.py            # Settings
│
├── 📂 models/                       # Trained models
│   └── pose_classification.pth      # ML weights (PyTorch)
│
├── 📂 datasets/                     # Training data
│   └── yoga_pose_keypoint.csv       # 1000+ samples
│
├── 📂 docs/                         # Documentation
│   └── pose_requirements.md         # Pose specifications & thresholds
│
├── 📂 images/                       # Sample images for testing
├── 📂 videos/                       # Sample videos for testing
├── 📂 results/                      # Output results
└── 📂 snapshots/                    # Video frame snapshots

└── app_ui.py                        # GUI Application (CustomTkinter)
```

---

## 🔄 Data Flow

```
┌─────────────────┐
│  Image / Video  │
└────────┬────────┘
         │
  ┌──────▼──────┐
  │   YOLOv8    │ (GPU-accelerated Pose Detection)
  │ 17 Keypoints│
  └──────┬──────┘
         │
         ├─────────────┬──────────────┐
         │             │              │
  ┌──────▼──────┐ ┌───▼────────┐    │
  │ML Classifier│ │  Geometry  │    │
  │  "Plank"    │ │ Override   │    │ 
  └──────┬──────┘ │ Check      │    │
         │        └─────┬──────┘    │
         └──────┬───────┘            │
                │                    │
         ┌──────▼────────┐           │
         │  Evaluator    │           │
         │  Multi-Crit   │           │
         │  Score: 85    │           │
         │  "Hip high⚠️" │           │
         └──────┬────────┘           │
                │                    │
         ┌──────▼────────────────────▼┐
         │      Visualization          │
         │  Skeleton + Score + Feedback│
         └─────────────────────────────┘
```

---

## 📋 Module Responsibilities

### 1. **detection/** - YOLOv8 Integration ✅
- Extract 17 COCO keypoints from image/video
- **GPU-accelerated** for maximum performance
- Returns normalized & absolute keypoints

### 2. **recognition/** - Hybrid ML + Geometric ✅
- **ML Classification**: Trained Neural Network (primary)
- **Geometric Overrides**: Fix common ML errors
  - Goddess bent-over detection
  - Downdog vs Plank (hip angle check)
  - Tree vs Goddess vs Warrior2 (leg position)
- Output: pose_name, confidence

### 3. **geometry/** - Geometric Analysis ✅
- **calculate_angle()**: Góc giữa 3 điểm
- **calculate_distance()**: Khoảng cách Euclidean
- Used in both recognition and evaluation

### 4. **evaluation/** - Multi-Criteria Scoring ✅
- **Pose-specific evaluation** với 2-3 tiêu chí mỗi pose
- **Weighted scoring** (e.g., Plank: 60% body + 40% arms)
- **Y-position checks** for Goddess arms (not just elbow angle)
- **Wrist distance check** for Tree arms (clasped hands)
- **Supporting check** for Plank (distinguish from lying flat)
- **Relaxed thresholds** to account for keypoint variance
- Output: score (0-100), Vietnamese feedback message

### 5. **visualization/** - Display ✅
- Draw skeleton with color-coded joints
- Score overlay with pose name
- Vietnamese feedback messages
- Support both image and video output

### 6. **config/** - Settings ✅
- App configuration
- Visualization parameters
- Processing constants

---

## 🎯 Implementation Status

### ✅ **Phase 1: Core Infrastructure** - COMPLETE
- [x] YOLOv8 Pose Detection (GPU-accelerated)
- [x] ML Classifier (Neural Network)
- [x] Basic pipeline (image + video)

### ✅ **Phase 2: Geometric Analysis** - COMPLETE
- [x] Geometry utilities (angles, distances)
- [x] Multi-criteria evaluation per pose
- [x] Weighted scoring system

### ✅ **Phase 3: Visualization** - COMPLETE
- [x] Skeleton drawer with color coding
- [x] Score overlay UI
- [x] Vietnamese feedback messages

### ✅ **Phase 4: Advanced Features** - COMPLETE
- [x] **Hybrid AI + Geometry:** Geometric overrides to fix ML errors
- [x] **Y-position checks:** For Goddess arm evaluation
- [x] **Wrist distance check:** For Tree clasped hands
- [x] **Plank variants:** Support both straight-arm and elbow plank
- [x] **Relaxed thresholds:** Account for ankle keypoint variance
- [x] **Async video processing:** Producer-Consumer architecture
- [x] **GUI Application:** CustomTkinter desktop app

### ✅ **Phase 5: Polish & Testing** - COMPLETE
- [x] Comprehensive testing with sample images/videos
- [x] Threshold fine-tuning
- [x] Documentation updates
- [x] Code cleanup

---

## 🔑 Design Decisions

### 1. Why Hybrid (ML + Geometric)?
- ✅ **ML for Recognition**: Fast, accurate classification (90%+)
- ✅ **Geometric for Overrides**: Fix systematic ML errors
- ✅ **Geometric for Scoring**: Meaningful scores based on actual angles
- ✅ **Best of both**: Speed + Accuracy + Interpretability

### 2. Why Multi-Criteria Evaluation?
- ✅ **Realistic scoring**: Multiple checks per pose (not just 1 angle)
- ✅ **Weighted importance**: Critical aspects weighted higher
- ✅ **Actionable feedback**: Specific issues identified

### 3. Why Geometric Overrides?
- ✅ **Fix systematic errors**: ML often confuses similar poses
- ✅ **Logic-based**: Use physical constraints (e.g., hip angle)
- ✅ **No retraining needed**: Quick fixes without ML overhead

### 4. Why Image + Video (no webcam)?
- ✅ **Easier to demo and test**
- ✅ **Save results for presentation**
- ✅ **Async processing** for smooth video playback

### 5. Why GPU Acceleration?
- ✅ **10x faster inference** on YOLOv8
- ✅ **Real-time video processing**
- ✅ **Better user experience**

---

## 🚀 Advanced Features

### 1. **Geometric Override Examples**

```python
# Goddess Bent-Over Detection
if AI_says_Goddess:
    if avg_shoulder_y > avg_hip_y + 15% body_height:
        return 'Unknown'  # Person is bent over!

# Downdog vs Plank (Hip Angle)
hip_angle = calculate_angle(shoulder, hip, knee)
if AI_says_Downdog and hip_angle > 140°:
    return 'Plank'  # Body is straight, not V-shaped!
```

### 2. **Y-Position Checks (Goddess Arms)**

```python
# OLD: Check elbow angle (WRONG - allows lowered arms!)
if elbow_angle >= 140°:
    score = 100

# NEW: Check Y position (RIGHT - arms must be raised!)
wrist_lift = shoulder_y - wrist_y  # Positive = raised
if wrist_lift > 20 and elbow >= 140°:
    score = 100
```

### 3. **Async Video Processing**

```
[Background Thread]          [UI Thread]
     Producer        Queue     Consumer
        │              │           │
    ┌───▼───┐      ┌──▼──┐    ┌───▼───┐
    │ Read  │─────▶│ 5   │───▶│Display│
    │Process│      │Frame│    │@ 30fps│
    │ AI    │      │Buffer    │       │
    └───────┘      └─────┘    └───────┘
                   (Non-blocking)
```

---

## 📐 Key Thresholds & Improvements

| Pose | Improvement | Before | After |
|------|------------|--------|-------|
| **Tree** | Standing leg | 165° | 155° (relaxed) |
| **Tree** | Arms clasped | elbow angle | wrist distance < 50% shoulder |
| **Plank** | Body angle | 170-180° | **160-175°** (correct range!) |
| **Plank** | Variants | straight-arm only | straight-arm OR elbow |
| **Plank** | Lying flat | no check | Y-position check |
| **Goddess** | Arms raised | elbow angle | **Y-position** (wrist > shoulder) |
| **Goddess** | Bent over | no check | shoulder vs hip check |
| **Downdog vs Plank** | Confusion | no fix | hip angle override |

---

## 📝 Documentation

- **README.md** - Project overview, setup, usage
- **PROJECT_STRUCTURE.md** - This file (architecture)
- **docs/pose_requirements.md** - Detailed pose specifications, thresholds, geometric overrides

---

**Status**: ✅ **COMPLETE** - Production-ready with advanced features

**Last Updated**: 2025-12-21 - All phases complete, hybrid architecture implemented
