# 🧘 HỆ THỐNG NHẬN DẠNG & ĐÁNH GIÁ TƯ THẾ YOGA

> **Đồ án cuối kỳ môn Xử lý ảnh số**  
> Ứng dụng YOLOv8, Machine Learning và Geometric Analysis  
> **NHÓM 18 - HCMUTE**

---

## 👥 **THÔNG TIN NHÓM**

**Trường:** Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)  
**Môn:** Xử lý ảnh số  
**Nhóm:** 18

**Thành viên:**
1. **Mai Hồng Hải** - MSSV: 22133014
2. **Nguyễn Tấn Hùng** - MSSV: 22133027
3. **Nguyễn Ngọc Hiếu Hảo** - MSSV: 22133015

---

## 🎯 **MỤC TIÊU**

Xây dựng hệ thống:
1. **Nhận dạng** tư thế Yoga từ ảnh/video
2. **Chấm điểm** độ chính xác (0-100) với nhiều tiêu chí
3. **Feedback** chi tiết về tư thế (tiếng Việt)

**5 Tư thế:**
- 🧘 **Plank** - Chống đẩy (Straight-arm & Elbow variants)
- 🌳 **Tree** - Cái cây
- ⚔️ **Warrior II** - Chiến binh 2
- 👸 **Goddess** - Nữ thần
- 🐕 **Downward Dog** - Chó úp mặt

---

## ✨ **TÍNH NĂNG NỔI BẬT**

### **🚀 Performance**
- ✅ **GPU Acceleration** - YOLOv8 trên CUDA (NVIDIA RTX)
- ✅ **Async Video Processing** - Producer-Consumer architecture
- ✅ **Real-time FPS** - 30 FPS video playback mượt mà

### **🎯 Accuracy**
- ✅ **Hybrid AI + Geometry** - ML classifier với geometric override rules
- ✅ **Multi-Criteria Evaluation** - 2-3 tiêu chí mỗi tư thế
- ✅ **Y-Position Checks** - Phát hiện chính xác vị trí tay/chân
- ✅ **Wrist Distance Check** - Kiểm tra tay có chắp lại không
- ✅ **Relaxed Thresholds** - Chấp nhận variance của keypoint detection

### **💻 User Experience**
- ✅ **Desktop GUI** - Giao diện đẹp với CustomTkinter
- ✅ **Dark/Light Mode** - Chuyển đổi giao diện
- ✅ **Keyboard Shortcuts** - Space (pause), S (snapshot)
- ✅ **Vietnamese Feedback** - Phản hồi chi tiết bằng tiếng Việt
- ✅ **Save Results** - Lưu ảnh kết quả và snapshots

---

## 🏗️ **KIẾN TRÚC**

### **Hybrid: ML + Geometric Analysis**

```
┌──────────────┐
│ Image/Video  │
└──────┬───────┘
       │
┌──────▼──────┐
│   YOLOv8    │  🚀 GPU-Accelerated
│ 17 Keypoints│  Deep Learning
└──────┬──────┘
       │
       ├──────────────┬─────────────┐
       │              │             │
┌──────▼──────┐ ┌────▼───────┐    │
│ML Classifier│ │  Geometry  │    │
│  "Plank"    │ │  Override  │    │
│   90% conf  │ │  Fix errors│    │
└──────┬──────┘ └─────┬──────┘    │
       │              │            │
       └──────┬───────┘            │
              │                    │
       ┌──────▼────────┐           │
       │  Evaluator    │           │
       │ Multi-Criteria│           │
       │  Score: 85    │           │
       │  "Hip high⚠️" │           │
       └──────┬────────┘           │
              │                    │
       ┌──────▼────────────────────▼┐
       │    Visualization            │
       │ Skeleton + Score + Feedback │
       └─────────────────────────────┘
```

---

## 📦 **CẤU TRÚC**

```
src/
├── detection/          # 1️⃣ YOLOv8 Pose Detection (GPU)
│   ├── pose_detector.py
│   └── keypoint_constants.py
│
├── recognition/        # 2️⃣ Hybrid ML + Geometric
│   ├── ml_classifier.py      → Neural Network
│   └── pose_recognizer.py    → AI + Override rules
│
├── geometry/          # 3️⃣ Geometric Analysis
│   └── geometry_utils.py     → Angles, distances
│
├── evaluation/        # 4️⃣ Multi-Criteria Scoring
│   └── pose_evaluator.py     → Detailed scoring
│
├── visualization/     # 5️⃣ Visualization
│   ├── skeleton_drawer.py    → Color-coded skeleton
│   └── overlay_ui.py         → Score overlay
│
└── config/
    └── app_config.py

models/
└── pose_classification.pth   # Trained ML model

datasets/
└── yoga_pose_keypoint.csv    # 1000+ samples

docs/
└── pose_requirements.md      # Pose specs & thresholds
```

---

## 🛠️ **CÔNG NGHỆ**

| Component | Technology | Details |
|-----------|-----------|---------|
| **Pose Detection** | YOLOv8 Pose | GPU-accelerated (CUDA) |
| **Classification** | PyTorch Neural Network | 90%+ accuracy |
| **Geometric Analysis** | NumPy | Angle & distance calculations |
| **Scoring** | Multi-criteria rules | 2-3 checks per pose |
| **Visualization** | OpenCV + PIL | Color-coded skeleton |
| **GUI** | CustomTkinter | Modern desktop UI |
| **Video Processing** | Threading + Queue | Async producer-consumer |
| **Language** | Python 3.12+ | Type hints, modern syntax |

---

## 🔬 **PHƯƠNG PHÁP**

### **1. YOLOv8 Pose Detection (GPU)**
- Model: `yolov8m-pose.pt`
- Output: 17 COCO keypoints
- **GPU**: 10x faster than CPU
- Normalized & absolute coordinates

### **2. Hybrid ML + Geometric**
```python
# ML Classification (Primary)
pose, conf = neural_network.predict(keypoints)

# Geometric Overrides (Fix ML errors)
if pose == 'Goddess' and shoulder_y > hip_y:
    pose = 'Unknown'  # Bent over detection
    
if pose == 'Downdog' and hip_angle > 140°:
    pose = 'Plank'  # Fix common confusion
```

### **3. Multi-Criteria Scoring**
**Ví dụ: Plank**
- **Body angle** (60%):
  - 160-175°: Score 100 ✅
  - 150-160°: Score 85 ⚠️
  - <150°: Score 70 ❌
- **Arms** (40%):
  - Supporting + straight: Score 100 ✅
  - Supporting + bent: Score 75 ⚠️
  - Not supporting: Score 30 ❌

### **4. Advanced Checks**
- **Y-Position**: Wrist phải CAO HƠN shoulder (Goddess)
- **Distance**: 2 wrists phải GẦN NHAU (Tree)
- **Supporting**: Elbow/Wrist phải thấp hơn shoulder (Plank)

---

## 🚀 **SỬ DỤNG**

### **Cài đặt**
```bash
# Clone repository
git clone https://github.com/your-repo/human-action-detector.git
cd human-action-detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Chạy GUI (Khuyến nghị)**
```bash
python app_ui.py
```

**Tính năng GUI:**
- 📷 **Image Mode** - Phân tích ảnh
- 🎥 **Video Mode** - Phân tích video (async processing)
- ⏸️ **Pause/Resume** - Space bar
- 📸 **Snapshot** - S key (lưu frame khi video)
- 💾 **Save Results** - Lưu ảnh kết quả
- 🌓 **Dark/Light Mode** - Chuyển đổi theme

### **Chạy CLI (Optional)**
```bash
# Xử lý ảnh
python src/main.py --input images/plank.jpg

# Xử lý video
python src/main.py --input videos/yoga.mp4 --output results/yoga_result.mp4
```

---

## 📊 **KẾT QUẢ**

### **Output Example:**
```
Pose: Plank
Score: 85/100
Feedback: "Tốt! Cần điều chỉnh: hông hơi cao ⚠️"
Details:
  - Body angle: 165° (target: 160-175°)
  - Arms: Straight & supporting ✅
```

### **Accuracy:**
- ✅ **ML Classification**: 90%+ accuracy
- ✅ **Geometric Overrides**: Fixed 15%+ systematic errors
- ✅ **Multi-Criteria Scoring**: Realistic 0-100 scores
- ✅ **Vietnamese Feedback**: Actionable suggestions

---

## 🎓 **ĐIỂM NỔI BẬT**

### **Technical Excellence:**
✅ **GPU Acceleration** - YOLOv8 CUDA  
✅ **Hybrid Architecture** - ML + Geometric rules  
✅ **Async Processing** - Smooth 30 FPS video  
✅ **Y-Position Checks** - Advanced geometric analysis  
✅ **Multi-Criteria Evaluation** - Weighted scoring  

### **Academic Value:**
✅ **Deep Learning Application** - YOLOv8 Pose  
✅ **Computer Vision** - Keypoint detection  
✅ **Geometric Calculations** - Angle & distance analysis  
✅ **Real-world Problem** - Yoga pose correction  
✅ **Production Quality** - GUI, error handling, async  

### **Innovation:**
✅ **Hybrid AI + Geometry** - Best of both worlds  
✅ **Geometric Overrides** - Fix ML without retraining  
✅ **Vietnamese Feedback** - Localized, actionable  
✅ **Multi-Variant Support** - Plank (straight-arm & elbow)  

---

## 📚 **DATASET**

- **Source**: Yoga Pose Classification (Kaggle)
- **Size**: 1000+ labeled samples
- **Format**: CSV with normalized keypoint coordinates
- **Classes**: 5 poses (balanced distribution)
- **Augmentation**: Applied during training

---

## � **YÊU CẦU HỆ THỐNG**

### **Minimum:**
- Python 3.8+
- 4GB RAM
- CPU: Multi-core

### **Recommended:**
- Python 3.12+
- 8GB+ RAM
- **GPU: NVIDIA with CUDA** (10x faster!)
- SSD storage

---

## 📖 **TÀI LIỆU THAM KHẢO**

1. **YOLOv8 Pose Documentation** - Ultralytics
2. **COCO Keypoints Specification** - COCO Dataset
3. **Yoga Pose Analysis** - Various yoga resources
4. **CustomTkinter** - Modern GUI framework
5. **PyTorch** - Deep learning framework

---

## 📝 **DOCUMENTATION**

- `README.md` - This file (overview, setup, usage)
- `PROJECT_STRUCTURE.md` - Architecture & design decisions
- `docs/pose_requirements.md` - Pose specifications & thresholds

---

## 🙏 **CREDITS**

**Developed by:** NHÓM 18 - HCMUTE  
**Course:** Xử lý ảnh số  
**Instructor:** [Tên Giảng Viên]  
**Institution:** Đại học Sư phạm Kỹ thuật TP.HCM

**Technologies:**
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- CustomTkinter
- NumPy

---

**🚀 Made with Python, Deep Learning & Geometric Analysis**

**Last Updated:** 2025-12-21 - Production-ready with advanced features
