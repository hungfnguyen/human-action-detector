# 🧘 HỆ THỐNG NHẬN DẠNG & ĐÁNH GIÁ TƯ THẾ YOGA

> **Đồ án cuối kỳ môn Xử lý ảnh số**  
> Ứng dụng YOLOv8, Machine Learning và Geometric Analysis

---

## 🎯 **MỤC TIÊU**

Xây dựng hệ thống:
1. **Nhận dạng** tư thế Yoga từ ảnh/video
2. **Chấm điểm** độ chính xác (0-100)
3. **Feedback** chi tiết về tư thế

**5 Tư thế:**
- 🧘 **Plank** - Chống đẩy
- 🌳 **Tree** - Cái cây
- ⚔️ **Warrior II** - Chiến binh 2
- 👸 **Goddess** - Nữ thần
- 🐕 **Downward Dog** - Chó úp mặt

---

## 🏗️ **KIẾN TRÚC**

### **Hybrid: ML + Geometric Analysis**

```
┌──────────────┐
│ Image/Video  │
└──────┬───────┘
       │
┌──────▼──────┐
│   YOLOv8    │  Deep Learning
│ 17 Keypoints│  (Pose Detection)
└──────┬──────┘
       │
       ├──────────────┬─────────────┐
       │              │             │
┌──────▼──────┐ ┌────▼───────┐    │
│ML Classifier│ │  Geometry  │    │
│  "Plank"    │ │Calc angles │    │
└──────┬──────┘ └─────┬──────┘    │
       │              │            │
       └──────┬───────┘            │
              │                    │
       ┌──────▼────────┐           │
       │  Evaluator    │           │
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
├── detection/          # 1️⃣ YOLOv8 Pose Detection
│   ├── pose_detector.py
│   └── keypoint_constants.py
│
├── recognition/        # 2️⃣ ML Classification
│   ├── ml_classifier.py      → Neural Network (CORE)
│   └── pose_recognizer.py
│
├── geometry/          # 3️⃣ Geometric Analysis (MINIMAL)
│   └── geometry_utils.py     → calculate_angle(), distance
│
├── evaluation/        # 4️⃣ Pose Scoring
│   └── pose_evaluator.py     → Score 0-100 + feedback
│
├── visualization/     # 5️⃣ Visualization
│   ├── skeleton_drawer.py
│   └── overlay_ui.py
│
├── config/
│   └── app_config.py
│
└── main.py           # 🚀 Main CLI

models/
└── pose_classification.pth   # Trained ML model

datasets/
└── yoga_pose_keypoint.csv    # 1000+ samples
```

---

## 🛠️ **CÔNG NGHỆ**

| Component | Technology |
|-----------|-----------|
| **Pose Detection** | YOLOv8 Pose |
| **Classification** | PyTorch Neural Network |
| **Geometric Analysis** | NumPy (angle calculation) |
| **Scoring** | Rule-based evaluation |
| **Visualization** | OpenCV |
| **Language** | Python 3.8+ |

---

## 🔬 **PHƯƠNG PHÁP**

### **1. YOLOv8 Pose Detection**
- Model: `yolov8m-pose.pt`
- Output: 17 COCO keypoints

### **2. ML Classification**
- Neural Network: 24 inputs → 256 hidden → 5 outputs
- Accuracy: ~90%+
- Fast: <10ms inference

### **3. Geometric Analysis (MINIMAL)**
```python
# Chỉ implement functions cơ bản
angle = GeometryUtils.calculate_angle(p1, p2, p3)
distance = GeometryUtils.calculate_distance(p1, p2)
```

### **4. Pose Scoring**
**Ví dụ: Plank**
- Angle Vai-Hông-Gối:
  - \>170°: Score 95 "Excellent ✅"
  - 160-170°: Score 80 "Hip slightly high ⚠️"
  - <160°: Score 65 "Hip too high ❌"

---

## 🚀 **SỬ DỤNG**

### **Cài đặt**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **Chạy**

**CLI - Xử lý ảnh:**
```bash
python src/main.py --input image.jpg
```

**CLI - Xử lý video:**
```bash
python src/main.py --input video.mp4 --output result.mp4
```

**GUI:**
```bash
python app_gui.py
```

---

## 📊 **KẾT QUẢ**

### **Output:**
```
Pose: Plank
Score: 85/100
Feedback: "Good! Hip slightly high ⚠️"
Angle: 165°
```

### **Chức năng:**
✅ Nhận dạng 5 tư thế  
✅ Chấm điểm 0-100  
✅ Feedback chi tiết  
✅ Hỗ trợ ảnh + video  
✅ GUI desktop  

---

## 🎓 **ĐIỂM NỔI BẬT**

### **Technical:**
✅ YOLOv8 Pose (SOTA)  
✅ Neural Network Classification  
✅ Geometric Analysis (angles)  
✅ Hybrid Architecture  

### **Academic:**
✅ Deep Learning application  
✅ Computer Vision  
✅ Geometric calculations  
✅ Real-world problem solving  

---

## 📚 **DATASET**

- **Source**: Yoga Pose Classification (Kaggle)
- **Size**: 1000+ labeled samples
- **Format**: CSV with keypoint coordinates
- **Classes**: 5 poses (balanced)

---

## 👨‍💻 **THÔNG TIN**

**Môn:** Xử lý ảnh số  
**Đề tài:** Nhận dạng & đánh giá tư thế Yoga  
**Trường:** Đại học Sư phạm Kỹ thuật TP.HCM  

---

**🚀 Made with Python, Deep Learning & Geometric Analysis**
