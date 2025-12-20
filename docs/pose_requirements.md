# Yêu Cầu Góc Độ Cho 5 Tư Thế Yoga

Tài liệu này mô tả các góc chuẩn và logic đánh giá cho từng tư thế yoga trong hệ thống.

**Version:** 2.0 - Updated with Hybrid AI + Geometry approach

---

## 1. Tree Pose (Tư thế Cái Cây)

**Đặc điểm:** Đứng trên một chân, bàn chân kia đặt lên đùi trong (hoặc bắp chân) của chân trụ, hai tay chắp trước ngực (hoặc vươn cao).

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Chân đứng** | Hông - Gối - Mắt cá | ≥ 155° | Chân trụ phải thẳng (relaxed from 165° due to ankle detection variance) |
| **Chân gập** | Hông - Gối - Mắt cá | < 120° | Chân kia gập vào đùi/bắp |
| **Tay chắp** | Khoảng cách 2 cổ tay | < 50% shoulder width | **Tay phải chắp lại** (distance check) |
| **Tay (elbow)** | Vai - Khuỷu - Cổ tay | < 130° | Tay gập (nếu chắp ngực) |

**Improvements:**
- ✅ **Relaxed standing leg threshold:** 165° → 155° để chấp nhận ankle keypoint variance
- ✅ **Wrist distance check:** Phát hiện tay tách ra (không chắp) bằng khoảng cách 2 cổ tay

---

## 2. Plank Pose (Tư thế Chống Đẩy)

**Đặc điểm:** Cơ thể tạo thành một đường thẳng từ đầu đến gót chân, chống đỡ bằng tay hoặc khuỷu tay.

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Thân** | Vai - Hông - Gối | **160-175°** | Plank CHUẨN (not 170-180°!) |
| **Tay chống** | Vai - Khuỷu - Cổ tay | ≥ 160° hoặc Elbow Plank | Tay duỗi thẳng HOẶC elbow plank |
| **Supporting check** | Elbow/Wrist Y vs Shoulder Y | Must support | Phân biệt plank vs nằm sấp |

**Improvements:**
- ✅ **Body angle CHUẨN: 160-175°** (không phải 170-180°) - hông hơi cao hơn nằm sấp
- ✅ **Support both variants:** Straight-arm plank VÀ elbow plank
- ✅ **Supporting check:** Elbow/Wrist phải thấp hơn shoulder (Y position) để phân biệt với nằm sấp

---

## 3. Warrior2 Pose (Chiến Binh 2)

**Đặc điểm:** Chân bước rộng, một chân gập gối vuông góc, hai tay dang ngang bằng vai, mắt nhìn theo tay trước.

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Chân trước (gập)** | Hông - Gối - Mắt cá | 80-125° | Gối gập ~90° |
| **Chân sau (thẳng)** | Hông - Gối - Mắt cá | ≥ 155° | Chân sau duỗi thẳng |
| **Tay dang ngang** | Vai phải - Vai trái - Khuỷu | ≥ 150° | Hai tay dang thẳng ngang |

---

## 4. Goddess Pose (Tư thế Nữ Thần)

**Đặc điểm:** Hai chân mở rộng, đầu gối gập sâu và hướng ra ngoài (giống tư thế squat), hai tay giơ lên cao vuông góc (dạng xương rồng).

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Chân squat** | Hông - Gối - Mắt cá | 70-140° | Cả 2 chân gập sâu ~90° |
| **Tay giơ cao** | **Wrist Y vs Shoulder Y** | **Wrist_Y < Shoulder_Y - 20px** | **CHECK Y POSITION** (not elbow angle!) |
| **Tay thẳng** | Vai - Khuỷu - Cổ tay | ≥ 140° | Tay duỗi thẳng khi giơ |

**Improvements:**
- ✅ **CHECK Y POSITION thay vì elbow angle:** Wrist phải CAO HƠN shoulder (Y nhỏ hơn) để đảm bảo tay giơ lên
- ✅ **Bent-over detection:** Reject nếu vai thấp hơn hông >15% (người cúi xuống)
- ✅ **Relaxed thresholds:** Wrist lift > 20px (instead of 50px) for realistic poses

---

## 5. Downdog Pose (Chó Úp Mặt)

**Đặc điểm:** Người tập tạo thành hình dạng chữ V ngược với hai tay và hai chân chống xuống sàn, hông nâng cao và đẩy ra sau.

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Chân thẳng** | Hông - Gối - Mắt cá | ≥ 165° | Cả 2 chân duỗi thẳng |
| **Tay thẳng** | Vai - Khuỷu - Cổ tay | ≥ 160° | Cả 2 tay duỗi thẳng |
| **Hông gập (chữ V)** | Vai - Hông - Gối | 70-110° | Hông nâng cao tạo chữ V |

---

## 🔧 Geometric Overrides (Hybrid AI + Geometry)

Hệ thống sử dụng **Hybrid Approach** để fix lỗi ML classifier:

### 1. Goddess Bent-Over Detection
```python
if AI_says_Goddess:
    if avg_shoulder_y > avg_hip_y + 15% body_height:
        return 'Unknown'  # Người đang cúi!
    if avg_wrist_y >= avg_hip_y:
        return 'Unknown'  # Tay không giơ lên!
```

### 2. Downdog vs Plank (Hip Angle)
```python
hip_angle = angle(shoulder, hip, knee)

if AI_says_Downdog and hip_angle > 140°:
    return 'Plank'  # Thân ngang → Plank!
    
if AI_says_Plank and hip_angle < 120°:
    return 'Downdog'  # Hông cao → Downdog!
```

### 3. Tree vs Goddess vs Warrior2
```python
if AI_says_Goddess and one_leg_straight:
    if ankle_distance < 2.5 × hip_width:
        return 'Tree'  # Chân khép
    else:
        return 'Warrior2'  # Chân rộng

if AI_says_Tree and both_legs_bent:
    return 'Goddess'
```

---

## Ghi Chú Chung

- **Keypoints** sử dụng chuẩn COCO 17 keypoints từ YOLOv8 Pose
- **Góc** được tính bằng công thức: `arccos((BA · BC) / (|BA| × |BC|))`
- **Threshold** đã được điều chỉnh để phù hợp với:
  - Ankle keypoint detection variance (±5-10°)
  - Realistic pose variations
  - Different camera angles
- **Y-axis coordinate:** Smaller Y = Higher position (screen coordinate system)
- **Hybrid Approach:** ML Classification + Geometric Rules = Higher accuracy

---

## 📊 Scoring Weights

| Pose | Criteria 1 | Criteria 2 | Criteria 3 |
|------|-----------|-----------|-----------|
| **Plank** | Body: 60% | Arms: 40% | - |
| **Tree** | Standing leg: 33% | Bent leg: 33% | Arms: 33% |
| **Warrior2** | Front leg: 40% | Back leg: 30% | Arms: 30% |
| **Goddess** | Legs: 70% | Arms: 30% | - |
| **Downdog** | Legs: 40% | Arms: 30% | Hip: 30% |

---

*Tài liệu này được sử dụng làm cơ sở cho module `pose_evaluator.py` và `pose_recognizer.py` trong hệ thống.*

**Last Updated:** 2025-12-21 - Improved with Y-position checks, relaxed thresholds, and geometric overrides
