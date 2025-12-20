# Yêu Cầu Góc Độ Cho 5 Tư Thế Yoga

Tài liệu này mô tả các góc chuẩn cần kiểm tra cho từng tư thế yoga trong hệ thống.

---

## 1. Tree Pose (Tư thế Cái Cây)

**Đặc điểm:** Đứng trên một chân, bàn chân kia đặt lên đùi trong (hoặc bắp chân) của chân trụ, hai tay chắp trước ngực (hoặc vươn cao).

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Chân đứng** | Hông - Gối - Mắt cá | ≥ 165° | Chân trụ phải thẳng |
| **Chân gập** | Hông - Gối - Mắt cá | < 120° | Chân kia gập vào đùi/bắp |
| **Tay** | Vai - Khuỷu - Cổ tay | < 100° | Tay chắp ngực hoặc giơ cao |

---

## 2. Plank Pose (Tư thế Chống Đẩy)

**Đặc điểm:** Cơ thể tạo thành một đường thẳng từ đầu đến gót chân, chống đỡ bằng khuỷu tay và ngón chân, giữ lưng thẳng.

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Thân** | Vai - Hông - Gối | ≥ 170° | Thân thẳng tắp |
| **Tay chống** | Vai - Khuỷu - Cổ tay | ≥ 160° | Tay duỗi thẳng |

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
| **Tay giơ vuông góc** | Vai - Khuỷu - Cổ tay | ≥ 140° | Tay duỗi thẳng, giơ lên |

---

## 5. Downdog Pose (Chó Úp Mặt)

**Đặc điểm:** Người tập tạo thành hình dạng chữ V ngược với hai tay và hai chân chống xuống sàn, hông nâng cao và đẩy ra sau. Đây là một tư thế nghỉ ngơi tích cực, kéo giãn toàn bộ cơ thể.

### Góc cần kiểm tra:

| Bộ phận | Keypoints | Góc yêu cầu | Ghi chú |
|---------|-----------|-------------|---------|
| **Chân thẳng** | Hông - Gối - Mắt cá | ≥ 165° | Cả 2 chân duỗi thẳng |
| **Tay thẳng** | Vai - Khuỷu - Cổ tay | ≥ 160° | Cả 2 tay duỗi thẳng |
| **Hông gập (chữ V)** | Vai - Hông - Gối | 70-110° | Hông nâng cao tạo chữ V |

---

## Ghi Chú Chung

- **Keypoints** sử dụng chuẩn COCO 17 keypoints từ YOLOv8 Pose
- **Góc** được tính bằng công thức: `arccos((BA · BC) / (|BA| × |BC|))`
- **Threshold** đã được điều chỉnh để phù hợp với thực tế đo góc từ keypoints
- Một số góc có thể khác với góc sinh lý học thực tế do phương pháp tính toán

---

*Tài liệu này được sử dụng làm cơ sở cho module `pose_evaluator.py` trong hệ thống.*
