import cv2
import glob
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Giữ nguyên các import logic của bạn
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification

# --- CẤU HÌNH UI ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
DISPLAY_SIZE = (500, 500) # Kích thước hiển thị ảnh trên App

# --- KHỞI TẠO MODEL (GIỮ NGUYÊN) ---
try:
    detection_keypoint = DetectKeypoint("yolov8m-pose.pt")
    classification_keypoint = KeypointClassification(
        "./models/pose_classification.pth"
    )
except Exception as e:
    print(f"Lỗi load model: {e}")
    # Có thể thêm thông báo lỗi GUI ở đây nếu cần

# --- HÀM XỬ LÝ LOGIC (CHUYỂN ĐỔI TỪ HÀM pose_classification CŨ) ---
def process_and_display(img_path):
    try:
        # 1. Xử lý ảnh đầu vào
        pil_image = Image.open(img_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Hiển thị ảnh gốc lên UI (bên trái)
        display_image_on_label(pil_image, label_img_original)

        # 2. Chạy Model Detection
        result = detection_keypoint(image_cv)
        keypoints = detection_keypoint.get_xy_keypoint(result)

        # 3. Kiểm tra kết quả
        if keypoints is None or result.boxes is None or len(result.boxes) == 0:
            lbl_status.config(text="Không phát hiện người!", fg="red")
            # Xóa ảnh kết quả cũ nếu có
            label_img_result.config(image='')
            return

        # 4. Chạy Model Classification
        input_classification = keypoints[10:]  # bỏ head (giữ nguyên logic)
        pose_label = classification_keypoint(input_classification)

        # 5. Vẽ Bounding Box và Text (Giữ nguyên logic vẽ)
        image_draw = result.plot(boxes=False)
        x_min, y_min, x_max, y_max = result.boxes.xyxy[0].cpu().numpy().astype(int)

        cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        label_text = pose_label.upper()
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(
            image_draw,
            (x_min, y_min - h - 8),
            (x_min + w + 6, y_min),
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            image_draw,
            label_text,
            (x_min + 3, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # 6. Hiển thị kết quả lên UI (bên phải)
        image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(image_draw)
        display_image_on_label(pil_result, label_img_result)

        # Cập nhật trạng thái text
        lbl_status.config(text=f"Pose Classification: {pose_label}", fg="green")

    except Exception as e:
        messagebox.showerror("Error", f"Có lỗi xảy ra: {str(e)}")

def display_image_on_label(pil_img, tk_label):
    """Hàm phụ trợ để resize và hiển thị ảnh lên Label Tkinter"""
    # Resize ảnh cho vừa khung hình hiển thị mà vẫn giữ tỷ lệ
    pil_img.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(pil_img)
    
    # Cập nhật label
    tk_label.config(image=tk_img)
    tk_label.image = tk_img # Giữ tham chiếu để không bị garbage collection xóa

def open_file_dialog():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        process_and_display(file_path)

# --- XÂY DỰNG GIAO DIỆN APP (TKINTER) ---
root = tk.Tk()
root.title("YOLOv8 Keypoint Yoga Pose Classification")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

# 1. Header
header_frame = tk.Frame(root, bg="#f0f2f6", pady=10)
header_frame.pack(fill="x")
tk.Label(header_frame, text="🧘 YOLOv8 Keypoint Yoga Pose Classification", 
         font=("Arial", 20, "bold"), bg="#f0f2f6").pack()
tk.Label(header_frame, text="Upload an image to classify basic yoga poses", 
         font=("Arial", 12), bg="#f0f2f6").pack()

# 2. Nút Upload
btn_frame = tk.Frame(root, pady=10)
btn_frame.pack()
tk.Button(btn_frame, text="📂 Chọn ảnh từ máy tính", command=open_file_dialog, 
          font=("Arial", 12), bg="#ff4b4b", fg="white", padx=20, pady=5).pack()

# 3. Khu vực hiển thị kết quả (Dùng Grid để chia 2 cột như st.columns)
content_frame = tk.Frame(root)
content_frame.pack(expand=True, fill="both", padx=20, pady=10)

# Cột 1: Ảnh gốc
frame_left = tk.Frame(content_frame, bd=2, relief="groove")
frame_left.pack(side="left", expand=True, fill="both", padx=10)
tk.Label(frame_left, text="Original Image", font=("Arial", 14, "bold")).pack(pady=5)
label_img_original = tk.Label(frame_left)
label_img_original.pack(expand=True)

# Cột 2: Ảnh kết quả
frame_right = tk.Frame(content_frame, bd=2, relief="groove")
frame_right.pack(side="right", expand=True, fill="both", padx=10)
tk.Label(frame_right, text="Keypoint Result 🔧", font=("Arial", 14, "bold")).pack(pady=5)
label_img_result = tk.Label(frame_right)
label_img_result.pack(expand=True)
lbl_status = tk.Label(frame_right, text="", font=("Arial", 14, "bold"))
lbl_status.pack(pady=10)

# 4. Khu vực Sample Images (Giữ nguyên logic glob)
sample_frame = tk.Frame(root, pady=10, height=150)
sample_frame.pack(fill="x", side="bottom")
tk.Label(sample_frame, text="Sample Images (Click to Run)", font=("Arial", 12, "bold")).pack(anchor="w", padx=20)

sample_container = tk.Frame(sample_frame)
sample_container.pack(padx=20, pady=5, anchor="w")

images = glob.glob("./images/*.jpeg")
# Giới hạn số lượng sample hiển thị để tránh tràn màn hình (ví dụ lấy 5 ảnh đầu)
for img_path in images[:6]: 
    try:
        # Tạo thumbnail nhỏ cho button
        img = Image.open(img_path)
        img.thumbnail((80, 80))
        photo = ImageTk.PhotoImage(img)
        
        # Tạo button hình ảnh
        btn = tk.Button(sample_container, image=photo, 
                        command=lambda p=img_path: process_and_display(p))
        btn.image = photo
        btn.pack(side="left", padx=5)
    except Exception as e:
        pass

# Chạy App
root.mainloop()