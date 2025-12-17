import cv2
import glob
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# --- IMPORT CŨ ---
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification
# --- IMPORT MỚI ---
from src.pose_corrector import PoseCorrector # <--- Thêm dòng này

# --- CONFIG ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
DISPLAY_SIZE = (500, 500)

# --- INIT MODELS ---
try:
    detection_keypoint = DetectKeypoint("yolov8m-pose.pt")
    # Lưu ý: Sửa lại đường dẫn model classification cho đúng với file của bạn (.pt hay .pth)
    classification_keypoint = KeypointClassification("./models/pose_classification.pth") 
    pose_corrector = PoseCorrector() # <--- Khởi tạo Corrector
except Exception as e:
    print(f"Lỗi load model: {e}")

def process_and_display(img_path):
    try:
        # 1. Load ảnh
        pil_image = Image.open(img_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        display_image_on_label(pil_image, label_img_original)

        # 2. Detect
        result = detection_keypoint(image_cv)
        # Lưu ý: detection_keypoint trả về list 34 giá trị (x,y normalized)
        keypoints = detection_keypoint.get_xy_keypoint(result) 

        if keypoints is None:
            lbl_status.config(text="Không phát hiện người!", fg="red")
            label_img_result.config(image='')
            lbl_feedback.config(text="") # Xóa feedback cũ
            return

        # 3. Classify
        # Keypoints gồm 17 điểm * 2 (x,y). Index 10 trở đi là từ vai trái (bỏ mắt, mũi, tai)
        input_classification = keypoints[10:] 
        pose_label = classification_keypoint(input_classification)

        # 4. Score & Correct (MỚI)
        # Truyền toàn bộ keypoints (bao gồm cả thân dưới) để tính góc
        score, feedbacks = pose_corrector.evaluate(pose_label, keypoints)

        # 5. Vẽ kết quả
        # Vẽ khung xương gốc từ YOLO
        image_draw = result.plot(boxes=False) 
        
        # Lấy tọa độ bounding box để vẽ text
        if result.boxes and len(result.boxes) > 0:
            x_min, y_min, x_max, y_max = result.boxes.xyxy[0].cpu().numpy().astype(int)
        else:
            h, w, _ = image_draw.shape
            x_min, y_min = 10, 10

        # Vẽ Label Pose
        text_pose = f"{pose_label} | Score: {score}/100"
        color = (0, 255, 0) if score > 80 else (0, 0, 255) # Xanh nếu tốt, Đỏ nếu tệ
        
        cv2.putText(image_draw, text_pose, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 6. Hiển thị UI
        image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(image_draw)
        display_image_on_label(pil_result, label_img_result)

        # Cập nhật Text Status & Feedback
        status_text = f"Pose: {pose_label} - Score: {score}"
        lbl_status.config(text=status_text, fg="blue")
        
        # Hiển thị danh sách lỗi
        feedback_str = "Feedback:\n" + "\n".join([f"- {fb}" for fb in feedbacks])
        lbl_feedback.config(text=feedback_str, fg="red" if score < 100 else "green")

    except Exception as e:
        messagebox.showerror("Error", f"Có lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

def display_image_on_label(pil_img, tk_label):
    pil_img.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(pil_img)
    tk_label.config(image=tk_img)
    tk_label.image = tk_img

def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_and_display(file_path)

# --- GUI SETUP ---
root = tk.Tk()
root.title("AI Yoga Trainer")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

# Header
tk.Label(root, text="🧘 AI Yoga Pose Correction", font=("Arial", 20, "bold")).pack(pady=10)
tk.Button(root, text="📂 Upload Image", command=open_file_dialog, bg="#4CAF50", fg="white", font=("Arial", 12)).pack()

# Content Area
content = tk.Frame(root)
content.pack(expand=True, fill="both", padx=10, pady=10)

# Left: Original
frame_l = tk.Frame(content)
frame_l.pack(side="left", expand=True)
tk.Label(frame_l, text="Original", font=("Arial", 12)).pack()
label_img_original = tk.Label(frame_l)
label_img_original.pack()

# Right: Result
frame_r = tk.Frame(content)
frame_r.pack(side="right", expand=True)
tk.Label(frame_r, text="Result Analysis", font=("Arial", 12)).pack()
label_img_result = tk.Label(frame_r)
label_img_result.pack()

# Info Area (Dưới ảnh kết quả)
lbl_status = tk.Label(frame_r, text="Ready", font=("Arial", 14, "bold"))
lbl_status.pack(pady=5)

lbl_feedback = tk.Label(frame_r, text="", font=("Arial", 12), justify="left")
lbl_feedback.pack(pady=5)

# Sample Images Footer
footer = tk.Frame(root, height=100)
footer.pack(side="bottom", fill="x", pady=10)
tk.Label(footer, text="Samples:").pack(anchor="w", padx=10)

# Load samples logic (giữ nguyên)
images = glob.glob("./images/*.jpeg")[:6]
for img_path in images:
    try:
        img = Image.open(img_path)
        img.thumbnail((60, 60))
        photo = ImageTk.PhotoImage(img)
        btn = tk.Button(footer, image=photo, command=lambda p=img_path: process_and_display(p))
        btn.image = photo
        btn.pack(side="left", padx=5)
    except: pass

root.mainloop()