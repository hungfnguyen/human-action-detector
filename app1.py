import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import os
import glob
import threading
import random
import time
from datetime import datetime

# --- CẤU HÌNH GIAO DIỆN ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

APP_NAME = "Yoga Pose AI Pro - Video Analytics"
WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768

# --- KHU VỰC IMPORT MODEL ---
try:
    from src.detection_keypoint import DetectKeypoint
    from src.classification_keypoint import KeypointClassification
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False
    print("⚠️ CẢNH BÁO: Đang chạy chế độ Demo (Không có Model thực).")

class YogaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup Cửa sổ chính
        self.title(APP_NAME)
        
        # Căn giữa màn hình
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (WINDOW_WIDTH / 2))
        y_cordinate = int((screen_height / 2) - (WINDOW_HEIGHT / 2))
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x_cordinate}+{y_cordinate}")
        self.minsize(1024, 600)
        
        # Load Model
        self.detection_model = None
        self.classification_model = None
        self.init_models()

        # Layout Chính
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR (Menu trái)
        self.create_sidebar()

        # 2. TABVIEW (Khu vực chính)
        self.create_main_view()

        # Biến trạng thái
        self.current_image = None
        self.current_pil_image = None
        self.current_result_image = None
        
        # Biến cho Video Control
        self.is_video_mode = False
        self.cap = None
        self.video_running = False
        self.is_paused = False
        self.video_delay = 30  # ms (default ~30fps)
        self.current_frame_cv = None # Lưu frame hiện tại để save

        # Load sample images ban đầu
        self.after(500, self.load_sample_images_ui)

    def init_models(self):
        if MODEL_LOADED:
            try:
                # Lưu ý: Với Video real-time, nên dùng yolov8n-pose.pt (nano) để nhanh hơn nếu máy yếu
                self.detection_model = DetectKeypoint("yolov8m-pose.pt")
                self.classification_model = KeypointClassification("./models/pose_classification.pth")
                print("✅ Models loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading models: {e}")

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Logo
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="🧘 YOGA COACH AI", 
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        # --- MODE CONTROL ---
        ctk.CTkLabel(self.sidebar_frame, text="CHẾ ĐỘ HOẠT ĐỘNG", font=("Arial", 12, "bold"), text_color="gray").grid(row=1, column=0, pady=(20, 5))
        
        self.btn_mode_image = ctk.CTkButton(self.sidebar_frame, text="📷 Phân tích Ảnh", 
                                            fg_color="#00ADB5",
                                            command=self.switch_to_image_mode)
        self.btn_mode_image.grid(row=2, column=0, padx=20, pady=10)

        self.btn_mode_video = ctk.CTkButton(self.sidebar_frame, text="🎥 Phân tích Video", 
                                            fg_color="transparent", border_width=2, border_color="#E63946", text_color="#E63946",
                                            command=self.switch_to_video_mode)
        self.btn_mode_video.grid(row=3, column=0, padx=20, pady=10)

        # Nút Upload (Dùng chung cho cả 2 chế độ)
        self.btn_upload = ctk.CTkButton(self.sidebar_frame, text="📂 Tải File Lên", 
                                        command=self.open_file_dialog,
                                        fg_color="#333", hover_color="#444")
        self.btn_upload.grid(row=4, column=0, padx=20, pady=20)

        # Info Box
        self.info_box = ctk.CTkTextbox(self.sidebar_frame, height=150, fg_color="transparent", text_color="gray")
        self.info_box.insert("0.0", "HƯỚNG DẪN:\n- Chọn chế độ Ảnh hoặc Video.\n- Bấm 'Tải File Lên' để chọn ảnh hoặc video cần phân tích.\n- AI sẽ chạy và đánh giá từng khung hình.")
        self.info_box.configure(state="disabled")
        self.info_box.grid(row=5, column=0, padx=20, pady=10)

        # Settings
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                      command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=7, column=0, padx=20, pady=(10, 30))

    def create_main_view(self):
        self.tab_view = ctk.CTkTabview(self, fg_color="transparent")
        self.tab_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=10)
        
        self.tab_dashboard = self.tab_view.add("🔍 Dashboard Giám Sát")
        self.setup_dashboard_tab(self.tab_dashboard)

    def setup_dashboard_tab(self, parent):
        parent.grid_columnconfigure((0, 1), weight=1, uniform="equal_cols")
        # Row 0: Area hiển thị (Ảnh/Video)
        parent.grid_rowconfigure(0, weight=1) 
        # Row 1: Video Controls (Ẩn hiện linh hoạt)
        parent.grid_rowconfigure(1, weight=0)
        # Row 2: Stats Panel
        parent.grid_rowconfigure(2, weight=0)
        # Row 3: Gallery
        parent.grid_rowconfigure(3, weight=0)

        # --- 1. KHUNG INPUT (Camera Raw / Ảnh gốc) ---
        self.frame_input = ctk.CTkFrame(parent, fg_color=("gray90", "#212121"))
        self.frame_input.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame_input.grid_columnconfigure(0, weight=1)
        self.frame_input.grid_rowconfigure(1, weight=1)

        self.lbl_input_title = ctk.CTkLabel(self.frame_input, text="ẢNH GỐC", font=("Arial", 14, "bold"), text_color="gray")
        self.lbl_input_title.grid(row=0, column=0, pady=5)
        
        self.lbl_img_input = ctk.CTkLabel(self.frame_input, text="Sẵn sàng...", corner_radius=0)
        self.lbl_img_input.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # --- 2. KHUNG OUTPUT (AI Processed) ---
        self.frame_output = ctk.CTkFrame(parent, fg_color=("gray90", "#212121"))
        # Mặc định ban đầu ở cột 1 (chế độ ảnh)
        self.frame_output.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.frame_output.grid_columnconfigure(0, weight=1)
        self.frame_output.grid_rowconfigure(1, weight=1)

        self.lbl_output_title = ctk.CTkLabel(self.frame_output, text="KẾT QUẢ AI", font=("Arial", 14, "bold"), text_color="#00ADB5")
        self.lbl_output_title.grid(row=0, column=0, pady=5)
        
        self.lbl_img_result = ctk.CTkLabel(self.frame_output, text="Kết quả hiển thị tại đây", corner_radius=0)
        self.lbl_img_result.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # --- 3. VIDEO CONTROLS (Thanh điều khiển Video) ---
        self.video_controls_frame = ctk.CTkFrame(parent, height=50, fg_color="transparent")
        # Mặc định ẩn, chỉ hiện khi switch sang Video Mode
        self.video_controls_frame.grid_remove() 
        
        # Nút Giảm Tốc
        ctk.CTkButton(self.video_controls_frame, text="⏪ Chậm", width=80, command=self.slow_down_video).pack(side="left", padx=10)
        # Nút Play/Pause
        self.btn_pause = ctk.CTkButton(self.video_controls_frame, text="⏸ Tạm Dừng", width=100, fg_color="#E63946", command=self.toggle_pause)
        self.btn_pause.pack(side="left", padx=10)
        # Nút Tăng Tốc
        ctk.CTkButton(self.video_controls_frame, text="Nhanh ⏩", width=80, command=self.speed_up_video).pack(side="left", padx=10)
        # Nút Chụp Ảnh
        ctk.CTkButton(self.video_controls_frame, text="📷 Lưu Ảnh", width=100, fg_color="#00ADB5", command=self.save_snapshot).pack(side="right", padx=10)
        
        self.lbl_speed = ctk.CTkLabel(self.video_controls_frame, text="Speed: 1x")
        self.lbl_speed.pack(side="left", padx=10)

        # --- 4. BẢNG ĐÁNH GIÁ (STATS PANEL) ---
        self.stats_panel = ctk.CTkFrame(parent, height=120, fg_color=("white", "#2B2B2B"), corner_radius=10)
        self.stats_panel.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.stats_panel.grid_columnconfigure(1, weight=1)

        # Cột 1: Tên tư thế
        self.lbl_pose_name = ctk.CTkLabel(self.stats_panel, text="---", font=("Arial", 28, "bold"), text_color="#E63946")
        self.lbl_pose_name.grid(row=0, column=0, rowspan=2, padx=30, pady=10)

        # Cột 2: Feedback
        ctk.CTkLabel(self.stats_panel, text="Đánh giá chi tiết:", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=10, pady=(10,0))
        self.lbl_feedback = ctk.CTkLabel(self.stats_panel, text="Chưa có dữ liệu phân tích.", text_color="orange", anchor="w", justify="left")
        self.lbl_feedback.grid(row=1, column=1, sticky="w", padx=10, pady=(0,10))

        # Cột 3: Confidence Score
        self.frame_score = ctk.CTkFrame(self.stats_panel, fg_color="transparent")
        self.frame_score.grid(row=0, column=2, rowspan=2, padx=30)
        
        ctk.CTkLabel(self.frame_score, text="Độ chính xác").pack()
        self.lbl_conf_val = ctk.CTkLabel(self.frame_score, text="0%", font=("Arial", 20, "bold"), text_color="#00ADB5")
        self.lbl_conf_val.pack()
        self.progress_conf = ctk.CTkProgressBar(self.frame_score, orientation="horizontal", width=150, height=10)
        self.progress_conf.set(0)
        self.progress_conf.pack(pady=5)

        # --- 5. GALLERY (Chỉ hiện ở chế độ ảnh) ---
        self.gallery_frame = ctk.CTkScrollableFrame(parent, height=80, orientation="horizontal", label_text="Ảnh Mẫu")
        self.gallery_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Bind resize
        self.lbl_img_input.bind("<Configure>", self.on_frame_configure)
        self.lbl_img_result.bind("<Configure>", self.on_frame_configure)

    # --- LOGIC CHUYỂN ĐỔI CHẾ ĐỘ & LAYOUT ---
    def switch_to_image_mode(self):
        self.is_video_mode = False
        self.stop_video()
        
        # 1. Khôi phục Layout Ảnh: 2 cột
        self.frame_input.grid(row=0, column=0, sticky="nsew") # Hiện khung Input
        self.frame_output.grid(row=0, column=1, columnspan=1, sticky="nsew") # Khung Output về cột 1
        
        # 2. Ẩn Video Controls
        self.video_controls_frame.grid_remove()
        
        # 3. UI Update khác
        self.btn_mode_image.configure(fg_color="#00ADB5")
        self.btn_mode_video.configure(fg_color="transparent")
        self.btn_upload.configure(text="📂 Tải Ảnh Lên")
        self.gallery_frame.grid() # Hiện gallery
        self.lbl_pose_name.configure(text="---")
        self.progress_conf.set(0)
        self.lbl_conf_val.configure(text="0%")
        self.lbl_output_title.configure(text="KẾT QUẢ AI")

    def switch_to_video_mode(self):
        self.is_video_mode = True
        self.stop_video()
        
        # 1. Thay đổi Layout Video: 1 khung lớn
        self.frame_input.grid_forget() # Ẩn khung Input
        self.frame_output.grid(row=0, column=0, columnspan=2, sticky="nsew") # Khung Output tràn 2 cột
        
        # 2. Hiện Video Controls
        self.video_controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # 3. UI Update khác
        self.btn_mode_image.configure(fg_color="transparent")
        self.btn_mode_video.configure(fg_color="#E63946")
        self.btn_upload.configure(text="📂 Tải Video Lên")
        self.gallery_frame.grid_remove() # Ẩn gallery
        self.lbl_pose_name.configure(text="---")
        self.progress_conf.set(0)
        self.lbl_conf_val.configure(text="0%")
        self.lbl_output_title.configure(text="PHÂN TÍCH VIDEO")

    # --- VIDEO CONTROLS LOGIC ---
    def toggle_pause(self):
        if not self.video_running: return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.configure(text="▶ Tiếp Tục", fg_color="#00C853")
        else:
            self.btn_pause.configure(text="⏸ Tạm Dừng", fg_color="#E63946")
            self.update_video_frame() # Gọi lại loop nếu đang dừng

    def slow_down_video(self):
        self.video_delay = min(500, self.video_delay + 20) # Tăng delay = chậm lại
        self.update_speed_label()

    def speed_up_video(self):
        self.video_delay = max(5, self.video_delay - 20) # Giảm delay = nhanh hơn
        self.update_speed_label()

    def update_speed_label(self):
        # 30ms chuẩn là 1x.
        speed_x = round(30 / self.video_delay, 1)
        self.lbl_speed.configure(text=f"Speed: {speed_x}x")

    def save_snapshot(self):
        if self.current_frame_cv is None:
            messagebox.showerror("Lỗi", "Không có khung hình để lưu!")
            return
            
        # Tự động tạo tên file theo thời gian
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(os.getcwd(), filename)
        
        try:
            # Save frame đang có (đã vẽ Bounding Box)
            cv2.imwrite(save_path, cv2.cvtColor(self.current_frame_cv, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

    # --- VIDEO FILE LOGIC ---
    def start_video(self, file_path):
        if self.video_running: return
        
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở file Video này!")
            return
            
        self.video_running = True
        self.is_paused = False
        self.video_delay = 30 # Reset tốc độ
        self.update_speed_label()
        self.btn_pause.configure(text="⏸ Tạm Dừng", fg_color="#E63946")
        self.lbl_feedback.configure(text="Đang phân tích Video...")
        
        # Chạy loop cập nhật frame
        self.update_video_frame()

    def stop_video(self):
        self.video_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.lbl_img_input.configure(image=None)
        self.lbl_img_result.configure(image=None)

    def update_video_frame(self):
        if not self.video_running or not self.is_video_mode:
            return
            
        if self.is_paused:
            return # Dừng gọi đệ quy nếu pause

        ret, frame = self.cap.read()
        if ret:
            # Xử lý AI ngay trên frame này
            self.process_frame_live(frame)
            
            # Lặp lại sau video_delay ms
            self.after(self.video_delay, self.update_video_frame)
        else:
            # Hết video -> Loop lại từ đầu hoặc dừng? Ở đây mình dừng.
            self.stop_video()
            self.lbl_feedback.configure(text="Đã hoàn thành phân tích video.")
            messagebox.showinfo("Hoàn tất", "Đã chạy hết video.")

    def process_frame_live(self, frame_cv):
        """Xử lý AI thời gian thực cho từng frame Video"""
        try:
            if not MODEL_LOADED:
                # Nếu không có model thì hiển thị frame gốc luôn
                img_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                self.current_frame_cv = img_rgb # Lưu để chụp ảnh
                pil_result = Image.fromarray(img_rgb)
                self.display_image_on_label(pil_result, self.lbl_img_result)
                return

            # 1. AI Detection & Classification
            result = self.detection_model(frame_cv)
            keypoints = self.detection_model.get_xy_keypoint(result)

            if keypoints is None or result.boxes is None or len(result.boxes) == 0:
                self.lbl_pose_name.configure(text="NO POSE", text_color="gray")
                # Hiển thị frame gốc nếu không thấy người
                img_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                self.current_frame_cv = img_rgb
                pil_result = Image.fromarray(img_rgb)
                self.display_image_on_label(pil_result, self.lbl_img_result)
                return

            input_classification = keypoints[10:]
            pose_label = self.classification_model(input_classification)

            # 2. Logic Đánh giá
            simulated_score = random.uniform(0.85, 0.99)
            feedback_text = self.get_ai_feedback(pose_label)
            
            # 3. Vẽ Skeleton & Visualization
            image_draw = result.plot(boxes=False)
            x_min, y_min, x_max, y_max = result.boxes.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

            image_draw_rgb = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
            self.current_frame_cv = image_draw_rgb # Lưu frame đã vẽ để chụp ảnh
            pil_result = Image.fromarray(image_draw_rgb)

            # 4. Cập nhật UI Kết quả
            self.display_image_on_label(pil_result, self.lbl_img_result)
            self.update_ui_result(pil_result, pose_label, simulated_score, feedback_text)

        except Exception as e:
            print(f"Video Error: {e}")

    def get_ai_feedback(self, label):
        """Hàm lấy feedback dựa trên nhãn tư thế"""
        feedback_db = {
            "downdog": "Lưng thẳng, gót chân chạm sàn. Hít thở sâu.",
            "warrior2": "Đầu gối vuông góc, mắt nhìn theo tay. Siết cơ đùi.",
            "tree": "Mắt tập trung một điểm. Giữ thăng bằng tốt.",
            "plank": "Siết cơ bụng, lưng không võng. Giữ thẳng người.",
            "cobra": "Mở rộng ngực, thả lỏng vai. Đừng ngửa cổ quá mức."
        }
        for key, text in feedback_db.items():
            if key in label.lower():
                return text
        return "Tư thế ổn định. Hãy duy trì nhịp thở đều."

    def update_ui_result(self, pil_img, label, score, feedback):
        """Cập nhật giao diện kết quả và thanh điểm số"""
        # Cập nhật thông số
        self.lbl_pose_name.configure(text=label.upper(), text_color="#00ADB5")
        
        # Cập nhật thanh Progress Bar và %
        self.progress_conf.set(score)
        self.lbl_conf_val.configure(text=f"{int(score*100)}%")
        
        if score > 0.9: self.progress_conf.configure(progress_color="#00E676")
        elif score > 0.7: self.progress_conf.configure(progress_color="#FFEA00")
        else: self.progress_conf.configure(progress_color="#FF3D00")

        self.lbl_feedback.configure(text=feedback)

    # --- IMAGE LOGIC & DIALOG ---
    def open_file_dialog(self):
        if self.is_video_mode:
            # Chọn Video
            file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
            if file_path:
                self.start_video(file_path)
        else:
            # Chọn Ảnh
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
            if file_path:
                self.current_image = file_path
                threading.Thread(target=self.process_image_logic, args=(file_path,)).start()

    def process_image_logic(self, img_path):
        try:
            pil_image = Image.open(img_path).convert("RGB")
            self.current_pil_image = pil_image
            
            self.after(0, lambda: self.display_image_on_label(pil_image, self.lbl_img_input))
            self.after(0, lambda: self.lbl_img_result.configure(text="Đang phân tích...", image=None))

            if not MODEL_LOADED: return

            image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            result = self.detection_model(image_cv)
            keypoints = self.detection_model.get_xy_keypoint(result)

            if keypoints is None or result.boxes is None or len(result.boxes) == 0:
                self.after(0, lambda: self.lbl_img_result.configure(text="Không tìm thấy người"))
                return

            input_classification = keypoints[10:]
            pose_label = self.classification_model(input_classification)
            
            # Logic đánh giá
            simulated_score = random.uniform(0.85, 0.99)
            feedback_text = self.get_ai_feedback(pose_label)

            image_draw = result.plot(boxes=False)
            x_min, y_min, x_max, y_max = result.boxes.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
            
            image_draw_rgb = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(image_draw_rgb)
            self.current_result_image = pil_result

            self.after(0, lambda: self.display_image_on_label(pil_result, self.lbl_img_result))
            self.after(0, lambda: self.update_ui_result(pil_result, pose_label, simulated_score, feedback_text))

        except Exception as e:
            print(f"Image Error: {e}")

    # --- UTILS ---
    def display_image_on_label(self, pil_img, ctk_label):
        w_widget = ctk_label.winfo_width()
        h_widget = ctk_label.winfo_height()
        if w_widget < 10 or h_widget < 10: return

        img_ratio = pil_img.width / pil_img.height
        widget_ratio = w_widget / h_widget

        if widget_ratio > img_ratio:
            display_h = h_widget
            display_w = int(h_widget * img_ratio)
        else:
            display_w = w_widget
            display_h = int(w_widget / img_ratio)

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(display_w, display_h))
        ctk_label.configure(image=ctk_img, text="")
        ctk_label.image = ctk_img

    def on_frame_configure(self, event):
        if not self.is_video_mode:
            if self.current_pil_image: self.display_image_on_label(self.current_pil_image, self.lbl_img_input)
            if self.current_result_image: self.display_image_on_label(self.current_result_image, self.lbl_img_result)
        else:
            # Trong chế độ video, nếu paused và có frame hiện tại thì redraw khi resize
            if self.is_paused and self.current_frame_cv is not None:
                 pil_img = Image.fromarray(self.current_frame_cv)
                 self.display_image_on_label(pil_img, self.lbl_img_result)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
    
    def load_sample_images_ui(self):
        images = glob.glob("./images/*.jpeg") + glob.glob("./images/*.jpg")
        for i, img_path in enumerate(images[:10]):
            try:
                img = Image.open(img_path)
                ctk_thumb = ctk.CTkImage(img, size=(80, 80))
                btn = ctk.CTkButton(self.gallery_frame, image=ctk_thumb, text="", width=90, height=90,
                                    fg_color="transparent", border_width=2, border_color="gray",
                                    command=lambda p=img_path: self.open_file_dialog_manual(p))
                btn.grid(row=0, column=i, padx=5, pady=5)
            except: pass
            
    def open_file_dialog_manual(self, path):
        self.current_image = path
        if self.is_video_mode: self.switch_to_image_mode()
        threading.Thread(target=self.process_image_logic, args=(path,)).start()

if __name__ == "__main__":
    app = YogaApp()
    app.mainloop()