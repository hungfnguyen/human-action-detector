import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import os
import glob
import threading
import time
from datetime import datetime
import sys
from pathlib import Path
import queue  # 🚀 For async video processing

# --- CẤU HÌNH GIAO DIỆN & XỬ LÝ ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

APP_NAME = "YOLOv8 Yoga Pose Assessment - HCMUTE Group 18"
WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768

# Kích thước chuẩn để xử lý AI (Tất cả ảnh/video sẽ được đưa về size này trước khi vẽ)
# 🚀 OPTIMIZED: Giảm từ 1280x720 xuống 960x540 để tăng FPS
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540

# --- IMPORT MODULES TỪ SRC ---
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.detection.pose_detector import PoseDetector
    from src.recognition.pose_recognizer import PoseRecognizer
    from src.evaluation.pose_evaluator import PoseEvaluator
    from src.visualization.skeleton_drawer import SkeletonDrawer
    from src.visualization.overlay_ui import OverlayUI
    MODEL_LOADED = True
except ImportError as e:
    MODEL_LOADED = False
    print(f"⚠️ LỖI IMPORT: {e}")
    print("Vui lòng kiểm tra cấu trúc thư mục 'src/' và 'config/'.")

class YogaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup Cửa sổ chính
        self.title(APP_NAME)
        
        # Cấu hình kích thước cơ bản
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Thiết lập geometry ban đầu (sẽ được override bởi state zoomed bên dưới)
        x_cordinate = int((screen_width / 2) - (WINDOW_WIDTH / 2))
        y_cordinate = int((screen_height / 2) - (WINDOW_HEIGHT / 2))
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x_cordinate}+{y_cordinate}")
        self.minsize(1024, 600)

        # --- TỰ ĐỘNG FULL MÀN HÌNH (MAXIMIZED) ---
        # Linux không hỗ trợ 'zoomed', dùng attributes() để maximize
        try:
            # Thử dùng attributes (Linux-friendly)
            self.attributes('-zoomed', True)
        except Exception:
            try:
                # Fallback: Windows/Mac style
                self.state('zoomed')
            except Exception:
                # Fallback cuối: Manual geometry
                self.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # --- KHỞI TẠO CÁC MODULE AI ---
        self.detector = None
        self.recognizer = None
        self.evaluator = None
        self.drawer = None
        self.overlay = None
        
        # Thread loading model
        self.after(100, self.init_models)

        # Layout Chính
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR (Menu trái)
        self.create_sidebar()

        # 2. TABVIEW (Khu vực chính)
        self.create_main_view()

        # Biến trạng thái
        self.current_image_path = None
        self.current_pil_image = None
        self.current_result_image = None # Lưu ảnh kết quả PIL (Full size đã xử lý)
        self.current_frame_processed = None # Lưu frame kết quả (OpenCV format)
        
        # Biến cho Video Control
        self.is_video_mode = False
        self.cap = None
        self.video_running = False
        self.is_paused = False
        self.video_delay = 30
        self.current_pose_name = "Unknown"  # Track current pose for snapshot naming
        self.current_score = 0  # Track current score for snapshot naming
        
        # 🚀 ASYNC VIDEO PROCESSING: Producer-Consumer Architecture
        self.frame_queue = queue.Queue(maxsize=5)  # Buffer 5 frames
        self.processing_thread = None
        self.frame_counter = 0
        self.process_every_n_frames = 1  # Xử lý mỗi N frame (1=all, 2=every other)
        
        # ⌨️ Keyboard shortcuts
        self.bind_all("<space>", lambda e: self.toggle_pause())
        self.bind_all("<KeyPress-s>", lambda e: self.save_snapshot())
        self.bind_all("<KeyPress-S>", lambda e: self.save_snapshot())
        
        # Load sample images ban đầu
        self.after(1000, self.load_sample_images_ui)

    def init_models(self):
        """Khởi tạo các class xử lý từ file src"""
        if MODEL_LOADED:
            try:
                print("⏳ Đang tải models...")
                yolo_path = "models/yolov8m-pose.pt"
                clf_path = "models/pose_classification.pth"
                
                if not os.path.exists(yolo_path):
                    yolo_path = "yolov8m-pose.pt" 
                
                # Enable GPU for maximum performance
                self.detector = PoseDetector(
                    model_path=yolo_path, 
                    confidence_threshold=0.5,
                    use_gpu=True  # 🚀 GPU Acceleration enabled
                )
                self.recognizer = PoseRecognizer(model_path=clf_path)
                self.evaluator = PoseEvaluator()
                self.drawer = SkeletonDrawer()
                self.overlay = OverlayUI()
                
                print("✅ Đã tải xong toàn bộ Models & Modules.")
                self.lbl_feedback.configure(text="Hệ thống đã sẵn sàng!", text_color="#00E676")
            except Exception as e:
                print(f"❌ Lỗi khởi tạo Model: {e}")
                self.lbl_feedback.configure(text="Lỗi tải Model AI", text_color="red")
                # Không hiển thị popup lỗi ngay lập tức để tránh block UI khi khởi động
                print(f"Chi tiết lỗi: {e}")

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        # --- 1. LOGO TRƯỜNG HCMUTE (Row 0) ---
        self.logo_image = None
        try:
            # Đảm bảo file 'logo.png' nằm cùng thư mục
            img_path = "logo.png" 
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # Resize logo cho vừa khung (ví dụ: 150x150)
                self.logo_image = ctk.CTkImage(light_image=img, dark_image=img, size=(160, 160))
            else:
                print("Không tìm thấy file logo 'logo.png'.")
        except Exception as e:
            print(f"Lỗi load logo: {e}")

        if self.logo_image:
            self.lbl_logo_img = ctk.CTkLabel(self.sidebar_frame, image=self.logo_image, text="")
            self.lbl_logo_img.grid(row=0, column=0, padx=20, pady=(20, 5))
        
        # --- 2. TÊN ĐỀ TÀI (Row 1 - Nằm dưới logo) ---
        # Sử dụng wraplength để tự động xuống dòng
        title_text = "🧘 Ứng dụng YOLOv8 trích xuất khung xương, nhận dạng & đánh giá Yoga"
        self.lbl_project_title = ctk.CTkLabel(self.sidebar_frame, 
                                            text=title_text,
                                            font=ctk.CTkFont(size=16, weight="bold"),
                                            wraplength=240, # Tự động xuống dòng nếu dài quá 240px
                                            justify="center")
        self.lbl_project_title.grid(row=1, column=0, padx=10, pady=(5, 20))
        
        # --- 3. MODE CONTROL (Row 2) ---
        ctk.CTkLabel(self.sidebar_frame, text="CHẾ ĐỘ HOẠT ĐỘNG", font=("Arial", 12, "bold"), text_color="gray").grid(row=2, column=0, pady=(10, 5))
        
        # --- 4. BUTTONS (Row 3, 4, 5) ---
        self.btn_mode_image = ctk.CTkButton(self.sidebar_frame, text="📷 Phân tích Ảnh", 
                                          fg_color="#00ADB5",
                                          command=self.switch_to_image_mode)
        self.btn_mode_image.grid(row=3, column=0, padx=20, pady=10)

        self.btn_mode_video = ctk.CTkButton(self.sidebar_frame, text="🎥 Phân tích Video", 
                                          fg_color="transparent", border_width=2, border_color="#E63946", text_color="#E63946",
                                          command=self.switch_to_video_mode)
        self.btn_mode_video.grid(row=4, column=0, padx=20, pady=10)

        self.btn_upload = ctk.CTkButton(self.sidebar_frame, text="📂 Tải File Lên", 
                                      command=self.open_file_dialog,
                                      fg_color="#333", hover_color="#444")
        self.btn_upload.grid(row=5, column=0, padx=20, pady=20)

        # --- 5. INFO BOX (Row 6) ---
        self.info_box = ctk.CTkTextbox(self.sidebar_frame, height=250, fg_color="transparent", text_color="gray", font=("Arial", 12))
        
        info_text = (
            "NHÓM 18 - HCMUTE:\n"
            "1. Mai Hồng Hải - MSSV: 22133014\n"
            "2. Nguyễn Tấn Hùng - MSSV: 22133027\n"
            "3. Nguyễn Ngọc Hiếu Hảo - MSSV: 22133015\n\n\n"
            "HƯỚNG DẪN:\n"
            "- Chọn Ảnh/Video để phân tích.\n"
            "- Hệ thống sẽ tự động resize ảnh về chuẩn HD để phân tích.\n"
            "- AI sẽ chấm điểm kỹ thuật.\n"
            "- Kết quả hiển thị bên trên màn hình."
        )
        
        self.info_box.insert("0.0", info_text)
        self.info_box.configure(state="disabled")
        self.info_box.grid(row=6, column=0, padx=20, pady=5)

        # Settings (Row 8 - đẩy xuống dưới cùng)
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                    command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=8, column=0, padx=20, pady=(0, 100))

    def create_main_view(self):
        # --- CẬP NHẬT: Xóa TabView, dùng Frame trực tiếp để tối đa không gian ---
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.setup_dashboard_view(self.main_view)

    def setup_dashboard_view(self, parent):
        parent.grid_columnconfigure((0, 1), weight=1, uniform="equal_cols")
        parent.grid_rowconfigure(0, weight=1) # Ảnh sẽ chiếm phần lớn không gian
        parent.grid_rowconfigure(1, weight=0) # Controls row
        parent.grid_rowconfigure(2, weight=0) 
        parent.grid_rowconfigure(3, weight=0) 

        # --- 1. KHUNG INPUT ---
        self.frame_input = ctk.CTkFrame(parent, fg_color=("gray90", "#212121"))
        # Giảm padding để khung to hơn
        self.frame_input.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.frame_input.grid_columnconfigure(0, weight=1)
        self.frame_input.grid_rowconfigure(1, weight=1)

        self.lbl_input_title = ctk.CTkLabel(self.frame_input, text="ẢNH GỐC", font=("Arial", 14, "bold"), text_color="gray")
        self.lbl_input_title.grid(row=0, column=0, pady=2)
        
        self.lbl_img_input = ctk.CTkLabel(self.frame_input, text="Sẵn sàng...", corner_radius=0)
        self.lbl_img_input.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # --- 2. KHUNG OUTPUT ---
        self.frame_output = ctk.CTkFrame(parent, fg_color=("gray90", "#212121"))
        # Giảm padding để khung to hơn
        self.frame_output.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.frame_output.grid_columnconfigure(0, weight=1)
        self.frame_output.grid_rowconfigure(1, weight=1)

        self.lbl_output_title = ctk.CTkLabel(self.frame_output, text="KẾT QUẢ AI", font=("Arial", 14, "bold"), text_color="#00ADB5")
        self.lbl_output_title.grid(row=0, column=0, pady=2)
        
        self.lbl_img_result = ctk.CTkLabel(self.frame_output, text="Kết quả hiển thị tại đây", corner_radius=0)
        self.lbl_img_result.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # --- 3a. VIDEO CONTROLS ---
        self.video_controls_frame = ctk.CTkFrame(parent, height=50, fg_color="transparent")
        self.video_controls_frame.grid_remove() # Ẩn mặc định
        
        # Pause/Resume button
        self.btn_pause = ctk.CTkButton(self.video_controls_frame, text="⏸ Tạm Dừng", width=120, fg_color="#E63946", command=self.toggle_pause)
        self.btn_pause.pack(side="left", padx=10)
        
        # Snapshot button
        ctk.CTkButton(self.video_controls_frame, text="📷 Lưu Ảnh", width=120, fg_color="#00ADB5", command=self.save_snapshot).pack(side="right", padx=10)

        # --- 3b. IMAGE CONTROLS (Nút lưu ảnh) ---
        self.image_controls_frame = ctk.CTkFrame(parent, height=50, fg_color="transparent")
        self.image_controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10)) # Mặc định hiện
        
        self.btn_save_image = ctk.CTkButton(self.image_controls_frame, text="💾 Lưu Kết Quả", 
                                          width=150, height=35,
                                          fg_color="#00ADB5", hover_color="#007d85",
                                          font=("Arial", 13, "bold"),
                                          command=self.save_image_result)
        self.btn_save_image.pack(pady=5)

        # --- 4. STATS PANEL ---
        self.stats_panel = ctk.CTkFrame(parent, height=120, fg_color=("white", "#2B2B2B"), corner_radius=10)
        self.stats_panel.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.stats_panel.grid_columnconfigure(1, weight=1)

        self.lbl_pose_name = ctk.CTkLabel(self.stats_panel, text="---", font=("Arial", 28, "bold"), text_color="#E63946")
        self.lbl_pose_name.grid(row=0, column=0, rowspan=2, padx=30, pady=10)

        ctk.CTkLabel(self.stats_panel, text="Đánh giá chi tiết:", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=10, pady=(10,0))
        self.lbl_feedback = ctk.CTkLabel(self.stats_panel, text="Chưa có dữ liệu phân tích.", text_color="orange", anchor="w", justify="left")
        self.lbl_feedback.grid(row=1, column=1, sticky="w", padx=10, pady=(0,10))

        self.frame_score = ctk.CTkFrame(self.stats_panel, fg_color="transparent")
        self.frame_score.grid(row=0, column=2, rowspan=2, padx=30)
        
        ctk.CTkLabel(self.frame_score, text="Độ chính xác").pack()
        self.lbl_conf_val = ctk.CTkLabel(self.frame_score, text="0%", font=("Arial", 20, "bold"), text_color="#00ADB5")
        self.lbl_conf_val.pack()
        self.progress_conf = ctk.CTkProgressBar(self.frame_score, orientation="horizontal", width=150, height=10)
        self.progress_conf.set(0)
        self.progress_conf.pack(pady=5)

        # --- 5. GALLERY ---
        self.gallery_frame = ctk.CTkScrollableFrame(parent, height=80, orientation="horizontal", label_text="Ảnh Mẫu")
        self.gallery_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Bind event resize để vẽ lại ảnh cho fit khung
        self.lbl_img_input.bind("<Configure>", self.on_frame_configure)
        self.lbl_img_result.bind("<Configure>", self.on_frame_configure)

    # --- LOGIC CHUYỂN ĐỔI CHẾ ĐỘ ---
    def switch_to_image_mode(self):
        self.is_video_mode = False
        self.stop_video()
        
        
        # Clear display when switching modes
        self.lbl_img_result.configure(image=None, text="")
        
        self.frame_input.grid(row=0, column=0, sticky="nsew")
        self.frame_output.grid(row=0, column=1, columnspan=1, sticky="nsew")
        
        self.video_controls_frame.grid_remove()
        self.image_controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10)) 
        
        self.btn_mode_image.configure(fg_color="#00ADB5")
        self.btn_mode_video.configure(fg_color="transparent")
        self.btn_upload.configure(text="📂 Tải Ảnh Lên")
        self.gallery_frame.grid() 
        self.lbl_output_title.configure(text="KẾT QUẢ AI")
        self.reset_stats()

    def switch_to_video_mode(self):
        self.is_video_mode = True
        self.stop_video()
        
        
        # Clear display when switching modes
        self.lbl_img_result.configure(image=None, text="")
        
        self.frame_input.grid_forget()
        self.frame_output.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        self.image_controls_frame.grid_remove() 
        self.video_controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        self.btn_mode_image.configure(fg_color="transparent")
        self.btn_mode_video.configure(fg_color="#E63946")
        self.btn_upload.configure(text="📂 Tải Video Lên")
        self.gallery_frame.grid_remove() 
        self.lbl_output_title.configure(text="PHÂN TÍCH VIDEO")
        self.reset_stats()

    def reset_stats(self):
        self.lbl_pose_name.configure(text="---")
        self.progress_conf.set(0)
        self.lbl_conf_val.configure(text="0%")
        self.lbl_feedback.configure(text="Đang chờ dữ liệu...")

    # --- VIDEO CONTROLS ---
    def toggle_pause(self):
        """Pause/Resume video playback"""
        if not self.video_running:
            return
        
        self.is_paused = not self.is_paused
        self.btn_pause.configure(
            text="▶ Tiếp Tục" if self.is_paused else "⏸ Tạm Dừng",
            fg_color="#00C853" if self.is_paused else "#E63946"
        )
        
        # Resume display worker if unpaused
        if not self.is_paused:
            self.display_video_worker()

    def save_snapshot(self):
        if self.current_frame_processed is None:
            messagebox.showerror("Lỗi", "Chưa có khung hình để lưu!")
            return
            
        # Create filename with pose name and score
        pose_clean = self.current_pose_name.lower().replace(" ", "_")
        score_str = f"{self.current_score}pct"  # e.g., "95pct"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{pose_clean}_{score_str}_{timestamp}.jpg"
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")
            
        save_path = os.path.join("snapshots", filename)
        
        try:
            cv2.imwrite(save_path, self.current_frame_processed)
            messagebox.showinfo("Đã lưu", f"Lưu ảnh thành công:\n{filename}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file: {e}")

    # -------------------------------------------------------------------------
    # HÀM RESIZE QUAN TRỌNG: Dùng chung cho cả xử lý AI và Hiển thị
    # -------------------------------------------------------------------------
    def resize_image_to_fixed_size(self, image_cv, target_size=(PROCESS_WIDTH, PROCESS_HEIGHT)):
        """
        Resize ảnh về kích thước cố định (target_size) mà giữ nguyên tỷ lệ.
        Phần thừa sẽ được thêm padding màu đen (Letterboxing).
        """
        target_w, target_h = target_size
        h, w = image_cv.shape[:2]
        
        # Tính tỷ lệ resize để fit vào target_size
        scale = min(target_w/w, target_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize ảnh
        resized = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Tạo canvas đen (target size)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Tính toán vị trí paste (center)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Paste ảnh vào canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

    def save_image_result(self):
        """Lưu ảnh kết quả vào thư mục results"""
        if self.current_frame_processed is None:
            messagebox.showerror("Lỗi", "Chưa có kết quả phân tích để lưu!")
            return
            
        filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Tạo thư mục results nếu chưa có
        if not os.path.exists("results"):
            os.makedirs("results")
            
        save_path = os.path.join("results", filename)
        try:
            # Lưu ảnh (đã được resize và vẽ sẵn từ pipeline xử lý)
            cv2.imwrite(save_path, self.current_frame_processed)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh kết quả tại:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file: {e}")

    # --- 🚀 ASYNC VIDEO PROCESSING ---
    def process_video_worker(self):
        """Background thread: Read + Process frames → Queue (Producer)"""
        frame_count = 0
        
        while self.video_running:
            # ✅ FIX: Check pause state
            if self.is_paused:
                time.sleep(0.1)  # Sleep when paused
                continue
            
            if self.cap is None or not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                self.video_running = False
                break
            
            frame_count += 1
            
            # Skip frames if needed
            if frame_count % self.process_every_n_frames != 0:
                continue
            
            try:
                # Resize
                frame_resized = self.resize_image_to_fixed_size(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
                
                # Process AI (YOLOv8, ML, Drawing) - All on GPU!
                if MODEL_LOADED and self.detector:
                    results = self.detector.predict(frame_resized)
                    kp_norm = self.detector.get_keypoints_normalized(results)
                    kp_abs = self.detector.get_keypoints_absolute(results)
                    
                    if kp_norm and kp_abs:
                        pose_name, confidence = self.recognizer.recognize(kp_norm)
                        score, feedback = self.evaluator.evaluate(pose_name, kp_abs)
                        frame_drawn = self.drawer.draw(frame_resized.copy(), kp_abs, score)
                        frame_final = self.overlay.draw_scoreboard(frame_drawn, pose_name, score, feedback)
                    else:
                        pose_name, score, feedback = "NO POSE", 0, "Không tìm thấy người"
                        frame_final = frame_resized
                else:
                    pose_name, score, feedback = "Loading", 0, "Đang tải..."
                    frame_final = frame_resized
                
                # Put processed frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame_final,
                        'pose': pose_name,
                        'score': score,
                        'feedback': feedback
                    })
                except queue.Full:
                    # Queue full, skip this frame
                    pass
                    
            except Exception as e:
                print(f"Processing error: {e}")
                continue
        
        # Signal end
        try:
            self.frame_queue.put_nowait(None)
        except:
            pass
    
    def display_video_worker(self):
        """UI thread: Get frame from queue → Display (Consumer)"""
        if not self.video_running or not self.is_video_mode:
            return
        
        try:
            # Get processed frame from queue (non-blocking)
            data = self.frame_queue.get_nowait()
            
            if data is None:
                # End of video
                self.stop_video()
                self.lbl_feedback.configure(text="Kết thúc Video.")
                messagebox.showinfo("Xong", "Đã phân tích xong video.")
                return
            
            # Display frame (FAST!)
            frame_cv = data['frame']
            pose_name = data['pose']
            score = data['score']
            feedback = data['feedback']
            
            # Track current pose and score for snapshots
            self.current_pose_name = pose_name
            self.current_score = score
            
            # Convert + Display
            self.current_frame_processed = frame_cv
            img_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(img_rgb)
            self.current_result_image = pil_result
            
            self.display_image_on_label(pil_result, self.lbl_img_result)
            self.update_stats(pose_name, score, feedback)
            
        except queue.Empty:
            # No frame ready, skip this cycle
            pass
        except Exception as e:
            print(f"Display error: {e}")
        
        # Schedule next display update (30 FPS = 33ms)
        if self.video_running and self.is_video_mode:
            self.after(33, self.display_video_worker)
    
    def start_video(self, file_path):
        """Start async video processing"""
        # ✅ FIX: Nếu đang có video (dù pause), stop nó trước
        if self.video_running:
            self.stop_video()
            # Wait a bit for cleanup
            import time
            time.sleep(0.3)
            
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở file Video!")
            return
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        self.video_running = True
        self.is_paused = False
        self.btn_pause.configure(text="⏸ Tạm Dừng", fg_color="#E63946")
        
        # Start processing thread (Background)
        self.processing_thread = threading.Thread(target=self.process_video_worker, daemon=True)
        self.processing_thread.start()
        
        # Start display worker (UI thread)
        self.display_video_worker()

    def stop_video(self):
        """Stop video processing"""
        self.video_running = False
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Release video capture
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        self.lbl_img_input.configure(image=None)
        self.lbl_img_result.configure(image=None)
        self.lbl_img_result.configure(text="Đã dừng.")

    # (Old update_video_frame removed - replaced by async architecture above)

    def process_image_logic(self, img_path):
        try:
            # 1. Load bằng PIL để xử lý xoay ảnh (EXIF)
            pil_image = Image.open(img_path).convert("RGB")
            pil_image = ImageOps.exif_transpose(pil_image)
            self.current_pil_image = pil_image 
            
            # 2. Hiển thị ảnh gốc lên UI (resize nhẹ để hiển thị nhanh)
            self.after(0, lambda: self.display_image_on_label(pil_image, self.lbl_img_input))
            self.after(0, lambda: self.lbl_img_result.configure(text="Đang phân tích AI...", image=None))

            if not MODEL_LOADED: return

            # 3. Chuyển sang OpenCV
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 4. --- QUAN TRỌNG: Resize về kích thước chuẩn TRƯỚC khi xử lý ---
            # Việc này đảm bảo bảng điểm và khung xương luôn có tỉ lệ đẹp
            frame_resized = self.resize_image_to_fixed_size(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            
            # 5. Đưa vào pipeline xử lý
            self.process_and_display(frame_resized, is_video=False)
            
        except Exception as e:
            print(f"Lỗi xử lý ảnh: {e}")
            self.after(0, lambda: self.lbl_feedback.configure(text=f"Lỗi: {e}"))

    def process_and_display(self, frame_cv, is_video=False):
        """
        Hàm này nhận vào frame đã được resize chuẩn (PROCESS_WIDTH x PROCESS_HEIGHT)
        """
        if not MODEL_LOADED or self.detector is None:
            return

        try:
            # 1. Detect
            results = self.detector.predict(frame_cv)
            kp_norm = self.detector.get_keypoints_normalized(results)
            kp_abs = self.detector.get_keypoints_absolute(results)

            if kp_norm is None or kp_abs is None:
                if is_video:
                    self.display_final_result(frame_cv, "NO POSE", 0, "Không tìm thấy người", is_video)
                else:
                    self.after(0, lambda: self.lbl_img_result.configure(text="Không tìm thấy người"))
                return

            # 2. Recognize & Evaluate
            pose_name, confidence = self.recognizer.recognize(kp_norm)
            score, feedback = self.evaluator.evaluate(pose_name, kp_abs)

            # 3. Draw & Overlay (Vẽ trực tiếp lên frame chuẩn)
            frame_drawn = self.drawer.draw(frame_cv.copy(), kp_abs, score)
            frame_final = self.overlay.draw_scoreboard(frame_drawn, pose_name, score, feedback)

            # 4. Lưu lại kết quả
            self.current_frame_processed = frame_final # Frame OpenCV (BGR)
            self.current_pose_name = pose_name  # Track for snapshots
            self.current_score = score  # Track for snapshots
            
            # 5. Hiển thị
            self.display_final_result(frame_final, pose_name, score, feedback, is_video)

        except Exception as e:
            print(f"Processing Error: {e}")

    def display_final_result(self, frame_cv, pose_name, score, feedback, is_video):
        # Convert BGR -> RGB để hiển thị lên UI
        img_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(img_rgb)
        
        self.current_result_image = pil_result 

        if is_video:
            # Video cần update liên tục
            self.display_image_on_label(pil_result, self.lbl_img_result)
            self.update_stats(pose_name, score, feedback)
        else:
            # Ảnh thì dùng after để tránh xung đột thread
            self.after(0, lambda: self.display_image_on_label(pil_result, self.lbl_img_result))
            self.after(0, lambda: self.update_stats(pose_name, score, feedback))

    def update_stats(self, pose, score, feedback):
        self.lbl_pose_name.configure(text=pose)
        
        score_val = score / 100.0
        self.progress_conf.set(score_val)
        self.lbl_conf_val.configure(text=f"{score}%")
        
        if score >= 80:
            self.progress_conf.configure(progress_color="#00E676")
        elif score >= 60:
            self.progress_conf.configure(progress_color="#FFEA00")
        else:
            self.progress_conf.configure(progress_color="#FF3D00")
            
        self.lbl_feedback.configure(text=feedback)

    def open_file_dialog(self):
        if self.is_video_mode:
            # Linux/Mac cần dùng space thay vì semicolon
            file_path = filedialog.askopenfilename(
                title="Chọn Video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                self.start_video(file_path)
        else:
            # Linux/Mac cần dùng space thay vì semicolon
            file_path = filedialog.askopenfilename(
                title="Chọn Ảnh",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                self.current_image_path = file_path
                threading.Thread(target=self.process_image_logic, args=(file_path,)).start()

    # -------------------------------------------------------------------------
    # HÀM HIỂN THỊ ẢNH LÊN UI (Tự động scale để FIT FULL khung)
    # -------------------------------------------------------------------------
    def display_image_on_label(self, pil_img, ctk_label):
        try:
            # 1. Lấy kích thước hiện tại của khung chứa (Label)
            w_widget = ctk_label.winfo_width()
            h_widget = ctk_label.winfo_height()
            
            if w_widget < 10 or h_widget < 10: return

            # Giảm kích thước hiển thị một chút để tránh chèn viền
            safe_w = w_widget - 4
            safe_h = h_widget - 4

            # 2. Tính toán tỷ lệ để "Contain" (Hiển thị toàn bộ ảnh)
            img_w, img_h = pil_img.size
            ratio_w = safe_w / img_w
            ratio_h = safe_h / img_h
            
            # Chọn tỷ lệ nhỏ hơn để đảm bảo ảnh nằm gọn trong khung
            scale = min(ratio_w, ratio_h)
            
            display_w = int(img_w * scale)
            display_h = int(img_h * scale)

            # 3. Tạo CTkImage với kích thước hiển thị đã tính
            # Lưu ý: light_image/dark_image giữ nguyên ảnh gốc (chất lượng cao)
            # size=(display_w, display_h) chỉ điều khiển việc hiển thị trên UI
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(display_w, display_h))
            
            ctk_label.configure(image=ctk_img, text="")
            ctk_label.image = ctk_img # Giữ reference để không bị Garbage Collection thu hồi
        except Exception as e:
            # print(f"Display error: {e}") 
            pass

    def on_frame_configure(self, event):
        """Khi resize cửa sổ, vẽ lại ảnh cho vừa khung"""
        if self.is_video_mode:
            if self.is_paused and self.current_result_image:
                self.display_image_on_label(self.current_result_image, self.lbl_img_result)
        else:
            if self.current_pil_image:
                self.display_image_on_label(self.current_pil_image, self.lbl_img_input)
            if self.current_result_image:
                self.display_image_on_label(self.current_result_image, self.lbl_img_result)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
    
    def load_sample_images_ui(self):
        # Load sample images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(glob.glob(os.path.join("images", ext)))
            
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
        self.current_image_path = path
        if self.is_video_mode: self.switch_to_image_mode()
        threading.Thread(target=self.process_image_logic, args=(path,)).start()

if __name__ == "__main__":
    app = YogaApp()
    app.mainloop()