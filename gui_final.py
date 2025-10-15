import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time, os, sys
import numpy as np

class PastelCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎀 Pastel Camera Recorder 🎀")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f2e9e4")

        self.cap = None
        self.current_cam_index = None
        self.recording = False
        self.start_time = None
        self.video_writer = None
        self.filename = None
        self.frame_size = (960, 720)

        # ========== MÀN HÌNH CHỌN CAMERA ==========
        self.select_frame = tk.Frame(self.root, bg="#f2e9e4")
        self.select_frame.pack(fill="both", expand=True)

        center_frame = tk.Frame(self.select_frame, bg="#f2e9e4")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        title = tk.Label(center_frame, text="🎥 Chọn nguồn camera 🎥",
                         bg="#f2e9e4", font=("Arial", 24, "bold"), fg="#2d3436")
        title.pack(pady=(0, 30))

        radio_frame = tk.Frame(center_frame, bg="#f2e9e4")
        radio_frame.pack(pady=10)

        tk.Label(radio_frame, text="Nguồn camera:", bg="#f2e9e4",
                 font=("Arial", 14, "bold")).grid(row=0, column=0, padx=(0, 15))

        self.cam_var = tk.StringVar(value="0")
        tk.Radiobutton(radio_frame, text="Camera máy tính (0)", variable=self.cam_var, value="0",
                       bg="#f2e9e4", font=("Arial", 13)).grid(row=0, column=1, padx=20)
        tk.Radiobutton(radio_frame, text="Camera rời (1)", variable=self.cam_var, value="1",
                       bg="#f2e9e4", font=("Arial", 13)).grid(row=0, column=2, padx=20)

        self.apply_btn = tk.Button(center_frame, text="✨ Áp dụng nguồn ✨",
                                   command=self.apply_camera_choice_and_go,
                                   bg="#ffffff", activebackground="#fde2e4",
                                   relief="raised", bd=3,
                                   font=("Arial", 14, "bold"), cursor="hand2",
                                   width=18, height=2)
        self.apply_btn.pack(pady=(30, 10))

        # ========== GIAO DIỆN CHÍNH ==========
        self.main_frame = tk.Frame(self.root, bg="#f2e9e4")

        top_bar = tk.Frame(self.main_frame, bg="#f2e9e4")
        top_bar.pack(fill="x", pady=(10, 0))

        self.back_btn = tk.Button(top_bar, text="⟵ Quay lại chọn nguồn",
                                  command=self.go_back_to_select,
                                  bg="#ffffff", activebackground="#fde2e4",
                                  relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
        self.back_btn.pack(side="left", padx=20, pady=5)

        # --- CHIA KHUNG CAMERA & BÁO CÁO ---
        content_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Cấu hình lưới để 2 cột cao bằng nhau
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=3)  # 75%
        content_frame.grid_columnconfigure(1, weight=1)  # 25%

        # Camera chiếm ~70%
        left_frame = tk.Frame(content_frame, bg="#f2e9e4")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        left_frame.pack_propagate(False)

  # --- HÀNG NÚT ĐIỀU KHIỂN (đặt NGANG HÀNG với nút Quay lại chọn nguồn) ---
        # --- Thanh nút nằm sát trên cùng cửa sổ chính ---
        btn_frame = tk.Frame(left_frame, bg="#f2e9e4")
        btn_frame.pack(pady=(5, 10))

        btn_style = {
            "width": 130, "height": 85, "compound": "top",
            "bg": "#ffffff", "activebackground": "#fde2e4",
            "relief": "raised", "bd": 4, "font": ("Arial", 11, "bold"),
            "cursor": "hand2"
        }

        try:
            start_icon = ImageTk.PhotoImage(Image.open("start.png").resize((60, 60)))
            stop_icon = ImageTk.PhotoImage(Image.open("stop.png").resize((60, 60)))
            report_icon = ImageTk.PhotoImage(Image.open("report.png").resize((60, 60)))
        except Exception:
            start_icon = stop_icon = report_icon = None

        def create_hover_button(parent, text, icon, command):
            btn = tk.Button(parent, text=text, image=icon, command=command, **btn_style)
            btn.image = icon
            btn.bind("<Enter>", lambda e: btn.config(bg="#ffe5ec", relief="groove"))
            btn.bind("<Leave>", lambda e: btn.config(bg="#ffffff", relief="raised"))
            return btn

        self.start_btn = create_hover_button(btn_frame, "Bắt Đầu", start_icon, self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=35)
        self.stop_btn = create_hover_button(btn_frame, "Dừng", stop_icon, self.stop_recording)
        self.stop_btn.grid(row=0, column=1, padx=35)
        self.report_btn = create_hover_button(btn_frame, "Xuất Báo Cáo", report_icon, self.export_report)
        self.report_btn.grid(row=0, column=2, padx=35)


        # Cửa sổ camera
        self.video_label = tk.Label(left_frame, bg="#c9ada7", relief="sunken", bd=3, width=1120, height=840) #320x240
        self.video_label.pack(padx=5, pady=5)

        # Bên phải: bảng báo cáo (25%)
        right_frame = tk.Frame(content_frame, bg="#f2e9e4")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="📋 Báo Cáo Theo Thời Gian Thực",
                 bg="#f2e9e4", font=("Arial", 18, "bold"), fg="#2d3436").pack(pady=(5, 10))

        self.log_text = tk.Text(right_frame, bg="#ffffff", fg="#2d3436",
                                font=("Consolas", 13), relief="solid", bd=2, height=15, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text.insert("end", "🟡 Chưa có hoạt động...\n")

        # Label hiển thị thời gian ghi hình
        self.timer_label = tk.Label(right_frame, text="⏱️ Chưa ghi hình",
                                    bg="#f2e9e4", font=("Arial", 16, "bold"), fg="#2d3436")
        self.timer_label.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_camera()

    # ================== CÁC HÀM CHÍNH ==================
    def show_main_ui(self):
        self.select_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    def go_back_to_select(self):
        if self.recording:
            self.log("⚠️ Dừng ghi trước khi quay lại.")
            return
        self.main_frame.pack_forget()
        self.select_frame.pack(fill="both", expand=True)

    def apply_camera_choice_and_go(self):
        cam_index = int(self.cam_var.get())
        if not self.init_camera(cam_index):
            alt = 1 - cam_index
            if self.init_camera(alt):
                self.cam_var.set(str(alt))
                self.log(f"⚠️ Không mở được camera {cam_index}, chuyển sang {alt}.")
            else:
                self.log("❌ Không mở được camera nào.")
                return
        self.show_main_ui()
        self.log(f"🟢 Camera {cam_index} đã sẵn sàng.")

    def init_camera(self, index: int) -> bool:
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        self.cap = cv2.VideoCapture(index, backend)
        ok = self.cap.isOpened()
        if ok:
            self.current_cam_index = index
        return ok

    def update_camera(self):
    
        if self.cap and self.cap.isOpened() and self.main_frame.winfo_ismapped():
            ret, frame = self.cap.read()
            if ret and frame is not None:
            # Kích thước thực tế của khung
                h0, w0 = frame.shape[:2]
                target_w, target_h = 1120, 840  

            # Giữ đúng tỉ lệ gốc ảnh
                frame_ratio = w0 / h0
                target_ratio = target_w / target_h
                if frame_ratio > target_ratio:
                
                    new_w = target_w
                    new_h = int(target_w / frame_ratio)
                else:
                # Ảnh cao hơn → thu theo chiều dọc
                    new_h = target_h
                    new_w = int(target_h * frame_ratio)

                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Tạo nền pastel để canh giữa (giữ đúng tỉ lệ)
                background = np.full((target_h, target_w, 3), (242, 233, 228), dtype=np.uint8)
                x_offset = (target_w - new_w) // 2
                y_offset = (target_h - new_h) // 2
                background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            # Hiển thị ảnh lên Tkinter
                frame_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

    # Gọi lại sau 30ms để cập nhật video liên tục
        self.root.after(30, self.update_camera)

    def start_recording(self):
        """Bắt đầu ghi video với kích thước đúng của frame"""
        if not (self.cap and self.cap.isOpened()):
            self.log("❌ Chưa có camera nào đang mở.")
            return
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            os.makedirs("recordings", exist_ok=True)
            self.filename = time.strftime("recordings/video_%Y%m%d_%H%M%S.avi")

            # Lấy kích thước thực tế từ frame đầu tiên
            ret, frame = self.cap.read()
            if not ret:
                self.log("⚠️ Không lấy được hình ảnh từ camera.")
                return
            h0, w0 = frame.shape[:2]
            self.frame_size = (w0, h0)

            # Tạo writer với kích thước đúng
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(self.filename, fourcc, 20.0, self.frame_size)
            self.log(f"🔴 Bắt đầu ghi hình ({self.filename})")
            self.update_timer()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            duration = int(time.time() - self.start_time)
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.log(f"🟡 Dừng ghi. Thời lượng: {duration} giây.")
            self.timer_label.config(text=f"🟡 Dừng ghi ({duration}s)", fg="orange")

    def update_timer(self):
        if self.recording:
            elapsed = int(time.time() - self.start_time)
            mm, ss = divmod(elapsed, 60)
            self.timer_label.config(text=f"⏱️ Thời gian ghi: {mm:02d}:{ss:02d}", fg="red")
            self.root.after(1000, self.update_timer)

    # ================== KHÁC ==================
    def log(self, text):
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} - {text}\n")
        self.log_text.see("end")

    def export_report(self):
        self.log("📤 Xuất báo cáo hoàn tất.")

    def on_close(self):
        if self.recording:
            self.stop_recording()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PastelCameraApp(root)
    root.mainloop()
