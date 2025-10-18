import cv2
import tkinter as tk
from PIL import Image, ImageTk, UnidentifiedImageError
import time
import os

class PastelCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎀 Camera Recorder – Pastel Pink Edition 🎀")
        self.root.state("zoomed")

        # === Màu pastel ===
        self.bg_main = "#FFD6E8"
        self.panel_color = "#FFE6EF"
        self.btn_color = "#FFB6C1"
        self.btn_hover = "#FF9EB8"
        self.text_color = "#5A3E3E"

        # === Canvas nền pastel ===
        self.canvas_bg = tk.Canvas(root, bg=self.bg_main, highlightthickness=0)
        self.canvas_bg.pack(fill="both", expand=True)

        # === Khung hiển thị camera ===
        self.frame_camera = tk.Label(self.canvas_bg, bg=self.panel_color)
        self.frame_camera.place(relx=0.015, rely=0.05, relwidth=0.70, relheight=0.9)

        # === Khung nút điều khiển ===
        self.frame_controls = tk.Frame(self.canvas_bg, bg=self.panel_color)
        self.frame_controls.place(relx=0.73, rely=0.25, relwidth=0.12, relheight=0.55)

        # === Khung thông tin ===
        self.frame_info = tk.Frame(self.canvas_bg, bg=self.panel_color)
        self.frame_info.place(relx=0.86, rely=0.25, relwidth=0.13, relheight=0.55)

        # === Thông tin camera ===
        self.lbl_status = tk.Label(
            self.frame_info,
            text="📷 Camera đang mở.\nChưa ghi hình.",
            bg=self.panel_color,
            fg=self.text_color,
            font=("Segoe UI", 12, "bold"),
            justify="center"
        )
        self.lbl_status.pack(padx=10, pady=10, fill="both", expand=True)

        
        def load_icon(name):
            try:
                return ImageTk.PhotoImage(Image.open(name).resize((52, 52)))
            except (FileNotFoundError, UnidentifiedImageError):
                img = Image.new("RGB", (52, 52), color=self.btn_color)
                return ImageTk.PhotoImage(img)

        
        self.start_icon = load_icon("start.png")
        self.stop_icon = load_icon("stop.png")
        self.report_icon = load_icon("report.png")

        self.create_hover_button(self.frame_controls, self.start_icon, "Bắt đầu", self.start_recording)
        self.create_hover_button(self.frame_controls, self.stop_icon, "Dừng", self.stop_recording)
        self.create_hover_button(self.frame_controls, self.report_icon, "Xuất Báo Cáo", self.export_report)

        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.lbl_status.config(text="❌ Không mở được camera. Kiểm tra thiết bị.")
            return

        self.recording = False
        self.start_time = 0
        self.video_writer = None

        
        self.root.after(500, self.update_camera)

    

    def create_hover_button(self, parent, icon, text, command):
        frame = tk.Frame(parent, bg=self.panel_color)
        frame.pack(pady=20)
        btn = tk.Button(
            frame,
            image=icon,
            text=" " + text,
            compound="left",
            font=("Segoe UI", 14, "bold"),
            bg=self.btn_color,
            fg=self.text_color,
            activebackground=self.btn_hover,
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=command
        )
        btn.pack()
        return btn

    def update_camera(self):
        if not self.cap.isOpened():
            self.lbl_status.config(text="🚫 Camera chưa sẵn sàng.")
            self.root.after(500, self.update_camera)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.lbl_status.config(text="⚠️ Không nhận được hình ảnh, đang thử lại...")
            self.root.after(500, self.update_camera)
            return

        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Lấy kích thước khung camera hiện tại
        w = int(self.frame_camera.winfo_width())
        h = int(self.frame_camera.winfo_height())
        if w > 0 and h > 0:
            img = img.resize((w, h))

        imgtk = ImageTk.PhotoImage(image=img)
        self.frame_camera.imgtk = imgtk
        self.frame_camera.configure(image=imgtk)

        # Ghi hình nếu đang quay
        if self.recording and self.video_writer:
            self.video_writer.write(cv2.flip(frame, 1))

        self.root.after(15, self.update_camera)

    def start_recording(self):
        if not self.recording:
            os.makedirs("videos", exist_ok=True)
            filename = time.strftime("videos/record_%Y%m%d_%H%M%S.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

            self.recording = True
            self.start_time = time.time()
            self.lbl_status.config(text="🔴 Đang ghi hình...")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
            duration = time.time() - self.start_time
            self.lbl_status.config(text=f"⏹ Đã dừng ghi.\nThời lượng: {duration:.1f} giây")

    def export_report(self):
        with open("report.txt", "w", encoding="utf-8") as f:
            f.write("📋 BÁO CÁO GHI HÌNH\n")
            f.write("===================\n")
            f.write(self.lbl_status.cget("text"))
        self.lbl_status.config(text="📄 Báo cáo đã xuất ra 'report.txt'")

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "video_writer") and self.video_writer:
            self.video_writer.release()


# === Khởi động chương trình ===
if __name__ == "__main__":
    root = tk.Tk()
    app = PastelCameraApp(root)
    root.mainloop()
