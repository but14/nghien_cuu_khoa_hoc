
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time, os
import sys

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
        self.recorded_time = 0
        self.filename = None
        self.frame_size = (960, 540)  

    
        self.select_frame = tk.Frame(self.root, bg="#f2e9e4")
        self.select_frame.pack(fill="both", expand=True)

        title = tk.Label(
            self.select_frame, text="Chọn nguồn camera",
            bg="#f2e9e4", font=("Arial", 18, "bold")
        )
        title.pack(pady=(30, 10))

        top_frame = tk.Frame(self.select_frame, bg="#f2e9e4")
        top_frame.pack(pady=(15, 5))

        tk.Label(
            top_frame, text="Nguồn camera:", bg="#f2e9e4",
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, padx=(0, 10))

        self.cam_var = tk.StringVar(value="0")  # 0: built-in, 1: external
        self.rb_internal = tk.Radiobutton(
            top_frame, text="Camera máy tính (0)",
            variable=self.cam_var, value="0",
            bg="#f2e9e4", font=("Arial", 11),
        )
        self.rb_external = tk.Radiobutton(
            top_frame, text="Camera rời (1)",
            variable=self.cam_var, value="1",
            bg="#f2e9e4", font=("Arial", 11),
        )
        self.rb_internal.grid(row=0, column=1, padx=10)
        self.rb_external.grid(row=0, column=2, padx=10)

        self.apply_btn = tk.Button(
            self.select_frame, text="Áp dụng nguồn",
            command=self.apply_camera_choice_and_go,
            bg="#ffffff", activebackground="#fde2e4",
            relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2"
        )
        self.apply_btn.pack(pady=20)

    
        self.main_frame = tk.Frame(self.root, bg="#f2e9e4")

        
        top_bar = tk.Frame(self.main_frame, bg="#f2e9e4")
        top_bar.pack(pady=(10, 0), fill="x")

        self.back_btn = tk.Button(
            top_bar, text="⟵ Quay lại chọn nguồn",
            command=self.go_back_to_select,
            bg="#ffffff", activebackground="#fde2e4",
            relief="raised", bd=2, font=("Arial", 11, "bold"), cursor="hand2"
        )
        self.back_btn.pack(side="left", padx=20)

        
        btn_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        btn_frame.pack(pady=12, fill="x")

        btn_style = {
            "width": 14, "height": 3, "compound": "top",
            "bg": "#fff", "activebackground": "#fde2e4",
            "relief": "raised", "bd": 2, "font": ("Arial", 11, "bold"),
            "cursor": "hand2"
        }

        
        try:
            start_icon = ImageTk.PhotoImage(Image.open("start.png").resize((64, 64)))
            stop_icon = ImageTk.PhotoImage(Image.open("stop.png").resize((64, 64)))
            report_icon = ImageTk.PhotoImage(Image.open("report.png").resize((64, 64)))
        except Exception:
            start_icon = stop_icon = report_icon = None

        def create_hover_button(parent, text, icon, command):
            btn = tk.Button(parent, text=text, image=icon, command=command, **btn_style)
            btn.image = icon
            btn.bind("<Enter>", lambda e: btn.config(bg="#ffe5ec", relief="groove"))
            btn.bind("<Leave>", lambda e: btn.config(bg="#ffffff", relief="raised"))
            return btn

        self.start_btn = create_hover_button(btn_frame, "Bắt đầu", start_icon, self.start_recording)
        self.start_btn.pack(side="left", padx=40)

        self.stop_btn = create_hover_button(btn_frame, "Dừng", stop_icon, self.stop_recording)
        self.stop_btn.pack(side="left", padx=40)

        self.report_btn = create_hover_button(btn_frame, "Báo cáo", report_icon, self.export_report)
        self.report_btn.pack(side="left", padx=40)

        
        self.video_label = tk.Label(self.main_frame, bg="#c9ada7")
        self.video_label.pack(pady=10, fill="both", expand=True)

        
        self.info_label = tk.Label(
            self.main_frame,
            text="🟡 Chưa mở camera. Hãy quay lại chọn nguồn và áp dụng.",
            bg="#f2e9e4", font=("Arial", 14, "bold"), fg="#2d3436"
        )
        self.info_label.pack(pady=8)

        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        
        self.update_camera()

    
    def show_main_ui(self):
        self.select_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    def go_back_to_select(self):
        if self.recording:
            messagebox.showwarning("Đang ghi hình", "Hãy dừng ghi hình trước khi quay lại màn chọn nguồn.")
            return
        self.main_frame.pack_forget()
        self.select_frame.pack(fill="both", expand=True)

    
    def apply_camera_choice_and_go(self):
        if self.recording:
            messagebox.showwarning("Đang ghi hình", "Vui lòng dừng ghi hình trước khi đổi nguồn.")
            return

        cam_index = int(self.cam_var.get())
        if not self.init_camera(cam_index):
            alt = 1 - cam_index
            if self.init_camera(alt):
                self.cam_var.set(str(alt))
                messagebox.showwarning(
                    "Chú ý",
                    f"Không mở được camera {cam_index}. Hệ thống đã tự chuyển sang camera {alt}."
                )
            else:
                messagebox.showerror(
                    "Lỗi",
                    f"Không thể mở camera {cam_index} (và cũng không mở được {alt})."
                )
                return

    
        self.show_main_ui()
        self.info_label.config(text=f"🟢 Camera {self.current_cam_index} đang bật – sẵn sàng ghi hình.")

    
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
        else:
            self.current_cam_index = None
        return ok

    
    def update_camera(self):
        if self.cap and self.cap.isOpened() and self.main_frame.winfo_ismapped():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                h0, w0 = frame.shape[:2]
                vw = self.video_label.winfo_width()
                vh = self.video_label.winfo_height()
                if vw <= 1 or vh <= 1:
                    vw, vh = self.frame_size

                scale = min(vw / w0, vh / h0)
                new_w = max(320, int(w0 * scale))
                new_h = max(240, int(h0 * scale))

                resized_bgr = cv2.resize(frame, (new_w, new_h))
                
                if self.recording:
                    if not self.video_writer:
                        os.makedirs("recordings", exist_ok=True)
                        if not self.filename:
                            self.filename = time.strftime("recordings/video_%Y%m%d_%H%M%S.avi")
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        self.video_writer = cv2.VideoWriter(self.filename, fourcc, 20.0, (new_w, new_h))
                    if self.video_writer:
                        self.video_writer.write(resized_bgr)

                frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.frame_size = (new_w, new_h)

                if self.current_cam_index is not None:
                    self.info_label.config(
                        text=f"🟢 Camera {self.current_cam_index} đang bật – sẵn sàng ghi hình."
                    )
            else:
                self.video_label.configure(image="")
        self.root.after(15, self.update_camera)

    
    def start_recording(self):
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Lỗi", "Chưa có camera nào đang mở.")
            return
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            
            os.makedirs("recordings", exist_ok=True)
            self.filename = time.strftime("recordings/video_%Y%m%d_%H%M%S.avi")
            self.info_label.config(text=f"🔴 Đang ghi hình (camera {self.current_cam_index})...")
            messagebox.showinfo("Ghi hình", "Bắt đầu ghi hình!")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.recorded_time = int(time.time() - self.start_time) if self.start_time else 0
            if self.video_writer:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None

            self.info_label.config(
                text=f"🟡 Đã dừng. Tổng thời lượng: {self.recorded_time} giây (camera {self.current_cam_index})"
            )
            messagebox.showinfo(
                "Hoàn tất",
                f"🎬 Đã lưu video:\n{self.filename}\n\n⏱️ Thời lượng: {self.recorded_time} giây"
            )
            with open("report_log.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Cam {self.current_cam_index} | Lưu: {self.filename} | Thời lượng: {self.recorded_time} giây\n"
                )
        else:
            messagebox.showinfo("Thông báo", "Chưa bắt đầu ghi hình.")

    
    def export_report(self):
        try:
            with open("report_log.txt", "r", encoding="utf-8") as f:
                report = f.read()
        except FileNotFoundError:
            report = "Chưa có báo cáo nào được ghi lại."
        messagebox.showinfo("📄 Báo cáo ghi hình", report)

    def on_close(self):
        if self.recording:
            if not messagebox.askyesno("Đang ghi hình", "Đang ghi hình. Muốn dừng và thoát?"):
                return
            self.stop_recording()
        try:
            if hasattr(self, "cap") and self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        if getattr(self, "video_writer", None):
            try:
                self.video_writer.release()
            except Exception:
                pass
        self.root.destroy()

    def __del__(self):
        try:
            if hasattr(self, "cap") and self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        if getattr(self, "video_writer", None):
            try:
                self.video_writer.release()
            except Exception:
                pass

if __name__ == "__main__":
    root = tk.Tk()
    app = PastelCameraApp(root)
    root.mainloop()