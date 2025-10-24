import os
import sys
import time
import logging
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, BooleanVar
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Matplotlib backend phải set TRƯỚC khi import pyplot
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

LOG = logging.getLogger("PastelCameraApp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PastelCameraApp:
    """Ứng dụng ghi video + nhận diện hành vi & học sinh (YOLOv8 x2 + Kết hợp + Tự lấy tên HS + Biểu đồ)."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎀 Pastel Camera Recorder v4 🎀")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f2e9e4")

        # Camera & video
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_cam_index: Optional[int] = None
        self.frame_size = (960, 540)

        # Recording / counting
        self.counting = False
        self.start_time: Optional[float] = None
        self.recorded_time = 0

        # Detection flags & models
        self.detection_enabled = BooleanVar(value=True)
        self.student_detection_enabled = BooleanVar(value=True)
        self.model = None
        self.student_model = None

        # Model paths
        self.model_path = self.resource_path("model.pt") #nhan_dien_hanh_vi
        self.student_model_path = self.resource_path("students.pt") #nhan_dien_hoc_sinh

        # Data structures for counting
        self.behavior_counts: Counter = Counter()
        self.student_behavior_counts: defaultdict = defaultdict(Counter)
        self.last_behavior_per_student: Dict[str, Tuple[str, float]] = {}
        self.detection_cooldown = 1.5
        self.student_names: List[str] = []

        # Font tiếng Việt
        self.font_path = self.resource_path("arial.ttf")
        self.font_size = 18
        self.pil_font: Optional[ImageFont.FreeTypeFont] = None
        try:
            if os.path.exists(self.font_path):
                self.pil_font = ImageFont.truetype(self.font_path, self.font_size)
                LOG.info("Đã tải font: %s", self.font_path)
            else:
                LOG.warning("Không tìm thấy font %s. Sẽ dùng CV2 (sai dấu).", self.font_path)
        except Exception as e:
            LOG.warning("Lỗi tải font %s: %s. Sẽ dùng CV2 (sai dấu).", self.font_path, e)

        self._check_dependencies_and_load_models()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_camera()

    @staticmethod
    def resource_path(relative_path: str) -> str:
        try:
            base_path = sys._MEIPASS  # PyInstaller
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def _check_dependencies_and_load_models(self):
        self.ultralytics_available = self._is_module_available("ultralytics")
        if not self.ultralytics_available:
            messagebox.showerror("Lỗi Thiếu Thư Viện", "Cần cài đặt ultralytics: pip install ultralytics")
            self.detection_enabled.set(False)
            self.student_detection_enabled.set(False)
            self.root.quit()
            return

        models_loaded = True
        if self.detection_enabled.get() and not self.load_model():
            self.detection_enabled.set(False)
            models_loaded = False
        if self.student_detection_enabled.get() and not self.load_student_model():
            self.student_detection_enabled.set(False)
            models_loaded = False

        if not models_loaded:
            messagebox.showwarning("Cảnh Báo", "Một hoặc cả hai model không tải được.")

    @staticmethod
    def _is_module_available(mod_name: str) -> bool:
        try:
            __import__(mod_name)
            return True
        except Exception:
            return False

    def _load_yolo_model(self, path: str, model_type: str) -> Optional[object]:
        if not os.path.exists(path):
            LOG.error("Model %s không tồn tại: %s", model_type, path)
            messagebox.showerror("Lỗi", f"Không tìm thấy file model {model_type}")
            return None
        if not self.ultralytics_available:
            LOG.error("Thiếu ultralytics")
            return None

        loading = None
        model = None
        try:
            loading = tk.Toplevel(self.root)
            loading.title(f"Tải model {model_type}")
            loading.geometry("320x100")
            loading.transient(self.root)
            tk.Label(loading, text=f"Đang tải model {model_type}...", font=("Arial", 11)).pack(pady=20)
            loading.update()

            from ultralytics import YOLO

            model = YOLO(path)
            LOG.info("Đã tải model %s từ %s", model_type, path)
        except Exception as e:
            LOG.exception("Lỗi tải model %s: %s", model_type, e)
            messagebox.showerror("Lỗi", f"Không tải được model {model_type}: {e}")
            model = None
        finally:
            if loading:
                loading.destroy()
        return model

    def load_model(self) -> bool:
        self.model = self._load_yolo_model(self.model_path, "hành vi")
        return self.model is not None

    def load_student_model(self) -> bool:
        model_temp = self._load_yolo_model(self.student_model_path, "học sinh")
        if model_temp:
            if hasattr(model_temp, "names") and isinstance(model_temp.names, dict):
                n = len(model_temp.names)
                if all(i in model_temp.names for i in range(n)):
                    self.student_names = [model_temp.names[i] for i in range(n)]
                    self.student_model = model_temp
                    LOG.info("Đã tải student model (%d lớp) và lấy tên từ model.", n)
                    return True
                messagebox.showerror("Lỗi Model", "ID lớp model học sinh không liên tục (không phải 0..n-1).")
            else:
                messagebox.showerror("Lỗi Model", "Model học sinh không có thuộc tính names (dict).")
        self.student_model = None
        return False

    def draw_text_pil(
        self, img: np.ndarray, text: str, pos: Tuple[int, int], color_rgb: Tuple[int, int, int]
    ) -> np.ndarray:
        if self.pil_font:
            try:
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text(pos, text, font=self.pil_font, fill=color_rgb)
                return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                LOG.warning("Lỗi vẽ PIL text: %s", e)

        # Fallback OpenCV (sai dấu tiếng Việt)
        cv2_color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.putText(
            img, text, (pos[0], pos[1] + self.font_size), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv2_color_bgr, 1, cv2.LINE_AA
        )
        return img

    def _build_ui(self):
        # Select frame
        self.select_frame = tk.Frame(self.root, bg="#f2e9e4")
        self.select_frame.pack(fill="both", expand=True)

        tk.Label(self.select_frame, text="Chọn nguồn camera", bg="#f2e9e4", font=("Arial", 18, "bold")).pack(
            pady=(30, 10)
        )

        top_frame = tk.Frame(self.select_frame, bg="#f2e9e4")
        top_frame.pack(pady=(15, 5))
        tk.Label(top_frame, text="Nguồn camera:", bg="#f2e9e4", font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=(0, 10)
        )

        self.cam_var = tk.StringVar(value="0")
        tk.Radiobutton(top_frame, text="Camera máy tính (0)", variable=self.cam_var, value="0", bg="#f2e9e4",
                       font=("Arial", 11)).grid(row=0, column=1, padx=10)
        tk.Radiobutton(top_frame, text="Camera rời (1)", variable=self.cam_var, value="1", bg="#f2e9e4",
                       font=("Arial", 11)).grid(row=0, column=2, padx=10)

        tk.Button(
            self.select_frame,
            text="Áp dụng nguồn",
            command=self.apply_camera_choice_and_go,
            bg="#ffffff",
            activebackground="#fde2e4",
            relief="raised",
            bd=2,
            font=("Arial", 12, "bold"),
            cursor="hand2",
        ).pack(pady=20)

        # Main frame
        self.main_frame = tk.Frame(self.root, bg="#f2e9e4")

        top_bar = tk.Frame(self.main_frame, bg="#f2e9e4")
        top_bar.pack(pady=(10, 0), fill="x")

        self.back_btn = tk.Button(
            top_bar,
            text="⟵ Quay lại chọn nguồn",
            command=self.go_back_to_select,
            bg="#ffffff",
            activebackground="#fde2e4",
            relief="raised",
            bd=2,
            font=("Arial", 11, "bold"),
            cursor="hand2",
        )
        self.back_btn.pack(side="left", padx=20)

        self.student_detection_cb = tk.Checkbutton(
            top_bar,
            text="Nhận diện học sinh",
            variable=self.student_detection_enabled,
            bg="#f2e9e4",
            font=("Arial", 11, "bold"),
            command=self.toggle_student_detection,
        )
        self.student_detection_cb.pack(side="right", padx=10)
        if self.student_detection_enabled.get():
            self.student_detection_cb.select()

        self.detection_cb = tk.Checkbutton(
            top_bar,
            text="Nhận diện hành vi",
            variable=self.detection_enabled,
            bg="#f2e9e4",
            font=("Arial", 11, "bold"),
            command=self.toggle_detection,
        )
        self.detection_cb.pack(side="right", padx=10)
        if self.detection_enabled.get():
            self.detection_cb.select()

        btn_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        btn_frame.pack(pady=12, fill="x")

        btn_style = {
            "width": 14,
            "height": 3,
            "compound": "top",
            "bg": "#fff",
            "activebackground": "#fde2e4",
            "relief": "raised",
            "bd": 2,
            "font": ("Arial", 11, "bold"),
            "cursor": "hand2",
        }

        start_icon = self._safe_load_icon("start.png", (64, 64))
        stop_icon = self._safe_load_icon("stop.png", (64, 64))
        report_icon = self._safe_load_icon("report.png", (64, 64))

        self.start_btn = tk.Button(btn_frame, text="Bắt đầu", image=start_icon, command=self.start_counting, **btn_style)
        self.start_btn.image = start_icon
        self.start_btn.pack(side="left", padx=40)

        self.stop_btn = tk.Button(btn_frame, text="Dừng", image=stop_icon, command=self.stop_counting, **btn_style)
        self.stop_btn.image = stop_icon
        self.stop_btn.pack(side="left", padx=40)

        self.report_btn = tk.Button(
            btn_frame, text="Báo cáo", image=report_icon, command=self.show_behavior_report, **btn_style
        )
        self.report_btn.image = report_icon
        self.report_btn.pack(side="left", padx=40)

        self.video_label = tk.Label(self.main_frame, bg="#c9ada7")
        self.video_label.pack(pady=10, fill="both", expand=True)

        self.count_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        self.count_frame.pack(pady=5, fill="x")

        self.count_label = tk.Label(
            self.count_frame,
            text="Đang chờ bắt đầu đếm...",
            justify=tk.LEFT,
            bg="#f2e9e4",
            font=("Arial", 12),
            fg="#2d3436",
        )
        self.count_label.pack(pady=5)

        self.info_label = tk.Label(
            self.main_frame,
            text="🟡 Chưa mở camera.",
            bg="#f2e9e4",
            font=("Arial", 14, "bold"),
            fg="#2d3436",
        )
        self.info_label.pack(pady=8)

    def _safe_load_icon(self, filename: str, size: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        try:
            path = self.resource_path(filename)
            if not os.path.exists(path):
                return None
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            img = Image.open(path).resize(size, resample)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            LOG.warning("Không tải được icon %s: %s", filename, e)
            return None

    def _iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def _associate_detections_iou(self, behavior_results, student_results, iou_threshold=0.3):
        associations: List[Tuple[str, str, float, List[int]]] = []
        behaviors: List[List[object]] = []
        students: List[List[object]] = []

        # Hành vi
        if behavior_results and hasattr(behavior_results[0], "boxes"):
            b_names_map = behavior_results[0].names
            for box in behavior_results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id in b_names_map:
                    behaviors.append([b_names_map[cls_id], conf, coords, -1])

        # Học sinh
        if student_results and hasattr(student_results[0], "boxes") and self.student_names:
            s_names_map = self.student_names
            student_conf_threshold = 0.5
            for box in student_results[0].boxes:
                conf = float(box.conf[0])
                if conf > student_conf_threshold:
                    coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    cls_id = int(box.cls[0])
                    if 0 <= cls_id < len(s_names_map):
                        students.append([s_names_map[cls_id], coords])
                    else:
                        LOG.warning("ID lớp HS không hợp lệ: %s", cls_id)

        # Ghép IoU
        if not students:
            for b_name, b_conf, b_box, _ in behaviors:
                associations.append(("Unknown", b_name, b_conf, b_box))
            return associations

        iou_matrix = np.zeros((len(behaviors), len(students)))
        for i in range(len(behaviors)):
            for j in range(len(students)):
                iou_matrix[i, j] = self._iou(behaviors[i][2], students[j][1])

        for i in range(len(behaviors)):
            best_iou = 0.0
            best_student_idx = -1
            for j in range(len(students)):
                if iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                    best_iou = iou_matrix[i, j]
                    best_student_idx = j
            student_name = students[best_student_idx][0] if best_student_idx != -1 else "Unknown"
            associations.append((student_name, behaviors[i][0], behaviors[i][1], behaviors[i][2]))
        return associations

    def detect_objects_and_count(self, frame: np.ndarray) -> np.ndarray:
        annotated_frame = frame.copy()
        now = time.time()

        behavior_results = None
        student_results = None
        b_conf_threshold = 0.3
        s_conf_threshold = 0.5

        if self.model and self.detection_enabled.get():
            try:
                behavior_results = self.model(frame, conf=b_conf_threshold, verbose=False)
            except Exception as e:
                LOG.error("Lỗi model hành vi: %s", e)

        if self.student_model and self.student_detection_enabled.get() and self.student_names:
            try:
                student_results = self.student_model(frame, conf=s_conf_threshold, verbose=False)
            except Exception as e:
                LOG.error("Lỗi model học sinh: %s", e)

        associations = self._associate_detections_iou(behavior_results, student_results, iou_threshold=0.3)

        processed_students_this_frame = set()
        for student_name, behavior_name, behavior_conf, behavior_box in associations:
            x1, y1, x2, y2 = behavior_box
            if student_name != "Unknown":
                color_bgr = (0, 255, 0)
                color_rgb = (0, 255, 0)
            else:
                color_bgr = (0, 255, 255)
                color_rgb = (255, 255, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)

            label_text = f"{student_name}: {behavior_name} ({behavior_conf:.2f})"
            text_y = y1 - self.font_size - 5 if y1 - self.font_size - 5 > 0 else y1 + 5
            annotated_frame = self.draw_text_pil(annotated_frame, label_text, (x1, text_y), color_rgb)

            if self.counting and student_name != "Unknown":
                last_behavior, last_time = self.last_behavior_per_student.get(student_name, (None, 0.0))
                if (now - last_time > self.detection_cooldown) and (behavior_name != last_behavior):
                    if student_name not in processed_students_this_frame:
                        self.behavior_counts[behavior_name] += 1
                        self.student_behavior_counts[student_name][behavior_name] += 1
                        self.last_behavior_per_student[student_name] = (behavior_name, now)
                        processed_students_this_frame.add(student_name)

        if self.counting and processed_students_this_frame:
            self.update_count_display()

        return annotated_frame

    def toggle_detection(self):
        if self.detection_enabled.get():
            if not self.ultralytics_available:
                self.detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu ultralytics")
                return
            if self.model is None and not self.load_model():
                self.detection_enabled.set(False)
                return
            messagebox.showinfo("Thông báo", "Đã bật nhận diện hành vi.")
        else:
            messagebox.showinfo("Thông báo", "Đã tắt nhận diện hành vi.")

    def toggle_student_detection(self):
        if self.student_detection_enabled.get():
            if not self.ultralytics_available:
                self.student_detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu ultralytics")
                return
            if self.student_model is None and not self.load_student_model():
                self.student_detection_enabled.set(False)
                return
            messagebox.showinfo("Thông báo", "Đã bật nhận diện học sinh.")
        else:
            messagebox.showinfo("Thông báo", "Đã tắt nhận diện học sinh.")

    def start_counting(self):
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Lỗi", "Chưa có camera.")
            return
        if not self.detection_enabled.get() and not self.student_detection_enabled.get():
            messagebox.showerror("Lỗi", "Bật ít nhất một nhận diện.")
            return
        if self.counting:
            messagebox.showinfo("Thông báo", "Đã đang đếm.")
            return

        self.behavior_counts = Counter()
        self.student_behavior_counts = defaultdict(Counter)
        self.last_behavior_per_student = {}
        self.counting = True
        self.start_time = time.time()
        self.recorded_time = 0

        status = f"🔴 Đang đếm... (camera {self.current_cam_index})"
        self.count_label.config(text="Bắt đầu đếm...")
        messagebox.showinfo("Thông báo", "Bắt đầu đếm!")
        self.info_label.config(text=status)

    def stop_counting(self):
        if not self.counting:
            messagebox.showinfo("Thông báo", "Chưa bắt đầu đếm.")
            return

        self.counting = False
        self.recorded_time = int(time.time() - self.start_time) if self.start_time else 0
        status = f"🟡 Đã dừng đếm. Thời gian: {self.recorded_time} giây (camera {self.current_cam_index})"
        self.info_label.config(text=status)
        self.last_behavior_per_student = {}
        self.count_label.config(text="Đã dừng đếm. Xem báo cáo.")

        self.show_behavior_report()
        self.save_behavior_report()

    def update_count_display(self):
        if not self.counting:
            return

        lines = ["Đang đếm..."]

        if self.student_behavior_counts:
            lines.append("--- Theo Học Sinh (Gần đây) ---")
            recent = sorted(self.last_behavior_per_student.items(), key=lambda i: i[1][1], reverse=True)[:5]
            if recent:
                for name, (beh, _) in recent:
                    counts = self.student_behavior_counts.get(name)
                    if counts:
                        total = sum(counts.values())
                        lines.append(f"{name}: {beh} (Tổng: {total})")
            else:
                lines.append("(Chưa ghi nhận HS)")

        if self.behavior_counts:
            lines.append("--- Tổng Hành Vi ---")
            for b, c in sorted(self.behavior_counts.items()):
                lines.append(f"- {b}: {c}")
        elif not self.student_behavior_counts:
            lines.append("(Chưa ghi nhận hành vi)")

        self.count_label.config(text="\n".join(lines))

    def save_behavior_report(self):
        os.makedirs("reports", exist_ok=True)
        fname = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.txt")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"BÁO CÁO HÀNH VI - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"T.gian đếm: {self.recorded_time}s\n")
                f.write(f"Camera: {self.current_cam_index}\n\n")

                if self.student_behavior_counts:
                    f.write("THEO HỌC SINH:\n=====================\n\n")
                    for student in sorted(self.student_behavior_counts.keys()):
                        behaviors = self.student_behavior_counts[student]
                        f.write(f"Học sinh: {student}\n")
                        if behaviors:
                            for b, c in sorted(behaviors.items()):
                                f.write(f"   - {b}: {c} lần\n")
                        else:
                            f.write("   Không phát hiện.\n")
                        f.write("\n")
                    f.write("\nTỔNG HỢP (ĐÃ NHẬN DIỆN HS):\n=====================\n\n")
                else:
                    f.write("TỔNG HỢP:\n=====================\n\n")

                if self.behavior_counts:
                    for b, c in sorted(self.behavior_counts.items()):
                        f.write(f"- {b}: {c} lần\n")
                else:
                    f.write("Không phát hiện.\n")

            messagebox.showinfo("Thông báo", f"Đã lưu báo cáo: {fname}")
        except Exception as e:
            LOG.exception("Lỗi save report: %s", e)
            messagebox.showerror("Lỗi", f"Không lưu được báo cáo: {e}")

    # ====== Vẽ biểu đồ trên cửa sổ báo cáo ======
    def _create_behavior_chart(self, parent_frame):
        """Tạo và nhúng biểu đồ tần suất hành vi vào frame."""
        try:
            if not self.behavior_counts:
                tk.Label(parent_frame, text="Không có dữ liệu hành vi để vẽ biểu đồ.", bg="#f2e9e4").pack(pady=10)
                return None

            labels = list(self.behavior_counts.keys())
            values = list(self.behavior_counts.values())

            fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
            fig.patch.set_facecolor("#f2e9e4")
            ax.set_facecolor("#ffffff")

            bars = ax.bar(labels, values)

            ax.set_xlabel("Hành vi", fontweight="bold")
            ax.set_ylabel("Số lần xuất hiện", fontweight="bold")
            ax.set_title("Thống kê tần suất hành vi", fontweight="bold")
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

            ax.bar_label(bars, padding=3)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()
            return canvas
        except Exception as e:
            LOG.exception("Lỗi khi tạo biểu đồ: %s", e)
            tk.Label(parent_frame, text=f"Lỗi khi tạo biểu đồ:\n{e}", fg="red", bg="#f2e9e4").pack(pady=10)
            return None

    def show_behavior_report(self):
        """Hiển thị báo cáo (text + biểu đồ) trong cửa sổ mới."""
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Báo cáo hành vi")
        report_window.geometry("800x850")
        report_window.configure(bg="#f2e9e4")

        chart_frame = tk.Frame(report_window, bg="#f2e9e4", height=350)
        chart_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 10))
        chart_frame.pack_propagate(False)

        text_frame = tk.Frame(report_window, bg="#f2e9e4")
        text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        # Vẽ chart
        self.report_chart_canvas = self._create_behavior_chart(chart_frame)

        # Text report
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget = tk.Text(text_frame, wrap="word", font=("Arial", 11), bg="#ffffff")
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        report = ["BÁO CÁO CHI TIẾT\n", f"Thời gian đếm: {self.recorded_time} giây\n"]
        if self.student_behavior_counts:
            report.append("\nTHEO HỌC SINH:\n=====================\n")
            for student in sorted(self.student_behavior_counts.keys()):
                behaviors = self.student_behavior_counts[student]
                report.append(f"Học sinh: {student}\n")
                if behaviors:
                    for b, c in sorted(behaviors.items()):
                        report.append(f"   - {b}: {c} lần\n")
                else:
                    report.append("   Không phát hiện.\n")
                report.append("\n")
            report.append("\nTỔNG HỢP (ĐÃ NHẬN DIỆN HS):\n=====================\n")
        else:
            report.append("\nTỔNG HỢP:\n=====================\n")

        if self.behavior_counts:
            for b, c in sorted(self.behavior_counts.items()):
                report.append(f"- {b}: {c} lần\n")
        else:
            report.append("Không phát hiện hành vi nào.")

        text_widget.insert("1.0", "".join(report))
        text_widget.config(state="disabled")

        # Buttons
        button_frame = tk.Frame(report_window, bg="#f2e9e4")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 15))

        tk.Button(
            button_frame,
            text="Lưu báo cáo Text",
            command=self.save_behavior_report,
            bg="#ffffff",
            activebackground="#fde2e4",
            relief="raised",
            bd=2,
            font=("Arial", 11, "bold"),
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=20)

        try:
            import pandas  # noqa: F401

            tk.Button(
                button_frame,
                text="Xuất Excel",
                command=self.export_excel_report,
                bg="#ffffff",
                activebackground="#d0f0c0",
                relief="raised",
                bd=2,
                font=("Arial", 11, "bold"),
                cursor="hand2",
            ).pack(side=tk.LEFT, padx=20)
        except ImportError:
            pass
        except Exception:
            pass

        # (Tuỳ chọn) đóng figure khi đóng cửa sổ để giải phóng bộ nhớ
        def on_report_close():
            try:
                if getattr(self, "report_chart_canvas", None):
                    plt.close(self.report_chart_canvas.figure)
            except Exception:
                pass
            report_window.destroy()

        report_window.protocol("WM_DELETE_WINDOW", on_report_close)

    def export_excel_report(self):
        try:
            import pandas as pd  # noqa
            import openpyxl  # noqa

            os.makedirs("reports", exist_ok=True)
            excel_file = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.xlsx")

            summary_data = [{"Hành vi": b, "Số lần xuất hiện": c} for b, c in sorted(self.behavior_counts.items())]
            summary_df = pd.DataFrame(summary_data)

            student_rows: List[Dict[str, object]] = []
            if self.student_behavior_counts:
                for student in sorted(self.student_behavior_counts.keys()):
                    behaviors = self.student_behavior_counts[student]
                    if behaviors:
                        for b, c in sorted(behaviors.items()):
                            student_rows.append({"Học sinh": student, "Hành vi": b, "Số lần xuất hiện": c})
            student_df = pd.DataFrame(student_rows)

            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Tổng hợp (All)", index=False)
                if not student_df.empty:
                    student_df.to_excel(writer, sheet_name="Theo học sinh (Known)", index=False)

            messagebox.showinfo("Thông báo", f"Đã xuất Excel: {excel_file}")
        except ImportError:
            messagebox.showerror("Lỗi", "Cần pandas và openpyxl: pip install pandas openpyxl")
        except Exception as e:
            LOG.exception("Lỗi export excel: %s", e)
            messagebox.showerror("Lỗi", f"Không xuất được Excel: {e}")

    def apply_camera_choice_and_go(self):
        if self.counting:
            messagebox.showwarning("Đang đếm", "Vui lòng dừng đếm trước.")
            return

        cam_index = int(self.cam_var.get())
        if not self.init_camera(cam_index):
            alt = 1 - cam_index
            if self.init_camera(alt):
                self.cam_var.set(str(alt))
                messagebox.showwarning("Chú ý", f"Không mở được camera {cam_index}. Đã thử chuyển sang {alt}.")
            else:
                messagebox.showerror("Lỗi", f"Không thể mở camera {cam_index} và {alt}.")
                return

        self.show_main_ui()

        status = f"🟢 Camera {self.current_cam_index} đang bật"
        parts = []
        if self.student_detection_enabled.get() and self.student_model:
            parts.append("Nhận diện học sinh")
        if self.detection_enabled.get() and self.model:
            parts.append("Nhận diện hành vi")
        status += f" với {' và '.join(parts)} đang hoạt động" if parts else " – sẵn sàng đếm."
        self.info_label.config(text=status)

    def init_camera(self, index: int) -> bool:
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            if not self.cap.isOpened():
                LOG.warning("Không mở được camera %s ngay.", index)
                self.current_cam_index = None
                return False
            self.current_cam_index = index
            LOG.info("Mở camera %s thành công", index)
            return True
        except Exception as e:
            LOG.exception("Lỗi init_camera: %s", e)
            return False

    def go_back_to_select(self):
        if self.counting:
            messagebox.showwarning("Đang đếm", "Hãy dừng đếm trước.")
            return
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.main_frame.pack_forget()
        self.select_frame.pack(fill="both", expand=True)
        self.info_label.config(text="🟡 Chưa mở camera.")

    def show_main_ui(self):
        self.select_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    def update_camera(self):
        try:
            if self.cap and self.cap.isOpened() and self.main_frame.winfo_ismapped():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h0, w0 = frame.shape[:2]
                    label_w = self.video_label.winfo_width()
                    label_h = self.video_label.winfo_height()
                    if label_w < 10 or label_h < 10:
                        label_w, label_h = 960, 540
                    scale = min(label_w / w0, label_h / h0) if w0 > 0 and h0 > 0 else 1.0
                    new_w = max(320, int(w0 * scale))
                    new_h = max(240, int(h0 * scale))
                    resized_bgr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    try:
                        annotated = self.detect_objects_and_count(resized_bgr)
                    except Exception as e:
                        LOG.exception("Lỗi detect: %s", e)
                        annotated = resized_bgr

                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                    self.frame_size = (new_w, new_h)

                    status = f"🟢 Camera {self.current_cam_index} bật"
                    if self.counting:
                        status += " (Đang đếm...)"
                    else:
                        parts = []
                        if self.student_detection_enabled.get() and self.student_model:
                            parts.append("Nhận diện HS: BẬT")
                        if self.detection_enabled.get() and self.model:
                            parts.append("Nhận diện HV: BẬT")
                        status += f" ({', '.join(parts)})" if parts else " – Sẵn sàng."
                    self.info_label.config(text=status)
                else:
                    self.video_label.configure(image="")
        except Exception as e:
            LOG.exception("Lỗi update_camera: %s", e)
        finally:
            self.root.after(30, self.update_camera)

    def on_close(self):
        if self.counting:
            if not messagebox.askyesno("Đang đếm", "Dừng đếm và thoát?"):
                return
            self.stop_counting()
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def __del__(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    # DPI scaling cho Windows
    if sys.platform == "win32":
        try:
            from ctypes import windll

            windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            LOG.warning("Không thể set DPI awareness: %s", e)

    root = tk.Tk()
    app = PastelCameraApp(root)
    root.mainloop()
