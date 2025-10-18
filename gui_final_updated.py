#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pastel Camera Recorder (Refactored)
Giữ nguyên chức năng của bản gốc, nhưng mã được tách thành các phần rõ ràng hơn,
sửa lỗi indentation, thêm logging nhẹ, và cải thiện xử lý lỗi khi load model / camera.
"""

import os
import sys
import time
import logging
from collections import Counter, defaultdict
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, BooleanVar
from PIL import Image, ImageTk

# Lưu ý: torch / ultralytics / torchvision chỉ cần import khi có trong môi trường.
# Chúng sẽ được import động trong các phương thức tương ứng.

LOG = logging.getLogger("PastelCameraApp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PastelCameraApp:
    """Ứng dụng ghi video + nhận diện hành vi & học sinh (refactored)."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎀 Pastel Camera Recorder 🎀")
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

        # Model paths (cấu hình theo file chạy)
        self.model_path = self.resource_path("model.pt")
        self.student_model_path = self.resource_path("face_classifier.pth")

        # Data structures for counting
        self.behavior_counts: Counter = Counter()
        self.student_behavior_counts: defaultdict = defaultdict(Counter)
        self.current_behaviors: dict = {}
        self.current_student: Optional[str] = None

        # Timing / cooldowns
        self.last_detected: Optional[str] = None
        self.last_detected_time = 0.0
        self.detection_cooldown = 1.0

        self.last_student_time = 0.0
        self.student_cooldown = 2.0

        # Student name list (placeholder; thay đổi khi train model có tên)
        self.student_names = [f"Học sinh {i+1}" for i in range(10)]

        # Kiểm tra dependencies & load models (không crash app nếu thiếu)
        self._check_dependencies_and_load_models()

        # Build UI
        self._build_ui()

        # Start camera update loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_camera()

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def resource_path(relative_path: str) -> str:
        """Lấy đường dẫn tuyệt đối đến tài nguyên (hỗ trợ PyInstaller)."""
        try:
            base_path = sys._MEIPASS  # type: ignore
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    # -------------------------
    # Dependency & model loading
    # -------------------------
    def _check_dependencies_and_load_models(self):
        """Kiểm tra thư viện cần thiết và cố gắng tải mô hình (nếu có)."""
        # Kiểm tra torch + ultralytics (không raise, chỉ cảnh báo và disable features)
        self.torch_available = self._is_module_available("torch")
        self.ultralytics_available = self._is_module_available("ultralytics")

        if not (self.torch_available and self.ultralytics_available):
            messagebox.showwarning(
                "Cảnh báo",
                "Thiếu thư viện nhận diện: torch và/hoặc ultralytics.\n"
                "Bạn có thể cài:\n\npip install torch ultralytics\n\n"
                "Các tính năng nhận diện sẽ bị tắt nếu thiếu."
            )
            # Nếu thiếu, tắt chế độ nhận diện
            if not self.torch_available:
                self.student_detection_enabled.set(False)
            if not (self.torch_available and self.ultralytics_available):
                self.detection_enabled.set(False)

        # Thử load mô hình hành vi / học sinh (không bắt buộc phải thành công)
        if self.detection_enabled.get():
            self.load_model()
        if self.student_detection_enabled.get():
            self.load_student_model()

    @staticmethod
    def _is_module_available(mod_name: str) -> bool:
        try:
            __import__(mod_name)
            return True
        except Exception:
            return False

    def load_model(self) -> bool:
        """Tải mô hình hành vi (YOLOv8 hoặc fallback YOLOv5) từ file model_path."""
        if not os.path.exists(self.model_path):
            LOG.warning("Model hành vi không tồn tại: %s", self.model_path)
            self.detection_enabled.set(False)
            return False

        if not (self.torch_available or self.ultralytics_available):
            LOG.warning("Không có thư viện để load model hành vi.")
            self.detection_enabled.set(False)
            return False

        # Hiển thị popup nhỏ trong UI khi load (nếu root đã sẵn sàng)
        loading = None
        try:
            loading = tk.Toplevel(self.root)
            loading.title("Đang tải mô hình hành vi")
            loading.geometry("320x100")
            loading.transient(self.root)
            tk.Label(loading, text="Đang tải mô hình hành vi...\nVui lòng đợi...", font=("Arial", 11)).pack(pady=20)
            loading.update()

            # Thử YOLOv8 (ultralytics)
            if self.ultralytics_available:
                try:
                    from ultralytics import YOLO  # local import
                    self.model = YOLO(self.model_path)
                    LOG.info("Đã tải model hành vi (YOLOv8) từ %s", self.model_path)
                    return True
                except Exception as e:
                    LOG.warning("Không thể load YOLOv8: %s", e)

            # Fallback: torch.hub yolov5
            if self.torch_available:
                try:
                    import torch
                    self.model = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path, verbose=False)
                    # đặt confidence mặc định nếu có
                    try:
                        self.model.conf = 0.45
                    except Exception:
                        pass
                    LOG.info("Đã tải model hành vi (YOLOv5) từ %s", self.model_path)
                    return True
                except Exception as e:
                    LOG.warning("Không thể load YOLOv5: %s", e)

            messagebox.showerror("Lỗi", "Không thể tải mô hình hành vi. Kiểm tra định dạng và thư viện.")
            self.detection_enabled.set(False)
            return False
        finally:
            if loading:
                loading.destroy()

    def load_student_model(self) -> bool:
        """Tải mô hình nhận diện học sinh (ResNet18) hoặc YOLO tùy file."""
        if not os.path.exists(self.student_model_path):
            LOG.info("Không tìm thấy student model: %s", self.student_model_path)
            self.student_detection_enabled.set(False)
            return False

        if not self.torch_available:
            LOG.warning("torch không có - không thể load student model.")
            self.student_detection_enabled.set(False)
            return False

        loading = None
        try:
            loading = tk.Toplevel(self.root)
            loading.title("Đang tải mô hình học sinh")
            loading.geometry("320x100")
            loading.transient(self.root)
            tk.Label(loading, text="Đang tải mô hình học sinh...\nVui lòng đợi...", font=("Arial", 11)).pack(pady=20)
            loading.update()

            # Thử load như ResNet (torch)
            try:
                import torch
                import torchvision.models as models

                # Lưu ý: bạn phải set đúng num_classes & tên lớp tương ứng với model đã train
                num_classes = max(10, len(self.student_names))
                model = models.resnet18(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                state = torch.load(self.student_model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state)
                model.eval()
                self.student_model = model
                LOG.info("Đã tải student model dạng ResNet từ %s", self.student_model_path)
                return True
            except Exception as e:
                LOG.warning("Không thể load student model dưới dạng ResNet: %s", e)

            # Fallback YOLO (ultralytics or yolov5)
            if self.ultralytics_available:
                try:
                    from ultralytics import YOLO
                    self.student_model = YOLO(self.student_model_path)
                    LOG.info("Đã tải student model (YOLOv8) từ %s", self.student_model_path)
                    return True
                except Exception as e:
                    LOG.warning("Không thể load student YOLOv8: %s", e)
            if self.torch_available:
                try:
                    import torch
                    self.student_model = torch.hub.load("ultralytics/yolov5", "custom", path=self.student_model_path, verbose=False)
                    LOG.info("Đã tải student model (YOLOv5) từ %s", self.student_model_path)
                    return True
                except Exception as e:
                    LOG.warning("Không thể load student YOLOv5: %s", e)

            messagebox.showwarning("Cảnh báo", "Không thể tải mô hình học sinh. Tính năng nhận diện học sinh sẽ bị tắt.")
            self.student_detection_enabled.set(False)
            return False
        finally:
            if loading:
                loading.destroy()

    # -------------------------
    # UI Building
    # -------------------------
    def _build_ui(self):
        """Tạo UI chính (2 màn: chọn nguồn & main)."""
        # --- Select frame ---
        self.select_frame = tk.Frame(self.root, bg="#f2e9e4")
        self.select_frame.pack(fill="both", expand=True)

        title = tk.Label(self.select_frame, text="Chọn nguồn camera", bg="#f2e9e4", font=("Arial", 18, "bold"))
        title.pack(pady=(30, 10))

        top_frame = tk.Frame(self.select_frame, bg="#f2e9e4")
        top_frame.pack(pady=(15, 5))

        tk.Label(top_frame, text="Nguồn camera:", bg="#f2e9e4", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=(0, 10))
        self.cam_var = tk.StringVar(value="0")

        rb_internal = tk.Radiobutton(top_frame, text="Camera máy tính (0)", variable=self.cam_var, value="0", bg="#f2e9e4", font=("Arial", 11))
        rb_external = tk.Radiobutton(top_frame, text="Camera rời (1)", variable=self.cam_var, value="1", bg="#f2e9e4", font=("Arial", 11))
        rb_internal.grid(row=0, column=1, padx=10)
        rb_external.grid(row=0, column=2, padx=10)

        apply_btn = tk.Button(self.select_frame, text="Áp dụng nguồn", command=self.apply_camera_choice_and_go,
                              bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
        apply_btn.pack(pady=20)

        # --- Main frame ---
        self.main_frame = tk.Frame(self.root, bg="#f2e9e4")

        # Top bar
        top_bar = tk.Frame(self.main_frame, bg="#f2e9e4")
        top_bar.pack(pady=(10, 0), fill="x")

        self.back_btn = tk.Button(top_bar, text="⟵ Quay lại chọn nguồn", command=self.go_back_to_select,
                                  bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, font=("Arial", 11, "bold"), cursor="hand2")
        self.back_btn.pack(side="left", padx=20)

        # Checkbuttons
        self.student_detection_cb = tk.Checkbutton(top_bar, text="Nhận diện học sinh", variable=self.student_detection_enabled,
                                                   bg="#f2e9e4", font=("Arial", 11, "bold"), command=self.toggle_student_detection)
        self.student_detection_cb.pack(side="right", padx=10)
        if self.student_detection_enabled.get():
            self.student_detection_cb.select()

        self.detection_cb = tk.Checkbutton(top_bar, text="Nhận diện hành vi", variable=self.detection_enabled,
                                           bg="#f2e9e4", font=("Arial", 11, "bold"), command=self.toggle_detection)
        self.detection_cb.pack(side="right", padx=10)
        if self.detection_enabled.get():
            self.detection_cb.select()

        # Buttons frame
        btn_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        btn_frame.pack(pady=12, fill="x")

        btn_style = {"width": 14, "height": 3, "compound": "top",
                     "bg": "#fff", "activebackground": "#fde2e4",
                     "relief": "raised", "bd": 2, "font": ("Arial", 11, "bold"),
                     "cursor": "hand2"}

        # Icons load (improve resilience)
        start_icon = self._safe_load_icon("start.png", (64, 64))
        stop_icon = self._safe_load_icon("stop.png", (64, 64))
        report_icon = self._safe_load_icon("report.png", (64, 64))

        self.start_btn = tk.Button(btn_frame, text="Bắt đầu", image=start_icon, command=self.start_counting, **btn_style)
        self.start_btn.image = start_icon
        self.start_btn.pack(side="left", padx=40)

        self.stop_btn = tk.Button(btn_frame, text="Dừng", image=stop_icon, command=self.stop_counting, **btn_style)
        self.stop_btn.image = stop_icon
        self.stop_btn.pack(side="left", padx=40)

        self.report_btn = tk.Button(btn_frame, text="Báo cáo", image=report_icon, command=self.show_behavior_report, **btn_style)
        self.report_btn.image = report_icon
        self.report_btn.pack(side="left", padx=40)

        # Video display
        self.video_label = tk.Label(self.main_frame, bg="#c9ada7")
        self.video_label.pack(pady=10, fill="both", expand=True)

        # Count frame
        self.count_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        self.count_frame.pack(pady=5, fill="x")

        self.count_label = tk.Label(self.count_frame, text="Đang chờ bắt đầu đếm...", bg="#f2e9e4", font=("Arial", 12), fg="#2d3436")
        self.count_label.pack(pady=5)

        self.info_label = tk.Label(self.main_frame, text="🟡 Chưa mở camera. Hãy quay lại chọn nguồn và áp dụng.",
                                   bg="#f2e9e4", font=("Arial", 14, "bold"), fg="#2d3436")
        self.info_label.pack(pady=8)

    def _safe_load_icon(self, name: str, size: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        """Tải icon nếu có, trả về None nếu lỗi (không raise)."""
        try:
            path = self.resource_path(name)
            img = Image.open(path).resize(size)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    # -------------------------
    # Camera control
    # -------------------------
    def init_camera(self, index: int) -> bool:
        """Khởi tạo camera - trả về True nếu thành công."""
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                try:
                    self.cap.release()
                except Exception:
                    pass
            # Chọn backend phù hợp
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(index)
            ok = self.cap.isOpened()
            if ok:
                self.current_cam_index = index
                LOG.info("Mở camera %s thành công", index)
            else:
                self.current_cam_index = None
                LOG.warning("Không mở được camera %s", index)
            return ok
        except Exception as e:
            LOG.exception("Lỗi init_camera: %s", e)
            return False

    def apply_camera_choice_and_go(self):
        """Xử lý khi ấn 'Áp dụng nguồn'."""
        if self.counting:
            messagebox.showwarning("Đang đếm", "Vui lòng dừng đếm trước khi đổi nguồn.")
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
        if self.student_detection_enabled.get() and self.detection_enabled.get():
            status += " với nhận diện học sinh và hành vi đang hoạt động"
        elif self.student_detection_enabled.get():
            status += " với nhận diện học sinh đang hoạt động"
        elif self.detection_enabled.get():
            status += " với nhận diện hành vi đang hoạt động"
        else:
            status += " – sẵn sàng đếm hành vi."
        self.info_label.config(text=status)

    def go_back_to_select(self):
        if self.counting:
            messagebox.showwarning("Đang đếm", "Hãy dừng đếm trước khi quay lại màn chọn nguồn.")
            return
        self.main_frame.pack_forget()
        self.select_frame.pack(fill="both", expand=True)

    def show_main_ui(self):
        self.select_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    # -------------------------
    # Detection pipeline
    # -------------------------
    def detect_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Dùng Haar Cascade để detect khuôn mặt, trả về (face_img, bbox) hoặc (None, None)."""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None, None
            # Lấy khuôn mặt lớn nhất
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_img = frame[y:y + h, x:x + w]
            return face_img, (x, y, w, h)
        except Exception as e:
            LOG.exception("Lỗi detect_face: %s", e)
            return None, None

    def preprocess_face_image(self, face_img: np.ndarray):
        """Tiền xử lý ảnh khuôn mặt cho ResNet (trả về tensor nếu torch có sẵn)."""
        try:
            import torch
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(face_img).unsqueeze(0)  # shape (1, C, H, W)
            return tensor
        except Exception as e:
            LOG.exception("Lỗi preprocess_face_image: %s", e)
            return None

    def detect_student(self, frame: np.ndarray) -> Optional[str]:
        """Nhận diện học sinh: hỗ trợ ResNet (classification) hoặc YOLO (detection)."""
        if self.student_model is None or not self.student_detection_enabled.get():
            return None

        now = time.time()
        if now - self.last_student_time < self.student_cooldown:
            return self.current_student  # return cached

        try:
            # Nếu student_model là một model torch classification (ResNet)
            if "ResNet" in str(type(self.student_model)):
                face_img, bbox = self.detect_face(frame)
                if face_img is None:
                    return None
                tensor = self.preprocess_face_image(face_img)
                if tensor is None:
                    return None

                import torch
                with torch.no_grad():
                    outputs = self.student_model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    conf_val = float(conf.item())
                    idx = int(pred.item())
                    if conf_val > 0.6 and idx < len(self.student_names):
                        name = self.student_names[idx]
                        # Draw box + label
                        if bbox:
                            x, y, w, h = bbox
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{name} ({conf_val:.2f})", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        self.current_student = name
                        self.last_student_time = now
                        return name
                    return None
            else:
                # Giả sử student_model là YOLO-like (ultralytics or yolov5)
                results = self.student_model(frame)
                # xử lý YOLOv8
                if "ultralytics.engine.results.Results" in str(type(results)):
                    if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        confidences = [float(b.conf[0]) for b in boxes]
                        best = int(np.argmax(confidences))
                        if confidences[best] > 0.6:
                            cls_id = int(boxes[best].cls[0])
                            if hasattr(results[0], "names") and cls_id in results[0].names:
                                name = results[0].names[cls_id]
                                self.current_student = name
                                self.last_student_time = now
                                return name
                # YOLOv5 format
                elif hasattr(results, "pred") and hasattr(results, "names") and len(results.pred) > 0:
                    pred = results.pred[0].cpu().numpy()
                    if len(pred) > 0:
                        # pred: [x1,y1,x2,y2,conf,class]
                        best_idx = int(np.argmax(pred[:, 4]))
                        best_det = pred[best_idx]
                        if best_det[4] > 0.6:
                            cls_id = int(best_det[5])
                            if cls_id in results.names:
                                name = results.names[cls_id]
                                self.current_student = name
                                self.last_student_time = now
                                return name
                return None
        except Exception as e:
            LOG.exception("Lỗi detect_student: %s", e)
            return None

    def detect_objects_and_count(self, frame: np.ndarray) -> np.ndarray:
        """
        Nhận diện hành vi từ model chính; trả về frame đã annotate.
        Đồng thời cập nhật counters nếu self.counting = True.
        """
        if self.model is None or not self.detection_enabled.get():
            # chỉ vẽ tên học sinh nếu có
            if self.current_student:
                annotated = frame.copy()
                cv2.putText(annotated, f"Học sinh: {self.current_student}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                return annotated
            return frame

        try:
            results = self.model(frame)
            annotated_frame = None
            detected_behaviors = []

            # YOLOv8 result support
            if "ultralytics.engine.results.Results" in str(type(results)):
                annotated_frame = results[0].plot()
                if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if hasattr(results[0], "names") and cls_id in results[0].names:
                            behavior_name = results[0].names[cls_id]
                            detected_behaviors.append(behavior_name)
                            # vẽ label student nếu có
                            if self.current_student:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                label = f"{self.current_student}: {behavior_name}"
                                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                # YOLOv5 or renderable results
                if hasattr(results, "render"):
                    annotated_frame = results.render()[0]
                    # lấy detections từ results.pred nếu có
                    if hasattr(results, "pred") and len(results.pred) > 0:
                        for det in results.pred[0]:
                            det = det.cpu().numpy()
                            if len(det) >= 6:
                                cls_id = int(det[5])
                                if cls_id in results.names:
                                    behavior_name = results.names[cls_id]
                                    detected_behaviors.append(behavior_name)
                else:
                    # Generic list-like handling (robust)
                    annotated_frame = frame.copy()
                    if isinstance(results, list) and len(results) > 0:
                        for det in results:
                            if hasattr(det, "boxes") and hasattr(det, "names"):
                                boxes = det.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    behavior_name = det.names[cls_id]
                                    detected_behaviors.append(behavior_name)
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"{behavior_name} {conf:.2f}"
                                    if self.current_student:
                                        label = f"{self.current_student}: {label}"
                                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update counts (throttled by detection_cooldown)
            now = time.time()
            if self.counting and detected_behaviors and (now - self.last_detected_time > self.detection_cooldown):
                for behavior in detected_behaviors:
                    if behavior != self.last_detected:
                        self.behavior_counts[behavior] += 1
                        if self.current_student:
                            self.student_behavior_counts[self.current_student][behavior] += 1
                            self.current_behaviors[self.current_student] = behavior
                        self.last_detected = behavior
                        self.last_detected_time = now
                        self.update_count_display()

            # Vẽ info student + current behavior
            if annotated_frame is None:
                annotated_frame = frame.copy()

            if self.current_student:
                behavior = self.current_behaviors.get(self.current_student, "")
                if behavior:
                    txt = f"Học sinh: {self.current_student} - Hành vi: {behavior}"
                else:
                    txt = f"Học sinh: {self.current_student}"
                cv2.putText(annotated_frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            return annotated_frame
        except Exception as e:
            LOG.exception("Lỗi detect_objects_and_count: %s", e)
            return frame

    # -------------------------
    # UI actions: toggle
    # -------------------------
    def toggle_detection(self):
        if self.detection_enabled.get():
            if not (self.torch_available or self.ultralytics_available):
                self.detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu thư viện để bật nhận diện hành vi: pip install torch ultralytics")
                return
            if self.model is None and not self.load_model():
                self.detection_enabled.set(False)
                return
            messagebox.showinfo("Thông báo", "Đã bật nhận diện hành vi.")
        else:
            messagebox.showinfo("Thông báo", "Đã tắt nhận diện hành vi.")

    def toggle_student_detection(self):
        if self.student_detection_enabled.get():
            if not self.torch_available:
                self.student_detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu torch: pip install torch")
                return
            if self.student_model is None and not self.load_student_model():
                self.student_detection_enabled.set(False)
                return
            messagebox.showinfo("Thông báo", "Đã bật nhận diện học sinh.")
        else:
            self.current_student = None
            messagebox.showinfo("Thông báo", "Đã tắt nhận diện học sinh.")

    # -------------------------
    # Counting control
    # -------------------------
    def start_counting(self):
        if not (self.cap and getattr(self.cap, "isOpened", lambda: False)()):
            messagebox.showerror("Lỗi", "Chưa có camera nào đang mở.")
            return
        if not self.detection_enabled.get():
            messagebox.showerror("Lỗi", "Vui lòng bật chế độ nhận diện hành vi trước khi đếm.")
            return
        if self.counting:
            messagebox.showinfo("Thông báo", "Đã đang đếm.")
            return

        # Reset counters
        self.behavior_counts = Counter()
        self.student_behavior_counts = defaultdict(Counter)
        self.current_behaviors = {}
        self.last_detected = None
        self.last_detected_time = 0.0

        self.counting = True
        self.start_time = time.time()
        self.recorded_time = 0
        if self.student_detection_enabled.get():
            status = f"🔴 Đang đếm hành vi theo học sinh (camera {self.current_cam_index})..."
            if self.current_student:
                self.count_label.config(text=f"Đang đếm hành vi cho học sinh: {self.current_student}...\nChưa phát hiện hành vi nào.")
            else:
                self.count_label.config(text="Đang đếm hành vi...\nĐang đợi nhận diện học sinh.")
            messagebox.showinfo("Thông báo", "Bắt đầu đếm hành vi theo học sinh!")
        else:
            status = f"🔴 Đang đếm hành vi (camera {self.current_cam_index})..."
            self.count_label.config(text="Đang đếm hành vi...\nChưa phát hiện hành vi nào.")
            messagebox.showinfo("Thông báo", "Bắt đầu đếm hành vi!")

        self.info_label.config(text=status)

    def stop_counting(self):
        if not self.counting:
            messagebox.showinfo("Thông báo", "Chưa bắt đầu đếm hành vi.")
            return
        self.counting = False
        self.recorded_time = int(time.time() - self.start_time) if self.start_time else 0
        status = f"🟡 Đã dừng đếm. Thời gian: {self.recorded_time} giây (camera {self.current_cam_index})"
        self.info_label.config(text=status)
        self.show_behavior_report()
        self.save_behavior_report()

    # -------------------------
    # Reporting
    # -------------------------
    def update_count_display(self):
        """Cập nhật label bên UI hiển thị kết quả đếm hiện tại."""
        if not self.counting:
            return
        if self.current_student:
            lines = [f"Học sinh hiện tại: {self.current_student}", "", "Số lần xuất hiện của các hành vi:"]
            beh = self.student_behavior_counts.get(self.current_student, {})
            if beh:
                for b, c in sorted(beh.items()):
                    lines.append(f"- {b}: {c} lần")
            else:
                lines.append("Chưa phát hiện hành vi nào.")
            cur = self.current_behaviors.get(self.current_student)
            if cur:
                lines.append("")
                lines.append(f"Hành vi hiện tại: {cur}")
        else:
            lines = ["Số lần xuất hiện của các hành vi:"]
            if self.behavior_counts:
                for b, c in sorted(self.behavior_counts.items()):
                    lines.append(f"- {b}: {c} lần")
            else:
                lines.append("Chưa phát hiện hành vi nào.")
            if self.student_detection_enabled.get():
                lines.append("")
                lines.append("(Đang đợi nhận diện học sinh...)")
        self.count_label.config(text="\n".join(lines))

    def save_behavior_report(self):
        """Lưu báo cáo text vào thư mục reports."""
        os.makedirs("reports", exist_ok=True)
        fname = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.txt")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"BÁO CÁO HÀNH VI - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Thời gian đếm: {self.recorded_time} giây\n")
                f.write(f"Camera: {self.current_cam_index}\n\n")
                if self.student_detection_enabled.get() and self.student_behavior_counts:
                    f.write("BÁO CÁO THEO HỌC SINH:\n=====================\n\n")
                    for student, behaviors in sorted(self.student_behavior_counts.items()):
                        f.write(f"Học sinh: {student}\n")
                        if behaviors:
                            for b, c in sorted(behaviors.items()):
                                f.write(f"  - {b}: {c} lần\n")
                        else:
                            f.write("  Không phát hiện hành vi nào.\n")
                        f.write("\n")
                    f.write("\nTỔNG HỢP TẤT CẢ HÀNH VI:\n=====================\n\n")
                f.write("Số lần xuất hiện của các hành vi:\n")
                if self.behavior_counts:
                    for b, c in sorted(self.behavior_counts.items()):
                        f.write(f"- {b}: {c} lần\n")
                else:
                    f.write("Không phát hiện hành vi nào trong thời gian đếm.\n")
            messagebox.showinfo("Thông báo", f"Đã lưu báo cáo vào file: {fname}")
        except Exception as e:
            LOG.exception("Lỗi save_behavior_report: %s", e)
            messagebox.showerror("Lỗi", f"Không lưu được báo cáo: {e}")

    def show_behavior_report(self):
        """Hiển thị báo cáo chi tiết trong một cửa sổ mới."""
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Báo cáo hành vi")
        report_window.geometry("700x700")
        report_window.configure(bg="#f2e9e4")

        text_frame = tk.Frame(report_window, bg="#f2e9e4")
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(text_frame, wrap="word", font=("Arial", 12), bg="#ffffff")
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        report = []
        report.append("BÁO CÁO HÀNH VI\n")
        if hasattr(self, "recorded_time") and self.recorded_time:
            report.append(f"Thời gian đếm: {self.recorded_time} giây\n")

        if self.student_detection_enabled.get() and self.student_behavior_counts:
            report.append("\nBÁO CÁO THEO HỌC SINH:\n=====================\n")
            for student, behaviors in sorted(self.student_behavior_counts.items()):
                report.append(f"Học sinh: {student}\n")
                if behaviors:
                    for b, c in sorted(behaviors.items()):
                        report.append(f"  - {b}: {c} lần\n")
                else:
                    report.append("  Không phát hiện hành vi nào.\n")
                report.append("\n")
            report.append("\nTỔNG HỢP TẤT CẢ HÀNH VI:\n=====================\n")

        report.append("\nSố lần xuất hiện của các hành vi:\n")
        if self.behavior_counts:
            for b, c in sorted(self.behavior_counts.items()):
                report.append(f"- {b}: {c} lần\n")
        else:
            report.append("Không phát hiện hành vi nào trong thời gian đếm.")

        text_widget.insert("1.0", "".join(report))
        text_widget.config(state="disabled")

        # Save & Export buttons
        save_btn = tk.Button(report_window, text="Lưu báo cáo", command=self.save_behavior_report,
                             bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
        save_btn.pack(pady=6)

        # Export to Excel nếu pandas có sẵn
        try:
            import pandas as pd  # noqa: F401
            excel_btn = tk.Button(report_window, text="Xuất Excel", command=self.export_excel_report,
                                  bg="#ffffff", activebackground="#d0f0c0", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
            excel_btn.pack(pady=6)
        except Exception:
            pass

    def export_excel_report(self):
        """Xuất báo cáo sang Excel nếu pandas có sẵn."""
        try:
            import pandas as pd
            os.makedirs("reports", exist_ok=True)
            excel_file = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.xlsx")

            summary_data = [{"Hành vi": b, "Số lần xuất hiện": c} for b, c in sorted(self.behavior_counts.items())]
            summary_df = pd.DataFrame(summary_data)

            student_data = []
            for student, behaviors in sorted(self.student_behavior_counts.items()):
                for b, c in sorted(behaviors.items()):
                    student_data.append({"Học sinh": student, "Hành vi": b, "Số lần xuất hiện": c})
            student_df = pd.DataFrame(student_data)

            with pd.ExcelWriter(excel_file) as writer:
                summary_df.to_excel(writer, sheet_name="Tổng hợp", index=False)
                if not student_df.empty:
                    student_df.to_excel(writer, sheet_name="Theo học sinh", index=False)

            messagebox.showinfo("Thông báo", f"Đã xuất báo cáo Excel: {excel_file}")
        except ImportError:
            messagebox.showerror("Lỗi", "Cần cài đặt pandas để xuất Excel:\n pip install pandas openpyxl")
        except Exception as e:
            LOG.exception("Lỗi export_excel_report: %s", e)
            messagebox.showerror("Lỗi", f"Không thể xuất Excel: {e}")

    # -------------------------
    # Main loop: camera frame update
    # -------------------------
    def update_camera(self):
        """Hàm được gọi định kỳ để đọc frame từ camera và hiển thị."""
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)() and self.main_frame.winfo_ismapped():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h0, w0 = frame.shape[:2]
                    vw = max(320, self.video_label.winfo_width())
                    vh = max(240, self.video_label.winfo_height())
                    scale = min(vw / w0, vh / h0) if w0 and h0 else 1.0
                    new_w = max(320, int(w0 * scale))
                    new_h = max(240, int(h0 * scale))

                    resized_bgr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Student detection (throttled)
                    now = time.time()
                    if self.student_detection_enabled.get() and self.student_model is not None and (now - self.last_student_time > self.student_cooldown):
                        try:
                            student = self.detect_student(resized_bgr)
                            if student:
                                # ensure entry exists
                                if student not in self.student_behavior_counts:
                                    self.student_behavior_counts[student] = Counter()
                        except Exception:
                            LOG.exception("Lỗi khi detect_student trong loop")

                    # Object detection & counting
                    try:
                        annotated = self.detect_objects_and_count(resized_bgr)
                    except Exception:
                        LOG.exception("Lỗi detect_objects_and_count")
                        annotated = resized_bgr

                    # Convert BGR->RGB for Tk display
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                    self.frame_size = (new_w, new_h)

                    # update status label
                    status = f"🟢 Camera {self.current_cam_index} đang bật"
                    if self.counting:
                        status += " (Đang đếm hành vi...)"
                    else:
                        parts = []
                        if self.student_detection_enabled.get():
                            parts.append("Nhận diện học sinh: BẬT")
                        if self.detection_enabled.get():
                            parts.append("Nhận diện hành vi: BẬT")
                        status += f" ({', '.join(parts)})" if parts else " – sẵn sàng đếm hành vi."
                    self.info_label.config(text=status)
                else:
                    # nếu không đọc được frame, show empty
                    self.video_label.configure(image="")
        except Exception:
            LOG.exception("Lỗi update_camera")
        finally:
            # call again after ~30 ms (~33 fps)
            self.root.after(30, self.update_camera)

    # -------------------------
    # Close / cleanup
    # -------------------------
    def on_close(self):
        if self.counting:
            if not messagebox.askyesno("Đang đếm", "Đang đếm hành vi. Muốn dừng và thoát?"):
                return
            self.stop_counting()
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    def __del__(self):
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                self.cap.release()
        except Exception:
            pass


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PastelCameraApp(root)
    root.mainloop()
