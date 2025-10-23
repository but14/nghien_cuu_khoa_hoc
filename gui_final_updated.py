import os
import sys
import time
import logging
from collections import Counter, defaultdict
from typing import Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, BooleanVar
from PIL import Image, ImageTk


LOG = logging.getLogger("PastelCameraApp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PastelCameraApp:
    """Ứng dụng ghi video + nhận diện hành vi & học sinh (improved version)."""

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

        # Model paths
        self.model_path = self.resource_path("model.pt")
        self.student_model_path = self.resource_path("face_classifier.pth")

        # Data structures for counting
        self.behavior_counts: Counter = Counter()
        self.student_behavior_counts: defaultdict = defaultdict(Counter)
        self.current_behaviors: dict = {}
        self.current_student: Optional[str] = None
        
        # NEW: Timeline tracking
        self.behavior_timeline = []  # Lưu chi tiết timeline

        # Timing / cooldowns
        self.last_detected: Optional[str] = None
        self.last_detected_time = 0.0
        self.detection_cooldown = 1.0

        self.last_student_time = 0.0
        self.student_cooldown = 1.5  # Giảm thời gian để phản hồi nhanh hơn
        
        # Cải thiện tracking
        self.student_tracking_history = []  # Lưu lịch sử tracking
        self.max_tracking_history = 10

        # Confidence threshold
        self.confidence_threshold = 0.65
        
        self.student_names = [f"Học sinh {i+1}" for i in range(10)]

        self._check_dependencies_and_load_models()
        self._build_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_camera()

    @staticmethod
    def resource_path(relative_path: str) -> str:
        """Lấy đường dẫn tuyệt đối đến tài nguyên (hỗ trợ PyInstaller)."""
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def _check_dependencies_and_load_models(self):
        """Kiểm tra thư viện cần thiết và tải mô hình."""
        self.torch_available = self._is_module_available("torch")
        self.ultralytics_available = self._is_module_available("ultralytics")

        if not (self.torch_available and self.ultralytics_available):
            messagebox.showwarning(
                "Cảnh báo",
                "Thiếu thư viện nhận diện: torch và/hoặc ultralytics.\n"
                "Bạn có thể cài:\n\npip install torch ultralytics\n\n"
                "Các tính năng nhận diện sẽ bị tắt nếu thiếu."
            )
            if not self.torch_available:
                self.student_detection_enabled.set(False)
            if not (self.torch_available and self.ultralytics_available):
                self.detection_enabled.set(False)

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
        """Tải mô hình hành vi (YOLOv8 hoặc YOLOv5)."""
        if not os.path.exists(self.model_path):
            LOG.warning("Model hành vi không tồn tại: %s", self.model_path)
            self.detection_enabled.set(False)
            return False

        if not (self.torch_available or self.ultralytics_available):
            LOG.warning("Không có thư viện để load model hành vi.")
            self.detection_enabled.set(False)
            return False

        loading = None
        try:
            loading = tk.Toplevel(self.root)
            loading.title("Đang tải mô hình hành vi")
            loading.geometry("320x100")
            loading.transient(self.root)
            tk.Label(loading, text="Đang tải mô hình hành vi...\nVui lòng đợi...", font=("Arial", 11)).pack(pady=20)
            loading.update()

            if self.ultralytics_available:
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    LOG.info("Đã tải model hành vi (YOLOv8) từ %s", self.model_path)
                    return True
                except Exception as e:
                    LOG.warning("Không thể load YOLOv8: %s", e)

            if self.torch_available:
                try:
                    import torch
                    self.model = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path, verbose=False)
                    try:
                        self.model.conf = 0.45
                    except Exception:
                        pass
                    LOG.info("Đã tải model hành vi (YOLOv5) từ %s", self.model_path)
                    return True
                except Exception as e:
                    LOG.warning("Không thể load YOLOv5: %s", e)

            messagebox.showerror("Lỗi", "Không thể tải mô hình hành vi.")
            self.detection_enabled.set(False)
            return False
        finally:
            if loading:
                loading.destroy()

    def load_student_model(self) -> bool:
        """Tải mô hình nhận diện học sinh (ResNet18)."""
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

            try:
                import torch
                import torchvision.models as models

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

            if self.ultralytics_available:
                try:
                    from ultralytics import YOLO
                    self.student_model = YOLO(self.student_model_path)
                    LOG.info("Đã tải student model (YOLOv8) từ %s", self.student_model_path)
                    return True
                except Exception as e:
                    LOG.warning("Không thể load student YOLOv8: %s", e)

            messagebox.showwarning("Cảnh báo", "Không thể tải mô hình học sinh.")
            self.student_detection_enabled.set(False)
            return False
        finally:
            if loading:
                loading.destroy()

    def _build_ui(self):
        """Tạo UI chính."""
        # Select frame
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

        # Main frame
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

        # NEW: Confidence threshold control
        conf_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        conf_frame.pack(pady=5)
        
        tk.Label(conf_frame, text="Ngưỡng tin cậy:", bg="#f2e9e4", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        self.conf_scale = tk.Scale(conf_frame, from_=0.5, to=0.95, resolution=0.05,
                                   orient="horizontal", bg="#f2e9e4", length=200,
                                   command=self.update_confidence_threshold)
        self.conf_scale.set(self.confidence_threshold)
        self.conf_scale.pack(side="left")
        
        self.conf_value_label = tk.Label(conf_frame, text=f"{self.confidence_threshold:.2f}", bg="#f2e9e4", font=("Arial", 10, "bold"))
        self.conf_value_label.pack(side="left", padx=5)

        # Buttons frame
        btn_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        btn_frame.pack(pady=12, fill="x")

        btn_style = {"width": 14, "height": 3, "compound": "top",
                     "bg": "#fff", "activebackground": "#fde2e4",
                     "relief": "raised", "bd": 2, "font": ("Arial", 11, "bold"),
                     "cursor": "hand2"}

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
        """Tải icon nếu có, trả về None nếu lỗi."""
        try:
            path = self.resource_path(name)
            img = Image.open(path).resize(size)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def update_confidence_threshold(self, value):
        """Cập nhật ngưỡng tin cậy."""
        self.confidence_threshold = float(value)
        self.conf_value_label.config(text=f"{self.confidence_threshold:.2f}")

    def init_camera(self, index: int) -> bool:
        """Khởi tạo camera."""
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                try:
                    self.cap.release()
                except Exception:
                    pass
            
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
                messagebox.showwarning("Chú ý", f"Không mở được camera {cam_index}. Đã chuyển sang {alt}.")
            else:
                messagebox.showerror("Lỗi", f"Không thể mở camera {cam_index} và {alt}.")
                return

        self.show_main_ui()
        status = f"🟢 Camera {self.current_cam_index} đang bật"
        if self.student_detection_enabled.get() and self.detection_enabled.get():
            status += " với nhận diện học sinh và hành vi"
        elif self.student_detection_enabled.get():
            status += " với nhận diện học sinh"
        elif self.detection_enabled.get():
            status += " với nhận diện hành vi"
        else:
            status += " – sẵn sàng đếm."
        self.info_label.config(text=status)

    def go_back_to_select(self):
        if self.counting:
            messagebox.showwarning("Đang đếm", "Hãy dừng đếm trước khi quay lại.")
            return
        self.main_frame.pack_forget()
        self.select_frame.pack(fill="both", expand=True)

    def show_main_ui(self):
        self.select_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    def detect_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Dùng Haar Cascade để detect khuôn mặt."""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                                 minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            if len(faces) == 0:
                return None, None
                
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            
            y_offset = int(h * 0.1)
            x_offset = int(w * 0.05)
            
            y1 = max(0, y - y_offset)
            y2 = min(frame.shape[0], y + h + y_offset)
            x1 = max(0, x - x_offset)
            x2 = min(frame.shape[1], x + w + x_offset)
            
            face_img = frame[y1:y2, x1:x2]
            return face_img, (x1, y1, x2-x1, y2-y1)
        except Exception as e:
            LOG.exception("Lỗi detect_face: %s", e)
            return None, None

    def preprocess_face_image(self, face_img: np.ndarray):
        """Tiền xử lý ảnh khuôn mặt cho ResNet."""
        try:
            import torch
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(face_img).unsqueeze(0)
            return tensor
        except Exception as e:
            LOG.exception("Lỗi preprocess_face_image: %s", e)
            return None

    @staticmethod
    def calculate_iou(box1, box2):
        """Tính IoU giữa 2 bounding boxes (x, y, w, h)."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(box1_x2, box2_x2)
        yi2 = min(box1_y2, box2_y2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def is_face_in_behavior_box(self, face_bbox, behavior_bbox, threshold=0.2):
        """Kiểm tra khuôn mặt có nằm trong vùng hành vi không."""
        if not face_bbox or not behavior_bbox:
            return False
        
        x1, y1, x2, y2 = behavior_bbox
        behavior_box_xywh = (x1, y1, x2-x1, y2-y1)
        
        iou = self.calculate_iou(face_bbox, behavior_box_xywh)
        return iou > threshold
    
    def calculate_distance_between_boxes(self, face_bbox, behavior_bbox):
        """Tính khoảng cách giữa center của hai bounding box."""
        if not face_bbox or not behavior_bbox:
            return float('inf')
        
        fx, fy, fw, fh = face_bbox
        bx1, by1, bx2, by2 = behavior_bbox
        
        # Center của face box
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2
        
        # Center của behavior box
        behavior_center_x = (bx1 + bx2) // 2
        behavior_center_y = (by1 + by2) // 2
        
        # Tính khoảng cách Euclidean
        distance = ((face_center_x - behavior_center_x) ** 2 + 
                   (face_center_y - behavior_center_y) ** 2) ** 0.5
        
        return distance
    
    def get_most_recent_student(self):
        """Lấy học sinh được nhận diện gần đây nhất."""
        if not self.student_tracking_history:
            return None
        
        # Lấy entry gần nhất
        most_recent = self.student_tracking_history[-1]
        now = time.time()
        
        # Kiểm tra xem có trong vòng 3 giây không
        if now - most_recent['timestamp'] < 3.0:
            return most_recent['name']
        
        return None

    def detect_objects_and_count(self, frame: np.ndarray) -> np.ndarray:
        """Nhận diện hành vi + học sinh và liên kết chúng."""
        annotated_frame = frame.copy()
        
        # Bước 1: Nhận diện khuôn mặt học sinh
        student_name = None
        face_bbox = None
        
        if self.student_detection_enabled.get() and self.student_model is not None:
            now = time.time()
            if now - self.last_student_time > self.student_cooldown:
                face_img, face_bbox = self.detect_face(frame)
                if face_img is not None and face_bbox is not None:
                    # Nhận diện bằng ResNet
                    if "ResNet" in str(type(self.student_model)):
                        tensor = self.preprocess_face_image(face_img)
                        if tensor is not None:
                            import torch
                            with torch.no_grad():
                                outputs = self.student_model(tensor)
                                probs = torch.nn.functional.softmax(outputs, dim=1)
                                conf, pred = torch.max(probs, dim=1)
                                conf_val = float(conf.item())
                                idx = int(pred.item())
                            
                            if conf_val > self.confidence_threshold and idx < len(self.student_names):
                                student_name = self.student_names[idx]
                                self.current_student = student_name
                                self.last_student_time = now
                                
                                # Cập nhật tracking history
                                self.student_tracking_history.append({
                                    'name': student_name,
                                    'confidence': conf_val,
                                    'timestamp': now,
                                    'bbox': face_bbox
                                })
                                
                                # Giới hạn lịch sử
                                if len(self.student_tracking_history) > self.max_tracking_history:
                                    self.student_tracking_history.pop(0)
                                
                                # Vẽ box khuôn mặt với màu sắc khác biệt
                                x, y, w, h = face_bbox
                                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 255), 3)
                                
                                # Vẽ label với background
                                label = f"{student_name} ({conf_val:.2f})"
                                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(annotated_frame, (x, y-label_h-15), (x+label_w+10, y-5), (255, 0, 255), -1)
                                cv2.putText(annotated_frame, label, (x+5, y-8), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Bước 2: Nhận diện hành vi
        if self.model is None or not self.detection_enabled.get():
            return annotated_frame

        try:
            results = self.model(frame)
            behavior_detections = []
            
            # Parse YOLO results
            if "ultralytics.engine.results.Results" in str(type(results)):
                if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if conf < self.confidence_threshold:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if hasattr(results[0], "names") and cls_id in results[0].names:
                            behavior_name = results[0].names[cls_id]
                            behavior_detections.append({
                                'name': behavior_name,
                                'conf': conf,
                                'bbox': (x1, y1, x2, y2)
                            })
            else:
                # YOLOv5 format
                if hasattr(results, "pred") and len(results.pred) > 0:
                    for det in results.pred[0]:
                        det = det.cpu().numpy()
                        if len(det) >= 6:
                            conf = float(det[4])
                            if conf < self.confidence_threshold:
                                continue
                                
                            cls_id = int(det[5])
                            x1, y1, x2, y2 = map(int, det[:4])
                            
                            if hasattr(results, "names") and cls_id in results.names:
                                behavior_name = results.names[cls_id]
                                behavior_detections.append({
                                    'name': behavior_name,
                                    'conf': conf,
                                    'bbox': (x1, y1, x2, y2)
                                })
            
            # Bước 3: Liên kết học sinh với hành vi
            matched_behaviors = []
            
            for detection in behavior_detections:
                behavior_name = detection['name']
                behavior_conf = detection['conf']
                behavior_bbox = detection['bbox']
                x1, y1, x2, y2 = behavior_bbox
                
                # Kiểm tra liên kết với học sinh (cải thiện logic)
                assigned_student = None
                if student_name and face_bbox:
                    # Kiểm tra IoU giữa khuôn mặt và hành vi
                    if self.is_face_in_behavior_box(face_bbox, behavior_bbox, threshold=0.15):
                        assigned_student = student_name
                    # Nếu không có IoU tốt, kiểm tra khoảng cách
                    elif self.calculate_distance_between_boxes(face_bbox, behavior_bbox) < 100:
                        # Kiểm tra lịch sử tracking gần đây
                        recent_student = self.get_most_recent_student()
                        if recent_student and recent_student == student_name:
                            assigned_student = student_name
                
                # Vẽ bounding box
                color = (0, 255, 0) if assigned_student else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Tạo label
                if assigned_student:
                    label = f"{assigned_student}: {behavior_name} {behavior_conf:.2f}"
                else:
                    label = f"{behavior_name} {behavior_conf:.2f}"
                
                # Vẽ label với background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                matched_behaviors.append({
                    'student': assigned_student or "Unknown",
                    'behavior': behavior_name,
                    'conf': behavior_conf,
                    'timestamp': time.time()
                })
            
            # Bước 4: Cập nhật đếm và timeline
            now = time.time()
            if self.counting and matched_behaviors:
                for match in matched_behaviors:
                    if now - self.last_detected_time > self.detection_cooldown:
                        behavior = match['behavior']
                        student = match['student']
                        
                        self.behavior_counts[behavior] += 1
                        
                        if student != "Unknown":
                            self.student_behavior_counts[student][behavior] += 1
                            self.current_behaviors[student] = behavior
                        
                        # Lưu vào timeline
                        self.behavior_timeline.append({
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'student': student,
                            'behavior': behavior,
                            'confidence': match['conf']
                        })
                        
                        self.last_detected = behavior
                        self.last_detected_time = now
                        self.update_count_display()
            
            return annotated_frame
            
        except Exception as e:
            LOG.exception("Lỗi detect_objects_and_count: %s", e)
            return annotated_frame

    def toggle_detection(self):
        if self.detection_enabled.get():
            if not (self.torch_available or self.ultralytics_available):
                self.detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu thư viện: pip install torch ultralytics")
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

    def start_counting(self):
        if not (self.cap and getattr(self.cap, "isOpened", lambda: False)()):
            messagebox.showerror("Lỗi", "Chưa có camera nào đang mở.")
            return
        if not self.detection_enabled.get():
            messagebox.showerror("Lỗi", "Vui lòng bật nhận diện hành vi trước khi đếm.")
            return
        if self.counting:
            messagebox.showinfo("Thông báo", "Đã đang đếm.")
            return

        # Reset counters
        self.behavior_counts = Counter()
        self.student_behavior_counts = defaultdict(Counter)
        self.current_behaviors = {}
        self.behavior_timeline = []
        self.last_detected = None
        self.last_detected_time = 0.0

        self.counting = True
        self.start_time = time.time()
        self.recorded_time = 0
        
        if self.student_detection_enabled.get():
            status = f"🔴 Đang đếm hành vi theo học sinh (camera {self.current_cam_index})..."
            if self.current_student:
                self.count_label.config(text=f"Đang đếm cho học sinh: {self.current_student}...\nChưa phát hiện hành vi.")
            else:
                self.count_label.config(text="Đang đếm hành vi...\nĐang đợi nhận diện học sinh.")
            messagebox.showinfo("Thông báo", "Bắt đầu đếm hành vi theo học sinh!")
        else:
            status = f"🔴 Đang đếm hành vi (camera {self.current_cam_index})..."
            self.count_label.config(text="Đang đếm hành vi...\nChưa phát hiện hành vi.")
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

    def update_count_display(self):
        """Cập nhật label hiển thị kết quả đếm."""
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
                f.write(f"Camera: {self.current_cam_index}\n")
                f.write(f"Ngưỡng tin cậy: {self.confidence_threshold:.2f}\n\n")
                
                if self.student_detection_enabled.get() and self.student_behavior_counts:
                    f.write("BÁO CÁO THEO HỌC SINH:\n=====================\n\n")
                    for student, behaviors in sorted(self.student_behavior_counts.items()):
                        f.write(f"Học sinh: {student}\n")
                        if behaviors:
                            total = sum(behaviors.values())
                            for b, c in sorted(behaviors.items()):
                                percentage = (c / total * 100) if total > 0 else 0
                                f.write(f"  - {b}: {c} lần ({percentage:.1f}%)\n")
                        else:
                            f.write("  Không phát hiện hành vi nào.\n")
                        f.write("\n")
                    f.write("\nTỔNG HỢP TẤT CẢ HÀNH VI:\n=====================\n\n")
                
                f.write("Số lần xuất hiện của các hành vi:\n")
                if self.behavior_counts:
                    for b, c in sorted(self.behavior_counts.items()):
                        f.write(f"- {b}: {c} lần\n")
                else:
                    f.write("Không phát hiện hành vi nào.\n")
                
                # Thêm timeline
                if self.behavior_timeline:
                    f.write("\n\nTIMELINE CHI TIẾT:\n=====================\n")
                    for entry in self.behavior_timeline:
                        f.write(f"{entry['time']} - {entry['student']}: {entry['behavior']} (conf: {entry['confidence']:.2f})\n")
            
            messagebox.showinfo("Thành công", f"Đã lưu báo cáo: {fname}")
        except Exception as e:
            LOG.exception("Lỗi save_behavior_report: %s", e)
            messagebox.showerror("Lỗi", f"Không lưu được báo cáo: {e}")

    def show_behavior_report(self):
        """Hiển thị báo cáo chi tiết."""
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Báo cáo hành vi")
        report_window.geometry("800x800")
        report_window.configure(bg="#f2e9e4")

        text_frame = tk.Frame(report_window, bg="#f2e9e4")
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(text_frame, wrap="word", font=("Arial", 11), bg="#ffffff")
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        report = []
        report.append("=" * 60 + "\n")
        report.append("BÁO CÁO HÀNH VI HỌC SINH\n")
        report.append("=" * 60 + "\n\n")
        
        if hasattr(self, "recorded_time") and self.recorded_time:
            report.append(f"⏱️  Thời gian đếm: {self.recorded_time} giây\n")
        report.append(f"🎯 Ngưỡng tin cậy: {self.confidence_threshold:.2f}\n")
        report.append(f"📷 Camera: {self.current_cam_index}\n\n")

        if self.student_detection_enabled.get() and self.student_behavior_counts:
            report.append("👥 BÁO CÁO THEO HỌC SINH:\n")
            report.append("=" * 60 + "\n\n")
            
            for student, behaviors in sorted(self.student_behavior_counts.items()):
                report.append(f"📌 {student}\n")
                report.append("-" * 40 + "\n")
                if behaviors:
                    total = sum(behaviors.values())
                    for b, c in sorted(behaviors.items(), key=lambda x: x[1], reverse=True):
                        percentage = (c / total * 100) if total > 0 else 0
                        bar = "█" * int(percentage / 5)
                        report.append(f"  • {b}: {c} lần ({percentage:.1f}%) {bar}\n")
                else:
                    report.append("  Không phát hiện hành vi nào.\n")
                report.append("\n")
            
            report.append("\n📊 TỔNG HỢP TẤT CẢ HÀNH VI:\n")
            report.append("=" * 60 + "\n\n")

        report.append("📈 Số lần xuất hiện:\n")
        report.append("-" * 40 + "\n")
        if self.behavior_counts:
            total_all = sum(self.behavior_counts.values())
            for b, c in sorted(self.behavior_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (c / total_all * 100) if total_all > 0 else 0
                bar = "█" * int(percentage / 5)
                report.append(f"  • {b}: {c} lần ({percentage:.1f}%) {bar}\n")
        else:
            report.append("  Không phát hiện hành vi nào.\n")

        # Timeline
        if self.behavior_timeline:
            report.append("\n\n⏰ TIMELINE CHI TIẾT:\n")
            report.append("=" * 60 + "\n\n")
            for entry in self.behavior_timeline[-20:]:  # Chỉ hiển thị 20 mục gần nhất
                report.append(f"{entry['time']} | {entry['student']:15} | {entry['behavior']:15} | {entry['confidence']:.2f}\n")
            if len(self.behavior_timeline) > 20:
                report.append(f"\n... và {len(self.behavior_timeline) - 20} mục khác\n")

        text_widget.insert("1.0", "".join(report))
        text_widget.config(state="disabled")

        # Buttons
        btn_frame = tk.Frame(report_window, bg="#f2e9e4")
        btn_frame.pack(pady=10)

        save_btn = tk.Button(btn_frame, text="💾 Lưu báo cáo", command=self.save_behavior_report,
                             bg="#ffffff", activebackground="#fde2e4", relief="raised", 
                             bd=2, font=("Arial", 11, "bold"), cursor="hand2", width=15)
        save_btn.pack(side="left", padx=5)

        try:
            import pandas as pd  # noqa: F401
            excel_btn = tk.Button(btn_frame, text="📊 Xuất Excel", command=self.export_excel_report,
                                  bg="#ffffff", activebackground="#d0f0c0", relief="raised", 
                                  bd=2, font=("Arial", 11, "bold"), cursor="hand2", width=15)
            excel_btn.pack(side="left", padx=5)
        except Exception:
            pass

    def export_excel_report(self):
        """Xuất báo cáo sang Excel với timeline."""
        try:
            import pandas as pd
            os.makedirs("reports", exist_ok=True)
            excel_file = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.xlsx")

            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Sheet 1: Tổng hợp
                summary_data = [{"Hành vi": b, "Số lần xuất hiện": c, 
                                "Tỷ lệ (%)": round(c/sum(self.behavior_counts.values())*100, 2) if sum(self.behavior_counts.values()) > 0 else 0} 
                               for b, c in sorted(self.behavior_counts.items())]
                if summary_data:
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Tổng hợp", index=False)
                
                # Sheet 2: Chi tiết theo học sinh
                student_data = []
                for student, behaviors in sorted(self.student_behavior_counts.items()):
                    total = sum(behaviors.values())
                    for b, c in sorted(behaviors.items()):
                        student_data.append({
                            "Học sinh": student,
                            "Hành vi": b,
                            "Số lần": c,
                            "Tỷ lệ (%)": round(c/total*100, 2) if total > 0 else 0
                        })
                
                if student_data:
                    pd.DataFrame(student_data).to_excel(writer, sheet_name="Theo học sinh", index=False)
                
                # Sheet 3: Timeline
                if self.behavior_timeline:
                    timeline_df = pd.DataFrame(self.behavior_timeline)
                    timeline_df.to_excel(writer, sheet_name="Timeline", index=False)
                
                # Sheet 4: Thống kê tổng quan
                stats_data = [{
                    "Chỉ số": "Tổng số sự kiện",
                    "Giá trị": sum(self.behavior_counts.values())
                }, {
                    "Chỉ số": "Số loại hành vi",
                    "Giá trị": len(self.behavior_counts)
                }, {
                    "Chỉ số": "Số học sinh",
                    "Giá trị": len(self.student_behavior_counts)
                }, {
                    "Chỉ số": "Thời gian (giây)",
                    "Giá trị": self.recorded_time
                }, {
                    "Chỉ số": "Ngưỡng tin cậy",
                    "Giá trị": self.confidence_threshold
                }]
                pd.DataFrame(stats_data).to_excel(writer, sheet_name="Thống kê", index=False)

            messagebox.showinfo("Thành công", f"Đã xuất báo cáo Excel:\n{excel_file}")
        except ImportError:
            messagebox.showerror("Lỗi", "Cần cài đặt pandas:\npip install pandas openpyxl")
        except Exception as e:
            LOG.exception("Lỗi export_excel_report: %s", e)
            messagebox.showerror("Lỗi", f"Không thể xuất Excel: {e}")

    def update_camera(self):
        """Hàm được gọi định kỳ để đọc frame từ camera."""
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

                    # Update status label
                    status = f"🟢 Camera {self.current_cam_index} đang bật"
                    if self.counting:
                        elapsed = int(time.time() - self.start_time) if self.start_time else 0
                        status += f" (Đang đếm - {elapsed}s)"
                    else:
                        parts = []
                        if self.student_detection_enabled.get():
                            parts.append("Nhận diện học sinh: BẬT")
                        if self.detection_enabled.get():
                            parts.append("Nhận diện hành vi: BẬT")
                        status += f" ({', '.join(parts)})" if parts else " – sẵn sàng đếm."
                    self.info_label.config(text=status)
                else:
                    self.video_label.configure(image="")
        except Exception:
            LOG.exception("Lỗi update_camera")
        finally:
            self.root.after(30, self.update_camera)

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


if __name__ == "__main__":
    root = tk.Tk()
    app = PastelCameraApp(root)
    root.mainloop()