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

# --- CẬP NHẬT: Thêm thư viện PIL để vẽ font Tiếng Việt ---
from PIL import Image, ImageTk, ImageDraw, ImageFont

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
        self.current_student: Optional[str] = None # Still used in this version

        # Timing / cooldowns
        self.last_detected: Optional[str] = None
        self.last_detected_time = 0.0
        self.detection_cooldown = 1.0

        self.last_student_time = 0.0
        self.student_cooldown = 2.0 # Still used in this version

        # --- CẬP NHẬT: Map 32 tên học sinh thật ---
        # (Thứ tự phải khớp chính xác với 32 folder lúc bạn train model)
        # (Bỏ số và dấu gạch dưới)
        self.student_names = [
            "QUYNH ANH", "TRAM ANH", "NAM ANH", "THAO ANH", "HONG ANH",
            "BAO CHAU", "NGOC DIEP", "KIEU DUYEN", "KHANH DAN", "XUAN HIEU",
            "DUC HUY", "GIA HUNG", "MINH KHANG", "DANG KHANG", "NGHIEM KHOA",
            "THIEN KIM", "BAO LAN", "GIA LINH", "PHUOC LOI", "BINH MINH",
            "GIA MINH", "BAO NGOC", "TUNG NHAN", "QUYNH NHU", "THANH TAM",
            "THIEN TAN", "GIA THI", "TRUNG TIN", "DINH TRONG", "ANH TRUNG",
            "TRUC VAN", "NHU Y"
        ]
        # --- KẾT THÚC MAP TÊN ---

        # --- CẬP NHẬT: Tải font cho Tiếng Việt ---
        # (Đảm bảo file "arial.ttf" nằm cùng thư mục với file .py)
        self.font_path = self.resource_path("arial.ttf")
        self.font_size = 18 # You can adjust this size
        try:
            self.pil_font = ImageFont.truetype(self.font_path, self.font_size)
            LOG.info("Đã tải font TrueType: %s", self.font_path)
        except IOError:
            LOG.warning("Không thể tải font %s. Sẽ dùng font CV2 (lỗi tiếng Việt).", self.font_path)
            self.pil_font = None
        # --- KẾT THÚC CẬP NHẬT FONT ---


        self._check_dependencies_and_load_models()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_camera()


    @staticmethod
    def resource_path(relative_path: str) -> str:
        """Lấy đường dẫn tuyệt đối đến tài nguyên (hỗ trợ PyInstaller)."""
        try:
            base_path = sys._MEIPASS  # type: ignore
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def _check_dependencies_and_load_models(self):
        """Kiểm tra thư viện cần thiết và cố gắng tải mô hình (nếu có)."""
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
        """Tải mô hình hành vi (YOLOv8 hoặc fallback YOLOv5) từ file model_path."""
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
                    from ultralytics import YOLO  # local import
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

            messagebox.showerror("Lỗi", "Không thể tải mô hình hành vi. Kiểm tra định dạng và thư viện.")
            self.detection_enabled.set(False)
            return False
        finally:
            if loading:
                loading.destroy()

    def load_student_model(self) -> bool:
        """Tải mô hình nhận diện học sinh (ResNet18).""" # Modified description
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

                # --- CẬP NHẬT: Số classes lấy từ len(self.student_names) ---
                num_classes = len(self.student_names) # Should be 32 now
                model = models.resnet18(weights=None) # Use weights=None
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                state = torch.load(self.student_model_path, map_location=torch.device("cpu"))
                # --- CẬP NHẬT: Load from state['model_state_dict'] ---
                model.load_state_dict(state['model_state_dict'])
                model.eval()
                self.student_model = model
                LOG.info("Đã tải student model dạng ResNet (lớp: %d) từ %s", num_classes, self.student_model_path)
                return True
            except Exception as e:
                LOG.warning("Không thể load student model dưới dạng ResNet: %s", e)


            # --- CẬP NHẬT: Xóa fallback loading bằng YOLO ---
            # (Vì biết file .pth là ResNet)

            messagebox.showwarning("Cảnh báo", "Không thể tải mô hình học sinh ResNet. Tính năng nhận diện học sinh sẽ bị tắt.")
            self.student_detection_enabled.set(False)
            return False
        finally:
            if loading:
                loading.destroy()


    # --- CẬP NHẬT: Hàm vẽ chữ Tiếng Việt bằng PIL ---
    def draw_text_pil(self, img: np.ndarray, text: str, pos: Tuple[int, int], color_rgb: Tuple[int, int, int]) -> np.ndarray:
        """(HÀM MỚI) Vẽ văn bản (hỗ trợ UTF-8) lên ảnh using PIL."""
        if self.pil_font:
            try:
                # Chuyển BGR (OpenCV) sang RGB (PIL)
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                # Vẽ chữ
                draw.text(pos, text, font=self.pil_font, fill=color_rgb)
                # Chuyển RGB (PIL) về BGR (OpenCV)
                return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                LOG.warning("Lỗi khi vẽ PIL text: %s", e)

        # Fallback (nếu không tải được font, sẽ báo lỗi font)
        # Bỏ qua text lỗi để tránh làm phiền
        # cv2.putText(img, f"FONT ERROR", (pos[0], pos[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return img
    # --- KẾT THÚC HÀM MỚI ---

    def _build_ui(self):
        """Tạo UI chính (2 màn: chọn nguồn & main)."""

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
        parts = []
        if self.student_detection_enabled.get(): parts.append("Nhận diện học sinh")
        if self.detection_enabled.get(): parts.append("Nhận diện hành vi")
        if parts:
             status += f" với { ' và '.join(parts) } đang hoạt động"
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


    def detect_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Dùng Haar Cascade để detect khuôn mặt, trả về (face_img, bbox) hoặc (None, None)."""
        # (Using Haar Cascade as per this code version)
        try:
            # Lazy load cascade to avoid error if file missing
            if not hasattr(self, 'face_cascade'):
                 cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                 if not os.path.exists(cascade_path):
                     LOG.error("Không tìm thấy file Haar Cascade: haarcascade_frontalface_default.xml")
                     self.face_cascade = None # Flag that loading failed
                 else:
                     self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade is None or self.face_cascade.empty():
                 return None, None # Cascade failed to load

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None, None
            # Lấy khuôn mặt lớn nhất
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            # Add padding to the bounding box if needed, e.g., slightly larger box
            # padding = 10
            # x = max(0, x - padding)
            # y = max(0, y - padding)
            # w = w + 2 * padding
            # h = h + 2 * padding
            face_img = frame[y:y + h, x:x + w]
            return face_img, (x, y, w, h)
        except Exception as e:
            LOG.exception("Lỗi detect_face: %s", e)
            return None, None

    def preprocess_face_image(self, face_img: np.ndarray):
        """Tiền xử lý ảnh khuôn mặt cho ResNet (trả về tensor nếu torch có sẵn)."""
        if face_img is None or face_img.size == 0: # Add check for empty image
            return None
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
        """Nhận diện học sinh: chỉ hỗ trợ ResNet trong phiên bản này."""
        # (Note: This function follows the OLD logic of detecting face separately)
        if self.student_model is None or not self.student_detection_enabled.get():
            return None

        now = time.time()
        # Reset current_student if cooldown passed without new detection
        if (now - self.last_student_time > self.student_cooldown * 2): # Longer timeout to clear name
             self.current_student = None

        # Only run detection if cooldown passed
        if now - self.last_student_time < self.student_cooldown:
            return self.current_student  # return cached

        # --- Detection logic starts here ---
        detected_name_this_run = None # Track if we detect *this* time
        try:
            if "ResNet" in str(type(self.student_model)):
                face_img, bbox = self.detect_face(frame)

                # Check if face_img is valid before proceeding
                if face_img is None or face_img.shape[0] < 1 or face_img.shape[1] < 1:
                    return self.current_student # Return cached if no face found

                tensor = self.preprocess_face_image(face_img)
                if tensor is None:
                    return self.current_student # Return cached on preprocess error

                import torch
                with torch.no_grad():
                    outputs = self.student_model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    conf_val = float(conf.item())
                    idx = int(pred.item())
                    if conf_val > 0.6 and idx < len(self.student_names):
                        # --- CẬP NHẬT: Lấy tên thật ---
                        name = self.student_names[idx]
                        detected_name_this_run = name # Store the detected name
                        # Update global state only if detected
                        self.current_student = name
                        self.last_student_time = now # Reset cooldown timer

                        # --- CẬP NHẬT: Vẽ bằng PIL (Moved to detect_objects_and_count) ---
                        # Drawing the student-only box here might conflict with the main loop's drawing
                        # if bbox:
                        #     x, y, w, h = bbox
                        #     label_text = f"{name} ({conf_val:.2f})"
                        #     text_pos = (x, y - self.font_size - 5 if y - self.font_size - 5 > 0 else y + 5)
                        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        #     frame = self.draw_text_pil(frame, label_text, text_pos, (0, 255, 0)) # Green

            # If no student detected THIS run, return the cached name (might be None)
            return detected_name_this_run if detected_name_this_run else self.current_student

        except Exception as e:
            LOG.exception("Lỗi detect_student: %s", e)
            return self.current_student # Return cached on error


    def detect_objects_and_count(self, frame: np.ndarray) -> np.ndarray:
        """Nhận diện hành vi (YOLO) và vẽ label (dùng tên học sinh nếu có)."""
        # (Note: This function follows the OLD logic)
        annotated_frame = frame.copy() # Start with a copy

        # --- CẬP NHẬT: Vẽ status text bằng PIL at the end ---
        # (Moved drawing the lone student name from here)

        if self.model is None or not self.detection_enabled.get():
            # Draw status text even if behavior detection is off
            if self.current_student:
                 status_text = f"Học sinh: {self.current_student}"
                 annotated_frame = self.draw_text_pil(annotated_frame, status_text, (10, 10), (255, 0, 0)) # Red
            return annotated_frame


        try:
            results = self.model(frame, conf=0.3) # Run YOLO behavior detection with conf threshold
            detected_behaviors = []

            # YOLOv8 result support
            if "ultralytics.engine.results.Results" in str(type(results)):
                # annotated_frame = results[0].plot() # DON'T use this - causes font error
                if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                    names = results[0].names
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        if cls_id in names:
                            behavior_name = names[cls_id]
                            detected_behaviors.append(behavior_name)

                            # --- CẬP NHẬT: Vẽ box (CV2) và label (PIL) ---
                            # Construct label using current student (if known)
                            label_text = f"{self.current_student if self.current_student else 'Unknown'}: {behavior_name} ({conf:.2f})"

                            # Determine color (Green if student known, Yellow otherwise)
                            if self.current_student:
                                rect_color_bgr = (0, 255, 0) # Green BGR
                                text_color_rgb = (0, 255, 0) # Green RGB
                            else:
                                rect_color_bgr = (0, 255, 255) # Yellow BGR
                                text_color_rgb = (255, 255, 0) # Yellow RGB

                            # Draw rectangle
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), rect_color_bgr, 2)
                            # Calculate text position
                            text_pos = (x1, y1 - self.font_size - 5 if y1 - self.font_size - 5 > 0 else y1 + 5)
                            # Draw text using PIL
                            annotated_frame = self.draw_text_pil(annotated_frame, label_text, text_pos, text_color_rgb)
                            # --- KẾT THÚC CẬP NHẬT VẼ ---

            # (Add similar manual drawing logic for YOLOv5 results if needed)
            # else: Handle other result types (e.g., YOLOv5) manually here

            # Counting logic (remains the same basic idea, check cooldown first)
            now = time.time()
            if self.counting and detected_behaviors: # Check if there are detections first
                 # Only check cooldown/update if NEW behaviors were detected
                 is_new_behavior_present = any(b != self.last_detected for b in detected_behaviors)

                 if (now - self.last_detected_time > self.detection_cooldown) and is_new_behavior_present:
                    processed_this_cooldown = False # Count only the first NEW one
                    for behavior in detected_behaviors:
                        if behavior != self.last_detected and not processed_this_cooldown:
                            self.behavior_counts[behavior] += 1
                            if self.current_student:
                                self.student_behavior_counts[self.current_student][behavior] += 1
                                self.current_behaviors[self.current_student] = behavior
                            self.last_detected = behavior # Update last detected
                            self.last_detected_time = now   # Reset cooldown timer
                            processed_this_cooldown = True
                            self.update_count_display()
                            break # Only count the first new behavior

            # --- CẬP NHẬT: Vẽ status text bằng PIL (Moved to end) ---
            if self.current_student:
                 behavior = self.current_behaviors.get(self.current_student, "")
                 status_text = f"Học sinh: {self.current_student}"
                 if behavior and self.counting: # Only show behavior if counting
                     status_text += f" - Hành vi: {behavior}"
                 # Draw text (PIL) - Red color RGB
                 annotated_frame = self.draw_text_pil(annotated_frame, status_text, (10, 10), (255, 0, 0))
            # --- KẾT THÚC CẬP NHẬT VẼ PIL ---

            return annotated_frame # Return the manually annotated frame

        except Exception as e:
            LOG.exception("Lỗi detect_objects_and_count: %s", e)
             # Draw status text even on error
            if self.current_student:
                 status_text = f"Học sinh: {self.current_student}"
                 annotated_frame = self.draw_text_pil(annotated_frame, status_text, (10, 10), (255, 0, 0)) # Red
            return annotated_frame # Return the frame with potential status text


    def toggle_detection(self):
        if self.detection_enabled.get():
            if not (self.torch_available or self.ultralytics_available):
                self.detection_enabled.set(False)
                messagebox.showerror("Lỗi", "Thiếu thư viện: pip install torch ultralytics")
                return
            if self.model is None and not self.load_model():
                self.detection_enabled.set(False); return
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
                self.student_detection_enabled.set(False); return
            messagebox.showinfo("Thông báo", "Đã bật nhận diện học sinh.")
        else:
            self.current_student = None # Clear student when disabled
            messagebox.showinfo("Thông báo", "Đã tắt nhận diện học sinh.")


    def start_counting(self):
        if not (self.cap and getattr(self.cap, "isOpened", lambda: False)()):
            messagebox.showerror("Lỗi", "Chưa có camera nào đang mở.")
            return
        # Allow starting even if student detection is off, but behavior must be on
        if not self.detection_enabled.get():
            messagebox.showerror("Lỗi", "Vui lòng bật chế độ nhận diện hành vi trước khi đếm.")
            return
        if self.counting:
            messagebox.showinfo("Thông báo", "Đã đang đếm."); return


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
            # Reset label
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
        # Clear current student display after stopping
        self.current_student = None
        self.current_behaviors = {}
        self.count_label.config(text="Đã dừng đếm. Xem báo cáo.") # Update label
        # Show/Save Report
        self.show_behavior_report()
        self.save_behavior_report()


    def update_count_display(self):
        """Cập nhật label bên UI hiển thị kết quả đếm hiện tại."""
        # (This uses self.current_student, which is updated by detect_student)
        if not self.counting:
            # If stopped, show a stopped message (handled in stop_counting)
            return

        # --- CẬP NHẬT: Hiển thị tên thật ---
        if self.current_student and self.student_detection_enabled.get():
            lines = [f"HS hiện tại: {self.current_student}"] # Shorter label
            beh = self.student_behavior_counts.get(self.current_student, {})
            if beh:
                lines.append("Hành vi đã đếm:")
                for b, c in sorted(beh.items()):
                    lines.append(f"- {b}: {c}") # Shorter line
            else:
                lines.append("Chưa đếm hành vi nào cho HS này.")
            # Show current detected behavior if available
            cur = self.current_behaviors.get(self.current_student)
            if cur:
                lines.append(f"HV hiện tại: {cur}")
        else: # Display overall counts if no current student or student detection off
            lines = ["Tổng số lần đếm:"]
            if self.behavior_counts:
                for b, c in sorted(self.behavior_counts.items()):
                    lines.append(f"- {b}: {c}")
            else:
                lines.append("Chưa đếm hành vi nào.")
            if self.student_detection_enabled.get(): # Add note if waiting for student
                lines.append("(Đang đợi nhận diện HS...)")
        self.count_label.config(text="\n".join(lines))
        # --- KẾT THÚC CẬP NHẬT TÊN THẬT ---

    def save_behavior_report(self):
        """Lưu báo cáo text vào thư mục reports."""
        os.makedirs("reports", exist_ok=True)
        fname = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.txt")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"BÁO CÁO HÀNH VI - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Thời gian đếm: {self.recorded_time} giây\n")
                f.write(f"Camera: {self.current_cam_index}\n\n")

                # --- CẬP NHẬT: Dùng tên thật ---
                if self.student_detection_enabled.get() and self.student_behavior_counts:
                    f.write("BÁO CÁO THEO HỌC SINH:\n=====================\n\n")
                    # Sort by student name
                    for student in sorted(self.student_behavior_counts.keys()):
                        behaviors = self.student_behavior_counts[student]
                        f.write(f"Học sinh: {student}\n") # Tên thật
                        if behaviors:
                            for b, c in sorted(behaviors.items()):
                                f.write(f"   - {b}: {c} lần\n")
                        else:
                            f.write("   Không phát hiện hành vi nào.\n")
                        f.write("\n")
                    f.write("\nTỔNG HỢP TẤT CẢ HÀNH VI (BAO GỒM 'Unknown'):\n=====================\n\n")
                else:
                    f.write("TỔNG HỢP TẤT CẢ HÀNH VI:\n=====================\n\n")
                # --- KẾT THÚC CẬP NHẬT TÊN THẬT ---

                # Overall counts
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

        # --- CẬP NHẬT: Dùng tên thật ---
        if self.student_detection_enabled.get() and self.student_behavior_counts:
            report.append("\nBÁO CÁO THEO HỌC SINH:\n=====================\n")
            for student in sorted(self.student_behavior_counts.keys()):
                 behaviors = self.student_behavior_counts[student]
                 report.append(f"Học sinh: {student}\n") # Tên thật
                 if behaviors:
                    for b, c in sorted(behaviors.items()):
                        report.append(f"   - {b}: {c} lần\n")
                 else:
                    report.append("   Không phát hiện hành vi nào.\n")
                 report.append("\n")
            report.append("\nTỔNG HỢP TẤT CẢ HÀNH VI (BAO GỒM 'Unknown'):\n=====================\n")
        else:
             report.append("\nTỔNG HỢP TẤT CẢ HÀNH VI:\n=====================\n")
        # --- KẾT THÚC CẬP NHẬT TÊN THẬT ---

        if self.behavior_counts:
            for b, c in sorted(self.behavior_counts.items()):
                report.append(f"- {b}: {c} lần\n")
        else:
            report.append("Không phát hiện hành vi nào trong thời gian đếm.")

        text_widget.insert("1.0", "".join(report))
        text_widget.config(state="disabled")

        save_btn = tk.Button(report_window, text="Lưu báo cáo", command=self.save_behavior_report,
                             bg="#ffffff", activebackground="#fde2e4", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
        save_btn.pack(pady=6)

        try:
            import pandas # Check if pandas exists
            excel_btn = tk.Button(report_window, text="Xuất Excel", command=self.export_excel_report,
                                  bg="#ffffff", activebackground="#d0f0c0", relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2")
            excel_btn.pack(pady=6)
        except ImportError: # Don't show button if pandas not installed
            pass
        except Exception: # Catch other potential errors
            pass


    def export_excel_report(self):
        """Xuất báo cáo sang Excel nếu pandas có sẵn."""
        try:
            import pandas as pd
            # Need openpyxl to write .xlsx files
            import openpyxl # noqa
            os.makedirs("reports", exist_ok=True)
            excel_file = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.xlsx")

            summary_data = [{"Hành vi": b, "Số lần xuất hiện": c} for b, c in sorted(self.behavior_counts.items())]
            summary_df = pd.DataFrame(summary_data)

            # --- CẬP NHẬT: Dùng tên thật ---
            student_data = []
            if self.student_detection_enabled.get():
                for student in sorted(self.student_behavior_counts.keys()):
                    behaviors = self.student_behavior_counts[student]
                    if behaviors: # Only add students with detected behaviors
                        for b, c in sorted(behaviors.items()):
                            student_data.append({"Học sinh": student, "Hành vi": b, "Số lần xuất hiện": c}) # Tên thật
            student_df = pd.DataFrame(student_data)
            # --- KẾT THÚC CẬP NHẬT TÊN THẬT ---


            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer: # Specify engine
                summary_df.to_excel(writer, sheet_name="Tổng hợp (All)", index=False)
                # Only write student sheet if data exists and detection was enabled
                if not student_df.empty and self.student_detection_enabled.get():
                    student_df.to_excel(writer, sheet_name="Theo học sinh (Known)", index=False)

            messagebox.showinfo("Thông báo", f"Đã xuất báo cáo Excel: {excel_file}")
        except ImportError:
            messagebox.showerror("Lỗi", "Cần cài đặt pandas và openpyxl để xuất Excel:\n pip install pandas openpyxl")
        except Exception as e:
            LOG.exception("Lỗi export_excel_report: %s", e)
            messagebox.showerror("Lỗi", f"Không thể xuất Excel: {e}")


    def update_camera(self):
        """Hàm được gọi định kỳ để đọc frame từ camera và hiển thị."""
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)() and self.main_frame.winfo_ismapped():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h0, w0 = frame.shape[:2]
                    # Simple resize to fit label, maintain aspect ratio
                    label_w = self.video_label.winfo_width()
                    label_h = self.video_label.winfo_height()
                    if label_w < 10 or label_h < 10: # Handle case where widget size is not yet known
                        label_w, label_h = 960, 540 # Default size
                    scale = min(label_w / w0, label_h / h0) if w0 > 0 and h0 > 0 else 1.0
                    new_w = max(320, int(w0 * scale))
                    new_h = max(240, int(h0 * scale))

                    resized_bgr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Make a copy for student detection to avoid drawing on the frame used by object detection
                    student_detect_frame = resized_bgr.copy()

                    # --- CẬP NHẬT: Gọi detect_student TRƯỚC detect_objects ---
                    if self.student_detection_enabled.get():
                         try:
                             # detect_student updates self.current_student but drawing is now handled later
                             _ = self.detect_student(student_detect_frame)
                         except Exception:
                             LOG.exception("Lỗi khi detect_student trong loop")
                    # --- KẾT THÚC CẬP NHẬT ---


                    # Object detection & counting (uses self.current_student if set)
                    try:
                        # Pass the ORIGINAL resized frame for behavior detection
                        annotated = self.detect_objects_and_count(resized_bgr)
                    except Exception:
                        LOG.exception("Lỗi detect_objects_and_count")
                        annotated = resized_bgr # Show resized if error

                    # Convert final annotated BGR->RGB for Tk display
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
                        if self.student_detection_enabled.get(): parts.append("Nhận diện HS: BẬT") # Shorter text
                        if self.detection_enabled.get(): parts.append("Nhận diện HV: BẬT") # Shorter text
                        status += f" ({', '.join(parts)})" if parts else " – sẵn sàng đếm hành vi."
                    self.info_label.config(text=status)
                else:
                    # Clear image if no frame read
                    self.video_label.configure(image="")
        except Exception as e: # Catch potential errors during resize/display
            LOG.exception("Lỗi update_camera: %s", e)
        finally:
            # call again after ~30 ms (~33 fps)
            self.root.after(30, self.update_camera)


    def on_close(self):
        if self.counting:
            if not messagebox.askyesno("Đang đếm", "Đang đếm hành vi. Muốn dừng và thoát?"):
                return
            self.stop_counting() # Ensure report is saved if counting
        try:
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                self.cap.release()
        except Exception:
            pass
        # Attempt to gracefully destroy tkinter window
        try:
            self.root.destroy()
        except tk.TclError:
            pass # Ignore error if window already destroyed


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