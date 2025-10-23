import os
import sys
import time
import logging
from collections import Counter, defaultdict
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, BooleanVar, ttk
from PIL import Image, ImageTk


LOG = logging.getLogger("PastelCameraApp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PastelCameraApp:
    """Ứng dụng ghi video + nhận diện hành vi & học sinh (phiên bản nâng cấp)."""

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
        
        # Danh sách tên học sinh mặc định
        self.student_names = [f"Học sinh {i+1}" for i in range(10)]
        
        # Tải cấu hình
        self.load_config()
        
        # Tải danh sách học sinh từ file (nếu có)
        self.load_student_names()
        
        # Kiểm tra dependencies và tải mô hình
        self._check_dependencies_and_load_models()

        # Build UI
        self._build_ui()
        
        # Set up window close handler
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