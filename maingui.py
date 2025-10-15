import cv2
import tkinter as tk
from tkinter import messagebox, BooleanVar, filedialog
from PIL import Image, ImageTk
import time, os
import sys
import numpy as np
from collections import Counter, defaultdict

class PastelCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎀 Pastel Camera Recorder 🎀")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f2e9e4")

        self.cap = None
        self.current_cam_index = None
        self.counting = False 
        self.start_time = None
        self.video_writer = None
        self.recorded_time = 0
        self.filename = None
        self.frame_size = (960, 540)  
        
        # Biến cho nhận diện hành vi
        self.detection_enabled = BooleanVar(value=True)
        self.model = None
        self.model_path = 'model.pt'  # Đường dẫn mô hình hành vi
        
        # Biến cho nhận diện học sinh
        self.student_detection_enabled = BooleanVar(value=True)
        self.student_model = None
        self.student_model_path = 'student_model.pt'  # Đường dẫn mô hình học sinh
        
        # Biến đếm hành vi
        self.behavior_counts = Counter()  # Đếm tổng số hành vi
        self.student_behavior_counts = defaultdict(Counter)  # Đếm hành vi theo học sinh {student_name: {behavior: count}}
        
        # Biến theo dõi hành vi
        self.last_detected = None
        self.last_detected_time = 0
        self.detection_cooldown = 1.0  # Thời gian chờ giữa các lần đếm hành vi (giây)
        
        # Biến theo dõi học sinh
        self.current_student = None
        self.last_student_time = 0
        self.student_cooldown = 2.0  # Thời gian chờ giữa các lần nhận diện học sinh (giây)
        
        # Kiểm tra thư viện cần thiết
        self.check_dependencies()
        
        # Tải mô hình
        self.load_model()
        self.load_student_model()

        # Khung chọn nguồn camera
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

        self.cam_var = tk.StringVar(value="0")
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

        # Khung chính
        self.main_frame = tk.Frame(self.root, bg="#f2e9e4")

        # Thanh công cụ trên cùng
        top_bar = tk.Frame(self.main_frame, bg="#f2e9e4")
        top_bar.pack(pady=(10, 0), fill="x")

        self.back_btn = tk.Button(
            top_bar, text="⟵ Quay lại chọn nguồn",
            command=self.go_back_to_select,
            bg="#ffffff", activebackground="#fde2e4",
            relief="raised", bd=2, font=("Arial", 11, "bold"), cursor="hand2"
        )
        self.back_btn.pack(side="left", padx=20)
        
        # Checkbox để bật/tắt nhận diện học sinh
        self.student_detection_cb = tk.Checkbutton(
            top_bar, text="Nhận diện học sinh",
            variable=self.student_detection_enabled,
            bg="#f2e9e4", font=("Arial", 11, "bold"),
            command=self.toggle_student_detection
        )
        self.student_detection_cb.pack(side="right", padx=10)
        self.student_detection_cb.select()  # Mặc định đã chọn
        
        # Checkbox để bật/tắt nhận diện hành vi
        self.detection_cb = tk.Checkbutton(
            top_bar, text="Nhận diện hành vi",
            variable=self.detection_enabled,
            bg="#f2e9e4", font=("Arial", 11, "bold"),
            command=self.toggle_detection
        )
        self.detection_cb.pack(side="right", padx=10)
        self.detection_cb.select()  # Mặc định đã chọn

        # Khung chứa các nút chức năng
        btn_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        btn_frame.pack(pady=12, fill="x")

        btn_style = {
            "width": 14, "height": 3, "compound": "top",
            "bg": "#fff", "activebackground": "#fde2e4",
            "relief": "raised", "bd": 2, "font": ("Arial", 11, "bold"),
            "cursor": "hand2"
        }

        # Tải biểu tượng cho các nút
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

        # Các nút chức năng
        self.start_btn = create_hover_button(btn_frame, "Bắt đầu", start_icon, self.start_counting)
        self.start_btn.pack(side="left", padx=40)

        self.stop_btn = create_hover_button(btn_frame, "Dừng", stop_icon, self.stop_counting)
        self.stop_btn.pack(side="left", padx=40)

        self.report_btn = create_hover_button(btn_frame, "Báo cáo", report_icon, self.show_behavior_report)
        self.report_btn.pack(side="left", padx=40)

        # Khung hiển thị video
        self.video_label = tk.Label(self.main_frame, bg="#c9ada7")
        self.video_label.pack(pady=10, fill="both", expand=True)

        # Khung hiển thị số lượng hành vi đã đếm được
        self.count_frame = tk.Frame(self.main_frame, bg="#f2e9e4")
        self.count_frame.pack(pady=5, fill="x")
        
        self.count_label = tk.Label(
            self.count_frame,
            text="Đang chờ bắt đầu đếm...",
            bg="#f2e9e4", font=("Arial", 12), fg="#2d3436"
        )
        self.count_label.pack(pady=5)
        
        # Nhãn hiển thị trạng thái
        self.info_label = tk.Label(
            self.main_frame,
            text="🟡 Chưa mở camera. Hãy quay lại chọn nguồn và áp dụng.",
            bg="#f2e9e4", font=("Arial", 14, "bold"), fg="#2d3436"
        )
        self.info_label.pack(pady=8)

        # Xử lý khi đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Bắt đầu cập nhật camera
        self.update_camera()
    
    def check_dependencies(self):
        """Kiểm tra các thư viện cần thiết"""
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            
        try:
            from ultralytics import YOLO
            self.ultralytics_available = True
        except ImportError:
            self.ultralytics_available = False
            
        if not self.torch_available or not self.ultralytics_available:
            messagebox.showwarning(
                "Cảnh báo",
                "Thiếu thư viện cần thiết cho nhận diện đối tượng.\n"
                "Vui lòng cài đặt bằng lệnh:\n\n"
                "pip install torch ultralytics\n\n"
                "Bạn vẫn có thể sử dụng ứng dụng mà không có tính năng nhận diện."
            )
            # Tắt chế độ nhận diện nếu thiếu thư viện
            self.detection_enabled.set(False)
            self.student_detection_enabled.set(False)
    
    def load_model(self):
        """Tải mô hình YOLO nhận diện hành vi từ file model.pt"""
        if not os.path.exists(self.model_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy file mô hình hành vi: {self.model_path}")
            self.detection_enabled.set(False)
            return False
            
        if not self.torch_available or not self.ultralytics_available:
            messagebox.showerror(
                "Lỗi",
                "Không thể tải mô hình nhận diện. Vui lòng cài đặt thư viện cần thiết:\n"
                "pip install torch ultralytics"
            )
            self.detection_enabled.set(False)
            return False
            
        try:
            # Hiển thị thông báo đang tải
            loading_window = tk.Toplevel(self.root)
            loading_window.title("Đang tải mô hình hành vi")
            loading_window.geometry("300x100")
            loading_window.resizable(False, False)
            loading_window.transient(self.root)
            loading_window.grab_set()
            
            tk.Label(
                loading_window, 
                text="Đang tải mô hình nhận diện hành vi...\nVui lòng đợi...",
                font=("Arial", 12)
            ).pack(pady=20)
            
            loading_window.update()
            
            # Thử tải mô hình bằng YOLOv8 (ultralytics)
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"Đã tải mô hình hành vi YOLOv8 thành công từ {self.model_path}!")
                loading_window.destroy()
                return True
            except Exception as e:
                print(f"Không thể tải mô hình hành vi YOLOv8: {e}")
                
                # Thử tải mô hình bằng YOLOv5
                try:
                    import torch
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                    self.model.conf = 0.45  # Ngưỡng tin cậy
                    print(f"Đã tải mô hình hành vi YOLOv5 thành công từ {self.model_path}!")
                    loading_window.destroy()
                    return True
                except Exception as e2:
                    print(f"Không thể tải mô hình hành vi YOLOv5: {e2}")
                    loading_window.destroy()
                    messagebox.showerror(
                        "Lỗi",
                        f"Không thể tải mô hình nhận diện hành vi:\n{str(e2)}\n\n"
                        "Vui lòng kiểm tra định dạng mô hình và thư viện."
                    )
                    self.detection_enabled.set(False)
                    return False
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tải mô hình hành vi: {str(e)}")
            self.detection_enabled.set(False)
            return False
    
    def load_student_model(self):
        """Tải mô hình YOLO nhận diện học sinh từ file student_model.pt"""
        if not os.path.exists(self.student_model_path):
            messagebox.showwarning(
                "Cảnh báo", 
                f"Không tìm thấy file mô hình học sinh: {self.student_model_path}\n"
                "Bạn vẫn có thể sử dụng ứng dụng mà không có tính năng nhận diện học sinh."
            )
            self.student_detection_enabled.set(False)
            return False
            
        if not self.torch_available or not self.ultralytics_available:
            self.student_detection_enabled.set(False)
            return False
            
        try:
            # Hiển thị thông báo đang tải
            loading_window = tk.Toplevel(self.root)
            loading_window.title("Đang tải mô hình học sinh")
            loading_window.geometry("300x100")
            loading_window.resizable(False, False)
            loading_window.transient(self.root)
            loading_window.grab_set()
            
            tk.Label(
                loading_window, 
                text="Đang tải mô hình nhận diện học sinh...\nVui lòng đợi...",
                font=("Arial", 12)
            ).pack(pady=20)
            
            loading_window.update()
            
            # Thử tải mô hình bằng YOLOv8 (ultralytics)
            try:
                from ultralytics import YOLO
                self.student_model = YOLO(self.student_model_path)
                print(f"Đã tải mô hình học sinh YOLOv8 thành công từ {self.student_model_path}!")
                loading_window.destroy()
                return True
            except Exception as e:
                print(f"Không thể tải mô hình học sinh YOLOv8: {e}")
                
                # Thử tải mô hình bằng YOLOv5
                try:
                    import torch
                    self.student_model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.student_model_path)
                    self.student_model.conf = 0.45  # Ngưỡng tin cậy
                    print(f"Đã tải mô hình học sinh YOLOv5 thành công từ {self.student_model_path}!")
                    loading_window.destroy()
                    return True
                except Exception as e2:
                    print(f"Không thể tải mô hình học sinh YOLOv5: {e2}")
                    loading_window.destroy()
                    messagebox.showwarning(
                        "Cảnh báo",
                        f"Không thể tải mô hình nhận diện học sinh:\n{str(e2)}\n\n"
                        "Bạn vẫn có thể sử dụng ứng dụng mà không có tính năng nhận diện học sinh."
                    )
                    self.student_detection_enabled.set(False)
                    return False
        except Exception as e:
            messagebox.showwarning("Cảnh báo", f"Lỗi khi tải mô hình học sinh: {str(e)}")
            self.student_detection_enabled.set(False)
            return False
    
    def toggle_detection(self):
        """Bật/tắt chế độ nhận diện hành vi"""
        if self.detection_enabled.get():
            if not self.torch_available or not self.ultralytics_available:
                self.detection_enabled.set(False)
                messagebox.showerror(
                    "Lỗi", 
                    "Không thể bật chế độ nhận diện hành vi.\n"
                    "Vui lòng cài đặt thư viện cần thiết:\n"
                    "pip install torch ultralytics"
                )
                return
                
            if self.model is None:
                if self.load_model():
                    messagebox.showinfo("Thông báo", "Đã bật chế độ nhận diện hành vi")
                else:
                    self.detection_enabled.set(False)
            else:
                messagebox.showinfo("Thông báo", "Đã bật chế độ nhận diện hành vi")
        else:
            messagebox.showinfo("Thông báo", "Đã tắt chế độ nhận diện hành vi")
    
    def toggle_student_detection(self):
        """Bật/tắt chế độ nhận diện học sinh"""
        if self.student_detection_enabled.get():
            if not self.torch_available or not self.ultralytics_available:
                self.student_detection_enabled.set(False)
                messagebox.showerror(
                    "Lỗi", 
                    "Không thể bật chế độ nhận diện học sinh.\n"
                    "Vui lòng cài đặt thư viện cần thiết:\n"
                    "pip install torch ultralytics"
                )
                return
                
            if self.student_model is None:
                if self.load_student_model():
                    messagebox.showinfo("Thông báo", "Đã bật chế độ nhận diện học sinh")
                else:
                    self.student_detection_enabled.set(False)
            else:
                messagebox.showinfo("Thông báo", "Đã bật chế độ nhận diện học sinh")
        else:
            self.current_student = None
            messagebox.showinfo("Thông báo", "Đã tắt chế độ nhận diện học sinh")

    def detect_student(self, frame):
        """Nhận diện học sinh trong khung hình"""
        if self.student_model is None or not self.student_detection_enabled.get():
            return None
        
        try:
            # Thực hiện nhận diện học sinh
            results = self.student_model(frame)
            
            # Xử lý kết quả nhận diện học sinh
            if 'ultralytics.engine.results.Results' in str(type(results)):
                # YOLOv8 - Lấy kết quả có độ tin cậy cao nhất
                if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # Sắp xếp các kết quả theo độ tin cậy
                    boxes = results[0].boxes
                    confidences = [float(box.conf[0]) for box in boxes]
                    
                    # Lấy kết quả có độ tin cậy cao nhất
                    max_conf_idx = np.argmax(confidences)
                    if confidences[max_conf_idx] > 0.6:  # Ngưỡng tin cậy
                        cls_id = int(boxes[max_conf_idx].cls[0])
                        if hasattr(results[0], 'names') and cls_id in results[0].names:
                            student_name = results[0].names[cls_id]
                            return student_name
            elif hasattr(results, 'names') and hasattr(results, 'pred') and len(results.pred) > 0:
                # YOLOv5
                if len(results.pred[0]) > 0:
                    # Sắp xếp các kết quả theo độ tin cậy
                    detections = results.pred[0].cpu().numpy()
                    if len(detections) > 0:
                        # Sắp xếp theo độ tin cậy (cột thứ 5)
                        sorted_indices = np.argsort(-detections[:, 4])
                        best_detection = detections[sorted_indices[0]]
                        
                        if best_detection[4] > 0.6:  # Ngưỡng tin cậy
                            cls_id = int(best_detection[5])
                            if cls_id in results.names:
                                student_name = results.names[cls_id]
                                return student_name
            
            return None
        except Exception as e:
            print(f"Lỗi khi nhận diện học sinh: {e}")
            return None

    def detect_objects(self, frame):
        """Nhận diện đối tượng trong khung hình và đếm số lần xuất hiện theo học sinh"""
        # Nhận diện học sinh trước nếu được bật
        current_time = time.time()
        if self.student_detection_enabled.get() and self.student_model is not None:
            if current_time - self.last_student_time > self.student_cooldown:
                student_name = self.detect_student(frame)
                if student_name:
                    self.current_student = student_name
                    self.last_student_time = current_time
                    
                    # Khởi tạo dict đếm hành vi cho học sinh nếu chưa có
                    if student_name not in self.student_behavior_counts:
                        self.student_behavior_counts[student_name] = Counter()
        
        # Nếu không bật nhận diện hành vi hoặc không có mô hình, trả về frame gốc
        if self.model is None or not self.detection_enabled.get():
            # Vẽ tên học sinh hiện tại lên khung hình nếu có
            if self.current_student:
                frame_with_text = frame.copy()
                cv2.putText(
                    frame_with_text, 
                    f"Học sinh: {self.current_student}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                return frame_with_text
            return frame
        
        try:
            # Thực hiện nhận diện hành vi
            results = self.model(frame)
            
            # Vẽ bounding boxes và đếm số lần xuất hiện của các hành vi
            if 'ultralytics.engine.results.Results' in str(type(results)):
                # YOLOv8 - Sử dụng phương thức plot() từ kết quả
                annotated_frame = results[0].plot()
                
                # Đếm số lần xuất hiện của các hành vi nếu đang trong chế độ đếm
                if self.counting:
                    current_time = time.time()
                    # Lấy các hành vi được phát hiện
                    detected_behaviors = []
                    
                    # Kiểm tra kết quả và lấy tên lớp
                    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            if hasattr(results[0], 'names') and cls_id in results[0].names:
                                behavior_name = results[0].names[cls_id]
                                detected_behaviors.append(behavior_name)
                    
                    # Nếu có hành vi được phát hiện và đã qua thời gian chờ
                    if detected_behaviors and (current_time - self.last_detected_time > self.detection_cooldown):
                        for behavior in detected_behaviors:
                            if self.last_detected != behavior:  # Chỉ đếm khi hành vi thay đổi
                                # Đếm cho tổng số
                                self.behavior_counts[behavior] += 1
                                
                                # Đếm cho học sinh hiện tại nếu có
                                if self.current_student:
                                    self.student_behavior_counts[self.current_student][behavior] += 1
                                
                                self.last_detected = behavior
                                self.last_detected_time = current_time
                                
                                # Cập nhật hiển thị số lượng
                                self.update_count_display()
                
                # Vẽ tên học sinh hiện tại lên khung hình
                if self.current_student:
                    cv2.putText(
                        annotated_frame, 
                        f"Học sinh: {self.current_student}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                
                return annotated_frame
            else:
                # YOLOv5 hoặc định dạng khác
                try:
                    # Nếu là YOLOv5
                    if hasattr(results, 'render'):
                        rendered_frame = results.render()[0]
                        
                        # Đếm số lần xuất hiện của các hành vi nếu đang trong chế độ đếm
                        if self.counting:
                            current_time = time.time()
                            detected_behaviors = []
                            
                            # Lấy các hành vi được phát hiện từ kết quả YOLOv5
                            if hasattr(results, 'names') and hasattr(results, 'pred') and len(results.pred) > 0:
                                for det in results.pred[0]:
                                    if len(det) >= 6:  # Kiểm tra xem có đủ thông tin không
                                        cls_id = int(det[5])
                                        if cls_id in results.names:
                                            behavior_name = results.names[cls_id]
                                            detected_behaviors.append(behavior_name)
                            
                            # Nếu có hành vi được phát hiện và đã qua thời gian chờ
                            if detected_behaviors and (current_time - self.last_detected_time > self.detection_cooldown):
                                for behavior in detected_behaviors:
                                    if self.last_detected != behavior:  # Chỉ đếm khi hành vi thay đổi
                                        # Đếm cho tổng số
                                        self.behavior_counts[behavior] += 1
                                        
                                        # Đếm cho học sinh hiện tại nếu có
                                        if self.current_student:
                                            self.student_behavior_counts[self.current_student][behavior] += 1
                                        
                                        self.last_detected = behavior
                                        self.last_detected_time = current_time
                                        
                                        # Cập nhật hiển thị số lượng
                                        self.update_count_display()
                        
                        # Vẽ tên học sinh hiện tại lên khung hình
                        if self.current_student:
                            cv2.putText(
                                rendered_frame, 
                                f"Học sinh: {self.current_student}", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 0, 255), 
                                2
                            )
                        
                        return rendered_frame
                    # Nếu là list (trường hợp khác)
                    else:
                        # Tự vẽ bounding box
                        img_copy = frame.copy()
                        
                        # Kiểm tra cấu trúc kết quả
                        if isinstance(results, list) and len(results) > 0:
                            # Lấy thông tin từ kết quả
                            detected_behaviors = []
                            
                            for det in results:
                                if hasattr(det, 'boxes') and hasattr(det, 'names'):
                                    # YOLOv8 format
                                    boxes = det.boxes
                                    for box in boxes:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        cls = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        label = f"{det.names[cls]} {conf:.2f}"
                                        behavior_name = det.names[cls]
                                        detected_behaviors.append(behavior_name)
                                        
                                        # Vẽ bounding box
                                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        
                                        # Vẽ nhãn
                                        cv2.putText(img_copy, label, (x1, y1 - 10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Đếm số lần xuất hiện của các hành vi nếu đang trong chế độ đếm
                            if self.counting and detected_behaviors:
                                current_time = time.time()
                                if current_time - self.last_detected_time > self.detection_cooldown:
                                    for behavior in detected_behaviors:
                                        if self.last_detected != behavior:  # Chỉ đếm khi hành vi thay đổi
                                            # Đếm cho tổng số
                                            self.behavior_counts[behavior] += 1
                                            
                                            # Đếm cho học sinh hiện tại nếu có
                                            if self.current_student:
                                                self.student_behavior_counts[self.current_student][behavior] += 1
                                            
                                            self.last_detected = behavior
                                            self.last_detected_time = current_time
                                            
                                            # Cập nhật hiển thị số lượng
                                            self.update_count_display()
                        
                        # Vẽ tên học sinh hiện tại lên khung hình
                        if self.current_student:
                            cv2.putText(
                                img_copy, 
                                f"Học sinh: {self.current_student}", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 0, 255), 
                                2
                            )
                        
                        return img_copy
                except Exception as e:
                    print(f"Lỗi khi xử lý kết quả nhận diện: {e}")
                    return frame
        except Exception as e:
            print(f"Lỗi khi nhận diện đối tượng: {e}")
            return frame
    
    def update_count_display(self):
        """Cập nhật hiển thị số lượng hành vi đã đếm được theo học sinh"""
        if not self.counting:
            return
        
        if self.current_student:
            count_text = f"Học sinh hiện tại: {self.current_student}\n\n"
            count_text += "Số lần xuất hiện của các hành vi:\n"
            
            if self.student_behavior_counts.get(self.current_student):
                for behavior, count in sorted(self.student_behavior_counts[self.current_student].items()):
                    count_text += f"- {behavior}: {count} lần\n"
            else:
                count_text += "Chưa phát hiện hành vi nào."
        else:
            count_text = "Số lần xuất hiện của các hành vi:\n"
            if self.behavior_counts:
                for behavior, count in sorted(self.behavior_counts.items()):
                    count_text += f"- {behavior}: {count} lần\n"
            else:
                count_text += "Chưa phát hiện hành vi nào."
            
            if self.student_detection_enabled.get():
                count_text += "\n(Đang đợi nhận diện học sinh...)"
            
        self.count_label.config(text=count_text)
    
    def show_main_ui(self):
        self.select_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    def go_back_to_select(self):
        if self.counting:
            messagebox.showwarning("Đang đếm", "Hãy dừng đếm trước khi quay lại màn chọn nguồn.")
            return
        self.main_frame.pack_forget()
        self.select_frame.pack(fill="both", expand=True)

    def apply_camera_choice_and_go(self):
        if self.counting:
            messagebox.showwarning("Đang đếm", "Vui lòng dừng đếm trước khi đổi nguồn.")
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
                
                # Thực hiện nhận diện đối tượng nếu được bật
                try:
                    resized_bgr = self.detect_objects(resized_bgr)
                except Exception as e:
                    print(f"Lỗi khi nhận diện: {e}")

                frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.frame_size = (new_w, new_h)

                if self.current_cam_index is not None:
                    status = "🟢 Camera {} đang bật".format(self.current_cam_index)
                    if self.counting:
                        status += " (Đang đếm hành vi...)"
                    else:
                        status_parts = []
                        if self.student_detection_enabled.get():
                            status_parts.append("Nhận diện học sinh: BẬT")
                        if self.detection_enabled.get():
                            status_parts.append("Nhận diện hành vi: BẬT")
                        
                        if status_parts:
                            status += f" ({', '.join(status_parts)})"
                        else:
                            status += " – sẵn sàng đếm hành vi."
                    self.info_label.config(text=status)
            else:
                self.video_label.configure(image="")
        self.root.after(15, self.update_camera)

    def start_counting(self):
        """Bắt đầu đếm số lần xuất hiện của các hành vi theo học sinh"""
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Lỗi", "Chưa có camera nào đang mở.")
            return
            
        if not self.detection_enabled.get():
            messagebox.showerror("Lỗi", "Vui lòng bật chế độ nhận diện hành vi trước khi đếm.")
            return
            
        if not self.counting:
            # Reset các biến đếm
            self.behavior_counts = Counter()
            self.student_behavior_counts = defaultdict(Counter)
            self.last_detected = None
            self.last_detected_time = 0
            
            self.counting = True
            self.start_time = time.time()
            
            if self.student_detection_enabled.get():
                status = f"🔴 Đang đếm hành vi theo học sinh (camera {self.current_cam_index})..."
                self.info_label.config(text=status)
                
                if self.current_student:
                    self.count_label.config(text=f"Đang đếm hành vi cho học sinh: {self.current_student}...\nChưa phát hiện hành vi nào.")
                else:
                    self.count_label.config(text="Đang đếm hành vi...\nĐang đợi nhận diện học sinh.")
                
                messagebox.showinfo("Thông báo", "Bắt đầu đếm hành vi theo học sinh!")
            else:
                status = f"🔴 Đang đếm hành vi (camera {self.current_cam_index})..."
                self.info_label.config(text=status)
                self.count_label.config(text="Đang đếm hành vi...\nChưa phát hiện hành vi nào.")
                messagebox.showinfo("Thông báo", "Bắt đầu đếm hành vi!")

    def stop_counting(self):
        """Dừng đếm số lần xuất hiện của các hành vi"""
        if self.counting:
            self.counting = False
            self.recorded_time = int(time.time() - self.start_time) if self.start_time else 0
            
            status = f"🟡 Đã dừng đếm. Thời gian: {self.recorded_time} giây (camera {self.current_cam_index})"
            self.info_label.config(text=status)
            
            # Hiển thị kết quả đếm
            self.show_behavior_report()
            
            # Lưu kết quả vào file
            self.save_behavior_report()
        else:
            messagebox.showinfo("Thông báo", "Chưa bắt đầu đếm hành vi.")

    def save_behavior_report(self):
        """Lưu báo cáo hành vi vào file"""
        os.makedirs("reports", exist_ok=True)
        report_file = time.strftime("reports/behavior_report_%Y%m%d_%H%M%S.txt")
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"BÁO CÁO HÀNH VI - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Thời gian đếm: {self.recorded_time} giây\n")
            f.write(f"Camera: {self.current_cam_index}\n\n")
            
            # Nếu có nhận diện học sinh
            if self.student_detection_enabled.get() and self.student_behavior_counts:
                f.write("BÁO CÁO THEO HỌC SINH:\n")
                f.write("=====================\n\n")
                
                for student, behaviors in sorted(self.student_behavior_counts.items()):
                    f.write(f"Học sinh: {student}\n")
                    if behaviors:
                        for behavior, count in sorted(behaviors.items()):
                            f.write(f"  - {behavior}: {count} lần\n")
                    else:
                        f.write("  Không phát hiện hành vi nào.\n")
                    f.write("\n")
                
                f.write("\nTỔNG HỢP TẤT CẢ HÀNH VI:\n")
                f.write("=====================\n\n")
            
            # Báo cáo tổng hợp
            f.write("Số lần xuất hiện của các hành vi:\n")
            if self.behavior_counts:
                for behavior, count in sorted(self.behavior_counts.items()):
                    f.write(f"- {behavior}: {count} lần\n")
            else:
                f.write("Không phát hiện hành vi nào trong thời gian đếm.\n")
        
        messagebox.showinfo("Thông báo", f"Đã lưu báo cáo vào file: {report_file}")
    
    def show_behavior_report(self):
        """Hiển thị báo cáo số lần xuất hiện của các hành vi"""
        # Tạo cửa sổ mới để hiển thị báo cáo chi tiết
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Báo cáo hành vi")
        report_window.geometry("600x700")
        report_window.configure(bg="#f2e9e4")
        
        # Tạo Text widget với thanh cuộn
        text_frame = tk.Frame(report_window, bg="#f2e9e4")
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        text_widget = tk.Text(text_frame, wrap="word", font=("Arial", 12), bg="#ffffff")
        text_widget.pack(side="left", fill="both", expand=True)
        
        scrollbar.config(command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Tạo nội dung báo cáo
        report = f"BÁO CÁO HÀNH VI\n\n"
        if hasattr(self, 'recorded_time') and self.recorded_time:
            report += f"Thời gian đếm: {self.recorded_time} giây\n\n"
        
        # Nếu có nhận diện học sinh
        if self.student_detection_enabled.get() and self.student_behavior_counts:
            report += "BÁO CÁO THEO HỌC SINH:\n"
            report += "=====================\n\n"
            
            for student, behaviors in sorted(self.student_behavior_counts.items()):
                report += f"Học sinh: {student}\n"
                if behaviors:
                    for behavior, count in sorted(behaviors.items()):
                        report += f"  - {behavior}: {count} lần\n"
                else:
                    report += "  Không phát hiện hành vi nào.\n"
                report += "\n"
            
            report += "\nTỔNG HỢP TẤT CẢ HÀNH VI:\n"
            report += "=====================\n\n"
        
        # Báo cáo tổng hợp
        report += "Số lần xuất hiện của các hành vi:\n"
        if self.behavior_counts:
            for behavior, count in sorted(self.behavior_counts.items()):
                report += f"- {behavior}: {count} lần\n"
        else:
            report += "Không phát hiện hành vi nào trong thời gian đếm."
        
        text_widget.insert("1.0", report)
        text_widget.config(state="disabled")  # Chỉ đọc
        
        # Nút để lưu báo cáo
        save_btn = tk.Button(
            report_window, text="Lưu báo cáo",
            command=self.save_behavior_report,
            bg="#ffffff", activebackground="#fde2e4",
            relief="raised", bd=2, font=("Arial", 12, "bold"), cursor="hand2"
        )
        save_btn.pack(pady=10)

    def on_close(self):
        if self.counting:
            if not messagebox.askyesno("Đang đếm", "Đang đếm hành vi. Muốn dừng và thoát?"):
                return
            self.stop_counting()
        try:
            if hasattr(self, "cap") and self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    def __del__(self):
        try:
            if hasattr(self, "cap") and self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = PastelCameraApp(root)
    root.mainloop()