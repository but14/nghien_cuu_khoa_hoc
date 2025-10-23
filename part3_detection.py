    def detect_face_improved(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Phương thức cải tiến để nhận diện khuôn mặt với độ chính xác cao hơn."""
        try:
            # Thử sử dụng face_recognition nếu có
            try:
                import face_recognition
                # Giảm kích thước frame để tăng tốc độ xử lý
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Tìm vị trí khuôn mặt
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if not face_locations:
                    return None, None
                    
                # Lấy khuôn mặt lớn nhất
                top, right, bottom, left = max(face_locations, key=lambda rect: (rect[2]-rect[0])*(rect[3]-rect[1]))
                
                # Chuyển về tọa độ frame gốc
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Mở rộng vùng khuôn mặt một chút để đảm bảo đủ thông tin
                h, w = frame.shape[:2]
                margin = 20
                top = max(0, top - margin)
                left = max(0, left - margin)
                bottom = min(h, bottom + margin)
                right = min(w, right + margin)
                
                face_img = frame[top:bottom, left:right]
                return face_img, (left, top, right-left, bottom-top)
                
            except ImportError:
                pass
                
            # Fallback: Sử dụng Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(faces) == 0:
                return None, None
                
            # Lấy khuôn mặt lớn nhất
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            # Mở rộng vùng khuôn mặt
            h_margin = int(h * 0.2)
            w_margin = int(w * 0.2)
            y = max(0, y - h_margin)
            x = max(0, x - w_margin)
            h = min(frame.shape[0] - y, h + 2*h_margin)
            w = min(frame.shape[1] - x, w + 2*w_margin)
            
            face_img = frame[y:y + h, x:x + w]
            return face_img, (x, y, w, h)
        except Exception as e:
            LOG.exception("Lỗi detect_face_improved: %s", e)
            return None, None

    def preprocess_face_image(self, face_img: np.ndarray):
        """Tiền xử lý ảnh khuôn mặt cho ResNet với cải tiến."""
        try:
            import torch
            import torchvision.transforms as transforms
            
            # Cải tiến: Thêm các bước tiền xử lý để tăng độ chính xác
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                # Cải tiến: Thêm các bước tiền xử lý
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
            # Phương pháp ResNet
            if "ResNet" in str(type(self.student_model)):
                # Cải tiến: Sử dụng thư viện face_recognition để detect khuôn mặt chính xác hơn
                face_img, bbox = self.detect_face_improved(frame)
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
                    
                    # Cải tiến: Ngưỡng tin cậy có thể điều chỉnh
                    confidence_threshold = 0.6
                    if conf_val > confidence_threshold and idx < len(self.student_names):
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
                # Phương pháp YOLO
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
                # YOLOv5 format
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
                    # Fallback
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

            # Cải tiến: Thêm logic để gán hành vi cho học sinh hiện tại một cách rõ ràng hơn
            now = time.time()
            if self.counting and detected_behaviors and (now - self.last_detected_time > self.detection_cooldown):
                for behavior in detected_behaviors:
                    if behavior != self.last_detected:
                        self.behavior_counts[behavior] += 1
                        if self.current_student:
                            self.student_behavior_counts[self.current_student][behavior] += 1
                            self.current_behaviors[self.current_student] = behavior
                            # Thêm log để debug
                            LOG.info(f"Đã ghi nhận: {self.current_student} - {behavior}")
                        self.last_detected = behavior
                        self.last_detected_time = now
                        self.update_count_display()

            # Fallback
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