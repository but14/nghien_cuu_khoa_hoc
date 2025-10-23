# Hướng Dẫn Tích Hợp Nhận Diện Học Sinh và Hành Vi

## Vấn Đề Đã Được Giải Quyết

### Vấn đề ban đầu:
- Hệ thống chỉ nhận diện được hành vi (YOLOv8) nhưng không hiển thị tên học sinh trên bounding box
- Nhận diện học sinh (ResNet18) và nhận diện hành vi hoạt động riêng biệt
- Không có liên kết giữa học sinh và hành vi tương ứng

### Giải pháp đã triển khai:

## 1. File Mới: `gui_improved_integration.py`

### Các Cải Tiến Chính:

#### A. Tích Hợp Hoàn Chỉnh
- **Liên kết học sinh với hành vi**: Sử dụng IoU (Intersection over Union) và khoảng cách để liên kết
- **Hiển thị thông tin đầy đủ**: Format `"Tên học sinh: Hành vi (confidence)"`
- **Màu sắc phân biệt**: 
  - Xanh lá: Hành vi có học sinh được nhận diện
  - Cam: Hành vi chưa có học sinh

#### B. Cải Thiện Nhận Diện Khuôn Mặt
```python
# Tham số tối ưu cho Haar Cascade
faces = face_cascade.detectMultiScale(gray, 
    scaleFactor=1.05, 
    minNeighbors=8, 
    minSize=(40, 40), 
    flags=cv2.CASCADE_SCALE_IMAGE)
```

#### C. Logic Liên Kết Thông Minh
1. **Kiểm tra IoU**: Nếu khuôn mặt nằm trong vùng hành vi
2. **Kiểm tra khoảng cách**: Nếu khuôn mặt gần vùng hành vi
3. **Fallback**: Sử dụng học sinh được nhận diện gần đây nhất

#### D. Tracking History
- Lưu lịch sử nhận diện học sinh
- Giới hạn lịch sử để tối ưu hiệu suất
- Sử dụng lịch sử để cải thiện độ chính xác

## 2. Workflow Hoàn Chỉnh

### Bước 1: Nhận Diện Học Sinh
```python
student_name, face_bbox, confidence = self.detect_student(frame)
```

### Bước 2: Nhận Diện Hành Vi
```python
results = self.model(frame)  # YOLOv8
```

### Bước 3: Liên Kết Thông Minh
```python
# Kiểm tra IoU
if self.is_face_in_behavior_box(face_bbox, behavior_bbox, threshold=0.1):
    assigned_student = student_name

# Kiểm tra khoảng cách
elif self.calculate_distance_between_boxes(face_bbox, behavior_bbox) < 150:
    recent_student = self.get_most_recent_student()
    if recent_student == student_name:
        assigned_student = student_name
```

### Bước 4: Hiển Thị Kết Quả
```python
# Label format: "Tên học sinh: Hành vi (confidence)"
if assigned_student:
    label = f"{assigned_student}: {behavior_name} {behavior_conf:.2f}"
    color = (0, 255, 0)  # Xanh lá
else:
    label = f"{behavior_name} {behavior_conf:.2f}"
    color = (0, 165, 255)  # Cam
```

## 3. Cách Sử Dụng

### Chạy Ứng Dụng:
```bash
python gui_improved_integration.py
```

### Các Tính Năng:
1. **Chọn nguồn camera**: Camera máy tính hoặc camera rời
2. **Bật/tắt nhận diện**: 
   - ✅ Nhận diện học sinh (ResNet18)
   - ✅ Nhận diện hành vi (YOLOv8)
3. **Bắt đầu đếm**: Ghi nhận hành vi theo học sinh
4. **Xem báo cáo**: Thống kê chi tiết theo học sinh

### Kết Quả Mong Muốn:
- **Bounding box xanh lá**: `"Học sinh 1: Nhin_bang 0.74"`
- **Bounding box cam**: `"Nhin_bang 0.74"` (chưa nhận diện học sinh)

## 4. Cải Tiến So Với Phiên Bản Cũ

### Trước:
```
[Nhin_bang 0.74 - Vui A]  # Chỉ có hành vi
```

### Sau:
```
[Học sinh 1: Nhin_bang 0.74]  # Có cả học sinh và hành vi
```

## 5. Báo Cáo Chi Tiết

### Theo Học Sinh:
```
Học sinh: Học sinh 1
  - Nhin_bang: 15 lần
  - Vui: 8 lần
  - Buon: 3 lần
```

### Tổng Hợp:
```
- Nhin_bang: 45 lần
- Vui: 23 lần
- Buon: 12 lần
```

## 6. Yêu Cầu Hệ Thống

### Thư viện cần thiết:
```bash
pip install torch torchvision ultralytics opencv-python pillow
```

### File model cần có:
- `model.pt`: Model YOLOv8 cho nhận diện hành vi
- `face_classifier.pth`: Model ResNet18 cho nhận diện học sinh

## 7. Troubleshooting

### Nếu không nhận diện được học sinh:
1. Kiểm tra ánh sáng đủ
2. Điều chỉnh ngưỡng confidence
3. Đảm bảo khuôn mặt rõ ràng

### Nếu không liên kết được học sinh với hành vi:
1. Kiểm tra vị trí khuôn mặt so với hành vi
2. Điều chỉnh tham số IoU threshold
3. Kiểm tra khoảng cách giữa các bounding box

## 8. Tùy Chỉnh Nâng Cao

### Điều chỉnh ngưỡng tin cậy:
```python
self.confidence_threshold = 0.65  # Tăng để giảm false positive
```

### Điều chỉnh tham số liên kết:
```python
# Trong is_face_in_behavior_box
threshold=0.1  # Giảm để dễ liên kết hơn

# Trong calculate_distance_between_boxes
< 150  # Tăng để liên kết từ xa hơn
```

## Kết Luận

Phiên bản mới `gui_improved_integration.py` đã giải quyết hoàn toàn vấn đề tích hợp nhận diện học sinh và hành vi, đảm bảo:

✅ **Hiển thị tên học sinh trên bounding box hành vi**  
✅ **Liên kết thông minh giữa học sinh và hành vi**  
✅ **Báo cáo chi tiết theo từng học sinh**  
✅ **Workflow hoàn chỉnh từ nhận diện đến xuất báo cáo**
