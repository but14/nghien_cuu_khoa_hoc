import os, cv2, shutil, random, json, hashlib
import numpy as np
from glob import glob
from collections import defaultdict
from pathlib import Path

# =======================
# CẤU HÌNH
# =======================
RAW_DIR   = r"D:\SRCCODE\nghien_cuu_khoa_hoc\FaceDS\dataset"           # Thư mục gốc dataset hiện tại (mỗi người 1 folder)
OUT_DIR   = r"D:\SRCCODE\nghien_cuu_khoa_hoc\FaceDS\fd"     # Thư mục xuất kết quả
IMG_SIZE  = 224                       # Kích thước đầu ra (IMG_SIZE x IMG_SIZE)
MARGIN    = 0.30                      # Tỉ lệ nới rộng quanh bbox khuôn mặt khi crop
MIN_FACE  = 60                        # Không nhận khuôn mặt nhỏ hơn (pixel)
MAX_PER_PERSON = 80                   # Giới hạn tối đa ảnh sau augment cho mỗi người
SPLIT = (0.7, 0.2, 0.1)               # train, val, test

# Augment mỗi ảnh gốc sẽ tạo thêm bao nhiêu biến thể (có thể bị chặn bởi MAX_PER_PERSON)
AUG_PER_IMAGE = 4

# Random seed để tái lập
random.seed(42)
np.random.seed(42)

# =======================
# CÔNG CỤ PHỤ TRỢ
# =======================
def safe_name(s: str) -> str:
    # Đổi "01. QUYNH AN" -> "01_QUYNH_AN"
    t = s.strip().replace(".", "_").replace("-", "_")
    t = "_".join(t.split())
    return t

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def largest_face(faces):
    # Trả về face có diện tích lớn nhất
    if len(faces) == 0: return None
    areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in faces]
    areas.sort(reverse=True, key=lambda t: t[0])
    return areas[0][1]

def crop_with_margin(img, box, margin=0.3):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw/2, y + bh/2
    side = int(max(bw, bh) * (1 + margin*2))
    x1 = int(max(0, cx - side/2))
    y1 = int(max(0, cy - side/2))
    x2 = int(min(w, cx + side/2))
    y2 = int(min(h, cy + side/2))
    return img[y1:y2, x1:x2]

def resize_square(img, size=224):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def normalize_jpg_quality(img):
    # Lưu rồi đọc lại để normalize jpeg artifacts (tùy chọn)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def sha1_of_image(img):
    # hash nhanh để lọc gần trùng
    return hashlib.sha1(img.tobytes()).hexdigest()

# ===== Augmentations cơ bản (không cần thư viện ngoài) =====
def aug_flip(img):
    return cv2.flip(img, 1)

def aug_rotate(img, angle=12):
    h, w = img.shape[:2]
    ang = random.uniform(-angle, angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_brightness_contrast(img, b_range=(-40,40), c_range=(0.9,1.2)):
    b = random.uniform(*b_range)
    c = random.uniform(*c_range)
    out = cv2.convertScaleAbs(img, alpha=c, beta=b)
    return out

def aug_blur(img, k_choices=(3,5)):
    k = random.choice(k_choices)
    return cv2.GaussianBlur(img, (k,k), 0)

def random_aug(img):
    ops = [aug_flip, aug_rotate, aug_brightness_contrast, aug_blur]
    # chọn ngẫu nhiên 2–3 phép
    random.shuffle(ops)
    out = img.copy()
    for op in ops[:random.randint(2,3)]:
        out = op(out)
    return out

# =======================
# PHÁT HIỆN KHUÔN MẶT
# =======================
def load_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError("Không tìm thấy HaarCascade trong cv2.data.haarcascades")
    return cv2.CascadeClassifier(cascade_path)

def detect_face(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE))
    if len(faces) == 0: return None
    return largest_face(faces)

# =======================
# QUY TRÌNH XỬ LÝ
# =======================
def collect_people(raw_dir):
    people = []
    for p in sorted(Path(raw_dir).glob("*")):
        if p.is_dir():
            # ít nhất 1 ảnh
            imgs = []
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
                imgs += glob(str(p / ext))
            if len(imgs) > 0:
                people.append((p.name, imgs))
    return people

def process_person(name, imgs, detector, out_base):
    label = safe_name(name)
    tmp_dir = Path(out_base) / "_tmp" / label
    ensure_dirs(tmp_dir)
    kept_hashes = set()
    saved = 0

    # 1) Detect & crop & save base
    for img_path in imgs:
        try:
            img = cv2.imread(img_path)
            if img is None: 
                continue
            box = detect_face(img, detector)
            if box is None:
                continue
            face = crop_with_margin(img, box, margin=MARGIN)
            if face.size == 0: 
                continue
            face = resize_square(face, IMG_SIZE)
            face = normalize_jpg_quality(face)

            h = sha1_of_image(face)
            if h in kept_hashes:
                continue
            kept_hashes.add(h)

            cv2.imwrite(str(tmp_dir / f"base_{saved:04d}.jpg"), face)
            saved += 1
        except Exception as e:
            print(f"[WARN] {img_path}: {e}")

    # 2) Augment (giới hạn MAX_PER_PERSON)
    aug_needed = max(0, min(MAX_PER_PERSON, saved * (1 + AUG_PER_IMAGE)) - saved)
    bases = sorted(glob(str(tmp_dir / "base_*.jpg")))
    idx = 0
    while aug_needed > 0 and bases:
        src = cv2.imread(bases[idx % len(bases)])
        aug_img = random_aug(src)
        h = sha1_of_image(aug_img)
        if h in kept_hashes:
            idx += 1
            continue
        kept_hashes.add(h)
        cv2.imwrite(str(tmp_dir / f"aug_{len(kept_hashes):04d}.jpg"), aug_img)
        aug_needed -= 1
        idx += 1

    return label, sorted(glob(str(tmp_dir / "*.jpg")))

def split_and_export(label, files, out_dir):
    random.shuffle(files)
    n = len(files)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    train, val, test = files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

    for subset, lst in [("train", train), ("val", val), ("test", test)]:
        d = Path(out_dir) / subset / label
        ensure_dirs(d)
        for i, f in enumerate(lst):
            shutil.copy2(f, str(d / f"{label}_{i:04d}.jpg"))
    return {"train": len(train), "val": len(val), "test": len(test), "total": n}

def main():
    print("[INFO] Bắt đầu xử lý dataset…")
    ensure_dirs(OUT_DIR)
    detector = load_detector()

    people = collect_people(RAW_DIR)
    if not people:
        print("[ERROR] Không tìm thấy thư mục người dùng/ảnh trong RAW_DIR.")
        return

    stats = {}
    total_kept = 0
    tmp_root = Path(OUT_DIR) / "_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    for name, imgs in people:
        label, processed = process_person(name, imgs, detector, OUT_DIR)
        if len(processed) == 0:
            print(f"[WARN] {name}: không dùng được ảnh nào (không thấy mặt hoặc lỗi).")
            continue
        subset_stats = split_and_export(label, processed, OUT_DIR)
        stats[label] = subset_stats
        total_kept += subset_stats["total"]
        print(f"[OK] {name} -> {label}: {subset_stats}")

    # Ghi thống kê
    with open(Path(OUT_DIR) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Tổng quan
    grand = defaultdict(int)
    for lb, st in stats.items():
        for k, v in st.items():
            grand[k] += v
    print("\n=== TỔNG KẾT ===")
    print(json.dumps(grand, indent=2, ensure_ascii=False))
    print(f"[DONE] Ảnh hợp lệ sau xử lý: {total_kept}")
    print(f"[OUT]  {OUT_DIR}")

if __name__ == "__main__":
    main()
