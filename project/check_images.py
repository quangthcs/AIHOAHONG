import os
from PIL import Image

DATASET_PATH = "datasets/rose"

print("🔍 ĐANG QUÉT KIỂM TRA TOÀN BỘ ẢNH TRONG DATASET...\n")
bad_files = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        # Chỉ kiểm tra các file có đuôi ảnh
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            file_path = os.path.join(root, file)
            try:
                # Thử mở file bằng chế độ đọc nhị phân giống hệt torchvision
                with open(file_path, "rb") as f:
                    img = Image.open(f)
                    img.verify()  # Kiểm tra tính toàn vẹn của dữ liệu ảnh
            except Exception as e:
                print(f"❌ PHÁT HIỆN FILE LỖI: {file_path}")
                print(f"   Chi tiết lỗi: {e}\n")
                bad_files.append(file_path)

if not bad_files:
    print("✅ Tuyệt vời! Toàn bộ ảnh đều hợp lệ và không bị hỏng.")
else:
    print(f"⚠️ Tổng cộng tìm thấy {len(bad_files)} file lỗi. Bạn hãy vào thư mục và xóa các file trên đi nhé!")