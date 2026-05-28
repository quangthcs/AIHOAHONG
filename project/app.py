import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm
from PIL import Image
import os
import glob
import time
import json

# --- Cấu hình đường dẫn và thông số ---
DATASET_PATH = "datasets/rose"
MODEL_SAVE_DIR = "outputs/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# Tạo thư mục nếu chưa tồn tại
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Các hàm tiện ích ---

def get_transform():
    """Hàm định nghĩa các phép biến đổi ảnh (Tiền xử lý)"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def train_model():
    """Hàm huấn luyện mô hình từ dữ liệu trong folder datasets/rose/train"""
    st.info(f"Đang chuẩn bị huấn luyện trên thiết bị: {DEVICE}")
    
    # 1. Load dữ liệu
    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir = os.path.join(DATASET_PATH, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        st.error("Lỗi: Không tìm thấy thư mục train hoặc val trong datasets/rose/")
        return
    
    train_dataset = datasets.ImageFolder(train_dir, transform=get_transform())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # 2. Khởi tạo mô hình (Sử dụng ResNet18 từ timm)
    model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Vòng lặp huấn luyện
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Đánh giá (Validation)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        status_text.text(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Acc: {acc:.4f}")
        progress_bar.progress((epoch + 1) / EPOCHS)
        
        if acc >= best_acc:
            best_acc = acc
            # Lưu model tốt nhất
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_path = os.path.join(MODEL_SAVE_DIR, f"rose_model_best.pt")
            
            # Lưu kèm class_names để sau này nhận diện đúng nhãn
            payload = {
                'model_state': model.state_dict(),
                'class_names': class_names,
                'accuracy': acc
            }
            torch.save(payload, model_path)
            
    st.success(f"Huấn luyện hoàn tất! Độ chính xác cao nhất: {best_acc:.4f}")

def load_latest_model():
    """Tìm và tải mô hình mới nhất trong thư mục models"""
    list_of_files = glob.glob(os.path.join(MODEL_SAVE_DIR, "*.pt"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    payload = torch.load(latest_file, map_location=DEVICE)
    
    # Khởi tạo lại cấu trúc mô hình
    num_classes = len(payload['class_names'])
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    model.load_state_dict(payload['model_state'])
    model.to(DEVICE)
    model.eval()
    
    return model, payload['class_names'], latest_file

def predict_image(image, model, class_names):
    """Dự đoán nhãn của một bức ảnh"""
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    prob_values, indices = torch.topk(probabilities, min(3, len(class_names)))
    
    results = []
    for i in range(len(indices)):
        results.append({
            "label": class_names[indices[i]],
            "confidence": prob_values[i].item()
        })
    return results

# --- Giao diện Streamlit ---

st.set_page_config(page_title="Rose AI Doctor", layout="wide")
st.title("🌹 AI Chẩn Đoán Sâu Bệnh Hoa Hồng")
st.markdown("Hệ thống nhận diện bệnh trên lá và hoa hồng sử dụng Deep Learning.")

tab_predict, tab_train = st.tabs(["🔍 Dự đoán bệnh", "⚙️ Huấn luyện Model"])

# --- Tab 1: Dự đoán ---
with tab_predict:
    st.header("Tải ảnh lên để bác sĩ AI kiểm tra")
    
    model_data = load_latest_model()
    
    if model_data is None:
        st.warning("⚠️ Chưa tìm thấy mô hình nào. Vui lòng sang tab 'Huấn luyện Model' để thực hiện train trước.")
    else:
        model, class_names, path = model_data
        st.info(f"Đang sử dụng mô hình: `{os.path.basename(path)}`")
        
        uploaded_file = st.file_uploader("Chọn ảnh lá hoặc hoa hồng...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            
            with col2:
                if st.button("Bắt đầu chẩn đoán", type="primary"):
                    with st.spinner("Đang phân tích..."):
                        predictions = predict_image(image, model, class_names)
                        
                        st.subheader("Kết quả phân tích:")
                        top_res = predictions[0]
                        st.metric(label="Dự đoán chính", value=top_res['label'], delta=f"{top_res['confidence']*100:.2f}% Confidence")
                        
                        st.write("---")
                        st.write("**Top khả năng có thể xảy ra:**")
                        for res in predictions:
                            st.write(f"- **{res['label']}**: {res['confidence']*100:.2f}%")
                            st.progress(res['confidence'])

# --- Tab 2: Huấn luyện ---
with tab_train:
    st.header("Cấu hình & Huấn luyện lại")
    st.write("Sử dụng khi bạn vừa thêm ảnh mới vào các thư mục `datasets/rose/train/`.")
    
    # Kiểm tra dataset hiện có
    if os.path.exists(os.path.join(DATASET_PATH, "train")):
        classes = os.listdir(os.path.join(DATASET_PATH, "train"))
        st.write(f"Tìm thấy **{len(classes)} lớp** bệnh: `{', '.join(classes)}`")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Bắt đầu Train lại toàn bộ", use_container_width=True):
                train_model()
    else:
        st.error("Không tìm thấy folder dữ liệu. Vui lòng kiểm tra lại cấu trúc thư mục `datasets/rose/train`.")

st.sidebar.title("Thông tin dự án")
st.sidebar.info("""
- **Đối tượng:** Lá & Hoa Hồng.
- **Mô hình:** ResNet18 (Transfer Learning).
- **Thư viện:** PyTorch, Timm, Streamlit.
""")