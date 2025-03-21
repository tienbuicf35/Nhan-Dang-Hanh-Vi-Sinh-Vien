import torch
import cv2
import numpy as np
import time
import csv
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Danh sách hành vi
class_names = ['book', 'using_phone', 'using_laptop', 'write', 'sleep']

# Thiết bị sử dụng (GPU nếu có, nếu không thì dùng CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# Load mô hình YOLOv7 đã huấn luyện
model = attempt_load('last.pt', map_location=device)
model.eval()

# Đường dẫn RTSP từ camera
rtsp_url = "rtsp://admin:danhphong123@192.168.0.107:554/onvif1"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Không thể mở RTSP stream.")
    exit()

# Kích thước ảnh đầu vào cho mô hình
img_size = 640  # Tăng độ phân giải để nhận dạng tốt hơn

# Mở file CSV để lưu kết quả
with open('behaviors.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Behavior", "Confidence"])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không lấy được khung hình.")
            break

        # Tiền xử lý ảnh
        img0 = frame.copy()
        img = letterbox(img0, img_size, stride=32, auto=True)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).float().to(device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # Dự đoán hành vi
        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.4)

        # Hiển thị kết quả lên màn hình
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"{class_names[int(cls)]} {conf:.2f}"

                    # Vẽ hình chữ nhật và nhãn
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img0, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(img0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Lưu thời gian và hành vi vào CSV
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())  # Lấy thời gian hiện tại
                    writer.writerow([current_time, class_names[int(cls)], conf.item()])

        # Hiển thị kết quả
        cv2.imshow("Nhận dạng hành vi học sinh", img0)

        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
