import torch
import cv2
import numpy as np
import csv
from datetime import datetime
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# --- Cài đặt ---
class_names = ['book', 'using_phone', 'using_laptop', 'write', 'sleep']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# Load model
model = attempt_load('best.pt', map_location=device)
model.eval()

# Đường dẫn video test
video_path = r'C:\Users\Lenovo\Python3.10\student_yolov7\yolov7\IMG_0539.MOV'  # 👉 Thay bằng đường dẫn video của bạn
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Không thể mở video.")
    exit()

# Thông số xử lý
img_size = 640
conf_thres = 0.5
iou_thres = 0.4

# Ghi file CSV
csv_file = 'behavior_from_video.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Label', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

last_label = None
last_write_time = None

def write_to_csv_if_new(label, conf, x1, y1, x2, y2):
    global last_label, last_write_time
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    if label != last_label or (last_write_time and (now - last_write_time).seconds >= 5):
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, label, f"{conf:.2f}", x1, y1, x2, y2])
        last_label = label
        last_write_time = now

# --- Vòng lặp xử lý ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Đã xử lý hết video.")
        break

    img0 = frame.copy()
    img = letterbox(img0, img_size, stride=32, auto=True)[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                label = class_names[int(cls)]
                label_text = f"{label} {conf:.2f}"

                write_to_csv_if_new(label, conf, x1, y1, x2, y2)

                # Vẽ bounding box
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img0, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(img0, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("🎬 Nhận diện hành vi từ video", img0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
