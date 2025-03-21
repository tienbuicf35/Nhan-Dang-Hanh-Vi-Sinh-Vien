import cv2

# Cấu hình camera Yoosee
USERNAME = "admin"
PASSWORD = "danhphong2004"
IP = "192.168.0.102"
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/onvif1"

# Mở luồng video
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Không thể kết nối đến camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận video.")
            break

        # Hiển thị video
        cv2.imshow('Camera Stream', frame)

        # Bấm 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()