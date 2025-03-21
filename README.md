<h1 align="center">NHẬN DẠNG HÀNH VI SINH VIÊN TRONG LỚP HỌC SỬ DỤNG CAMERA THÔNG MINH</h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5a919d1-d081-4d12-819e-5fb18ce91a68" width="200"/>
  <img src="https://github.com/user-attachments/assets/59dec55d-7825-422f-b80c-ac6915e3775a" width="170"/>
</p>
<p align="center">
  <a href="https://www.facebook.com/DNUAIoTLab">
    <img src="https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge" alt="Made by AIoTLab"/>
  </a>
  <a href="https://fitdnu.net/">
    <img src="https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge" alt="Fit DNU"/>
  </a>
  <a href="https://dainam.edu.vn">
    <img src="https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge" alt="DaiNam University"/>
  </a>
</p>

## **Giới thiệu**  
Hệ thống Nhận dạng Hành vi Sinh viên trong Lớp học sử dụng Camera và trí tuệ nhân tạo (AI) để phát hiện và nhận diện hành vi của sinh viên trong lớp học thông qua luồng video trực tiếp. Hệ thống phù hợp cho các ứng dụng giám sát hành vi học tập, giúp theo dõi và đánh giá hành vi sinh viên trong môi trường học đường.

Hệ thống hoạt động theo cơ chế:

Nhận diện hành vi của sinh viên trong video bằng mô hình YOLOv7.

Theo dõi và phân tích hành vi của từng sinh viên trong lớp học.

Cung cấp thông tin hành vi và thống kê các hoạt động của sinh viên trong lớp học theo thời gian thực.

**Công nghệ sử dụng:**
- **YOLOv7**: Nhận diện hành vi sinh viên từ video trực tiếp và sử dụng PyTorch..
- **OpenCV**: Xử lý hình ảnh và video để phát hiện và phân tích hành vi. 
- **PyTorch Framework**: train dữ liệu 
---

## **Thiết bị sử dụng trong bài**
**Phần cứng**
- Camera Fnkvison. 
- Laptop để dữ liệu từ camera

**Phần mềm**
- Hệ điều hành: Windows.
- Môi trường Python: Python hoặc anaconda).

---
## **Yêu cầu hệ thống**  
- **Python** 3.10 trở lên   
- Bạn cần pip install requirements.txt trong file để khởi tạo


## **Hướng dẫn cài đặt**  

### **1. Cài đặt các thư viện cần thiết**  
Chạy lệnh sau để cài đặt các thư viện Python yêu cầu:  
```
pip install -r requirements.txt
```

### **2. Hướng dẫn thực hiện**  
Sơ đồ cấu trúc:
![image](https://github.com/user-attachments/assets/6316f2b9-8f30-4047-ae53-01c06700e9c5)


#### **2.1. Sử dụng phần mềm ODM để lấy địa chỉ và port/onvif1**  
Định dạng nguồn: 
```
rtsp://[username]:[password]@[Địa-chỉ-IP]:554/onvif
```
Ví dụ: Tên username bắt buộc phải là admin
```
rtsp://admin:danhphong2004@192.168.0.53:554/onvif1
```
#### **2.6. Tiến hành train mô hình** 
**Bước 1: Sử dụng data mà bạn đã thu nhập được từ video lớp học**

![image](https://github.com/user-attachments/assets/e77c5d69-de7b-4648-9876-e4c4d00c28a6)

**Bước 2: Sử dụng python để tách ảnh ra từng frame**
**Bước 3: Tiến hành gán nhãn thủ công sử dụng LabelImg**
![test_batch0_labels](https://github.com/user-attachments/assets/c1ac25d0-e30e-4c36-bc97-7c3c7db7dc73)
Gán nhãn nó sẽ tự động sinh ra file Txt cho từng ảnh
![image](https://github.com/user-attachments/assets/8191290e-6ed8-481a-bd72-68f1bcc7e5b7)
**Bước 4: Bắt đầu train mô hình**
```
python train.py
```
Sau khi train thành công sẽ có 2 file: "Best.pt và Last.pt" nên dùng file best.pt để nhận được chính xác nhất
![image](https://github.com/user-attachments/assets/525aaf1a-a756-4d44-afd3-70ab4d53c65c)
#### **2.7. Sau khi huấn luyện xong bạn có thể dùng file yolov7/Dectect.py để nhận dạng thử xem mô hình của mình train có tốt không** 
```
python Detect.py
```
#### **2.8. Sau khi test xong bạn có thể dùng file Dectect.py để nhận diện hành vi sinh viên trong lớp học bằng camera** 
```
python Detect.py
```
## **Ghi chú: bạn nên cài Cuda về để chạy PyTorch**  
---

