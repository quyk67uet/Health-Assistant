# Health Assistant

Health Assistant là một ứng dụng web giúp dự đoán nguy cơ mắc các bệnh như tiểu đường (diabetes), bệnh tim (heart disease), và Parkinson. Ngoài ra, ứng dụng còn có một Medical Bot để hỗ trợ người dùng hỏi đáp về tình trạng bệnh, cách chữa trị, và phòng ngừa.

## Tính Năng

- **Dự Đoán Bệnh**: Sử dụng các mô hình Machine Learning để dự đoán nguy cơ mắc bệnh diabetes, heart, và parkinson dựa trên các thông số y tế.
- **Medical Bot**: Trợ lý ảo y tế giúp người dùng tìm hiểu về bệnh tình của mình, cung cấp các thông tin hữu ích về điều trị và phòng ngừa.
- **Giao Diện Web**: Ứng dụng được xây dựng bằng Streamlit để dễ dàng tương tác với người dùng.
- **API**: Sử dụng FastAPI để cung cấp các API endpoint cho việc dự đoán và tương tác với Medical Bot.
- **Triển Khai Bằng Docker**: Ứng dụng được container hóa và triển khai bằng Docker, sử dụng Docker Compose để quản lý các container.

### Mô Hình Dự Đoán
![Mô Hình Dự Đoán](https://drive.google.com/uc?id=1tLwc4d5TTcHH86r2nHlgP9tNBnwIKSfb)

### Medical Bot
![Medical Bot](https://drive.google.com/uc?id=1x2swrQZ9ugbVC4iRubLONNUg7oe6-ZZP)

### Database sử dụng MongoDB
![Giao Diện Ứng Dụng](https://drive.google.com/uc?id=1n1nCpqQIZtRLfXqJbyiLNuBWGepeGDTu)

## Yêu Cầu Hệ Thống

- Python 3.8+
- Docker & Docker Compose

## Cài Đặt

1. **Clone Repository**:
    ```bash
    git clone https://github.com/yourusername/health-assistant.git
    cd health-assistant
    ```

2. **Cài Đặt Các Thư Viện Cần Thiết**:
    Nếu không sử dụng Docker, bạn có thể cài đặt các thư viện cần thiết bằng pip:
    ```bash
    pip install -r requirements.txt
    ```

3. **Cấu Hình Docker Compose**:
    Đảm bảo Docker và Docker Compose đã được cài đặt trên hệ thống của bạn. Chạy lệnh sau để khởi chạy ứng dụng:
    ```bash
    docker-compose up --build
    ```

## Sử Dụng

1. **Dự Đoán Bệnh**:
   - Nhập các thông số y tế cần thiết vào các trường trên giao diện.
   - Bấm nút "Dự Đoán" để nhận kết quả từ mô hình.

2. **Tương Tác Với Medical Bot**:
   - Nhập câu hỏi của bạn về tình trạng bệnh, cách chữa trị, hoặc phòng ngừa vào ô chat.
   - Medical Bot sẽ phản hồi ngay lập tức với thông tin hữu ích.

