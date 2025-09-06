# Sử dụng Python 3.9 slim base image
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các dependencies hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 2002

# Chạy ứng dụng bằng Flask
CMD ["python", "main.py"]
