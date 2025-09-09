# Sử dụng Python 3.9 slim base image
FROM python:3.9-slim

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
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR $HOME/app

# Copy requirements và cài đặt dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 7860

# Chạy ứng dụng bằng Flask
CMD ["python", "main.py"]