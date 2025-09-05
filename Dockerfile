# Sử dụng Python 3.9 slim base image
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các dependencies hệ thống cần thiết cho OpenCV và image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker layer caching
COPY requirements.txt .

# Cập nhật requirements.txt để bao gồm các dependencies cần thiết
RUN pip install --no-cache-dir \
    flask \
    Pillow \
    opennsfw2 \
    requests \
    opencv-python-headless \
    gunicorn

# Copy toàn bộ source code
COPY . .

# Tạo user non-root để chạy ứng dụng (security best practice)
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 2002

# Sử dụng gunicorn cho production
CMD ["gunicorn", "--bind", "0.0.0.0:2002", "--workers", "2", "--timeout", "120", "main:app"]