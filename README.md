---
title: He Thong Kiem Duyet
emoji: 🏆
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# Hệ thống Kiểm duyệt Nội dung NSFW

Hệ thống kiểm duyệt nội dung NSFW sử dụng **opennsfw2** và **Flask** để phân tích hình ảnh và video, hỗ trợ nhiều định dạng file khác nhau.

## 🚀 Tính năng mới

### 🖼️ Hỗ trợ nhiều định dạng Hình ảnh
- **Định dạng**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.ico`, `.ppm`, `.pgm`, `.pbm`, `.pnm`
- **MIME types**: `image/jpeg`, `image/png`, `image/gif`, `image/bmp`, `image/tiff`, `image/webp`, `image/x-icon`
- **Tự động chuyển đổi**: Hỗ trợ alpha channel và palette mode
- **Validation**: Kiểm tra định dạng và kích thước file

### 🎥 Hỗ trợ nhiều định dạng Video  
- **Định dạng**: `.mp4`, `.avi`, `.mov`, `.wmv`, `.flv`, `.webm`, `.mkv`, `.m4v`, `.3gp`, `.3g2`, `.asf`, `.rm`, `.rmvb`, `.vob`
- **Thống kê chi tiết**: Số frame NSFW, timestamps, duration, file size
- **Phân loại risk**: High risk và medium risk frames

### 🔍 API mở rộng
- `GET /supported_formats` - Xem định dạng được hỗ trợ
- `GET /health` - Health check
- Cải thiện error handling và validation
- Response chi tiết hơn với metadata

## Docker Deployment

### Chạy với Docker Compose (khuyến nghị)

```bash
# Build và chạy ứng dụng
docker-compose up --build

# Chạy ở background
docker-compose up -d --build

# Dừng ứng dụng
docker-compose down
```

### Chạy với Nginx reverse proxy (tùy chọn)

```bash
# Chạy cả app và nginx
docker-compose --profile with-nginx up --build

# Hoặc chỉ chạy nginx
docker-compose --profile with-nginx up -d nginx
```

### Chạy container riêng lẻ

```bash
# Build image
docker build -t nsfw-detector .

# Chạy container
docker run -p 7860:7860 nsfw-detector
```

## API Endpoints

### 1. `POST /predict` - Kiểm duyệt hình ảnh
**Input:**
- **Multipart form-data**: `file` (ảnh upload)
- **JSON**: `{"url": "https://example.com/image.jpg"}` (URL ảnh)

**Output:**
```json
{
  "success": true,
  "source": "upload",
  "result": {
    "nsfw_probability": 0.1234,
    "is_nsfw": false,
    "image_mode": "RGB",
    "image_size": [1920, 1080],
    "original_format": "JPEG",
    "filename": "example.jpg"
  }
}
```

### 2. `POST /predict_video` - Kiểm duyệt video
**Input:**
- **Multipart form-data**: `file` (video upload)

**Output:**
```json
{
  "success": true,
  "source": "upload", 
  "result": {
    "is_nsfw": false,
    "max_nsfw_probability": 0.3456,
    "min_nsfw_probability": 0.0123,
    "avg_nsfw_probability": 0.1234,
    "total_frames": 150,
    "nsfw_frames_count": 5,
    "high_risk_frames": 1,
    "medium_risk_frames": 4,
    "nsfw_timestamps_seconds": [12.5, 25.3, 45.1],
    "filename": "video.mp4",
    "file_size_mb": 15.6,
    "video_duration_seconds": 60.0
  }
}
```

### 3. `GET /supported_formats` - Thông tin định dạng
```json
{
  "supported_image_extensions": [".jpg", ".png", "..."],
  "supported_video_extensions": [".mp4", ".avi", "..."],
  "max_file_size_mb": 10,
  "nsfw_threshold": 0.5
}
```

### 4. `GET /health` - Health check
```json
{
  "status": "healthy",
  "service": "NSFW Detection API",
  "version": "2.0"
}
```

### 5. Legacy endpoints
- `GET /` - Giao diện web (cải tiến với hiển thị định dạng hỗ trợ)
- `GET /api` - Thông tin API

## Local Development

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy server
```bash
python main.py
```

Server sẽ chạy tại: `http://localhost:7860`

### Test API
```bash
python test_formats.py
```

## Sử dụng

Sau khi chạy, truy cập:
- http://localhost:7860 (development)
- http://localhost:2002 (docker)
- http://localhost (qua nginx nếu bật profile with-nginx)

## Cấu hình

- **Port**: 7860 (dev), 2002 (docker)
- **Memory limit**: 2GB
- **CPU limit**: 1 core  
- **Upload limit**: 10MB images, 20MB videos
- **Workers**: 2 gunicorn workers
- **NSFW Threshold**: 0.5 (có thể thay đổi)

## Xử lý lỗi cải tiến

### Validation tự động
- Kiểm tra định dạng file dựa trên extension và MIME type
- Kiểm tra kích thước file
- Kiểm tra tính hợp lệ của image/video
- Tự động chuyển đổi color mode

### Error responses chi tiết
```json
{
  "error": "Unsupported image format",
  "supported_formats": [".jpg", ".png", "..."],
  "received_filename": "test.xyz", 
  "received_content_type": "application/octet-stream"
}
```

## Logs

```bash
# Xem logs
docker-compose logs -f

# Xem logs của service cụ thể
docker-compose logs -f nsfw-detector
```

## Troubleshooting

Nếu gặp lỗi memory, tăng memory limit trong docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

## Performance & Tối ưu

- Tự động cleanup temporary files
- Optimized image processing với fallback
- Efficient video frame analysis
- Better error handling và logging
- Security improvements với filename sanitization