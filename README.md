# NSFW Content Detector

Hệ thống kiểm duyệt nội dung NSFW sử dụng Flask và opennsfw2.

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
docker run -p 2002:2002 nsfw-detector
```

## API Endpoints

- `GET /` - Giao diện web
- `POST /predict` - Kiểm tra hình ảnh (multipart file hoặc JSON với URL)
- `POST /predict_video` - Kiểm tra video (multipart file)
- `GET /api` - Thông tin API

## Sử dụng

Sau khi chạy, truy cập:
- http://localhost:2002 (direct access)
- http://localhost (qua nginx nếu bật profile with-nginx)

## Cấu hình

- **Port**: 2002 (có thể thay đổi trong docker-compose.yml)
- **Memory limit**: 2GB
- **CPU limit**: 1 core
- **Upload limit**: 10MB (có thể thay đổi trong main.py)
- **Workers**: 2 gunicorn workers

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