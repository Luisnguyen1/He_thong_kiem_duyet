from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import io
import os
import tempfile
import traceback
import opennsfw2 as n2
import requests
import logging
import sys
import mimetypes
import time
from werkzeug.utils import secure_filename

# Chọn path ghi được
home = os.path.expanduser("~")  # sẽ là /home/user
custom_weights_path = os.path.join(home, ".opennsfw2")

# Tạo nếu chưa tồn tại
os.makedirs(custom_weights_path, exist_ok=True)
os.environ["OPENNSFW2_WEIGHTS_PATH"] = custom_weights_path

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)

def is_supported_image(filename, content_type=None):
    """Kiểm tra xem file có phải là ảnh được hỗ trợ không"""
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            return True
    
    if content_type and content_type.lower() in SUPPORTED_IMAGE_MIMES:
        return True
    
    return False

def is_supported_video(filename, content_type=None):
    """Kiểm tra xem file có phải là video được hỗ trợ không"""
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in SUPPORTED_VIDEO_EXTENSIONS:
            return True
    
    if content_type and content_type.lower() in SUPPORTED_VIDEO_MIMES:
        return True
    
    return False

app = Flask(__name__)

# cấu hình
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB giới hạn kích thước upload
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def validate_file_size(file_size, max_size=MAX_CONTENT_LENGTH):
    """Kiểm tra kích thước file"""
    return file_size <= max_size

# ngưỡng để gắn nhãn NSFW (có thể thay đổi tuỳ ứng dụng)
NSFW_THRESHOLD = 0.5

# Các định dạng file được hỗ trợ
SUPPORTED_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', 
    '.webp', '.ico', '.ppm', '.pgm', '.pbm', '.pnm'
}

SUPPORTED_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', 
    '.m4v', '.3gp', '.3g2', '.asf', '.rm', '.rmvb', '.vob'
}

SUPPORTED_IMAGE_MIMES = {
    'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 
    'image/tiff', 'image/webp', 'image/x-icon', 'image/vnd.microsoft.icon'
}

SUPPORTED_VIDEO_MIMES = {
    'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
    'video/x-flv', 'video/webm', 'video/x-matroska', 'video/3gpp',
    'video/x-ms-wmv', 'video/x-ms-asf'
}

# --- Optional: preload model (nếu opennsfw2 hỗ trợ preload, gọi theo docs) ---
# Nếu opennsfw2 có hàm load_model bạn có thể gọi ở đây. Nếu không có, predict_image
# thường sẽ tự lo. Mình giữ dòng comment để bạn bật khi cần:
# n2.load_model(device='cpu')  # ví dụ

def predict_pil_image(pil_img):
    """
    Nhận PIL.Image, trả dict kết quả từ opennsfw2.predict_image.
    opennsfw2.accepts pillow Image objects or file path.
    """
    try:
        # Chuyển đổi sang RGB nếu cần thiết (cho các định dạng có alpha channel hoặc palette)
        if pil_img.mode not in ('RGB', 'L'):
            pil_img = pil_img.convert('RGB')
        
        # some versions accept PIL.Image directly
        prob = n2.predict_image(pil_img)
        # đảm bảo trả float
        prob = float(prob)
        return {
            "nsfw_probability": prob,
            "is_nsfw": prob >= NSFW_THRESHOLD,
            "image_mode": pil_img.mode,
            "image_size": pil_img.size
        }
    except Exception as e:
        logger.warning(f"Direct PIL prediction failed: {e}, trying file fallback")
        # fallback: lưu tạm và gọi bằng file path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        try:
            # Đảm bảo ảnh ở định dạng RGB trước khi lưu
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            pil_img.save(tmp, format="JPEG", quality=95)
            tmp.close()
            prob = float(n2.predict_image(tmp.name))
            return {
                "nsfw_probability": prob,
                "is_nsfw": prob >= NSFW_THRESHOLD,
                "image_mode": pil_img.mode,
                "image_size": pil_img.size
            }
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

@app.route('/predict', methods=['POST'])
def predict():
    """
    Hỗ trợ:
    - multipart/form-data với field 'file' (upload file ảnh)
    - JSON { "url": "https://..." } => server sẽ fetch và dùng ảnh đó
    Trả JSON: { nsfw_probability: 0.123, is_nsfw: true, detail: ... }
    """
    logger.info("Received prediction request")
    try:
        # 1) check file in form-data
        if 'file' in request.files:
            f = request.files['file']
            if f.filename == '':
                return jsonify({"error": "No filename provided"}), 400

            # Kiểm tra định dạng file
            if not is_supported_image(f.filename, f.content_type):
                return jsonify({
                    "error": "Unsupported image format", 
                    "supported_formats": list(SUPPORTED_IMAGE_EXTENSIONS),
                    "received_filename": f.filename,
                    "received_content_type": f.content_type
                }), 400

            # read bytes
            data = f.read()
            if not data:
                return jsonify({"error": "Empty file"}), 400

            # Kiểm tra kích thước file
            if not validate_file_size(len(data)):
                return jsonify({
                    "error": "File too large", 
                    "max_size_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
                    "received_size_mb": len(data) // (1024 * 1024)
                }), 400

            # open with PIL to avoid "file signature not found" errors
            try:
                img = Image.open(io.BytesIO(data))
                # Lưu thông tin gốc trước khi verify
                original_format = img.format
                original_mode = img.mode
                img.verify()  # verify file integrity (does not load image fully)
            except UnidentifiedImageError:
                return jsonify({"error": "Uploaded file is not a valid image"}), 400
            except Exception as e:
                return jsonify({"error": f"Cannot process image: {str(e)}"}), 400
            
            # reopen to get usable Image object (verify() can make file unusable)
            try:
                img = Image.open(io.BytesIO(data))
                result = predict_pil_image(img)
                result["original_format"] = original_format
                result["filename"] = secure_filename(f.filename)
                return jsonify({"success": True, "source": "upload", "result": result}), 200
            except Exception as e:
                return jsonify({"error": f"Cannot process image after verification: {str(e)}"}), 400

        # 2) check JSON with url
        if request.is_json:
            body = request.get_json()
            url = body.get("url")
            if url:
                # fetch remote image
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    resp = requests.get(url, timeout=15, headers=headers, stream=True)
                    resp.raise_for_status()
                    
                    # Kiểm tra content-type
                    content_type = resp.headers.get('content-type', '')
                    if content_type and not any(mime in content_type.lower() for mime in SUPPORTED_IMAGE_MIMES):
                        return jsonify({
                            "error": "URL does not point to a supported image format",
                            "content_type": content_type,
                            "supported_mimes": list(SUPPORTED_IMAGE_MIMES)
                        }), 400
                    
                    data = resp.content
                    
                    # Kiểm tra kích thước
                    if not validate_file_size(len(data)):
                        return jsonify({
                            "error": "Remote file too large", 
                            "max_size_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
                            "received_size_mb": len(data) // (1024 * 1024)
                        }), 400
                    
                    # try open with PIL
                    try:
                        img = Image.open(io.BytesIO(data))
                        original_format = img.format
                        img.verify()
                        # reopen after verify
                        img = Image.open(io.BytesIO(data))
                    except UnidentifiedImageError:
                        return jsonify({"error": "Fetched resource is not a valid image"}), 400
                    except Exception as e:
                        return jsonify({"error": f"Cannot process remote image: {str(e)}"}), 400
                    
                    result = predict_pil_image(img)
                    result["original_format"] = original_format
                    result["content_type"] = content_type
                    return jsonify({"success": True, "source": url, "result": result}), 200
                except requests.RequestException as e:
                    return jsonify({"error": "Failed to fetch image from url", "detail": str(e)}), 400

        return jsonify({"error": "No image provided. Send multipart form-data 'file' or JSON with 'url'."}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "Internal server error", "detail": str(e), "trace": tb}), 500


def process_video_file(video_path, filename="video"):
    """
    Xử lý file video và trả về kết quả kiểm duyệt
    """
    try:
        # Kiểm tra kích thước file
        file_size = os.path.getsize(video_path)
        max_video_size = MAX_CONTENT_LENGTH * 3  # Video cho phép lớn hơn: 30MB
        
        if not validate_file_size(file_size, max_video_size):
            return {
                "error": "Video file too large", 
                "max_size_mb": max_video_size // (1024 * 1024),
                "received_size_mb": file_size // (1024 * 1024)
            }, 400

        logger.info(f"Processing video: {filename}, size: {file_size} bytes")
        
        # Gọi hàm xử lý video của opennsfw2
        # Lưu ý: hàm này có thể tốn nhiều thời gian và CPU/RAM
        elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(video_path)

        if not nsfw_probabilities:
            return {
                "success": True,
                "result": {
                    "is_nsfw": False,
                    "max_nsfw_probability": 0.0,
                    "total_frames": 0,
                    "filename": secure_filename(filename),
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "details": "Video is empty or could not be processed."
                }
            }, 200

        # Xử lý kết quả
        max_prob = max(nsfw_probabilities)
        min_prob = min(nsfw_probabilities)
        avg_prob = sum(nsfw_probabilities) / len(nsfw_probabilities)
        is_nsfw = max_prob >= NSFW_THRESHOLD

        # Đếm các frame có khả năng là NSFW
        nsfw_frames_count = sum(1 for prob in nsfw_probabilities if prob >= NSFW_THRESHOLD)
        
        # Thống kê thêm
        high_risk_frames = sum(1 for p in nsfw_probabilities if p >= 0.8)
        medium_risk_frames = sum(1 for p in nsfw_probabilities if 0.5 <= p < 0.8)

        result = {
            "is_nsfw": is_nsfw,
            "max_nsfw_probability": float(max_prob),
            "min_nsfw_probability": float(min_prob),
            "avg_nsfw_probability": float(avg_prob),
            "total_frames": len(nsfw_probabilities),
            "nsfw_frames_count": nsfw_frames_count,
            "high_risk_frames": high_risk_frames,
            "medium_risk_frames": medium_risk_frames,
            "filename": secure_filename(filename),
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "video_duration_seconds": round(max(elapsed_seconds) if elapsed_seconds else 0, 2)
        }

        logger.info(f"Video processing completed: {result['total_frames']} frames, {result['nsfw_frames_count']} NSFW frames")
        return {"success": True, "result": result}, 200
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Video processing error: {e}")
        return {"error": "Internal server error during video processing", "detail": str(e), "trace": tb}, 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Hỗ trợ:
    - multipart/form-data với field 'file' (upload file video)
    - JSON { "url": "https://..." } => server sẽ fetch và dùng video đó
    Xử lý video, trả về kết quả kiểm duyệt.
    Lưu ý: Chức năng này yêu cầu `opencv-python`.
    """
    logger.info("Received video prediction request")
    
    try:
        # 1) check file in form-data
        if 'file' in request.files:
            video_file = request.files['file']
            
            if video_file.filename == '':
                return jsonify({"error": "No filename provided"}), 400

            # Kiểm tra định dạng video
            if not is_supported_video(video_file.filename, video_file.content_type):
                return jsonify({
                    "error": "Unsupported video format", 
                    "supported_formats": list(SUPPORTED_VIDEO_EXTENSIONS),
                    "received_filename": video_file.filename,
                    "received_content_type": video_file.content_type
                }), 400

            # Lấy extension gốc để giữ định dạng
            original_ext = os.path.splitext(video_file.filename.lower())[1] or '.mp4'
            
            # Lưu video vào file tạm để xử lý
            # Sử dụng delete=False và tự xóa để tương thích với Windows
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=original_ext)
            try:
                video_file.save(tmp.name)
                tmp.close()  # Đóng file để process khác có thể mở
                
                response_data, status_code = process_video_file(tmp.name, video_file.filename)
                response_data["source"] = "upload"
                return jsonify(response_data), status_code
                
            finally:
                # Dọn dẹp file tạm
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        # 2) check JSON with url
        if request.is_json:
            body = request.get_json()
            url = body.get("url")
            if url:
                # fetch remote video
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    # Kiểm tra URL trước khi tải
                    logger.info(f"Fetching video from URL: {url}")
                    resp = requests.head(url, timeout=10, headers=headers, allow_redirects=True)
                    
                    # Kiểm tra content-type
                    content_type = resp.headers.get('content-type', '')
                    if content_type and not any(mime in content_type.lower() for mime in SUPPORTED_VIDEO_MIMES):
                        # Thử kiểm tra extension từ URL nếu content-type không rõ ràng
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        ext = os.path.splitext(parsed_url.path.lower())[1]
                        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                            return jsonify({
                                "error": "URL does not point to a supported video format",
                                "content_type": content_type,
                                "url_extension": ext,
                                "supported_mimes": list(SUPPORTED_VIDEO_MIMES),
                                "supported_extensions": list(SUPPORTED_VIDEO_EXTENSIONS)
                            }), 400
                    
                    # Kiểm tra content-length nếu có
                    content_length = resp.headers.get('content-length')
                    if content_length:
                        file_size = int(content_length)
                        max_size = MAX_CONTENT_LENGTH * 3  # 30MB cho video
                        if file_size > max_size:
                            return jsonify({
                                "error": "Remote video file too large", 
                                "max_size_mb": max_size // (1024 * 1024),
                                "remote_size_mb": file_size // (1024 * 1024)
                            }), 400
                    
                    # Tải video xuống
                    logger.info(f"Downloading video from: {url}")
                    resp = requests.get(url, timeout=60, headers=headers, stream=True)
                    resp.raise_for_status()
                    
                    # Xác định extension từ URL hoặc content-type
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    url_ext = os.path.splitext(parsed_url.path.lower())[1]
                    
                    if url_ext in SUPPORTED_VIDEO_EXTENSIONS:
                        video_ext = url_ext
                    elif 'mp4' in content_type:
                        video_ext = '.mp4'
                    elif 'webm' in content_type:
                        video_ext = '.webm'
                    elif 'avi' in content_type:
                        video_ext = '.avi'
                    else:
                        video_ext = '.mp4'  # default
                    
                    # Lưu vào file tạm
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=video_ext)
                    try:
                        total_size = 0
                        max_size = MAX_CONTENT_LENGTH * 3  # 30MB
                        
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                total_size += len(chunk)
                                if total_size > max_size:
                                    return jsonify({
                                        "error": "Remote video file too large during download", 
                                        "max_size_mb": max_size // (1024 * 1024)
                                    }), 400
                                tmp.write(chunk)
                        
                        tmp.close()
                        
                        # Xử lý video
                        filename = os.path.basename(parsed_url.path) or "remote_video" + video_ext
                        response_data, status_code = process_video_file(tmp.name, filename)
                        response_data["source"] = url
                        response_data["result"]["content_type"] = content_type
                        return jsonify(response_data), status_code
                        
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass
                            
                except requests.RequestException as e:
                    return jsonify({"error": "Failed to fetch video from url", "detail": str(e)}), 400

        return jsonify({"error": "No video provided. Send multipart form-data 'file' or JSON with 'url'."}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "Internal server error", "detail": str(e), "trace": tb}), 500


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return jsonify({
        "message": "opennsfw2 Flask server. POST /predict with multipart 'file' or JSON {url}",
        "endpoints": {
            "/predict": "Image NSFW detection",
            "/predict_video": "Video NSFW detection",
            "/supported_formats": "Get supported file formats",
            "/health": "Health check"
        }
    }), 200

@app.route('/supported_formats')
def supported_formats():
    """Trả về danh sách các định dạng file được hỗ trợ"""
    return jsonify({
        "supported_image_extensions": sorted(list(SUPPORTED_IMAGE_EXTENSIONS)),
        "supported_video_extensions": sorted(list(SUPPORTED_VIDEO_EXTENSIONS)),
        "supported_image_mimes": sorted(list(SUPPORTED_IMAGE_MIMES)),
        "supported_video_mimes": sorted(list(SUPPORTED_VIDEO_MIMES)),
        "max_file_size_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
        "nsfw_threshold": NSFW_THRESHOLD
    }), 200

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "NSFW Detection API",
        "version": "2.0",
        "timestamp": str(time.time())
    }), 200


if __name__ == '__main__':
    # Thêm logging khi start server
    logger.info("Starting NSFW Detection Flask Server...")
    logger.info(f"Server will run on host=0.0.0.0, port=2002")
    logger.info(f"NSFW Threshold: {NSFW_THRESHOLD}")
    
    # chạy dev server, production nên dùng gunicorn/uvicorn
    app.run(host='0.0.0.0', port=7860, debug=True)