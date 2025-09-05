from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import io
import os
import tempfile
import traceback
import opennsfw2 as n2
import requests

app = Flask(__name__)

# cấu hình
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB giới hạn kích thước upload
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ngưỡng để gắn nhãn NSFW (có thể thay đổi tuỳ ứng dụng)
NSFW_THRESHOLD = 0.5

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
        # some versions accept PIL.Image directly
        prob = n2.predict_image(pil_img)
        # đảm bảo trả float
        prob = float(prob)
        return {
            "nsfw_probability": prob,
            "is_nsfw": prob >= NSFW_THRESHOLD
        }
    except Exception:
        # fallback: lưu tạm và gọi bằng file path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        try:
            pil_img.save(tmp, format="JPEG")
            tmp.close()
            prob = float(n2.predict_image(tmp.name))
            return {
                "nsfw_probability": prob,
                "is_nsfw": prob >= NSFW_THRESHOLD
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
    try:
        # 1) check file in form-data
        if 'file' in request.files:
            f = request.files['file']
            if f.filename == '':
                return jsonify({"error": "No filename provided"}), 400

            # read bytes
            data = f.read()
            if not data:
                return jsonify({"error": "Empty file"}), 400

            # open with PIL to avoid "file signature not found" errors
            try:
                img = Image.open(io.BytesIO(data))
                img.verify()  # verify file integrity (does not load image fully)
            except UnidentifiedImageError:
                return jsonify({"error": "Uploaded file is not a valid image"}), 400
            except Exception:
                # some images require re-opening after verify
                try:
                    img = Image.open(io.BytesIO(data)).convert('RGB')
                except Exception:
                    return jsonify({"error": "Cannot open image file"}), 400
            else:
                # reopen to get usable Image object (verify() can make file unusable)
                img = Image.open(io.BytesIO(data)).convert('RGB')
                result = predict_pil_image(img)
                return jsonify({"success": True, "source": "upload", "result": result}), 200

        # 2) check JSON with url
        if request.is_json:
            body = request.get_json()
            url = body.get("url")
            if url:
                # fetch remote image
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    data = resp.content
                    # try open with PIL
                    try:
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                    except UnidentifiedImageError:
                        return jsonify({"error": "Fetched resource is not a valid image"}), 400
                    result = predict_pil_image(img)
                    return jsonify({"success": True, "source": url, "result": result}), 200
                except requests.RequestException as e:
                    return jsonify({"error": "Failed to fetch image from url", "detail": str(e)}), 400

        return jsonify({"error": "No image provided. Send multipart form-data 'file' or JSON with 'url'."}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "Internal server error", "detail": str(e), "trace": tb}), 500


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Hỗ trợ upload video qua multipart/form-data với field 'file'.
    Xử lý video, trả về kết quả kiểm duyệt.
    Lưu ý: Chức năng này yêu cầu `opencv-python`.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['file']

    if video_file.filename == '':
        return jsonify({"error": "No filename provided"}), 400

    # Lưu video vào file tạm để xử lý
    # Sử dụng delete=False và tự xóa để tương thích với Windows
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        video_file.save(tmp.name)
        tmp.close()  # Đóng file để process khác có thể mở

        # Gọi hàm xử lý video của opennsfw2
        # Lưu ý: hàm này có thể tốn nhiều thời gian và CPU/RAM
        elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(tmp.name)

        if not nsfw_probabilities:
            return jsonify({
                "success": True,
                "source": "upload",
                "result": {
                    "is_nsfw": False,
                    "max_nsfw_probability": 0.0,
                    "details": "Video is empty or could not be processed."
                }
            }), 200

        # Xử lý kết quả
        max_prob = max(nsfw_probabilities)
        is_nsfw = max_prob >= NSFW_THRESHOLD

        # Tìm các frame có khả năng là NSFW
        nsfw_timestamps = [
            round(elapsed_seconds[i], 2) for i, prob in enumerate(nsfw_probabilities)
            if prob >= NSFW_THRESHOLD
        ]

        result = {
            "is_nsfw": is_nsfw,
            "max_nsfw_probability": float(max_prob),
            "nsfw_frames_count": len(nsfw_timestamps),
            "nsfw_timestamps_seconds": nsfw_timestamps,
        }

        return jsonify({"success": True, "source": "upload", "result": result}), 200

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "Internal server error during video processing", "detail": str(e), "trace": tb}), 500
    finally:
        # Dọn dẹp file tạm
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return jsonify({"message": "opennsfw2 Flask server. POST /predict with multipart 'file' or JSON {url}"}), 200


if __name__ == '__main__':
    # chạy dev server, production nên dùng gunicorn/uvicorn
    app.run(host='0.0.0.0', port=2002, debug=True)
