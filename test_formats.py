#!/usr/bin/env python3
"""
Script để test các định dạng file được hỗ trợ
"""

import requests
import json
import os

# URL của API
API_URL = "http://localhost:7860"

def test_supported_formats():
    """Test endpoint /supported_formats"""
    try:
        response = requests.get(f"{API_URL}/supported_formats")
        if response.status_code == 200:
            data = response.json()
            print("=== SUPPORTED FORMATS ===")
            print(f"Image extensions: {data['supported_image_extensions']}")
            print(f"Video extensions: {data['supported_video_extensions']}")
            print(f"Max file size: {data['max_file_size_mb']} MB")
            print(f"NSFW threshold: {data['nsfw_threshold']}")
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_health_check():
    """Test endpoint /health"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("=== HEALTH CHECK ===")
            print(f"Status: {data['status']}")
            print(f"Service: {data['service']}")
            print(f"Version: {data['version']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_image_prediction(image_path):
    """Test prediction với file ảnh local"""
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"=== IMAGE PREDICTION: {os.path.basename(image_path)} ===")
            print(f"Success: {data['success']}")
            result = data['result']
            print(f"NSFW Probability: {result['nsfw_probability']:.4f}")
            print(f"Is NSFW: {result['is_nsfw']}")
            print(f"Image Size: {result.get('image_size', 'Unknown')}")
            print(f"Image Mode: {result.get('image_mode', 'Unknown')}")
            print(f"Original Format: {result.get('original_format', 'Unknown')}")
            return True
        else:
            print(f"Prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Prediction error: {e}")
        return False

def test_url_prediction(image_url):
    """Test prediction với URL ảnh"""
    try:
        payload = {"url": image_url}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"=== URL PREDICTION: {image_url} ===")
            print(f"Success: {data['success']}")
            result = data['result']
            print(f"NSFW Probability: {result['nsfw_probability']:.4f}")
            print(f"Is NSFW: {result['is_nsfw']}")
            print(f"Image Size: {result.get('image_size', 'Unknown')}")
            print(f"Content Type: {result.get('content_type', 'Unknown')}")
            return True
        else:
            print(f"URL prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"URL prediction error: {e}")
        return False

if __name__ == "__main__":
    print("Testing NSFW Detection API...")
    print("="*50)
    
    # Test basic endpoints
    test_health_check()
    print()
    test_supported_formats()
    print()
    
    # Test với URL ảnh mẫu (safe image)
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    test_url_prediction(sample_url)
    print()
    
    # Test với file local nếu có
    test_files = [
        "test_image.jpg",
        "test_image.png", 
        "test_image.gif",
        "test_image.webp"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            test_image_prediction(test_file)
            print()
    
    print("Testing completed!")