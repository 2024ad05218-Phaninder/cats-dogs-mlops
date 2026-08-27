"""
Smoke tests for deployed API
Runs basic checks to ensure service is operational
"""

import requests
import json
import sys
from PIL import Image
import io

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed")

def test_model_info():
    """Test model info endpoint"""
    print("Testing /model-info endpoint...")
    response = requests.get(f"{BASE_URL}/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "classes" in data
    assert data["classes"] == ["cat", "dog"]
    print("✅ Model info check passed")

def test_prediction():
    """Test prediction endpoint"""
    print("Testing /predict endpoint...")

    # Create test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Send prediction request
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert data["predicted_class"] in ["cat", "dog"]
    assert 0 <= data["confidence"] <= 1
    print("✅ Prediction check passed")

def test_batch_prediction():
    """Test batch prediction"""
    print("Testing batch prediction...")

    images = []
    for i in range(3):
        img = Image.new('RGB', (224, 224), color=f'#{i*80:02x}{i*80:02x}{i*80:02x}')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        images.append(("files", ("test_{}.jpg".format(i), img_bytes, "image/jpeg")))

    response = requests.post(
        f"{BASE_URL}/batch-predict",
        files=images
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    for result in data:
        assert "predicted_class" in result
        assert "confidence" in result
    print("✅ Batch prediction check passed")

def test_metrics():
    """Test metrics endpoint"""
    print("Testing /metrics endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "fastapi" in response.text or "python" in response.text
    print("✅ Metrics endpoint check passed")

def test_response_time():
    """Test response time is acceptable"""
    print("Testing response time...")
    import time

    start = time.time()
    response = requests.get(f"{BASE_URL}/health")
    elapsed = time.time() - start

    assert elapsed < 1.0  # Should respond in less than 1 second
    print(f"✅ Response time check passed ({elapsed:.3f}s)")

if __name__ == "__main__":
    try:
        print("Starting smoke tests...\n")

        test_health()
        test_model_info()
        test_prediction()
        test_batch_prediction()
        test_metrics()
        test_response_time()

        print("\n" + "="*50)
        print("✅ All smoke tests passed!")
        print("="*50)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
