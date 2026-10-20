import io

from fastapi.testclient import TestClient
from PIL import Image

try:
    from backend.main import app
except ImportError:
    from main import app

client = TestClient(app)


def test_rejects_non_image():
    response = client.post('/predict', files={'file': ('notes.txt', b'not an image', 'text/plain')})
    assert response.status_code == 415


def test_rejects_invalid_image_bytes():
    response = client.post('/predict', files={'file': ('leaf.jpg', b'not an image', 'image/jpeg')})
    assert response.status_code == 400


def test_health_reports_model_configuration():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['ok'] is True


def test_valid_image_requires_real_checkpoint():
    image = Image.new('RGB', (16, 16), 'green')
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    response = client.post('/predict', files={'file': ('leaf.png', stream.getvalue(), 'image/png')})
    assert response.status_code == 503
