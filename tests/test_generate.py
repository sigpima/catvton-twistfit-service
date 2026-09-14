import io
import os

from fastapi.testclient import TestClient
from PIL import Image

os.environ["CATVTON_API_KEY"] = "test-secret-key"

from app.main import app  # noqa: E402

client = TestClient(app)


def _fake_image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _stub_generate_fn(person_bytes: bytes, garment_bytes: bytes, cloth_type: str) -> bytes:
    return _fake_image_bytes((1, 2, 3))


def setup_module():
    app.state.generate_fn = _stub_generate_fn


def test_generate_requires_api_key():
    response = client.post(
        "/generate",
        files={
            "person_image": ("person.png", _fake_image_bytes((255, 0, 0)), "image/png"),
            "garment_image": ("garment.png", _fake_image_bytes((0, 255, 0)), "image/png"),
        },
        data={"cloth_type": "upper"},
    )
    assert response.status_code == 401


def test_generate_returns_image_bytes_from_the_injected_generator():
    response = client.post(
        "/generate",
        headers={"X-API-Key": "test-secret-key"},
        files={
            "person_image": ("person.png", _fake_image_bytes((255, 0, 0)), "image/png"),
            "garment_image": ("garment.png", _fake_image_bytes((0, 255, 0)), "image/png"),
        },
        data={"cloth_type": "upper"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == _fake_image_bytes((1, 2, 3))


def test_generate_rejects_an_invalid_cloth_type():
    response = client.post(
        "/generate",
        headers={"X-API-Key": "test-secret-key"},
        files={
            "person_image": ("person.png", _fake_image_bytes((255, 0, 0)), "image/png"),
            "garment_image": ("garment.png", _fake_image_bytes((0, 255, 0)), "image/png"),
        },
        data={"cloth_type": "not-a-real-type"},
    )
    assert response.status_code == 422
