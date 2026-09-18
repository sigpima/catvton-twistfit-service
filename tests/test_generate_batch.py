import base64
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


def _stub_generate_batch_fn(person_bytes_list: list[bytes], garment_bytes: bytes, cloth_type: str) -> list[bytes]:
    return [_fake_image_bytes((1, 2, 3)) for _ in person_bytes_list]


def setup_module():
    app.state.generate_batch_fn = _stub_generate_batch_fn


def _files_for(count: int):
    return [
        ("person_images", (f"person{i}.png", _fake_image_bytes((255, 0, 0)), "image/png")) for i in range(count)
    ] + [("garment_image", ("garment.png", _fake_image_bytes((0, 255, 0)), "image/png"))]


def test_generate_batch_requires_api_key():
    response = client.post("/generate-batch", files=_files_for(2), data={"cloth_type": "upper"})
    assert response.status_code == 401


def test_generate_batch_returns_one_image_per_person_image():
    response = client.post(
        "/generate-batch",
        headers={"X-API-Key": "test-secret-key"},
        files=_files_for(2),
        data={"cloth_type": "upper"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["images"]) == 2
    expected = base64.b64encode(_fake_image_bytes((1, 2, 3))).decode("ascii")
    assert body["images"] == [expected, expected]


def test_generate_batch_works_with_a_single_person_image():
    response = client.post(
        "/generate-batch",
        headers={"X-API-Key": "test-secret-key"},
        files=_files_for(1),
        data={"cloth_type": "lower"},
    )
    assert response.status_code == 200
    assert len(response.json()["images"]) == 1


def test_generate_batch_rejects_an_invalid_cloth_type():
    response = client.post(
        "/generate-batch",
        headers={"X-API-Key": "test-secret-key"},
        files=_files_for(2),
        data={"cloth_type": "not-a-real-type"},
    )
    assert response.status_code == 422
