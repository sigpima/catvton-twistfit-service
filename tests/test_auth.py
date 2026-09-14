import os

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import require_api_key

os.environ["CATVTON_API_KEY"] = "test-secret-key"

probe_app = FastAPI()


@probe_app.get("/protected", dependencies=[Depends(require_api_key)])
def protected_route():
    return {"ok": True}


client = TestClient(probe_app)


def test_rejects_request_with_no_api_key():
    response = client.get("/protected")
    assert response.status_code == 401


def test_rejects_request_with_wrong_api_key():
    response = client.get("/protected", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_accepts_request_with_correct_api_key():
    response = client.get("/protected", headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
