import base64
import json

from fastapi.testclient import TestClient

from vial_classifier.serving import app as serving

# Not used as a context manager, so the lifespan (scoring server) is not started
client = TestClient(serving.app)


def test_health_is_loading_while_scoring_server_is_down(monkeypatch):
    monkeypatch.setattr(serving, "_upstream_health", lambda: None)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_health_is_ok_when_scoring_server_is_up(monkeypatch):
    monkeypatch.setattr(serving, "_upstream_health", lambda: 200)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_reads_version_file(tmp_path, monkeypatch):
    (tmp_path / "version.json").write_text(json.dumps({"model_version": "3", "trained_at": None}))
    monkeypatch.setattr(serving, "MODEL_PATH", tmp_path)
    assert client.get("/version").json()["model_version"] == "3"


def test_raw_image_bytes_are_wrapped_as_base64_instances():
    body, content_type = serving._to_scoring_payload(b"fake-jpeg-bytes", "image/jpeg")
    assert content_type == "application/json"
    assert json.loads(body) == {"instances": [base64.b64encode(b"fake-jpeg-bytes").decode()]}


def test_json_payload_is_passed_through_unchanged():
    body = b'{"instances": ["abc"]}'
    assert serving._to_scoring_payload(body, "application/json") == (body, "application/json")