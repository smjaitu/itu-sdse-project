import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from vial_classifier.serving import app as serving
from vial_classifier.serving.drift import detect_drift, filter_logs_by_time

BASELINE = {
    "confidence_statistics": {"mean": 0.9, "std": 0.05, "min": 0.7, "max": 1.0},
    "class_distribution": {"class_0_count": 22, "class_1_count": 86, "total": 108},
}


def make_logs(n, confidence, prediction, age_days=0):
    timestamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    return [
        {"timestamp": timestamp, "prediction": prediction, "confidence": confidence}
        for _ in range(n)
    ]


def test_no_logs_means_no_drift():
    result = detect_drift([], baseline_stats=BASELINE)
    assert result["overall_drift_detected"] is False
    assert result["num_samples"] == 0


def test_old_logs_are_filtered_out():
    logs = make_logs(5, 0.9, 1, age_days=10) + make_logs(3, 0.9, 1)
    assert len(filter_logs_by_time(logs, days=7)) == 3


def test_large_shift_in_confidence_and_classes_is_detected():
    result = detect_drift(make_logs(30, 0.2, 0), baseline_stats=BASELINE)
    assert result["overall_drift_detected"] is True
    assert result["confidence_drift"]["drift_detected"] is True
    assert result["class_distribution_drift"]["drift_detected"] is True


def test_drift_endpoint_reads_log_and_baseline(tmp_path, monkeypatch):
    log_path = tmp_path / "inference_log.json"
    log_path.write_text(json.dumps(make_logs(30, 0.2, 0)))
    (tmp_path / "drift_detection").mkdir()
    (tmp_path / "drift_detection" / "drift_baseline_stats.json").write_text(json.dumps(BASELINE))
    monkeypatch.setattr(serving, "INFERENCE_LOG_PATH", log_path)
    monkeypatch.setattr(serving, "MODEL_PATH", tmp_path)

    response = TestClient(serving.app).get("/drift")

    assert response.status_code == 200
    assert response.json()["num_samples"] == 30
    assert response.json()["overall_drift_detected"] is True