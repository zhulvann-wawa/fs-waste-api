from fastapi.testclient import TestClient

from app.main import app, get_classifier


class DummyClassifier:
    image_size = 224
    class_names = ["Kaca", "Kardus", "Kertas", "Logam", "Plastik", "Residu"]

    def predict(self, image_bytes: bytes):
        from app.inference import Prediction

        return [
            Prediction(label="Plastik", confidence=0.9),
            Prediction(label="Residu", confidence=0.1),
        ]


def test_model_info():
    app.dependency_overrides[get_classifier] = lambda: DummyClassifier()
    client = TestClient(app)

    response = client.get("/model")

    assert response.status_code == 200
    assert response.json()["classes"] == DummyClassifier.class_names
    app.dependency_overrides.clear()


def test_classes():
    app.dependency_overrides[get_classifier] = lambda: DummyClassifier()
    client = TestClient(app)

    response = client.get("/classes")

    assert response.status_code == 200
    assert response.json() == {"classes": DummyClassifier.class_names}
    app.dependency_overrides.clear()


def test_rejects_non_image_upload():
    app.dependency_overrides[get_classifier] = lambda: DummyClassifier()
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not-image", "text/plain")},
    )

    assert response.status_code == 415
    app.dependency_overrides.clear()
