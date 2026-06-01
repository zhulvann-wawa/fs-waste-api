from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.inference import InvalidImageError, ModelLoadError, WasteClassifier
from app.schemas import ClassScore, ClassesResponse, HealthResponse, ModelInfoResponse, PredictionResponse
from app.settings import Settings, get_settings

app = FastAPI(
    title="FS Waste Classifier API",
    version="1.0.0",
    description="API klasifikasi sampah untuk kelas Kaca, Kardus, Kertas, Logam, Plastik, dan Residu.",
)


@lru_cache
def get_classifier() -> WasteClassifier:
    settings = get_settings()
    return WasteClassifier(
        model_path=settings.model_path,
        class_names_path=settings.class_names_path,
        image_size=settings.image_size,
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    try:
        get_classifier()
        return HealthResponse(status="ok", model_loaded=True)
    except ModelLoadError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/model", response_model=ModelInfoResponse, tags=["model"])
def model_info(classifier: WasteClassifier = Depends(get_classifier)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_type="tflite",
        image_size=classifier.image_size,
        classes=classifier.class_names,
    )


@app.get("/classes", response_model=ClassesResponse, tags=["model"])
def classes(classifier: WasteClassifier = Depends(get_classifier)) -> ClassesResponse:
    return ClassesResponse(classes=classifier.class_names)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    classifier: WasteClassifier = Depends(get_classifier),
) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File harus berupa gambar.",
        )

    payload = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Ukuran file maksimal {settings.max_upload_mb} MB.",
        )

    try:
        predictions = classifier.predict(payload)
    except InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    top_prediction = predictions[0]
    return PredictionResponse(
        label=top_prediction.label,
        confidence=top_prediction.confidence,
        predictions=[
            ClassScore(label=item.label, confidence=item.confidence)
            for item in predictions
        ],
    )
