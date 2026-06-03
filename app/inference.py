from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class ModelLoadError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class WasteClassifier:
    def __init__(self, model_path: Path, class_names_path: Path, image_size: int = 224) -> None:
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.image_size = image_size
        self.class_names = self._load_class_names(self.class_names_path)
        self.interpreter = self._load_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def _load_class_names(path: Path) -> list[str]:
        if not path.exists():
            raise ModelLoadError(f"Class names file not found: {path}")

        class_names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not class_names:
            raise ModelLoadError("Class names file is empty")
        return class_names

    @staticmethod
    def _load_interpreter(model_path: Path):
        if not model_path.exists():
            raise ModelLoadError(f"TFLite model not found: {model_path}")

        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError as exc:
                raise ModelLoadError(
                    "Install either tflite-runtime or tensorflow to run inference."
                ) from exc

        return Interpreter(model_path=str(model_path))

    def predict(self, image_bytes: bytes) -> list[Prediction]:
        image = self._preprocess(image_bytes)
        input_detail = self.input_details[0]
        output_detail = self.output_details[0]

        if input_detail["dtype"] == np.uint8:
            scale, zero_point = input_detail["quantization"]
            if scale:
                image = (image / scale + zero_point).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        else:
            image = image.astype(input_detail["dtype"])

        self.interpreter.set_tensor(input_detail["index"], image)
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        scores = self._softmax_if_needed(scores)

        predictions = [
            Prediction(label=label, confidence=float(scores[index]))
            for index, label in enumerate(self.class_names)
        ]
        predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)

    top = predictions[0]
    if top.confidence < 0.85:
        return [Prediction(label="Tidak Dikenali", confidence=top.confidence)]
    
    return predictions

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("Uploaded file is not a valid image.") from exc

        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)
        return array

    @staticmethod
    def _softmax_if_needed(scores: np.ndarray) -> np.ndarray:
        if np.isclose(scores.sum(), 1.0, atol=1e-3) and np.all(scores >= 0):
            return scores

        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)
