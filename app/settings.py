from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FS Waste Classifier API"
    model_path: Path = Path("models/waste_classifier.tflite")
    class_names_path: Path = Path("models/class_names.txt")
    image_size: int = 224
    max_upload_mb: int = 8

    model_config = SettingsConfigDict(env_prefix="FS_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
