from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to .env — works regardless of where uvicorn is launched from
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Force .env to override any stale Windows system environment variables
load_dotenv(dotenv_path=_ENV_FILE, override=True)


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    max_ocr_retries: int = 3
    max_structuring_retries: int = 2
    ocr_confidence_threshold: float = 0.65

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
