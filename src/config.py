"""Configuration management for the project."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # Google Gemini API
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_model: str = "gemini-2.0-flash"
    google_max_tokens: int = 2000

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_path: Path = project_root / "data" / "raw"
    data_processed_path: Path = project_root / "data" / "processed"
    reports_path: Path = project_root / "reports"

    # Feature generation
    feature_cache_enabled: bool = (
        os.getenv("FEATURE_CACHE_ENABLED", "true").lower() == "true"
    )
    feature_cache_ttl: int = int(os.getenv("FEATURE_CACHE_TTL", "86400"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Testing
    testing: bool = os.getenv("TESTING", "false").lower() == "true"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()

# Ensure directories exist
settings.data_raw_path.mkdir(parents=True, exist_ok=True)
settings.data_processed_path.mkdir(parents=True, exist_ok=True)
settings.reports_path.mkdir(parents=True, exist_ok=True)
