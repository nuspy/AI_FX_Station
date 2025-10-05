"""
API Configuration

Configuration settings for ForexGPT API service.
"""
from pydantic import BaseSettings, Field
from typing import Optional


class APISettings(BaseSettings):
    """API configuration settings"""

    # Server
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=False, env="API_RELOAD")
    workers: int = Field(default=4, env="API_WORKERS")

    # CORS
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")

    # Cache
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL")  # 5 minutes

    # Database
    database_url: str = Field(
        default="sqlite:///forexgpt.db",
        env="DATABASE_URL"
    )

    # Sentiment
    sentiment_update_interval_seconds: int = Field(default=3600, env="SENTIMENT_UPDATE_INTERVAL")

    # Calendar
    calendar_update_interval_seconds: int = Field(default=1800, env="CALENDAR_UPDATE_INTERVAL")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = APISettings()
