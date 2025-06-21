"""
AICUBE Embedding2Embedding API - Configuration

Centralized application configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class AICUBESettings(BaseSettings):
    """
    AICUBE Embedding2Embedding application settings
    """
    
    # Application information
    API_NAME: str = "aicube-embedding2embedding"
    API_VERSION: str = "v1"
    ENVIRONMENT: str = Field(default="development", description="Execution environment")
    DEBUG: bool = Field(default=True, description="Debug mode")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Logging format")
    LOG_FILE: Optional[str] = Field(default=None, description="Log file")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed origins for CORS"
    )
    
    # Model settings
    MODELS_PATH: str = Field(default="./aicube-models", description="Models directory")
    MAX_EMBEDDING_DIMENSION: int = Field(default=4096, description="Maximum embedding dimension")
    MAX_BATCH_SIZE: int = Field(default=32, description="Maximum batch size")
    
    # Performance settings
    DEVICE: str = Field(default="cpu", description="Processing device (cpu/cuda)")
    MAX_WORKERS: int = Field(default=4, description="Maximum number of workers")
    
    # Rate limiting (for future implementation)
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Request limit per minute")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Time window in seconds")
    
    # Cache settings
    ENABLE_MODEL_CACHE: bool = Field(default=True, description="Enable model caching")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Monitoring settings
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=8001, description="Metrics port")
    
    # Timeout settings
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")
    MODEL_LOAD_TIMEOUT: int = Field(default=120, description="Model loading timeout")
    
    # AICUBE specific settings
    AICUBE_TECHNOLOGY_NAME: str = Field(default="AICUBE TECHNOLOGY", description="Technology name")
    AICUBE_MODELS: List[str] = Field(
        default=[
            "FastAPI Framework",
            "PyTorch Deep Learning",
            "MLP Neural Networks",
            "Advanced ML Algorithms"
        ],
        description="Technologies used in AICUBE implementation"
    )
    
    class Config:
        env_file = "aicube.env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
aicube_settings = AICUBESettings()


def get_aicube_settings() -> AICUBESettings:
    """
    Function to get AICUBE settings (useful for dependency injection)
    """
    return aicube_settings