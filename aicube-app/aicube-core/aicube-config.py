"""
AICUBE Embedding2Embedding API - Configuration

Configurações centralizadas da aplicação utilizando Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class AICUBESettings(BaseSettings):
    """
    Configurações da aplicação AICUBE Embedding2Embedding
    """
    
    # Informações da aplicação
    API_NAME: str = "aicube-embedding2embedding"
    API_VERSION: str = "v1"
    ENVIRONMENT: str = Field(default="development", description="Ambiente de execução")
    DEBUG: bool = Field(default=True, description="Modo debug")
    
    # Configurações de servidor
    HOST: str = Field(default="0.0.0.0", description="Host do servidor")
    PORT: int = Field(default=8000, description="Porta do servidor")
    
    # Configurações de logging
    LOG_LEVEL: str = Field(default="INFO", description="Nível de logging")
    LOG_FORMAT: str = Field(default="json", description="Formato de logging")
    LOG_FILE: Optional[str] = Field(default=None, description="Arquivo de log")
    
    # Configurações de CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Origens permitidas para CORS"
    )
    
    # Configurações de modelos
    MODELS_PATH: str = Field(default="./aicube-models", description="Diretório dos modelos")
    MAX_EMBEDDING_DIMENSION: int = Field(default=4096, description="Dimensão máxima de embedding")
    MAX_BATCH_SIZE: int = Field(default=32, description="Tamanho máximo de batch")
    
    # Configurações de performance
    DEVICE: str = Field(default="cpu", description="Dispositivo de processamento (cpu/cuda)")
    MAX_WORKERS: int = Field(default=4, description="Número máximo de workers")
    
    # Rate limiting (para implementação futura)
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Limite de requisições por minuto")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Janela de tempo em segundos")
    
    # Configurações de cache
    ENABLE_MODEL_CACHE: bool = Field(default=True, description="Habilitar cache de modelos")
    CACHE_TTL: int = Field(default=3600, description="TTL do cache em segundos")
    
    # Configurações de monitoramento
    ENABLE_METRICS: bool = Field(default=True, description="Habilitar métricas Prometheus")
    METRICS_PORT: int = Field(default=8001, description="Porta das métricas")
    
    # Configurações de timeouts
    REQUEST_TIMEOUT: int = Field(default=30, description="Timeout de requisição em segundos")
    MODEL_LOAD_TIMEOUT: int = Field(default=120, description="Timeout para carregamento de modelo")
    
    # Configurações específicas AICUBE
    AICUBE_TECHNOLOGY_NAME: str = Field(default="AICUBE TECHNOLOGY", description="Nome da tecnologia")
    AICUBE_MODELS: List[str] = Field(
        default=[
            "Qube LCM Model",
            "Qube Neural Memory",
            "Qube Agentic Workflows",
            "Qube Computer Vision"
        ],
        description="Modelos AICUBE utilizados"
    )
    
    class Config:
        env_file = "aicube.env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instância global das configurações
aicube_settings = AICUBESettings()


def get_aicube_settings() -> AICUBESettings:
    """
    Função para obter as configurações AICUBE (útil para dependency injection)
    """
    return aicube_settings