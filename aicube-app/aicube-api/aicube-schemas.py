"""
AICUBE Embedding2Embedding API - Pydantic Schemas

Esquemas de dados para requisições e respostas da API utilizando Pydantic.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


class AICUBEEmbeddingTranslateRequest(BaseModel):
    """
    Schema para requisição de tradução de embedding
    """
    origem: str = Field(..., description="Modelo de origem do embedding")
    destino: str = Field(..., description="Modelo de destino para tradução")
    embedding: Union[List[float], List[List[float]]] = Field(
        ..., 
        description="Embedding(s) para tradução - pode ser um vetor único ou batch"
    )
    include_metadata: bool = Field(
        default=False, 
        description="Incluir metadados adicionais na resposta"
    )
    
    @validator('embedding')
    def validate_embedding(cls, v):
        if not v:
            raise ValueError("Embedding não pode estar vazio")
        
        # Verificar se é lista de floats ou lista de listas de floats
        if isinstance(v[0], (int, float)):
            # Embedding único
            if len(v) == 0:
                raise ValueError("Embedding deve ter pelo menos uma dimensão")
        elif isinstance(v[0], list):
            # Batch de embeddings
            if len(v) == 0:
                raise ValueError("Batch de embeddings não pode estar vazio")
            for i, emb in enumerate(v):
                if not emb:
                    raise ValueError(f"Embedding no índice {i} está vazio")
        else:
            raise ValueError("Embedding deve ser uma lista de números ou lista de listas de números")
        
        return v
    
    @validator('origem', 'destino')
    def validate_model_names(cls, v):
        if not v or not v.strip():
            raise ValueError("Nome do modelo não pode estar vazio")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "origem": "bert_base_uncased",
                "destino": "t5_base",
                "embedding": [0.12, 0.45, -0.67, 0.89, -0.23],
                "include_metadata": True
            }
        }


class AICUBEEmbeddingTranslateResponse(BaseModel):
    """
    Schema para resposta de tradução de embedding
    """
    origem: str = Field(..., description="Modelo de origem")
    destino: str = Field(..., description="Modelo de destino")
    embedding_traduzido: Union[List[float], List[List[float]]] = Field(
        ..., 
        description="Embedding(s) traduzido(s)"
    )
    aicube_technology: str = Field(default="AICUBE TECHNOLOGY LLC", description="Tecnologia utilizada")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadados da tradução")
    
    class Config:
        schema_extra = {
            "example": {
                "origem": "bert_base_uncased",
                "destino": "t5_base",
                "embedding_traduzido": [0.08, 0.33, -0.10, 0.71, -0.15],
                "aicube_technology": "AICUBE TECHNOLOGY LLC",
                "metadata": {
                    "aicube_model_id": "aicube-bert-to-t5",
                    "duration_ms": 15.2,
                    "cosine_similarity": 0.94,
                    "powered_by": ["Qube LCM Model", "Qube Neural Memory"]
                }
            }
        }


class AICUBEModelInfo(BaseModel):
    """
    Informações de um modelo de tradução
    """
    model_id: str = Field(..., description="ID único do modelo")
    source_model: str = Field(..., description="Modelo de origem")
    target_model: str = Field(..., description="Modelo de destino")
    source_dimension: int = Field(..., description="Dimensão do embedding de origem")
    target_dimension: int = Field(..., description="Dimensão do embedding de destino")
    version: str = Field(..., description="Versão do modelo")
    description: str = Field(..., description="Descrição do modelo")
    aicube_technology: List[str] = Field(..., description="Tecnologias AICUBE utilizadas")
    created_by: str = Field(default="AICUBE TECHNOLOGY LLC", description="Criado por")


class AICUBEModelsListResponse(BaseModel):
    """
    Lista de modelos disponíveis
    """
    aicube_models: List[str] = Field(..., description="Lista de modelos disponíveis")
    aicube_model_pairs: List[Dict[str, str]] = Field(..., description="Pares de modelos suportados")
    total_models: int = Field(..., description="Total de modelos disponíveis")
    aicube_technology: str = Field(default="AICUBE TECHNOLOGY LLC", description="Tecnologia")
    powered_by: List[str] = Field(..., description="Tecnologias AICUBE utilizadas")
    
    class Config:
        schema_extra = {
            "example": {
                "aicube_models": ["aicube-bert-to-t5", "aicube-mpnet-to-ada002"],
                "aicube_model_pairs": [
                    {
                        "source": "bert_base_uncased",
                        "target": "t5_base",
                        "aicube_translator_id": "aicube-bert-to-t5"
                    }
                ],
                "total_models": 2,
                "aicube_technology": "AICUBE TECHNOLOGY LLC",
                "powered_by": ["Qube LCM Model", "Qube Neural Memory", "Qube Agentic Workflows", "Qube Computer Vision"]
            }
        }


class AICUBEHealthResponse(BaseModel):
    """
    Resposta do health check
    """
    status: str = Field(..., description="Status do serviço")
    aicube_service: str = Field(..., description="Nome do serviço AICUBE")
    version: str = Field(..., description="Versão da API")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp da verificação")
    aicube_technology: str = Field(default="AICUBE TECHNOLOGY LLC", description="Tecnologia")
    models_loaded: int = Field(..., description="Número de modelos carregados")
    powered_by: List[str] = Field(..., description="Tecnologias AICUBE utilizadas")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "aicube_service": "aicube-embedding2embedding",
                "version": "v1",
                "timestamp": "2023-06-19T12:00:00Z",
                "aicube_technology": "AICUBE TECHNOLOGY LLC",
                "models_loaded": 3,
                "powered_by": ["Qube LCM Model", "Qube Neural Memory", "Qube Agentic Workflows", "Qube Computer Vision"]
            }
        }


class AICUBEErrorResponse(BaseModel):
    """
    Schema padrão para respostas de erro
    """
    error: str = Field(..., description="Tipo do erro")
    message: str = Field(..., description="Mensagem detalhada do erro")
    aicube_service: str = Field(default="aicube-embedding2embedding", description="Serviço AICUBE")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp do erro")
    request_id: Optional[str] = Field(default=None, description="ID da requisição para rastreamento")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Modelo de origem não suportado: invalid_model",
                "aicube_service": "aicube-embedding2embedding",
                "timestamp": "2023-06-19T12:00:00Z",
                "request_id": "req_12345"
            }
        }


class AICUBEBatchTranslateRequest(BaseModel):
    """
    Schema para tradução em lote
    """
    origem: str = Field(..., description="Modelo de origem")
    destino: str = Field(..., description="Modelo de destino")
    embeddings: List[List[float]] = Field(..., description="Lista de embeddings para tradução")
    include_metadata: bool = Field(default=False, description="Incluir metadados")
    
    @validator('embeddings')
    def validate_embeddings_batch(cls, v):
        if not v:
            raise ValueError("Lista de embeddings não pode estar vazia")
        
        if len(v) > 32:  # Limitar batch size
            raise ValueError("Tamanho máximo de batch é 32 embeddings")
        
        for i, emb in enumerate(v):
            if not emb:
                raise ValueError(f"Embedding no índice {i} está vazio")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "origem": "bert_base_uncased",
                "destino": "t5_base",
                "embeddings": [
                    [0.12, 0.45, -0.67],
                    [0.89, -0.23, 0.56]
                ],
                "include_metadata": True
            }
        }


class AICUBEBatchTranslateResponse(BaseModel):
    """
    Schema para resposta de tradução em lote
    """
    origem: str = Field(..., description="Modelo de origem")
    destino: str = Field(..., description="Modelo de destino")
    embeddings_traduzidos: List[List[float]] = Field(..., description="Lista de embeddings traduzidos")
    batch_size: int = Field(..., description="Tamanho do batch processado")
    aicube_technology: str = Field(default="AICUBE TECHNOLOGY LLC", description="Tecnologia utilizada")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadados da tradução em lote")
    
    class Config:
        schema_extra = {
            "example": {
                "origem": "bert_base_uncased",
                "destino": "t5_base",
                "embeddings_traduzidos": [
                    [0.08, 0.33, -0.10],
                    [0.71, -0.15, 0.42]
                ],
                "batch_size": 2,
                "aicube_technology": "AICUBE TECHNOLOGY LLC",
                "metadata": {
                    "total_duration_ms": 25.6,
                    "average_cosine_similarity": 0.93
                }
            }
        }


class AICUBEStatisticsResponse(BaseModel):
    """
    Estatísticas de uso da API
    """
    aicube_loaded_models: int = Field(..., description="Modelos carregados")
    aicube_available_models: int = Field(..., description="Modelos disponíveis")
    aicube_usage_stats: Dict[str, int] = Field(..., description="Estatísticas de uso por modelo")
    aicube_technology: str = Field(default="AICUBE TECHNOLOGY LLC", description="Tecnologia")
    aicube_version: str = Field(..., description="Versão AICUBE")
    powered_by: List[str] = Field(..., description="Tecnologias utilizadas")
    
    class Config:
        schema_extra = {
            "example": {
                "aicube_loaded_models": 3,
                "aicube_available_models": 5,
                "aicube_usage_stats": {
                    "aicube-bert-to-t5": 150,
                    "aicube-mpnet-to-ada002": 89
                },
                "aicube_technology": "AICUBE TECHNOLOGY LLC",
                "aicube_version": "1.0.0-aicube",
                "powered_by": ["Qube LCM Model", "Qube Neural Memory"]
            }
        }