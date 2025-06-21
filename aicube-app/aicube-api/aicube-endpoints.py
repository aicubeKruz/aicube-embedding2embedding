"""
AICUBE Embedding2Embedding API - Endpoints

Endpoints REST para tradução de embeddings utilizando tecnologia AICUBE.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
import numpy as np
from typing import List, Dict, Any
import time
import structlog
from datetime import datetime

from aicube_app.aicube_api.aicube_schemas import (
    AICUBEEmbeddingTranslateRequest,
    AICUBEEmbeddingTranslateResponse,
    AICUBEModelsListResponse,
    AICUBEHealthResponse,
    AICUBEErrorResponse,
    AICUBEBatchTranslateRequest,
    AICUBEBatchTranslateResponse,
    AICUBEStatisticsResponse
)
from aicube_app.aicube_core.aicube_config import aicube_settings, get_aicube_settings, AICUBESettings
from aicube_app.aicube_core.aicube_model_manager import aicube_model_manager
from aicube_app.aicube_core.aicube_logging import aicube_logger

logger = structlog.get_logger("aicube.api.endpoints")

# Criar router AICUBE
aicube_router = APIRouter()


@aicube_router.post(
    "/translate",
    response_model=AICUBEEmbeddingTranslateResponse,
    status_code=status.HTTP_200_OK,
    summary="Traduzir Embedding",
    description="Traduz um embedding de um modelo de origem para um modelo de destino usando tecnologia AICUBE",
    response_description="Embedding traduzido com metadados opcionais",
    tags=["AICUBE Translation"]
)
async def aicube_translate_embedding(
    request: AICUBEEmbeddingTranslateRequest,
    settings: AICUBESettings = Depends(get_aicube_settings)
):
    """
    Endpoint principal para tradução de embeddings AICUBE
    """
    start_time = time.time()
    
    try:
        # Validar dimensões do embedding
        if isinstance(request.embedding[0], (int, float)):
            # Embedding único
            embedding_array = np.array(request.embedding, dtype=np.float32)
            is_batch = False
        else:
            # Batch de embeddings
            embedding_array = np.array(request.embedding, dtype=np.float32)
            is_batch = True
        
        # Verificar dimensões máximas
        if embedding_array.shape[-1] > settings.MAX_EMBEDDING_DIMENSION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dimensão do embedding ({embedding_array.shape[-1]}) excede o máximo permitido ({settings.MAX_EMBEDDING_DIMENSION})"
            )
        
        # Verificar tamanho do batch
        if is_batch and len(embedding_array) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tamanho do batch ({len(embedding_array)}) excede o máximo permitido ({settings.MAX_BATCH_SIZE})"
            )
        
        # Executar tradução
        translated_embedding, metadata = await aicube_model_manager.aicube_translate_embedding(
            embedding=embedding_array,
            source_model=request.origem,
            target_model=request.destino
        )
        
        # Converter resultado para lista
        if is_batch:
            translated_list = translated_embedding.tolist()
        else:
            translated_list = translated_embedding.tolist()
        
        # Preparar resposta
        response_data = {
            "origem": request.origem,
            "destino": request.destino,
            "embedding_traduzido": translated_list,
            "aicube_technology": settings.AICUBE_TECHNOLOGY_NAME
        }
        
        # Incluir metadados se solicitado
        if request.include_metadata:
            response_data["metadata"] = metadata
        
        # Log da operação
        duration = time.time() - start_time
        aicube_logger.log_translation(
            source_model=request.origem,
            target_model=request.destino,
            embedding_dim=embedding_array.shape[-1],
            batch_size=len(embedding_array) if is_batch else 1,
            duration=duration,
            cosine_similarity=metadata.get("cosine_similarity")
        )
        
        return AICUBEEmbeddingTranslateResponse(**response_data)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        aicube_logger.log_error(e, {"endpoint": "/translate", "request": request.dict()})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno do servidor AICUBE"
        )


@aicube_router.post(
    "/translate/batch",
    response_model=AICUBEBatchTranslateResponse,
    status_code=status.HTTP_200_OK,
    summary="Traduzir Embeddings em Lote",
    description="Traduz múltiplos embeddings de uma vez usando tecnologia AICUBE",
    tags=["AICUBE Translation"]
)
async def aicube_translate_batch(
    request: AICUBEBatchTranslateRequest,
    settings: AICUBESettings = Depends(get_aicube_settings)
):
    """
    Endpoint para tradução em lote de embeddings
    """
    start_time = time.time()
    
    try:
        # Converter para numpy array
        embeddings_array = np.array(request.embeddings, dtype=np.float32)
        
        # Validações
        if embeddings_array.shape[-1] > settings.MAX_EMBEDDING_DIMENSION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dimensão do embedding excede o máximo permitido"
            )
        
        # Executar tradução
        translated_embeddings, metadata = await aicube_model_manager.aicube_translate_embedding(
            embedding=embeddings_array,
            source_model=request.origem,
            target_model=request.destino
        )
        
        # Preparar resposta
        batch_metadata = {
            "total_duration_ms": round((time.time() - start_time) * 1000, 2),
            "average_cosine_similarity": metadata.get("cosine_similarity"),
            "batch_processed": len(request.embeddings)
        }
        
        response_data = {
            "origem": request.origem,
            "destino": request.destino,
            "embeddings_traduzidos": translated_embeddings.tolist(),
            "batch_size": len(request.embeddings),
            "aicube_technology": settings.AICUBE_TECHNOLOGY_NAME
        }
        
        if request.include_metadata:
            response_data["metadata"] = batch_metadata
        
        return AICUBEBatchTranslateResponse(**response_data)
        
    except Exception as e:
        aicube_logger.log_error(e, {"endpoint": "/translate/batch"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno do servidor AICUBE"
        )


@aicube_router.get(
    "/models",
    response_model=AICUBEModelsListResponse,
    status_code=status.HTTP_200_OK,
    summary="Listar Modelos Disponíveis",
    description="Retorna lista de modelos de tradução de embeddings disponíveis na tecnologia AICUBE",
    tags=["AICUBE Models"]
)
async def aicube_list_models(
    settings: AICUBESettings = Depends(get_aicube_settings)
):
    """
    Listar modelos de tradução disponíveis
    """
    try:
        available_models = aicube_model_manager.get_available_models()
        model_pairs = aicube_model_manager.aicube_get_supported_model_pairs()
        
        return AICUBEModelsListResponse(
            aicube_models=available_models,
            aicube_model_pairs=model_pairs,
            total_models=len(available_models),
            aicube_technology=settings.AICUBE_TECHNOLOGY_NAME,
            powered_by=settings.AICUBE_MODELS
        )
        
    except Exception as e:
        aicube_logger.log_error(e, {"endpoint": "/models"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao obter lista de modelos AICUBE"
        )


@aicube_router.get(
    "/models/{model_id}",
    summary="Obter Informações de Modelo",
    description="Retorna informações detalhadas de um modelo específico AICUBE",
    tags=["AICUBE Models"]
)
async def aicube_get_model_info(model_id: str):
    """
    Obter informações detalhadas de um modelo específico
    """
    try:
        model_info = await aicube_model_manager.aicube_get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modelo AICUBE não encontrado: {model_id}"
            )
        
        return {
            "model_id": model_info.model_id,
            "source_model": model_info.source_model,
            "target_model": model_info.target_model,
            "source_dimension": model_info.source_dimension,
            "target_dimension": model_info.target_dimension,
            "version": model_info.version,
            "description": model_info.description,
            "aicube_technology": model_info.aicube_technology,
            "created_by": model_info.created_by
        }
        
    except HTTPException:
        raise
    except Exception as e:
        aicube_logger.log_error(e, {"endpoint": f"/models/{model_id}"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao obter informações do modelo AICUBE"
        )


@aicube_router.get(
    "/health",
    response_model=AICUBEHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Verifica o status de saúde do serviço AICUBE",
    tags=["AICUBE Health"]
)
async def aicube_health_check(
    settings: AICUBESettings = Depends(get_aicube_settings)
):
    """
    Health check do serviço AICUBE
    """
    try:
        models_loaded = len(aicube_model_manager.aicube_models)
        
        return AICUBEHealthResponse(
            status="healthy",
            aicube_service=settings.API_NAME,
            version=settings.API_VERSION,
            timestamp=datetime.utcnow(),
            aicube_technology=settings.AICUBE_TECHNOLOGY_NAME,
            models_loaded=models_loaded,
            powered_by=settings.AICUBE_MODELS
        )
        
    except Exception as e:
        aicube_logger.log_error(e, {"endpoint": "/health"})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "aicube_service": settings.API_NAME,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@aicube_router.get(
    "/statistics",
    response_model=AICUBEStatisticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Estatísticas de Uso",
    description="Retorna estatísticas de uso dos modelos AICUBE",
    tags=["AICUBE Statistics"]
)
async def aicube_get_statistics(
    settings: AICUBESettings = Depends(get_aicube_settings)
):
    """
    Obter estatísticas de uso da API AICUBE
    """
    try:
        stats = aicube_model_manager.aicube_get_statistics()
        
        return AICUBEStatisticsResponse(
            aicube_loaded_models=stats["aicube_loaded_models"],
            aicube_available_models=stats["aicube_available_models"],
            aicube_usage_stats=stats["aicube_usage_stats"],
            aicube_technology=stats["aicube_technology"],
            aicube_version=stats["aicube_version"],
            powered_by=stats["aicube_powered_by"]
        )
        
    except Exception as e:
        aicube_logger.log_error(e, {"endpoint": "/statistics"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao obter estatísticas AICUBE"
        )


# Handler de erro personalizado para este router
@aicube_router.exception_handler(HTTPException)
async def aicube_http_exception_handler(request, exc: HTTPException):
    """
    Handler personalizado para exceções HTTP
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "aicube_service": aicube_settings.API_NAME,
            "timestamp": datetime.utcnow().isoformat(),
            "aicube_technology": "AICUBE TECHNOLOGY"
        }
    )