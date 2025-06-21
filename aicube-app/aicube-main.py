"""
AICUBE Embedding2Embedding API - Main Application

Aplicação principal da API para tradução de embeddings entre diferentes espaços vetoriais.
Utiliza FastAPI com arquiteturas modulares e suporte a múltiplos modelos de tradução.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn
from contextlib import asynccontextmanager

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aicube_app.aicube_api.aicube_endpoints import aicube_router
from aicube_app.aicube_core.aicube_config import aicube_settings
from aicube_app.aicube_core.aicube_model_manager import aicube_model_manager
from aicube_app.aicube_core.aicube_logging import aicube_setup_logging

# Configurar logging estruturado
logger = structlog.get_logger("aicube.embedding2embedding.main")


@asynccontextmanager
async def aicube_lifespan(app: FastAPI):
    """
    Lifecycle manager para inicialização e limpeza da aplicação AICUBE
    """
    logger.info("Iniciando aplicação AICUBE Embedding2Embedding...")
    
    # Inicializar logging
    aicube_setup_logging()
    
    # Carregar modelos pré-treinados
    await aicube_model_manager.initialize_models()
    
    logger.info("Aplicação AICUBE inicializada com sucesso", 
                version=aicube_settings.API_VERSION,
                models_loaded=len(aicube_model_manager.get_available_models()))
    
    yield
    
    # Cleanup
    logger.info("Encerrando aplicação AICUBE Embedding2Embedding...")
    await aicube_model_manager.cleanup()


# Criar aplicação FastAPI
aicube_app = FastAPI(
    title="AICUBE Embedding2Embedding API",
    description="""
    API para tradução de embeddings entre diferentes espaços vetoriais de modelos de linguagem natural.
    
    Desenvolvido pela AICUBE TECHNOLOGY utilizando:
    - Qube LCM Model
    - Qube Neural Memory  
    - Qube Agentic Workflows
    - Qube Computer Vision
    
    Esta API permite converter embeddings de um modelo de origem para um modelo de destino,
    preservando o significado semântico através de técnicas avançadas de alinhamento de espaços vetoriais.
    """,
    version=aicube_settings.API_VERSION,
    docs_url="/aicube-docs",
    redoc_url="/aicube-redoc",
    openapi_url="/aicube-openapi.json",
    lifespan=aicube_lifespan
)

# Configurar CORS
aicube_app.add_middleware(
    CORSMiddleware,
    allow_origins=aicube_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Registrar routers
aicube_app.include_router(aicube_router, prefix="/api/v1")


@aicube_app.get("/", tags=["Root"])
async def aicube_root():
    """
    Endpoint raiz da API AICUBE Embedding2Embedding
    """
    return {
        "message": "AICUBE Embedding2Embedding API",
        "version": aicube_settings.API_VERSION,
        "technology": "AICUBE TECHNOLOGY",
        "powered_by": [
            "Qube LCM Model",
            "Qube Neural Memory",
            "Qube Agentic Workflows", 
            "Qube Computer Vision"
        ],
        "docs": "/aicube-docs",
        "health": "/api/v1/health"
    }


@aicube_app.exception_handler(Exception)
async def aicube_global_exception_handler(request, exc):
    """
    Handler global para exceções não tratadas
    """
    logger.error("Erro não tratado na aplicação AICUBE", 
                error=str(exc), 
                path=str(request.url))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor AICUBE",
            "message": "Por favor, entre em contato com o suporte técnico",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "aicube-main:aicube_app",
        host="0.0.0.0",
        port=8000,
        reload=aicube_settings.DEBUG,
        log_level=aicube_settings.LOG_LEVEL.lower()
    )