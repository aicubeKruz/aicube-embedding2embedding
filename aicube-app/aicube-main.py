"""
AICUBE Embedding2Embedding API - Main Application

Main application for the API for translating embeddings between different vector spaces.
Uses FastAPI with modular architectures and support for multiple translation models.
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

# Configure structured logging
logger = structlog.get_logger("aicube.embedding2embedding.main")


@asynccontextmanager
async def aicube_lifespan(app: FastAPI):
    """
    Lifecycle manager for AICUBE application initialization and cleanup
    """
    logger.info("Starting AICUBE Embedding2Embedding application...")
    
    # Initialize logging
    aicube_setup_logging()
    
    # Load pre-trained models
    await aicube_model_manager.initialize_models()
    
    logger.info("AICUBE application initialized successfully", 
                version=aicube_settings.API_VERSION,
                models_loaded=len(aicube_model_manager.get_available_models()))
    
    yield
    
    # Cleanup
    logger.info("Shutting down AICUBE Embedding2Embedding application...")
    await aicube_model_manager.cleanup()


# Create FastAPI application
aicube_app = FastAPI(
    title="AICUBE Embedding2Embedding API",
    description="""
    API for translating embeddings between different vector spaces of natural language models.
    
    Developed by AICUBE TECHNOLOGY LLC LLC using:
    - FastAPI framework for high-performance REST API
    - PyTorch for deep learning model implementation
    - MLP architecture with residual connections
    - Advanced neural network techniques for vector space alignment
    
    This API allows converting embeddings from a source model to a target model,
    preserving semantic meaning through sophisticated machine learning algorithms.
    """,
    version=aicube_settings.API_VERSION,
    docs_url="/aicube-docs",
    redoc_url="/aicube-redoc",
    openapi_url="/aicube-openapi.json",
    lifespan=aicube_lifespan
)

# Configure CORS
aicube_app.add_middleware(
    CORSMiddleware,
    allow_origins=aicube_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Register routers
aicube_app.include_router(aicube_router, prefix="/api/v1")


@aicube_app.get("/", tags=["Root"])
async def aicube_root():
    """
    Root endpoint of the AICUBE Embedding2Embedding API
    """
    return {
        "message": "AICUBE Embedding2Embedding API",
        "version": aicube_settings.API_VERSION,
        "technology": "AICUBE TECHNOLOGY LLC",
        "powered_by": [
            "FastAPI",
            "PyTorch", 
            "MLP Neural Networks",
            "Advanced ML Algorithms"
        ],
        "docs": "/aicube-docs",
        "health": "/api/v1/health",
        "license": "MIT License (Non-Commercial Use)",
        "copyright": "Copyright (c) 2024 AICUBE TECHNOLOGY LLC"
    }


@aicube_app.exception_handler(Exception)
async def aicube_global_exception_handler(request, exc):
    """
    Global handler for unhandled exceptions
    """
    logger.error("Unhandled error in AICUBE application", 
                error=str(exc), 
                path=str(request.url))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "AICUBE internal server error",
            "message": "Please contact technical support",
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