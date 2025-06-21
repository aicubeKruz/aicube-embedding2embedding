"""
AICUBE Embedding2Embedding API - Logging Configuration

Structured logging system using structlog for monitoring and auditing.
"""

import structlog
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from aicube_app.aicube_core.aicube_config import aicube_settings


def aicube_setup_logging(log_file: Optional[str] = None) -> None:
    """
    Configurar logging estruturado para a aplicação AICUBE
    
    Args:
        log_file: Caminho para arquivo de log (opcional)
    """
    
    # Configurar formato baseado no ambiente
    if aicube_settings.LOG_FORMAT == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            aicube_add_context,
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            aicube_add_context,
            structlog.dev.ConsoleRenderer()
        ]
    
    # Configurar structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configurar logging padrão
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, aicube_settings.LOG_LEVEL),
    )
    
    # Configurar arquivo de log se especificado
    if log_file or aicube_settings.LOG_FILE:
        log_path = Path(log_file or aicube_settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, aicube_settings.LOG_LEVEL))
        
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def aicube_add_context(logger, method_name, event_dict):
    """
    Adicionar contexto AICUBE aos logs
    """
    event_dict["aicube_service"] = aicube_settings.API_NAME
    event_dict["aicube_version"] = aicube_settings.API_VERSION
    event_dict["aicube_technology"] = aicube_settings.AICUBE_TECHNOLOGY_NAME
    event_dict["environment"] = aicube_settings.ENVIRONMENT
    
    return event_dict


class AICUBELogger:
    """
    Classe utilitária para logging padronizado AICUBE
    """
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(f"aicube.{name}")
    
    def log_request(self, method: str, path: str, duration: float, status_code: int):
        """Log de requisições HTTP"""
        self.logger.info(
            "AICUBE HTTP Request",
            method=method,
            path=path,
            duration_ms=round(duration * 1000, 2),
            status_code=status_code
        )
    
    def log_model_operation(self, operation: str, model_id: str, duration: float, success: bool):
        """Log de operações de modelo"""
        level = "info" if success else "error"
        getattr(self.logger, level)(
            "AICUBE Model Operation",
            operation=operation,
            model_id=model_id,
            duration_ms=round(duration * 1000, 2),
            success=success
        )
    
    def log_translation(self, source_model: str, target_model: str, embedding_dim: int, 
                       batch_size: int, duration: float, cosine_similarity: Optional[float] = None):
        """Log de traduções de embedding"""
        self.logger.info(
            "AICUBE Embedding Translation",
            source_model=source_model,
            target_model=target_model,
            embedding_dimension=embedding_dim,
            batch_size=batch_size,
            duration_ms=round(duration * 1000, 2),
            cosine_similarity=cosine_similarity
        )
    
    def log_error(self, error: Exception, context: dict = None):
        """Log de erros com contexto"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            **(context or {})
        }
        self.logger.error("AICUBE Application Error", **error_context)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log de métricas de performance"""
        self.logger.info(
            "AICUBE Performance Metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow().isoformat()
        )


# Instância global do logger
aicube_logger = AICUBELogger("embedding2embedding")