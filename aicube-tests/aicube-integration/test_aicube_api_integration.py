"""
AICUBE Embedding2Embedding API - Integration Tests

Testes de integração para os endpoints da API AICUBE.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
import numpy as np
from unittest.mock import patch, MagicMock

from aicube_app.aicube_main import aicube_app
from aicube_app.aicube_core.aicube_config import aicube_settings


@pytest.fixture
def aicube_client():
    """Fixture para cliente de teste AICUBE"""
    return TestClient(aicube_app)


@pytest.fixture
def aicube_async_client():
    """Fixture para cliente assíncrono de teste AICUBE"""
    return AsyncClient(app=aicube_app, base_url="http://test")


@pytest.fixture
def aicube_sample_embedding():
    """Fixture para embedding de exemplo"""
    return [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.7, -0.8]


@pytest.fixture
def aicube_sample_batch_embeddings():
    """Fixture para batch de embeddings de exemplo"""
    return [
        [0.1, 0.2, -0.3, 0.4],
        [0.5, -0.6, 0.7, -0.8],
        [-0.1, 0.2, 0.3, -0.4]
    ]


class TestAICUBEHealthEndpoint:
    """Testes para endpoint de health check AICUBE"""
    
    def test_aicube_health_endpoint(self, aicube_client):
        """Testar endpoint de health check"""
        response = aicube_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["aicube_service"] == aicube_settings.API_NAME
        assert data["version"] == aicube_settings.API_VERSION
        assert data["aicube_technology"] == aicube_settings.AICUBE_TECHNOLOGY_NAME
        assert "models_loaded" in data
        assert "powered_by" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_aicube_health_async(self, aicube_async_client):
        """Testar health check assíncrono"""
        async with aicube_async_client as client:
            response = await client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


class TestAICUBEModelsEndpoints:
    """Testes para endpoints de modelos AICUBE"""
    
    def test_aicube_list_models(self, aicube_client):
        """Testar listagem de modelos"""
        response = aicube_client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "aicube_models" in data
        assert "aicube_model_pairs" in data
        assert "total_models" in data
        assert data["aicube_technology"] == aicube_settings.AICUBE_TECHNOLOGY_NAME
        assert "powered_by" in data
        assert isinstance(data["aicube_models"], list)
        assert isinstance(data["aicube_model_pairs"], list)
    
    @patch('aicube_app.aicube_core.aicube_model_manager.aicube_model_manager.aicube_get_model_info')
    def test_aicube_get_model_info_success(self, mock_get_info, aicube_client):
        """Testar obtenção de informações de modelo específico"""
        # Mock da resposta
        mock_model_info = MagicMock()
        mock_model_info.model_id = "aicube-test-model"
        mock_model_info.source_model = "bert_base"
        mock_model_info.target_model = "t5_base"
        mock_model_info.source_dimension = 768
        mock_model_info.target_dimension = 768
        mock_model_info.version = "1.0.0-aicube"
        mock_model_info.description = "Test model"
        mock_model_info.aicube_technology = ["Qube LCM Model"]
        mock_model_info.created_by = "AICUBE TECHNOLOGY LLC"
        
        mock_get_info.return_value = mock_model_info
        
        response = aicube_client.get("/api/v1/models/aicube-test-model")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_id"] == "aicube-test-model"
        assert data["source_model"] == "bert_base"
        assert data["target_model"] == "t5_base"
        assert data["created_by"] == "AICUBE TECHNOLOGY LLC"
    
    @patch('aicube_app.aicube_core.aicube_model_manager.aicube_model_manager.aicube_get_model_info')
    def test_aicube_get_model_info_not_found(self, mock_get_info, aicube_client):
        """Testar modelo não encontrado"""
        mock_get_info.return_value = None
        
        response = aicube_client.get("/api/v1/models/inexistent-model")
        
        assert response.status_code == 404
        data = response.json()
        assert "Modelo AICUBE não encontrado" in data["detail"]


class TestAICUBETranslationEndpoints:
    """Testes para endpoints de tradução AICUBE"""
    
    @patch('aicube_app.aicube_core.aicube_model_manager.aicube_model_manager.aicube_translate_embedding')
    def test_aicube_translate_embedding_success(self, mock_translate, aicube_client, aicube_sample_embedding):
        """Testar tradução de embedding com sucesso"""
        # Mock da resposta
        mock_translated = np.array([0.2, 0.4, -0.1, 0.8, -0.3, 0.6])
        mock_metadata = {
            "aicube_model_id": "aicube-test-model",
            "duration_ms": 15.2,
            "cosine_similarity": 0.94
        }
        mock_translate.return_value = (mock_translated, mock_metadata)
        
        request_data = {
            "origem": "bert_base_uncased",
            "destino": "t5_base",
            "embedding": aicube_sample_embedding,
            "include_metadata": True
        }
        
        response = aicube_client.post("/api/v1/translate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["origem"] == "bert_base_uncased"
        assert data["destino"] == "t5_base"
        assert data["aicube_technology"] == aicube_settings.AICUBE_TECHNOLOGY_NAME
        assert "embedding_traduzido" in data
        assert "metadata" in data
        assert len(data["embedding_traduzido"]) == 6
    
    def test_aicube_translate_embedding_validation_error(self, aicube_client):
        """Testar erro de validação na tradução"""
        request_data = {
            "origem": "",  # Nome vazio - deve causar erro de validação
            "destino": "t5_base",
            "embedding": [0.1, 0.2, 0.3]
        }
        
        response = aicube_client.post("/api/v1/translate", json=request_data)
        
        assert response.status_code == 422  # Validation Error
    
    def test_aicube_translate_embedding_missing_fields(self, aicube_client):
        """Testar campos obrigatórios ausentes"""
        request_data = {
            "origem": "bert_base_uncased"
            # Faltando destino e embedding
        }
        
        response = aicube_client.post("/api/v1/translate", json=request_data)
        
        assert response.status_code == 422
    
    @patch('aicube_app.aicube_core.aicube_model_manager.aicube_model_manager.aicube_translate_embedding')
    def test_aicube_translate_batch_success(self, mock_translate, aicube_client, aicube_sample_batch_embeddings):
        """Testar tradução em lote com sucesso"""
        # Mock da resposta
        mock_translated = np.array([
            [0.2, 0.4, -0.1, 0.8],
            [0.3, -0.5, 0.7, -0.2],
            [-0.1, 0.6, 0.4, -0.9]
        ])
        mock_metadata = {
            "aicube_model_id": "aicube-test-model",
            "duration_ms": 25.6,
            "cosine_similarity": 0.93
        }
        mock_translate.return_value = (mock_translated, mock_metadata)
        
        request_data = {
            "origem": "bert_base_uncased",
            "destino": "t5_base",
            "embeddings": aicube_sample_batch_embeddings,
            "include_metadata": True
        }
        
        response = aicube_client.post("/api/v1/translate/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["origem"] == "bert_base_uncased"
        assert data["destino"] == "t5_base"
        assert data["batch_size"] == 3
        assert data["aicube_technology"] == aicube_settings.AICUBE_TECHNOLOGY_NAME
        assert len(data["embeddings_traduzidos"]) == 3
        assert len(data["embeddings_traduzidos"][0]) == 4
    
    def test_aicube_translate_batch_empty_list(self, aicube_client):
        """Testar batch vazio"""
        request_data = {
            "origem": "bert_base_uncased",
            "destino": "t5_base",
            "embeddings": []  # Lista vazia
        }
        
        response = aicube_client.post("/api/v1/translate/batch", json=request_data)
        
        assert response.status_code == 422
    
    def test_aicube_translate_batch_too_large(self, aicube_client):
        """Testar batch muito grande"""
        # Criar batch maior que o limite
        large_batch = [[0.1, 0.2, 0.3, 0.4] for _ in range(50)]
        
        request_data = {
            "origem": "bert_base_uncased",
            "destino": "t5_base",
            "embeddings": large_batch
        }
        
        response = aicube_client.post("/api/v1/translate/batch", json=request_data)
        
        assert response.status_code == 422
    
    @patch('aicube_app.aicube_core.aicube_model_manager.aicube_model_manager.aicube_translate_embedding')
    def test_aicube_translate_model_not_found(self, mock_translate, aicube_client, aicube_sample_embedding):
        """Testar modelo não encontrado na tradução"""
        mock_translate.side_effect = ValueError("Não há tradutor AICUBE disponível para invalid_model -> t5_base")
        
        request_data = {
            "origem": "invalid_model",
            "destino": "t5_base",
            "embedding": aicube_sample_embedding
        }
        
        response = aicube_client.post("/api/v1/translate", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Não há tradutor AICUBE disponível" in data["detail"]


class TestAICUBEStatisticsEndpoint:
    """Testes para endpoint de estatísticas AICUBE"""
    
    @patch('aicube_app.aicube_core.aicube_model_manager.aicube_model_manager.aicube_get_statistics')
    def test_aicube_get_statistics(self, mock_get_stats, aicube_client):
        """Testar obtenção de estatísticas"""
        mock_stats = {
            "aicube_loaded_models": 3,
            "aicube_available_models": 5,
            "aicube_usage_stats": {"aicube-bert-to-t5": 150},
            "aicube_technology": "AICUBE TECHNOLOGY LLC",
            "aicube_version": "1.0.0-aicube",
            "aicube_powered_by": ["Qube LCM Model"]
        }
        mock_get_stats.return_value = mock_stats
        
        response = aicube_client.get("/api/v1/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["aicube_loaded_models"] == 3
        assert data["aicube_available_models"] == 5
        assert data["aicube_technology"] == "AICUBE TECHNOLOGY LLC"
        assert "aicube_usage_stats" in data


class TestAICUBERootEndpoint:
    """Testes para endpoint raiz AICUBE"""
    
    def test_aicube_root_endpoint(self, aicube_client):
        """Testar endpoint raiz"""
        response = aicube_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "AICUBE Embedding2Embedding API"
        assert data["version"] == aicube_settings.API_VERSION
        assert data["technology"] == "AICUBE TECHNOLOGY LLC"
        assert "powered_by" in data
        assert "Qube LCM Model" in data["powered_by"]
        assert "docs" in data
        assert "health" in data


class TestAICUBEErrorHandling:
    """Testes para tratamento de erros AICUBE"""
    
    def test_aicube_404_error(self, aicube_client):
        """Testar erro 404"""
        response = aicube_client.get("/api/v1/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    def test_aicube_method_not_allowed(self, aicube_client):
        """Testar método não permitido"""
        response = aicube_client.delete("/api/v1/health")
        
        assert response.status_code == 405
    
    @patch('aicube_app.aicube_core.aicube_model_manager.aicube_model_manager.aicube_translate_embedding')
    def test_aicube_internal_server_error(self, mock_translate, aicube_client, aicube_sample_embedding):
        """Testar erro interno do servidor"""
        mock_translate.side_effect = Exception("Internal error")
        
        request_data = {
            "origem": "bert_base_uncased",
            "destino": "t5_base",
            "embedding": aicube_sample_embedding
        }
        
        response = aicube_client.post("/api/v1/translate", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "Erro interno do servidor AICUBE" in data["detail"]


class TestAICUBECORSHeaders:
    """Testes para headers CORS AICUBE"""
    
    def test_aicube_cors_headers(self, aicube_client):
        """Testar headers CORS"""
        response = aicube_client.options("/api/v1/health")
        
        # Verificar se headers CORS estão presentes
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


class TestAICUBEContentTypes:
    """Testes para tipos de conteúdo AICUBE"""
    
    def test_aicube_json_content_type(self, aicube_client):
        """Testar content-type JSON"""
        response = aicube_client.get("/api/v1/health")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_aicube_invalid_content_type(self, aicube_client):
        """Testar content-type inválido"""
        response = aicube_client.post(
            "/api/v1/translate", 
            data="invalid data",
            headers={"content-type": "text/plain"}
        )
        
        assert response.status_code == 422


class TestAICUBEAsyncEndpoints:
    """Testes assíncronos para endpoints AICUBE"""
    
    @pytest.mark.asyncio
    async def test_aicube_async_health_check(self, aicube_async_client):
        """Testar health check assíncrono"""
        async with aicube_async_client as client:
            response = await client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["aicube_service"] == aicube_settings.API_NAME
    
    @pytest.mark.asyncio
    async def test_aicube_async_models_list(self, aicube_async_client):
        """Testar listagem de modelos assíncrona"""
        async with aicube_async_client as client:
            response = await client.get("/api/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            assert "aicube_models" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])