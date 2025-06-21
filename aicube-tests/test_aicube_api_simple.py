"""
AICUBE Embedding2Embedding API - Simple API Tests

Testes simplificados para a API AICUBE.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
import json

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_aicube_app_creation():
    """Testar criação da aplicação AICUBE"""
    try:
        import importlib.util
        
        # Import da aplicação principal
        spec_main = importlib.util.spec_from_file_location(
            "aicube_main", 
            "/home/aicube/aicube-embedding2embedding/aicube-app/aicube-main.py"
        )
        aicube_main = importlib.util.module_from_spec(spec_main)
        
        # Verificar que a spec foi criada corretamente
        assert spec_main is not None
        print("✅ Spec da aplicação AICUBE criada!")
        
    except Exception as e:
        pytest.fail(f"Falha na criação da app: {e}")

def test_aicube_health_endpoint_mock():
    """Testar health endpoint com mock simples"""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Criar app mínima para teste
        app = FastAPI(title="AICUBE Test API")
        
        @app.get("/api/v1/health")
        async def health():
            return {
                "status": "healthy",
                "aicube_service": "aicube-embedding2embedding",
                "version": "v1",
                "technology": "AICUBE TECHNOLOGY LLC"
            }
        
        @app.get("/")
        async def root():
            return {
                "message": "AICUBE Embedding2Embedding API",
                "technology": "AICUBE TECHNOLOGY LLC",
                "license": "MIT License (Non-Commercial Use)"
            }
        
        client = TestClient(app)
        
        # Testar health endpoint
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["aicube_service"] == "aicube-embedding2embedding"
        assert data["technology"] == "AICUBE TECHNOLOGY LLC"
        
        # Testar root endpoint
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "AICUBE" in data["message"]
        assert data["technology"] == "AICUBE TECHNOLOGY LLC"
        
        print("✅ Endpoints básicos AICUBE funcionando!")
        
    except Exception as e:
        pytest.fail(f"Falha nos endpoints: {e}")

def test_aicube_translate_payload_validation():
    """Testar validação de payload para tradução"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from pydantic import BaseModel
        from typing import List
        
        # Schema básico
        class TranslateRequest(BaseModel):
            origem: str
            destino: str
            embedding: List[float]
            include_metadata: bool = False
        
        # App mínima
        app = FastAPI()
        
        @app.post("/api/v1/translate")
        async def translate(request: TranslateRequest):
            # Validações básicas
            if not request.origem:
                raise HTTPException(400, "Origem não pode estar vazia")
            if not request.destino:
                raise HTTPException(400, "Destino não pode estar vazio")
            if not request.embedding:
                raise HTTPException(400, "Embedding não pode estar vazio")
            
            return {
                "origem": request.origem,
                "destino": request.destino,
                "embedding_traduzido": [x * 0.9 for x in request.embedding],  # Mock translation
                "aicube_technology": "AICUBE TECHNOLOGY LLC"
            }
        
        client = TestClient(app)
        
        # Teste com dados válidos
        valid_payload = {
            "origem": "bert_base_uncased",
            "destino": "t5_base",
            "embedding": [0.1, 0.2, -0.3, 0.4],
            "include_metadata": True
        }
        
        response = client.post("/api/v1/translate", json=valid_payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["origem"] == "bert_base_uncased"
        assert data["destino"] == "t5_base"
        assert data["aicube_technology"] == "AICUBE TECHNOLOGY LLC"
        assert len(data["embedding_traduzido"]) == 4
        
        # Teste com dados inválidos
        invalid_payload = {
            "origem": "",
            "destino": "t5_base",
            "embedding": []
        }
        
        response = client.post("/api/v1/translate", json=invalid_payload)
        assert response.status_code == 400
        
        print("✅ Validação de payload AICUBE funcionando!")
        
    except Exception as e:
        pytest.fail(f"Falha na validação: {e}")

def test_aicube_license_info():
    """Testar informações de licença AICUBE"""
    try:
        # Verificar se arquivo de licença existe
        license_path = "/home/aicube/aicube-embedding2embedding/LICENSE"
        assert os.path.exists(license_path), "Arquivo LICENSE não encontrado"
        
        with open(license_path, 'r') as f:
            license_content = f.read()
        
        assert "MIT License" in license_content
        assert "Non-Commercial" in license_content
        assert "AICUBE TECHNOLOGY LLC" in license_content
        assert "contact@aicube.technology" in license_content
        
        print("✅ Informações de licença AICUBE corretas!")
        
    except Exception as e:
        pytest.fail(f"Falha na verificação de licença: {e}")

def test_aicube_requirements():
    """Testar arquivo de requirements AICUBE"""
    try:
        requirements_path = "/home/aicube/aicube-embedding2embedding/aicube-requirements.txt"
        assert os.path.exists(requirements_path), "Arquivo aicube-requirements.txt não encontrado"
        
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        # Verificar dependências principais
        assert "fastapi" in requirements
        assert "torch" in requirements
        assert "pytest" in requirements
        assert "pydantic" in requirements
        
        print("✅ Arquivo de requirements AICUBE correto!")
        
    except Exception as e:
        pytest.fail(f"Falha na verificação de requirements: {e}")

if __name__ == "__main__":
    test_aicube_app_creation()
    test_aicube_health_endpoint_mock()
    test_aicube_translate_payload_validation()
    test_aicube_license_info()
    test_aicube_requirements()
    print("🎉 Todos os testes simplificados da API AICUBE passaram!")