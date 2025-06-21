"""
AICUBE Embedding2Embedding API - Basic Tests

Testes básicos para verificar funcionamento da API AICUBE.
"""

import pytest
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Testar se os imports básicos funcionam"""
    try:
        # Test importing torch
        import torch
        assert torch.__version__ is not None
        
        # Test importing numpy
        import numpy as np
        assert np.__version__ is not None
        
        # Test importing fastapi
        from fastapi import FastAPI
        
        # Test importing pydantic
        from pydantic import BaseModel
        
        print("✅ Todos os imports básicos funcionaram!")
        
    except ImportError as e:
        pytest.fail(f"Falha no import: {e}")

def test_aicube_model_creation():
    """Testar criação básica do modelo AICUBE"""
    try:
        # Import usando importlib devido aos hífens nos nomes dos arquivos
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "aicube_embedding_translator", 
            "/home/aicube/aicube-embedding2embedding/aicube-app/aicube-models/aicube-embedding-translator.py"
        )
        aicube_embedding_translator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aicube_embedding_translator)
        
        AICUBEEmbeddingTranslator = aicube_embedding_translator.AICUBEEmbeddingTranslator
        
        # Criar modelo
        model = AICUBEEmbeddingTranslator(
            source_dim=128,
            target_dim=256,
            hidden_dim=192
        )
        
        assert model.source_dim == 128
        assert model.target_dim == 256
        assert model.hidden_dim == 192
        
        print("✅ Modelo AICUBE criado com sucesso!")
        
    except Exception as e:
        pytest.fail(f"Falha na criação do modelo: {e}")

def test_aicube_config():
    """Testar configurações AICUBE"""
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "aicube_config", 
            "/home/aicube/aicube-embedding2embedding/aicube-app/aicube-core/aicube-config.py"
        )
        aicube_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aicube_config)
        
        settings = aicube_config.aicube_settings
        
        assert settings.API_NAME == "aicube-embedding2embedding"
        assert settings.API_VERSION == "v1"
        assert settings.AICUBE_TECHNOLOGY_NAME == "AICUBE TECHNOLOGY LLC"
        
        print("✅ Configurações AICUBE funcionando!")
        
    except Exception as e:
        pytest.fail(f"Falha nas configurações: {e}")

def test_aicube_model_forward():
    """Testar forward pass do modelo AICUBE"""
    try:
        import torch
        import numpy as np
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "aicube_embedding_translator", 
            "/home/aicube/aicube-embedding2embedding/aicube-app/aicube-models/aicube-embedding-translator.py"
        )
        aicube_embedding_translator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aicube_embedding_translator)
        
        AICUBEEmbeddingTranslator = aicube_embedding_translator.AICUBEEmbeddingTranslator
        
        # Criar modelo
        model = AICUBEEmbeddingTranslator(
            source_dim=64,
            target_dim=128
        )
        
        # Criar embedding de teste
        embedding = torch.randn(1, 64)
        
        # Forward pass
        output = model(embedding)
        
        assert output.shape == (1, 128)
        assert not torch.isnan(output).any()
        
        print("✅ Forward pass do modelo AICUBE funcionou!")
        
    except Exception as e:
        pytest.fail(f"Falha no forward pass: {e}")

if __name__ == "__main__":
    test_imports()
    test_aicube_model_creation()
    test_aicube_config()
    test_aicube_model_forward()
    print("🎉 Todos os testes básicos passaram!")