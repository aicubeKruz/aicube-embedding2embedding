"""
AICUBE Embedding2Embedding API - Unit Tests for Embedding Translator

Testes unitários para o modelo de tradução de embeddings AICUBE.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from aicube_app.aicube_models.aicube_embedding_translator import (
    AICUBEEmbeddingTranslator,
    AICUBESiLUActivation,
    AICUBEResidualBlock,
    AICUBEEmbeddingTranslatorEnsemble
)


class TestAICUBESiLUActivation:
    """Testes para ativação SiLU AICUBE"""
    
    def test_aicube_silu_forward(self):
        """Testar forward pass da ativação SiLU"""
        activation = AICUBESiLUActivation()
        
        # Teste com tensor simples
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        output = activation(x)
        
        # Verificar que a saída tem a mesma forma
        assert output.shape == x.shape
        
        # Verificar valores específicos
        expected = x * torch.sigmoid(x)
        torch.testing.assert_close(output, expected)
    
    def test_aicube_silu_gradient(self):
        """Testar que a ativação SiLU é diferenciável"""
        activation = AICUBESiLUActivation()
        
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        output = activation(x)
        loss = output.sum()
        
        # Verificar que gradiente pode ser calculado
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestAICUBEResidualBlock:
    """Testes para bloco residual AICUBE"""
    
    def test_aicube_residual_block_creation(self):
        """Testar criação do bloco residual"""
        dim = 128
        block = AICUBEResidualBlock(dim)
        
        assert block.linear1.in_features == dim
        assert block.linear1.out_features == dim
        assert block.linear2.in_features == dim
        assert block.linear2.out_features == dim
        assert isinstance(block.activation, AICUBESiLUActivation)
    
    def test_aicube_residual_forward(self):
        """Testar forward pass do bloco residual"""
        dim = 64
        batch_size = 4
        block = AICUBEResidualBlock(dim)
        
        x = torch.randn(batch_size, dim)
        output = block(x)
        
        # Verificar forma da saída
        assert output.shape == (batch_size, dim)
        
        # Verificar que não é idêntico à entrada (a menos que pesos sejam zero)
        # mas mantém dimensionalidade
        assert output.shape == x.shape
    
    def test_aicube_residual_gradient_flow(self):
        """Testar fluxo de gradiente através do bloco residual"""
        dim = 32
        block = AICUBEResidualBlock(dim)
        
        x = torch.randn(2, dim, requires_grad=True)
        output = block(x)
        loss = output.sum()
        
        loss.backward()
        assert x.grad is not None


class TestAICUBEEmbeddingTranslator:
    """Testes para tradutor de embeddings AICUBE"""
    
    def test_aicube_translator_creation(self):
        """Testar criação do tradutor"""
        source_dim = 768
        target_dim = 1024
        hidden_dim = 512
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim
        )
        
        assert translator.source_dim == source_dim
        assert translator.target_dim == target_dim
        assert translator.hidden_dim == hidden_dim
        
        # Verificar componentes
        assert translator.aicube_input_adapter[0].in_features == source_dim
        assert translator.aicube_input_adapter[0].out_features == hidden_dim
        assert translator.aicube_output_adapter[0].in_features == hidden_dim
        assert translator.aicube_output_adapter[0].out_features == target_dim
    
    def test_aicube_translator_forward_single(self):
        """Testar forward pass com embedding único"""
        source_dim = 100
        target_dim = 200
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim
        )
        
        # Teste com batch_size=1
        x = torch.randn(1, source_dim)
        output = translator(x)
        
        assert output.shape == (1, target_dim)
    
    def test_aicube_translator_forward_batch(self):
        """Testar forward pass com batch"""
        source_dim = 256
        target_dim = 512
        batch_size = 8
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim
        )
        
        x = torch.randn(batch_size, source_dim)
        output = translator(x)
        
        assert output.shape == (batch_size, target_dim)
    
    def test_aicube_translator_dimension_validation(self):
        """Testar validação de dimensões"""
        source_dim = 128
        target_dim = 256
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim
        )
        
        # Testar com dimensão incorreta
        wrong_dim_input = torch.randn(1, source_dim + 10)
        
        with pytest.raises(ValueError, match="Dimensão de entrada esperada"):
            translator(wrong_dim_input)
    
    def test_aicube_translate_single_numpy(self):
        """Testar tradução de embedding único com NumPy"""
        source_dim = 64
        target_dim = 128
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim
        )
        
        # Embedding de entrada
        embedding = np.random.randn(source_dim).astype(np.float32)
        
        # Traduzir
        translated = translator.aicube_translate_single(embedding)
        
        assert isinstance(translated, np.ndarray)
        assert translated.shape == (target_dim,)
        assert translated.dtype == np.float32
    
    def test_aicube_translate_batch_numpy(self):
        """Testar tradução de batch com NumPy"""
        source_dim = 32
        target_dim = 64
        batch_size = 5
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim
        )
        
        # Batch de embeddings
        embeddings = np.random.randn(batch_size, source_dim).astype(np.float32)
        
        # Traduzir
        translated = translator.aicube_translate_batch(embeddings)
        
        assert isinstance(translated, np.ndarray)
        assert translated.shape == (batch_size, target_dim)
        assert translated.dtype == np.float32
    
    def test_aicube_model_info(self):
        """Testar informações do modelo"""
        source_dim = 100
        target_dim = 200
        hidden_dim = 150
        num_layers = 2
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        info = translator.aicube_get_model_info()
        
        assert info["source_dimension"] == source_dim
        assert info["target_dimension"] == target_dim
        assert info["hidden_dimension"] == hidden_dim
        assert info["num_layers"] == num_layers
        assert "total_parameters" in info
        assert "aicube_technology" in info
        assert "aicube_version" in info
    
    def test_aicube_compute_translation_fidelity(self):
        """Testar cálculo de fidelidade da tradução"""
        source_dim = 50
        target_dim = 75
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim
        )
        
        # Embeddings de teste
        source_embeddings = np.random.randn(3, source_dim).astype(np.float32)
        translated_embeddings = np.random.randn(3, target_dim).astype(np.float32)
        
        metrics = translator.aicube_compute_translation_fidelity(
            source_embeddings, translated_embeddings
        )
        
        assert "source_norm" in metrics
        assert "target_norm" in metrics
        assert "dimension_preservation" in metrics
        assert "aicube_technology" in metrics
        assert metrics["dimension_preservation"] == target_dim / source_dim
    
    @patch('torch.save')
    def test_aicube_save_model(self, mock_save):
        """Testar salvamento do modelo"""
        translator = AICUBEEmbeddingTranslator(
            source_dim=128,
            target_dim=256
        )
        
        path = "/tmp/aicube_test_model.pt"
        translator.aicube_save_model(path)
        
        # Verificar que torch.save foi chamado
        mock_save.assert_called_once()
        
        # Verificar estrutura dos dados salvos
        save_args = mock_save.call_args[0]
        saved_data = save_args[0]
        saved_path = save_args[1]
        
        assert saved_path == path
        assert "aicube_model_state_dict" in saved_data
        assert "aicube_model_info" in saved_data
        assert "aicube_version" in saved_data
        assert "aicube_technology" in saved_data
    
    @patch('torch.load')
    def test_aicube_load_model(self, mock_load):
        """Testar carregamento do modelo"""
        # Mock dos dados salvos
        mock_data = {
            'aicube_model_state_dict': {},
            'aicube_model_info': {
                'source_dimension': 128,
                'target_dimension': 256,
                'hidden_dimension': 192,
                'num_layers': 3
            },
            'aicube_version': '1.0.0-aicube'
        }
        mock_load.return_value = mock_data
        
        path = "/tmp/aicube_test_model.pt"
        
        with patch.object(AICUBEEmbeddingTranslator, 'load_state_dict'):
            model = AICUBEEmbeddingTranslator.aicube_load_model(path)
            
            assert isinstance(model, AICUBEEmbeddingTranslator)
            assert model.source_dim == 128
            assert model.target_dim == 256
            assert model.hidden_dim == 192
            assert model.num_layers == 3


class TestAICUBEEmbeddingTranslatorEnsemble:
    """Testes para ensemble de tradutores AICUBE"""
    
    def test_aicube_ensemble_creation(self):
        """Testar criação do ensemble"""
        translators = [
            AICUBEEmbeddingTranslator(64, 128),
            AICUBEEmbeddingTranslator(64, 128),
            AICUBEEmbeddingTranslator(64, 128)
        ]
        
        ensemble = AICUBEEmbeddingTranslatorEnsemble(translators)
        
        assert ensemble.aicube_num_translators == 3
        assert len(ensemble.aicube_translators) == 3
    
    def test_aicube_ensemble_forward(self):
        """Testar forward pass do ensemble"""
        source_dim = 32
        target_dim = 64
        batch_size = 4
        
        translators = [
            AICUBEEmbeddingTranslator(source_dim, target_dim),
            AICUBEEmbeddingTranslator(source_dim, target_dim)
        ]
        
        ensemble = AICUBEEmbeddingTranslatorEnsemble(translators)
        
        x = torch.randn(batch_size, source_dim)
        output = ensemble(x)
        
        assert output.shape == (batch_size, target_dim)
    
    def test_aicube_ensemble_translate_batch(self):
        """Testar tradução em batch do ensemble"""
        source_dim = 16
        target_dim = 32
        batch_size = 3
        
        translators = [
            AICUBEEmbeddingTranslator(source_dim, target_dim),
            AICUBEEmbeddingTranslator(source_dim, target_dim)
        ]
        
        ensemble = AICUBEEmbeddingTranslatorEnsemble(translators)
        
        embeddings = np.random.randn(batch_size, source_dim).astype(np.float32)
        translated = ensemble.aicube_translate_batch(embeddings)
        
        assert isinstance(translated, np.ndarray)
        assert translated.shape == (batch_size, target_dim)


class TestAICUBETranslatorIntegration:
    """Testes de integração para o tradutor AICUBE"""
    
    def test_aicube_end_to_end_translation(self):
        """Teste end-to-end de tradução"""
        source_dim = 768
        target_dim = 1024
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim,
            num_layers=2
        )
        
        # Simular embedding de entrada (similar ao BERT)
        embedding = np.random.randn(source_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalizar
        
        # Traduzir
        translated = translator.aicube_translate_single(embedding)
        
        # Verificações
        assert translated.shape == (target_dim,)
        assert not np.any(np.isnan(translated))
        assert not np.any(np.isinf(translated))
        
        # Verificar que a tradução não é idêntica à entrada
        # (mesmo com dimensões diferentes, não deve ser zero)
        assert np.linalg.norm(translated) > 0
    
    def test_aicube_batch_consistency(self):
        """Testar consistência entre tradução única e em batch"""
        source_dim = 256
        target_dim = 512
        
        translator = AICUBEEmbeddingTranslator(
            source_dim=source_dim,
            target_dim=target_dim
        )
        
        # Embedding de teste
        embedding = np.random.randn(source_dim).astype(np.float32)
        
        # Tradução única
        single_result = translator.aicube_translate_single(embedding)
        
        # Tradução em batch (com apenas um embedding)
        batch_result = translator.aicube_translate_batch(embedding.reshape(1, -1))
        
        # Devem ser praticamente idênticos
        np.testing.assert_allclose(single_result, batch_result[0], rtol=1e-5)
    
    def test_aicube_device_handling(self):
        """Testar handling de dispositivos (CPU/CUDA)"""
        translator = AICUBEEmbeddingTranslator(
            source_dim=128,
            target_dim=256,
            device="cpu"
        )
        
        # Verificar que o modelo está no dispositivo correto
        assert next(translator.parameters()).device.type == "cpu"
        
        # Testar tradução
        embedding = np.random.randn(128).astype(np.float32)
        result = translator.aicube_translate_single(embedding)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])