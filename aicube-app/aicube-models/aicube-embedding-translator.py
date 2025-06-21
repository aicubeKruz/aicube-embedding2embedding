"""
AICUBE Embedding2Embedding API - Embedding Translator Model

Modelo de tradução de embeddings baseado em MLP com conexões residuais,
utilizando tecnologias AICUBE para alinhamento de espaços vetoriais.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class AICUBESiLUActivation(nn.Module):
    """
    Ativação SiLU (Swish) conforme especificação AICUBE
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class AICUBEResidualBlock(nn.Module):
    """
    Bloco residual para o tradutor de embeddings AICUBE
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = AICUBESiLUActivation()
    
    def forward(self, x):
        residual = x
        
        # Primeira camada
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Segunda camada
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Conexão residual
        return x + residual


class AICUBEEmbeddingTranslator(nn.Module):
    """
    Modelo de tradução de embeddings AICUBE
    
    Arquitetura baseada em MLP com:
    - Adapter de entrada
    - Backbone compartilhado com blocos residuais
    - Adapter de saída
    - Normalização de camadas
    - Ativação SiLU
    
    Implementa a especificação técnica de Jha et al. (2025) adaptada para AICUBE.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Adapter de entrada - projeta do espaço de origem para espaço latente
        self.aicube_input_adapter = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            AICUBESiLUActivation(),
            nn.Dropout(dropout)
        )
        
        # Backbone compartilhado - processa no espaço latente
        self.aicube_backbone_layers = nn.ModuleList([
            AICUBEResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Camada de projeção central (se necessário)
        if num_layers > 1:
            self.aicube_central_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                AICUBESiLUActivation(),
                nn.Dropout(dropout)
            )
        else:
            self.aicube_central_projection = nn.Identity()
        
        # Adapter de saída - projeta do espaço latente para espaço de destino
        self.aicube_output_adapter = nn.Sequential(
            nn.Linear(hidden_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
        
        # Inicialização dos pesos
        self._aicube_initialize_weights()
        
        # Mover para dispositivo
        self.to(device)
    
    def _aicube_initialize_weights(self):
        """
        Inicialização de pesos conforme boas práticas AICUBE
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Inicialização Xavier/Glorot modificada
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do tradutor AICUBE
        
        Args:
            x: Tensor de embeddings de entrada [batch_size, source_dim]
            
        Returns:
            Tensor de embeddings traduzidos [batch_size, target_dim]
        """
        # Validação de entrada
        if x.size(-1) != self.source_dim:
            raise ValueError(f"Dimensão de entrada esperada: {self.source_dim}, recebida: {x.size(-1)}")
        
        # Adapter de entrada
        x = self.aicube_input_adapter(x)
        
        # Backbone com blocos residuais
        for layer in self.aicube_backbone_layers:
            x = layer(x)
        
        # Projeção central
        x = self.aicube_central_projection(x)
        
        # Adapter de saída
        x = self.aicube_output_adapter(x)
        
        return x
    
    def aicube_translate_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Traduzir batch de embeddings
        
        Args:
            embeddings: Array numpy de embeddings [batch_size, source_dim]
            
        Returns:
            Array numpy de embeddings traduzidos [batch_size, target_dim]
        """
        was_training = self.training
        self.eval()
        
        try:
            # Converter para tensor
            x = torch.from_numpy(embeddings).float().to(self.device)
            
            with torch.no_grad():
                translated = self.forward(x)
            
            # Converter de volta para numpy
            return translated.cpu().numpy()
        
        finally:
            if was_training:
                self.train()
    
    def aicube_translate_single(self, embedding: np.ndarray) -> np.ndarray:
        """
        Traduzir um único embedding
        
        Args:
            embedding: Array numpy de embedding [source_dim]
            
        Returns:
            Array numpy de embedding traduzido [target_dim]
        """
        # Adicionar dimensão de batch
        embedding_batch = embedding.reshape(1, -1)
        translated_batch = self.aicube_translate_batch(embedding_batch)
        
        # Remover dimensão de batch
        return translated_batch.squeeze(0)
    
    def aicube_compute_translation_fidelity(
        self, 
        source_embeddings: np.ndarray, 
        translated_embeddings: np.ndarray
    ) -> dict:
        """
        Computar métricas de fidelidade da tradução
        
        Args:
            source_embeddings: Embeddings originais
            translated_embeddings: Embeddings traduzidos
            
        Returns:
            Dict com métricas de fidelidade
        """
        # Normalizar embeddings
        source_norm = source_embeddings / np.linalg.norm(source_embeddings, axis=-1, keepdims=True)
        
        # Para calcular fidelidade, precisaríamos do tradutor reverso
        # Por enquanto, retornamos estrutura básica
        metrics = {
            "source_norm": float(np.mean(np.linalg.norm(source_embeddings, axis=-1))),
            "target_norm": float(np.mean(np.linalg.norm(translated_embeddings, axis=-1))),
            "dimension_preservation": translated_embeddings.shape[-1] / source_embeddings.shape[-1],
            "aicube_technology": "Qube LCM Model + Qube Neural Memory"
        }
        
        return metrics
    
    def aicube_get_model_info(self) -> dict:
        """
        Retornar informações do modelo
        """
        return {
            "source_dimension": self.source_dim,
            "target_dimension": self.target_dim,
            "hidden_dimension": self.hidden_dim,
            "num_layers": self.num_layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "device": str(next(self.parameters()).device),
            "aicube_technology": ["Qube LCM Model", "Qube Neural Memory", "Qube Agentic Workflows", "Qube Computer Vision"],
            "aicube_version": "1.0.0-aicube"
        }
    
    def aicube_save_model(self, path: str):
        """
        Salvar modelo AICUBE
        """
        model_info = self.aicube_get_model_info()
        
        save_dict = {
            'aicube_model_state_dict': self.state_dict(),
            'aicube_model_info': model_info,
            'aicube_version': '1.0.0-aicube',
            'aicube_architecture': 'AICUBEEmbeddingTranslator',
            'aicube_technology': 'AICUBE TECHNOLOGY'
        }
        
        torch.save(save_dict, path)
    
    @classmethod
    def aicube_load_model(cls, path: str, device: str = "cpu"):
        """
        Carregar modelo AICUBE
        """
        checkpoint = torch.load(path, map_location=device)
        model_info = checkpoint['aicube_model_info']
        
        # Recriar modelo
        model = cls(
            source_dim=model_info['source_dimension'],
            target_dim=model_info['target_dimension'],
            hidden_dim=model_info['hidden_dimension'],
            num_layers=model_info['num_layers'],
            device=device
        )
        
        # Carregar pesos
        model.load_state_dict(checkpoint['aicube_model_state_dict'])
        
        return model


class AICUBEEmbeddingTranslatorEnsemble(nn.Module):
    """
    Ensemble de tradutores AICUBE para maior robustez
    """
    
    def __init__(self, aicube_translators: list):
        super().__init__()
        self.aicube_translators = nn.ModuleList(aicube_translators)
        self.aicube_num_translators = len(aicube_translators)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass com ensemble averaging
        """
        outputs = []
        
        for translator in self.aicube_translators:
            output = translator(x)
            outputs.append(output)
        
        # Média das saídas
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        return ensemble_output
    
    def aicube_translate_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Traduzir batch usando ensemble
        """
        was_training = self.training
        self.eval()
        
        try:
            x = torch.from_numpy(embeddings).float()
            
            with torch.no_grad():
                translated = self.forward(x)
            
            return translated.cpu().numpy()
        
        finally:
            if was_training:
                self.train()