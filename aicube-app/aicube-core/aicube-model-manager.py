"""
AICUBE Embedding2Embedding API - Model Manager

Gerenciador de modelos de tradução de embeddings utilizando tecnologias AICUBE.
"""

import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import structlog
from concurrent.futures import ThreadPoolExecutor
import json
import time

from aicube_app.aicube_core.aicube_config import aicube_settings
from aicube_app.aicube_models.aicube_embedding_translator import AICUBEEmbeddingTranslator

logger = structlog.get_logger("aicube.model_manager")


@dataclass
class AICUBEModelInfo:
    """
    Informações de um modelo de tradução AICUBE
    """
    model_id: str
    source_model: str
    target_model: str
    source_dimension: int
    target_dimension: int
    version: str
    description: str
    created_by: str = "AICUBE TECHNOLOGY"
    aicube_technology: List[str] = None
    
    def __post_init__(self):
        if self.aicube_technology is None:
            self.aicube_technology = aicube_settings.AICUBE_MODELS


class AICUBEModelManager:
    """
    Gerenciador de modelos de tradução de embeddings AICUBE
    
    Responsável por:
    - Carregar e gerenciar modelos de tradução
    - Cache de modelos em memória
    - Seleção automática de modelos baseado em origem/destino
    - Métricas e monitoramento de uso
    """
    
    def __init__(self):
        self.aicube_models: Dict[str, AICUBEEmbeddingTranslator] = {}
        self.aicube_model_info: Dict[str, AICUBEModelInfo] = {}
        self.aicube_model_cache: Dict[str, Any] = {}
        self.aicube_load_times: Dict[str, float] = {}
        self.aicube_usage_stats: Dict[str, int] = {}
        self.aicube_executor = ThreadPoolExecutor(max_workers=aicube_settings.MAX_WORKERS)
        self._aicube_initialized = False
    
    async def initialize_models(self) -> None:
        """
        Inicializar modelos pré-treinados AICUBE
        """
        if self._aicube_initialized:
            return
        
        logger.info("Inicializando modelos AICUBE...")
        
        # Carregar configuração de modelos
        await self._aicube_load_model_configurations()
        
        # Carregar modelos essenciais (lazy loading para outros)
        aicube_essential_models = [
            "aicube-bert-to-t5",
            "aicube-mpnet-to-ada002"
        ]
        
        for model_id in aicube_essential_models:
            if model_id in self.aicube_model_info:
                try:
                    await self._aicube_load_model(model_id)
                    logger.info(f"Modelo essencial AICUBE carregado: {model_id}")
                except Exception as e:
                    logger.warning(f"Falha ao carregar modelo essencial AICUBE {model_id}: {e}")
        
        self._aicube_initialized = True
        logger.info(f"Gerenciador de modelos AICUBE inicializado com {len(self.aicube_models)} modelos")
    
    async def _aicube_load_model_configurations(self) -> None:
        """
        Carregar configurações dos modelos disponíveis
        """
        models_path = Path(aicube_settings.MODELS_PATH)
        config_file = models_path / "aicube-model-registry.json"
        
        if not config_file.exists():
            # Criar configuração padrão se não existir
            await self._aicube_create_default_model_config(config_file)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            for model_config in config_data.get('aicube_models', []):
                model_info = AICUBEModelInfo(**model_config)
                self.aicube_model_info[model_info.model_id] = model_info
                
        except Exception as e:
            logger.error(f"Erro ao carregar configurações de modelo AICUBE: {e}")
            # Fallback para modelos padrão
            self._aicube_setup_default_models()
    
    async def _aicube_create_default_model_config(self, config_file: Path) -> None:
        """
        Criar configuração padrão dos modelos AICUBE
        """
        aicube_default_config = {
            "aicube_version": "1.0.0-aicube",
            "created_by": "AICUBE TECHNOLOGY",
            "aicube_technologies": aicube_settings.AICUBE_MODELS,
            "aicube_models": [
                {
                    "model_id": "aicube-bert-to-t5",
                    "source_model": "bert_base_uncased",
                    "target_model": "t5_base",
                    "source_dimension": 768,
                    "target_dimension": 768,
                    "version": "1.0.0-aicube",
                    "description": "Tradutor AICUBE BERT para T5 usando Qube LCM Model"
                },
                {
                    "model_id": "aicube-mpnet-to-ada002",
                    "source_model": "sentence_transformers_mpnet",
                    "target_model": "openai_ada002",
                    "source_dimension": 768,
                    "target_dimension": 1536,
                    "version": "1.0.0-aicube",
                    "description": "Tradutor AICUBE MPNet para OpenAI Ada-002 usando Qube Neural Memory"
                },
                {
                    "model_id": "aicube-roberta-to-gpt2",
                    "source_model": "roberta_base",
                    "target_model": "gpt2_base",
                    "source_dimension": 768,
                    "target_dimension": 768,
                    "version": "1.0.0-aicube",
                    "description": "Tradutor AICUBE RoBERTa para GPT-2 usando Qube Agentic Workflows"
                }
            ]
        }
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(aicube_default_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuração padrão AICUBE criada: {config_file}")
    
    def _aicube_setup_default_models(self) -> None:
        """
        Configurar modelos padrão em caso de falha
        """
        aicube_default_models = [
            AICUBEModelInfo(
                model_id="aicube-bert-to-t5",
                source_model="bert_base_uncased",
                target_model="t5_base",
                source_dimension=768,
                target_dimension=768,
                version="1.0.0-aicube",
                description="Tradutor AICUBE BERT para T5 usando Qube LCM Model"
            )
        ]
        
        for model_info in aicube_default_models:
            self.aicube_model_info[model_info.model_id] = model_info
    
    async def _aicube_load_model(self, model_id: str) -> None:
        """
        Carregar um modelo específico de forma assíncrona
        """
        if model_id in self.aicube_models:
            return
        
        start_time = time.time()
        
        try:
            model_info = self.aicube_model_info[model_id]
            
            # Executar carregamento em thread separada
            model = await asyncio.get_event_loop().run_in_executor(
                self.aicube_executor,
                self._aicube_load_model_sync,
                model_info
            )
            
            self.aicube_models[model_id] = model
            load_time = time.time() - start_time
            self.aicube_load_times[model_id] = load_time
            
            logger.info(
                "Modelo AICUBE carregado",
                model_id=model_id,
                source_model=model_info.source_model,
                target_model=model_info.target_model,
                load_time_seconds=round(load_time, 2)
            )
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo AICUBE {model_id}: {e}")
            raise
    
    def _aicube_load_model_sync(self, model_info: AICUBEModelInfo) -> AICUBEEmbeddingTranslator:
        """
        Carregar modelo de forma síncrona (executado em thread separada)
        """
        model_path = Path(aicube_settings.MODELS_PATH) / "aicube-pretrained" / f"{model_info.model_id}.pt"
        
        # Criar tradutor
        translator = AICUBEEmbeddingTranslator(
            source_dim=model_info.source_dimension,
            target_dim=model_info.target_dimension,
            hidden_dim=max(model_info.source_dimension, model_info.target_dimension),
            device=aicube_settings.DEVICE
        )
        
        # Carregar pesos se existirem
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=aicube_settings.DEVICE)
                if 'aicube_model_state_dict' in state_dict:
                    translator.load_state_dict(state_dict['aicube_model_state_dict'])
                else:
                    translator.load_state_dict(state_dict)
                logger.info(f"Pesos AICUBE carregados para {model_info.model_id}")
            except Exception as e:
                logger.warning(f"Não foi possível carregar pesos AICUBE para {model_info.model_id}: {e}")
                logger.info(f"Usando modelo AICUBE com inicialização aleatória para {model_info.model_id}")
        else:
            logger.info(f"Arquivo de modelo AICUBE não encontrado, usando inicialização aleatória: {model_path}")
        
        translator.eval()
        return translator
    
    async def aicube_get_model(self, model_id: str) -> AICUBEEmbeddingTranslator:
        """
        Obter modelo por ID, carregando se necessário
        """
        if model_id not in self.aicube_models:
            await self._aicube_load_model(model_id)
        
        # Atualizar estatísticas de uso
        self.aicube_usage_stats[model_id] = self.aicube_usage_stats.get(model_id, 0) + 1
        
        return self.aicube_models[model_id]
    
    async def aicube_translate_embedding(
        self,
        embedding: np.ndarray,
        source_model: str,
        target_model: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Traduzir embedding entre modelos usando tecnologia AICUBE
        
        Returns:
            Tuple[np.ndarray, Dict]: (embedding_traduzido, metadados)
        """
        start_time = time.time()
        
        # Encontrar modelo tradutor apropriado
        model_id = await self._aicube_find_translator_model(source_model, target_model)
        
        if not model_id:
            raise ValueError(f"Não há tradutor AICUBE disponível para {source_model} -> {target_model}")
        
        # Obter modelo
        translator = await self.aicube_get_model(model_id)
        
        # Preparar input
        if embedding.ndim == 1:
            embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0)
        else:
            embedding_tensor = torch.from_numpy(embedding).float()
        
        if aicube_settings.DEVICE == "cuda" and torch.cuda.is_available():
            embedding_tensor = embedding_tensor.cuda()
        
        # Executar tradução
        with torch.no_grad():
            translated_tensor = translator(embedding_tensor)
            translated_embedding = translated_tensor.cpu().numpy()
        
        # Remover dimensão batch se foi adicionada
        if embedding.ndim == 1:
            translated_embedding = translated_embedding.squeeze(0)
        
        duration = time.time() - start_time
        
        # Calcular similaridade cosseno (opcional)
        cosine_sim = None
        try:
            # Normalizar embeddings para cálculo de similaridade
            original_norm = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
            translated_norm = translated_embedding / np.linalg.norm(translated_embedding, axis=-1, keepdims=True)
            
            if embedding.ndim == 1:
                cosine_sim = float(np.dot(original_norm, translated_norm))
            else:
                cosine_sim = float(np.mean(np.sum(original_norm * translated_norm, axis=-1)))
        except:
            pass
        
        # Metadados
        metadata = {
            "aicube_model_id": model_id,
            "source_model": source_model,
            "target_model": target_model,
            "duration_ms": round(duration * 1000, 2),
            "cosine_similarity": cosine_sim,
            "source_dimension": embedding.shape[-1],
            "target_dimension": translated_embedding.shape[-1],
            "aicube_technology": "AICUBE TECHNOLOGY",
            "powered_by": aicube_settings.AICUBE_MODELS
        }
        
        logger.info(
            "Embedding traduzido com tecnologia AICUBE",
            **metadata
        )
        
        return translated_embedding, metadata
    
    async def _aicube_find_translator_model(self, source_model: str, target_model: str) -> Optional[str]:
        """
        Encontrar modelo tradutor AICUBE apropriado para os modelos especificados
        """
        for model_id, model_info in self.aicube_model_info.items():
            if (model_info.source_model == source_model and 
                model_info.target_model == target_model):
                return model_id
        
        # Tentar encontrar tradutor reverso
        for model_id, model_info in self.aicube_model_info.items():
            if (model_info.source_model == target_model and 
                model_info.target_model == source_model):
                return model_id
        
        return None
    
    def get_available_models(self) -> List[str]:
        """
        Retornar lista de modelos AICUBE disponíveis
        """
        return list(self.aicube_model_info.keys())
    
    def aicube_get_supported_model_pairs(self) -> List[Dict[str, str]]:
        """
        Retornar pares de modelos suportados pela tecnologia AICUBE
        """
        pairs = []
        for model_info in self.aicube_model_info.values():
            pairs.append({
                "source": model_info.source_model,
                "target": model_info.target_model,
                "aicube_translator_id": model_info.model_id,
                "aicube_technology": model_info.aicube_technology
            })
        return pairs
    
    async def aicube_get_model_info(self, model_id: str) -> Optional[AICUBEModelInfo]:
        """
        Obter informações de um modelo AICUBE específico
        """
        return self.aicube_model_info.get(model_id)
    
    async def cleanup(self) -> None:
        """
        Limpeza de recursos AICUBE
        """
        logger.info("Limpando recursos do gerenciador de modelos AICUBE...")
        
        # Limpar modelos da memória
        for model_id in list(self.aicube_models.keys()):
            del self.aicube_models[model_id]
        
        self.aicube_models.clear()
        self.aicube_model_cache.clear()
        
        # Fechar executor
        self.aicube_executor.shutdown(wait=True)
        
        logger.info("Recursos AICUBE limpos com sucesso")
    
    def aicube_get_statistics(self) -> Dict[str, Any]:
        """
        Obter estatísticas de uso dos modelos AICUBE
        """
        return {
            "aicube_loaded_models": len(self.aicube_models),
            "aicube_available_models": len(self.aicube_model_info),
            "aicube_usage_stats": self.aicube_usage_stats.copy(),
            "aicube_load_times": self.aicube_load_times.copy(),
            "aicube_technology": aicube_settings.AICUBE_TECHNOLOGY_NAME,
            "aicube_powered_by": aicube_settings.AICUBE_MODELS,
            "aicube_version": "1.0.0-aicube"
        }


# Instância global do gerenciador
aicube_model_manager = AICUBEModelManager()