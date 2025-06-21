"""
AICUBE Embedding2Embedding API - Validators

Utilitários de validação para dados da API AICUBE.
"""

import numpy as np
from typing import List, Union, Tuple, Optional
import re


class AICUBEValidators:
    """
    Classe com validadores para a API AICUBE Embedding2Embedding
    """
    
    @staticmethod
    def aicube_validate_embedding_format(embedding: Union[List[float], List[List[float]]]) -> Tuple[bool, str]:
        """
        Validar formato do embedding
        
        Args:
            embedding: Embedding ou lista de embeddings
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if not embedding:
                return False, "Embedding não pode estar vazio"
            
            # Verificar se é lista de números ou lista de listas
            if isinstance(embedding[0], (int, float)):
                # Embedding único
                for i, val in enumerate(embedding):
                    if not isinstance(val, (int, float)):
                        return False, f"Valor no índice {i} não é um número válido"
                    if np.isnan(val) or np.isinf(val):
                        return False, f"Valor no índice {i} é NaN ou Infinito"
                
                if len(embedding) == 0:
                    return False, "Embedding deve ter pelo menos uma dimensão"
                    
            elif isinstance(embedding[0], list):
                # Batch de embeddings
                if len(embedding) == 0:
                    return False, "Batch de embeddings não pode estar vazio"
                
                first_dim = len(embedding[0])
                for i, emb in enumerate(embedding):
                    if not isinstance(emb, list):
                        return False, f"Embedding no índice {i} deve ser uma lista"
                    
                    if len(emb) != first_dim:
                        return False, f"Todos os embeddings devem ter a mesma dimensão. Esperado: {first_dim}, encontrado: {len(emb)} no índice {i}"
                    
                    for j, val in enumerate(emb):
                        if not isinstance(val, (int, float)):
                            return False, f"Valor no embedding {i}, posição {j} não é um número válido"
                        if np.isnan(val) or np.isinf(val):
                            return False, f"Valor no embedding {i}, posição {j} é NaN ou Infinito"
            else:
                return False, "Embedding deve ser uma lista de números ou lista de listas de números"
            
            return True, ""
            
        except Exception as e:
            return False, f"Erro na validação do embedding: {str(e)}"
    
    @staticmethod
    def aicube_validate_model_name(model_name: str) -> Tuple[bool, str]:
        """
        Validar nome do modelo
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not model_name or not isinstance(model_name, str):
            return False, "Nome do modelo deve ser uma string não vazia"
        
        model_name = model_name.strip()
        
        if not model_name:
            return False, "Nome do modelo não pode estar vazio"
        
        # Verificar caracteres válidos
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', model_name):
            return False, "Nome do modelo contém caracteres inválidos. Use apenas letras, números, underscore, hífen e ponto"
        
        if len(model_name) > 100:
            return False, "Nome do modelo não pode ter mais de 100 caracteres"
        
        return True, ""
    
    @staticmethod
    def aicube_validate_embedding_dimensions(
        embedding: np.ndarray, 
        expected_dim: Optional[int] = None,
        max_dim: int = 4096
    ) -> Tuple[bool, str]:
        """
        Validar dimensões do embedding
        
        Args:
            embedding: Array numpy do embedding
            expected_dim: Dimensão esperada (opcional)
            max_dim: Dimensão máxima permitida
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if embedding.ndim == 0:
                return False, "Embedding deve ter pelo menos uma dimensão"
            
            if embedding.ndim > 2:
                return False, "Embedding não pode ter mais de 2 dimensões"
            
            # Obter dimensão do embedding
            if embedding.ndim == 1:
                dim = embedding.shape[0]
            else:
                dim = embedding.shape[1]
            
            if dim == 0:
                return False, "Dimensão do embedding não pode ser zero"
            
            if dim > max_dim:
                return False, f"Dimensão do embedding ({dim}) excede o máximo permitido ({max_dim})"
            
            if expected_dim is not None and dim != expected_dim:
                return False, f"Dimensão do embedding ({dim}) não corresponde à esperada ({expected_dim})"
            
            return True, ""
            
        except Exception as e:
            return False, f"Erro na validação das dimensões: {str(e)}"
    
    @staticmethod
    def aicube_validate_batch_size(batch_size: int, max_batch_size: int = 32) -> Tuple[bool, str]:
        """
        Validar tamanho do batch
        
        Args:
            batch_size: Tamanho do batch
            max_batch_size: Tamanho máximo permitido
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if batch_size <= 0:
            return False, "Tamanho do batch deve ser maior que zero"
        
        if batch_size > max_batch_size:
            return False, f"Tamanho do batch ({batch_size}) excede o máximo permitido ({max_batch_size})"
        
        return True, ""
    
    @staticmethod
    def aicube_validate_cosine_similarity(similarity: float) -> Tuple[bool, str]:
        """
        Validar valor de similaridade cosseno
        
        Args:
            similarity: Valor de similaridade
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not isinstance(similarity, (int, float)):
            return False, "Similaridade deve ser um número"
        
        if np.isnan(similarity) or np.isinf(similarity):
            return False, "Similaridade não pode ser NaN ou Infinito"
        
        if similarity < -1.0 or similarity > 1.0:
            return False, f"Similaridade cosseno deve estar entre -1 e 1, recebido: {similarity}"
        
        return True, ""
    
    @staticmethod
    def aicube_sanitize_model_name(model_name: str) -> str:
        """
        Sanitizar nome do modelo
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            str: Nome sanitizado
        """
        if not isinstance(model_name, str):
            return ""
        
        # Remover espaços e converter para minúsculas
        sanitized = model_name.strip().lower()
        
        # Substituir espaços por underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        
        # Remover caracteres especiais
        sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '', sanitized)
        
        return sanitized
    
    @staticmethod
    def aicube_normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        Normalizar embedding (norma L2)
        
        Args:
            embedding: Array numpy do embedding
            
        Returns:
            np.ndarray: Embedding normalizado
        """
        try:
            norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
            
            # Evitar divisão por zero
            norm = np.where(norm == 0, 1, norm)
            
            return embedding / norm
            
        except Exception:
            # Em caso de erro, retornar o embedding original
            return embedding
    
    @staticmethod
    def aicube_compute_cosine_similarity(
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Computar similaridade cosseno entre dois embeddings
        
        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding
            
        Returns:
            float: Similaridade cosseno
        """
        try:
            # Normalizar embeddings
            emb1_norm = AICUBEValidators.aicube_normalize_embedding(embedding1)
            emb2_norm = AICUBEValidators.aicube_normalize_embedding(embedding2)
            
            # Calcular produto escalar
            if emb1_norm.ndim == 1 and emb2_norm.ndim == 1:
                similarity = np.dot(emb1_norm, emb2_norm)
            else:
                # Para batch
                similarity = np.mean(np.sum(emb1_norm * emb2_norm, axis=-1))
            
            return float(similarity)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def aicube_validate_request_data(data: dict) -> Tuple[bool, str]:
        """
        Validar dados de requisição completos
        
        Args:
            data: Dados da requisição
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        required_fields = ['origem', 'destino', 'embedding']
        
        # Verificar campos obrigatórios
        for field in required_fields:
            if field not in data:
                return False, f"Campo obrigatório ausente: {field}"
        
        # Validar modelo de origem
        is_valid, error = AICUBEValidators.aicube_validate_model_name(data['origem'])
        if not is_valid:
            return False, f"Modelo de origem inválido: {error}"
        
        # Validar modelo de destino
        is_valid, error = AICUBEValidators.aicube_validate_model_name(data['destino'])
        if not is_valid:
            return False, f"Modelo de destino inválido: {error}"
        
        # Validar embedding
        is_valid, error = AICUBEValidators.aicube_validate_embedding_format(data['embedding'])
        if not is_valid:
            return False, f"Embedding inválido: {error}"
        
        return True, ""


# Instância global dos validadores
aicube_validators = AICUBEValidators()