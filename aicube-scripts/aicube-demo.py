#!/usr/bin/env python3
"""
AICUBE Embedding2Embedding API - Demo Script

Script de demonstração da API AICUBE mostrando diferentes funcionalidades.
"""

import requests
import numpy as np
import time
import json
from typing import List, Dict, Any
import argparse


class AICUBEEmbeddingClient:
    """
    Cliente demonstração para a API AICUBE Embedding2Embedding
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "AICUBE-Demo-Client/1.0"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar saúde da API AICUBE"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def list_models(self) -> Dict[str, Any]:
        """Listar modelos disponíveis"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Obter informações de modelo específico"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models/{model_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def translate_embedding(
        self, 
        origem: str, 
        destino: str, 
        embedding: List[float],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Traduzir um embedding"""
        try:
            payload = {
                "origem": origem,
                "destino": destino,
                "embedding": embedding,
                "include_metadata": include_metadata
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/translate",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def translate_batch(
        self,
        origem: str,
        destino: str, 
        embeddings: List[List[float]],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Traduzir múltiplos embeddings"""
        try:
            payload = {
                "origem": origem,
                "destino": destino,
                "embeddings": embeddings,
                "include_metadata": include_metadata
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/translate/batch",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obter estatísticas de uso"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/statistics")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}


def print_header(title: str):
    """Imprimir cabeçalho formatado"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title: str):
    """Imprimir seção formatada"""
    print(f"\n🔹 {title}")
    print("-" * 40)


def print_success(message: str):
    """Imprimir mensagem de sucesso"""
    print(f"✅ {message}")


def print_error(message: str):
    """Imprimir mensagem de erro"""
    print(f"❌ {message}")


def print_info(message: str):
    """Imprimir mensagem informativa"""
    print(f"ℹ️  {message}")


def generate_sample_embedding(dim: int = 768) -> List[float]:
    """Gerar embedding de exemplo"""
    # Simular embedding normalizado
    embedding = np.random.randn(dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def demo_health_check(client: AICUBEEmbeddingClient):
    """Demonstrar health check"""
    print_section("Health Check AICUBE")
    
    health = client.health_check()
    
    if "error" in health:
        print_error(f"Falha no health check: {health['error']}")
        return False
    
    print_success("API AICUBE está saudável!")
    print(f"   📊 Serviço: {health.get('aicube_service', 'N/A')}")
    print(f"   🏷️  Versão: {health.get('version', 'N/A')}")
    print(f"   🏢 Tecnologia: {health.get('aicube_technology', 'N/A')}")
    print(f"   📈 Modelos carregados: {health.get('models_loaded', 0)}")
    
    powered_by = health.get('powered_by', [])
    if powered_by:
        print(f"   ⚡ Powered by: {', '.join(powered_by)}")
    
    return True


def demo_list_models(client: AICUBEEmbeddingClient):
    """Demonstrar listagem de modelos"""
    print_section("Modelos AICUBE Disponíveis")
    
    models = client.list_models()
    
    if "error" in models:
        print_error(f"Erro ao listar modelos: {models['error']}")
        return []
    
    aicube_models = models.get('aicube_models', [])
    model_pairs = models.get('aicube_model_pairs', [])
    
    print_success(f"Total de modelos AICUBE: {len(aicube_models)}")
    
    print("\n📋 Modelos disponíveis:")
    for model in aicube_models:
        print(f"   • {model}")
    
    print("\n🔄 Pares de tradução suportados:")
    for pair in model_pairs:
        source = pair.get('source', 'N/A')
        target = pair.get('target', 'N/A')
        translator = pair.get('aicube_translator_id', 'N/A')
        tech = pair.get('aicube_technology', [])
        
        print(f"   • {source} → {target}")
        print(f"     Tradutor: {translator}")
        if tech:
            print(f"     Tecnologia: {', '.join(tech)}")
    
    return aicube_models


def demo_model_info(client: AICUBEEmbeddingClient, model_id: str):
    """Demonstrar informações de modelo"""
    print_section(f"Informações do Modelo: {model_id}")
    
    info = client.get_model_info(model_id)
    
    if "error" in info:
        print_error(f"Erro ao obter informações: {info['error']}")
        return
    
    print_success("Informações obtidas com sucesso!")
    print(f"   🆔 ID: {info.get('model_id', 'N/A')}")
    print(f"   📥 Modelo origem: {info.get('source_model', 'N/A')}")
    print(f"   📤 Modelo destino: {info.get('target_model', 'N/A')}")
    print(f"   📐 Dimensão origem: {info.get('source_dimension', 'N/A')}")
    print(f"   📏 Dimensão destino: {info.get('target_dimension', 'N/A')}")
    print(f"   🏷️  Versão: {info.get('version', 'N/A')}")
    print(f"   📝 Descrição: {info.get('description', 'N/A')}")
    print(f"   👨‍💻 Criado por: {info.get('created_by', 'N/A')}")
    
    tech = info.get('aicube_technology', [])
    if tech:
        print(f"   ⚡ Tecnologia AICUBE: {', '.join(tech)}")


def demo_single_translation(client: AICUBEEmbeddingClient):
    """Demonstrar tradução de embedding único"""
    print_section("Tradução de Embedding Único")
    
    # Parâmetros de teste
    origem = "bert_base_uncased"
    destino = "t5_base"
    embedding = generate_sample_embedding(768)
    
    print_info(f"Traduzindo embedding de {origem} para {destino}")
    print(f"   📐 Dimensão original: {len(embedding)}")
    print(f"   🔢 Primeiros valores: {embedding[:5]}")
    
    start_time = time.time()
    result = client.translate_embedding(origem, destino, embedding)
    duration = time.time() - start_time
    
    if "error" in result:
        print_error(f"Erro na tradução: {result['error']}")
        return
    
    print_success(f"Tradução concluída em {duration:.2f}s!")
    
    translated = result.get('embedding_traduzido', [])
    print(f"   📏 Dimensão traduzida: {len(translated)}")
    print(f"   🔢 Primeiros valores: {translated[:5]}")
    print(f"   🏢 Tecnologia: {result.get('aicube_technology', 'N/A')}")
    
    metadata = result.get('metadata', {})
    if metadata:
        print("\n📊 Metadados:")
        print(f"   🤖 Modelo usado: {metadata.get('aicube_model_id', 'N/A')}")
        print(f"   ⏱️  Duração: {metadata.get('duration_ms', 'N/A')}ms")
        print(f"   📈 Similaridade cosseno: {metadata.get('cosine_similarity', 'N/A')}")


def demo_batch_translation(client: AICUBEEmbeddingClient):
    """Demonstrar tradução em lote"""
    print_section("Tradução em Lote")
    
    # Parâmetros de teste
    origem = "bert_base_uncased"
    destino = "t5_base"
    batch_size = 3
    embeddings = [generate_sample_embedding(768) for _ in range(batch_size)]
    
    print_info(f"Traduzindo batch de {batch_size} embeddings")
    print(f"   📦 Tamanho do batch: {len(embeddings)}")
    print(f"   📐 Dimensão: {len(embeddings[0])}")
    
    start_time = time.time()
    result = client.translate_batch(origem, destino, embeddings)
    duration = time.time() - start_time
    
    if "error" in result:
        print_error(f"Erro na tradução em lote: {result['error']}")
        return
    
    print_success(f"Tradução em lote concluída em {duration:.2f}s!")
    
    translated_batch = result.get('embeddings_traduzidos', [])
    print(f"   📦 Embeddings traduzidos: {len(translated_batch)}")
    print(f"   📏 Dimensão traduzida: {len(translated_batch[0]) if translated_batch else 0}")
    print(f"   🏢 Tecnologia: {result.get('aicube_technology', 'N/A')}")
    
    metadata = result.get('metadata', {})
    if metadata:
        print("\n📊 Metadados do batch:")
        print(f"   ⏱️  Duração total: {metadata.get('total_duration_ms', 'N/A')}ms")
        print(f"   📈 Similaridade média: {metadata.get('average_cosine_similarity', 'N/A')}")


def demo_statistics(client: AICUBEEmbeddingClient):
    """Demonstrar estatísticas"""
    print_section("Estatísticas de Uso AICUBE")
    
    stats = client.get_statistics()
    
    if "error" in stats:
        print_error(f"Erro ao obter estatísticas: {stats['error']}")
        return
    
    print_success("Estatísticas obtidas com sucesso!")
    print(f"   📈 Modelos carregados: {stats.get('aicube_loaded_models', 0)}")
    print(f"   📋 Modelos disponíveis: {stats.get('aicube_available_models', 0)}")
    print(f"   🏢 Tecnologia: {stats.get('aicube_technology', 'N/A')}")
    print(f"   🏷️  Versão: {stats.get('aicube_version', 'N/A')}")
    
    usage_stats = stats.get('aicube_usage_stats', {})
    if usage_stats:
        print("\n📊 Estatísticas de uso por modelo:")
        for model, count in usage_stats.items():
            print(f"   • {model}: {count} usos")
    
    powered_by = stats.get('powered_by', [])
    if powered_by:
        print(f"\n⚡ Powered by: {', '.join(powered_by)}")


def run_comprehensive_demo(base_url: str):
    """Executar demonstração completa"""
    print_header("AICUBE Embedding2Embedding API - Demo Completa")
    print("Desenvolvido pela AICUBE TECHNOLOGY LLC")
    print("Powered by: Qube LCM Model, Qube Neural Memory, Qube Agentic Workflows, Qube Computer Vision")
    
    # Inicializar cliente
    client = AICUBEEmbeddingClient(base_url)
    
    # 1. Health Check
    if not demo_health_check(client):
        print_error("API não está disponível. Verifique se o serviço está rodando.")
        return
    
    # 2. Listar modelos
    models = demo_list_models(client)
    
    # 3. Informações de modelo (se houver modelos)
    if models:
        demo_model_info(client, models[0])
    
    # 4. Tradução única
    demo_single_translation(client)
    
    # 5. Tradução em lote
    demo_batch_translation(client)
    
    # 6. Estatísticas
    demo_statistics(client)
    
    print_header("Demo AICUBE Concluída!")
    print("🎉 Todas as funcionalidades foram demonstradas com sucesso!")
    print("\nPara mais informações:")
    print(f"   📚 Documentação: {base_url}/aicube-docs")
    print(f"   🏥 Health Check: {base_url}/api/v1/health")
    print("   🌐 AICUBE TECHNOLOGY: https://aicube.technology")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Demo da API AICUBE Embedding2Embedding"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="URL base da API (padrão: http://localhost:8000)"
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Executar apenas health check"
    )
    parser.add_argument(
        "--models-only",
        action="store_true", 
        help="Listar apenas modelos disponíveis"
    )
    
    args = parser.parse_args()
    
    client = AICUBEEmbeddingClient(args.url)
    
    if args.health_only:
        print_header("AICUBE Health Check")
        demo_health_check(client)
    elif args.models_only:
        print_header("AICUBE Modelos Disponíveis")
        demo_list_models(client)
    else:
        run_comprehensive_demo(args.url)


if __name__ == "__main__":
    main()