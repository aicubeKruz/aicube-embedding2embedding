# AICUBE Embedding2Embedding API

## Visão Geral

A **AICUBE Embedding2Embedding API** é um serviço desenvolvido pela **AICUBE TECHNOLOGY** que permite traduzir embeddings entre diferentes espaços vetoriais de modelos de linguagem natural. Utilizando tecnologias avançadas como **Qube LCM Model**, **Qube Neural Memory**, **Qube Agentic Workflows** e **Qube Computer Vision**, a API preserva o significado semântico durante a conversão.

## Tecnologias AICUBE Utilizadas

- **Qube LCM Model**: Modelo de linguagem contextual avançado
- **Qube Neural Memory**: Sistema de memória neural para preservação semântica
- **Qube Agentic Workflows**: Fluxos de trabalho inteligentes e autônomos
- **Qube Computer Vision**: Processamento visual avançado para análise multimodal

## Características Principais

### 🚀 Performance Otimizada
- Arquitetura baseada em MLP com conexões residuais
- Ativação SiLU para melhor convergência
- Normalização de camadas para estabilidade
- Suporte a processamento em batch

### 🔄 Interoperabilidade
- Tradução entre modelos populares (BERT, T5, RoBERTa, GPT-2, OpenAI Ada-002)
- Preservação de similaridade semântica (>90% de similaridade cosseno)
- Suporte a diferentes dimensionalidades de embedding

### 📊 Monitoramento Avançado
- Logging estruturado com métricas detalhadas
- Health checks automáticos
- Estatísticas de uso em tempo real
- Rastreamento de performance

### 🔧 Flexibilidade
- Configuração via variáveis de ambiente
- Carregamento dinâmico de modelos
- Cache inteligente para otimização
- API RESTful com documentação OpenAPI

## Instalação e Configuração

### Pré-requisitos
- Python 3.11+
- Docker (opcional)
- CUDA (opcional, para aceleração GPU)

### Instalação com Docker (Recomendado)

```bash
# Clone o repositório
git clone <repository-url>
cd aicube-embedding2embedding

# Build da imagem
docker build -f aicube-Dockerfile -t aicube-embedding2embedding .

# Executar com Docker Compose
docker-compose -f aicube-docker-compose.yml up -d
```

### Instalação Manual

```bash
# Criar ambiente virtual
python -m venv aicube-env
source aicube-env/bin/activate  # Linux/Mac
# ou
aicube-env\Scripts\activate  # Windows

# Instalar dependências
pip install -r aicube-requirements.txt

# Configurar variáveis de ambiente
cp aicube.env.example aicube.env
# Editar aicube.env conforme necessário

# Executar aplicação
python aicube-app/aicube-main.py
```

## Uso da API

### Endpoints Principais

#### 1. Traduzir Embedding
```http
POST /api/v1/translate
Content-Type: application/json

{
  "origem": "bert_base_uncased",
  "destino": "t5_base", 
  "embedding": [0.12, 0.45, -0.67, 0.89, -0.23],
  "include_metadata": true
}
```

**Resposta:**
```json
{
  "origem": "bert_base_uncased",
  "destino": "t5_base",
  "embedding_traduzido": [0.08, 0.33, -0.10, 0.71, -0.15],
  "aicube_technology": "AICUBE TECHNOLOGY",
  "metadata": {
    "aicube_model_id": "aicube-bert-to-t5",
    "duration_ms": 15.2,
    "cosine_similarity": 0.94,
    "powered_by": ["Qube LCM Model", "Qube Neural Memory"]
  }
}
```

#### 2. Tradução em Lote
```http
POST /api/v1/translate/batch
Content-Type: application/json

{
  "origem": "bert_base_uncased",
  "destino": "t5_base",
  "embeddings": [
    [0.12, 0.45, -0.67],
    [0.89, -0.23, 0.56]
  ],
  "include_metadata": true
}
```

#### 3. Listar Modelos Disponíveis
```http
GET /api/v1/models
```

#### 4. Health Check
```http
GET /api/v1/health
```

#### 5. Estatísticas de Uso
```http
GET /api/v1/statistics
```

### Modelos Suportados

| Modelo de Origem | Modelo de Destino | Tradutor AICUBE | Tecnologia |
|-----------------|------------------|-----------------|------------|
| bert_base_uncased | t5_base | aicube-bert-to-t5 | Qube LCM Model |
| sentence_transformers_mpnet | openai_ada002 | aicube-mpnet-to-ada002 | Qube Neural Memory |
| roberta_base | gpt2_base | aicube-roberta-to-gpt2 | Qube Agentic Workflows |

## Arquitetura

### Camadas da Aplicação

```
┌─────────────────────────────────────┐
│        AICUBE API Layer             │
│  (FastAPI + Pydantic Validation)   │
└─────────────────────────────────────┘
                  │
┌─────────────────────────────────────┐
│      AICUBE Business Logic          │
│   (Model Manager + Validators)     │
└─────────────────────────────────────┘
                  │
┌─────────────────────────────────────┐
│     AICUBE ML Translation Layer     │
│  (Neural Networks + PyTorch)       │
└─────────────────────────────────────┘
```

### Componentes Principais

- **aicube-main.py**: Aplicação FastAPI principal
- **aicube-model-manager.py**: Gerenciador de modelos de tradução
- **aicube-embedding-translator.py**: Modelos neurais de tradução
- **aicube-endpoints.py**: Definição dos endpoints REST
- **aicube-schemas.py**: Validação de dados com Pydantic
- **aicube-logging.py**: Sistema de logging estruturado

## Configuração

### Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `AICUBE_API_NAME` | Nome da API | aicube-embedding2embedding |
| `AICUBE_MODELS_PATH` | Caminho dos modelos | ./aicube-models |
| `AICUBE_MAX_EMBEDDING_DIMENSION` | Dimensão máxima | 4096 |
| `AICUBE_MAX_BATCH_SIZE` | Tamanho máximo do batch | 32 |
| `AICUBE_DEVICE` | Dispositivo (cpu/cuda) | cpu |

### Logging

O sistema utiliza logging estruturado com contexto AICUBE:

```json
{
  "timestamp": "2023-06-19T12:00:00Z",
  "level": "info",
  "message": "AICUBE Embedding Translation",
  "aicube_service": "aicube-embedding2embedding",
  "aicube_technology": "AICUBE TECHNOLOGY",
  "source_model": "bert_base_uncased",
  "target_model": "t5_base",
  "duration_ms": 15.2,
  "cosine_similarity": 0.94
}
```

## Desenvolvimento

### Estrutura do Projeto

```
aicube-embedding2embedding/
├── aicube-app/                    # Código principal
│   ├── aicube-api/               # Endpoints REST
│   ├── aicube-core/              # Lógica central
│   ├── aicube-models/            # Modelos ML
│   └── aicube-utils/             # Utilitários
├── aicube-tests/                 # Testes
├── aicube-models/                # Modelos pré-treinados
├── aicube-docs/                  # Documentação
├── aicube-configs/               # Configurações
└── aicube-scripts/               # Scripts auxiliares
```

### Executar Testes

```bash
# Testes unitários
pytest aicube-tests/aicube-unit/ -v

# Testes de integração
pytest aicube-tests/aicube-integration/ -v

# Coverage
pytest --cov=aicube-app aicube-tests/
```

### Adicionar Novo Modelo

1. Treinar modelo de tradução
2. Salvar no formato AICUBE (.pt)
3. Adicionar entrada em `aicube-model-registry.json`
4. Atualizar documentação

## Métricas e Monitoramento

### Métricas Disponíveis

- **Latência**: Tempo de resposta por requisição
- **Throughput**: Requisições por segundo
- **Similaridade**: Qualidade da tradução (similaridade cosseno)
- **Uso de Modelos**: Estatísticas por modelo
- **Erros**: Taxa de erro e tipos

### Dashboards

A API expõe métricas no formato Prometheus em `/metrics` (porta 8001):

```
# HELP aicube_translation_duration_seconds Time spent translating embeddings
# TYPE aicube_translation_duration_seconds histogram
aicube_translation_duration_seconds_bucket{model="aicube-bert-to-t5",le="0.01"} 150
```

## Performance

### Benchmarks

| Modelo | Dimensão | Latência (ms) | Similaridade | Throughput (req/s) |
|--------|----------|---------------|--------------|-------------------|
| BERT→T5 | 768→768 | 15.2 | 0.94 | 65 |
| MPNet→Ada002 | 768→1536 | 23.1 | 0.93 | 43 |
| RoBERTa→GPT2 | 768→768 | 18.7 | 0.92 | 53 |

### Otimizações

- **Batch Processing**: Processe múltiplos embeddings juntos
- **Model Caching**: Mantenha modelos carregados em memória
- **GPU Acceleration**: Use CUDA quando disponível
- **Quantização**: Modelos quantizados para menor latência

## Casos de Uso

### 1. Setor Financeiro
- Análise de sentimento em notícias financeiras
- Detecção de fraudes em transações
- Chatbots especializados em finanças

### 2. Setor de Seguros
- Processamento de sinistros
- Análise multimodal (texto + imagem)
- Classificação automática de reclamações

### 3. Varejo
- Recomendações de produtos
- Busca semântica multilíngue
- Análise de reviews de clientes

### 4. Jurídico
- Pesquisa de jurisprudência
- Análise de contratos
- E-discovery automatizado

## Suporte e Contribuição

### Reportar Issues
Para reportar bugs ou solicitar funcionalidades, use o sistema de issues com as tags:
- `aicube-bug`: Para bugs
- `aicube-enhancement`: Para novas funcionalidades
- `aicube-question`: Para dúvidas

### Contribuir
1. Fork o repositório
2. Crie branch com prefixo `aicube-feature/`
3. Implemente seguindo os padrões AICUBE
4. Adicione testes com prefixo `aicube_test_`
5. Submeta pull request

### Licença
Este projeto é propriedade da **AICUBE TECHNOLOGY** e utiliza tecnologias proprietárias.

---

**Desenvolvido por AICUBE TECHNOLOGY**  
Powered by Qube LCM Model, Qube Neural Memory, Qube Agentic Workflows e Qube Computer Vision