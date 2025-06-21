# AICUBE Embedding2Embedding API Reference

## Visão Geral

A **AICUBE Embedding2Embedding API** fornece endpoints RESTful para tradução de embeddings entre diferentes espaços vetoriais. Desenvolvida pela **AICUBE TECHNOLOGY** utilizando tecnologias avançadas.

## Base URL

```
http://localhost:8000
```

## Autenticação

Atualmente a API não implementa autenticação conforme especificado nos requisitos (foco no core). Para implementação futura, será utilizado JWT ou API Keys.

## Headers Comuns

Todas as requisições devem incluir:

```http
Content-Type: application/json
Accept: application/json
```

## Endpoints

### 1. Health Check

Verifica o status de saúde do serviço AICUBE.

**Endpoint:** `GET /api/v1/health`

**Resposta:**
```json
{
  "status": "healthy",
  "aicube_service": "aicube-embedding2embedding",
  "version": "v1",
  "timestamp": "2023-06-19T12:00:00Z",
  "aicube_technology": "AICUBE TECHNOLOGY",
  "models_loaded": 3,
  "powered_by": [
    "Qube LCM Model",
    "Qube Neural Memory", 
    "Qube Agentic Workflows",
    "Qube Computer Vision"
  ]
}
```

**Status Codes:**
- `200`: Serviço saudável
- `503`: Serviço indisponível

---

### 2. Traduzir Embedding

Traduz um embedding de um modelo de origem para um modelo de destino.

**Endpoint:** `POST /api/v1/translate`

**Request Body:**
```json
{
  "origem": "bert_base_uncased",
  "destino": "t5_base",
  "embedding": [0.12, 0.45, -0.67, 0.89, -0.23],
  "include_metadata": true
}
```

**Parâmetros:**
- `origem` (string, obrigatório): Modelo de origem do embedding
- `destino` (string, obrigatório): Modelo de destino para tradução
- `embedding` (array, obrigatório): Vetor de embedding ou lista de vetores
- `include_metadata` (boolean, opcional): Incluir metadados na resposta (padrão: false)

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
    "source_dimension": 768,
    "target_dimension": 768,
    "powered_by": ["Qube LCM Model", "Qube Neural Memory"]
  }
}
```

**Status Codes:**
- `200`: Tradução realizada com sucesso
- `400`: Erro de validação ou modelo não suportado
- `422`: Dados de entrada inválidos
- `500`: Erro interno do servidor

---

### 3. Tradução em Lote

Traduz múltiplos embeddings de uma vez.

**Endpoint:** `POST /api/v1/translate/batch`

**Request Body:**
```json
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

**Parâmetros:**
- `origem` (string, obrigatório): Modelo de origem
- `destino` (string, obrigatório): Modelo de destino
- `embeddings` (array, obrigatório): Lista de vetores de embedding
- `include_metadata` (boolean, opcional): Incluir metadados

**Resposta:**
```json
{
  "origem": "bert_base_uncased",
  "destino": "t5_base",
  "embeddings_traduzidos": [
    [0.08, 0.33, -0.10],
    [0.71, -0.15, 0.42]
  ],
  "batch_size": 2,
  "aicube_technology": "AICUBE TECHNOLOGY",
  "metadata": {
    "total_duration_ms": 25.6,
    "average_cosine_similarity": 0.93
  }
}
```

**Limitações:**
- Máximo de 32 embeddings por batch
- Dimensão máxima de 4096 por embedding

---

### 4. Listar Modelos

Retorna lista de modelos de tradução disponíveis.

**Endpoint:** `GET /api/v1/models`

**Resposta:**
```json
{
  "aicube_models": [
    "aicube-bert-to-t5",
    "aicube-mpnet-to-ada002",
    "aicube-roberta-to-gpt2"
  ],
  "aicube_model_pairs": [
    {
      "source": "bert_base_uncased",
      "target": "t5_base",
      "aicube_translator_id": "aicube-bert-to-t5",
      "aicube_technology": ["Qube LCM Model"]
    }
  ],
  "total_models": 3,
  "aicube_technology": "AICUBE TECHNOLOGY",
  "powered_by": [
    "Qube LCM Model",
    "Qube Neural Memory",
    "Qube Agentic Workflows", 
    "Qube Computer Vision"
  ]
}
```

---

### 5. Informações de Modelo

Retorna informações detalhadas de um modelo específico.

**Endpoint:** `GET /api/v1/models/{model_id}`

**Parâmetros:**
- `model_id` (string): ID do modelo AICUBE

**Resposta:**
```json
{
  "model_id": "aicube-bert-to-t5",
  "source_model": "bert_base_uncased",
  "target_model": "t5_base",
  "source_dimension": 768,
  "target_dimension": 768,
  "version": "1.0.0-aicube",
  "description": "Tradutor AICUBE BERT para T5 usando Qube LCM Model",
  "aicube_technology": ["Qube LCM Model"],
  "created_by": "AICUBE TECHNOLOGY"
}
```

**Status Codes:**
- `200`: Modelo encontrado
- `404`: Modelo não encontrado

---

### 6. Estatísticas

Retorna estatísticas de uso dos modelos.

**Endpoint:** `GET /api/v1/statistics`

**Resposta:**
```json
{
  "aicube_loaded_models": 3,
  "aicube_available_models": 5,
  "aicube_usage_stats": {
    "aicube-bert-to-t5": 150,
    "aicube-mpnet-to-ada002": 89
  },
  "aicube_technology": "AICUBE TECHNOLOGY",
  "aicube_version": "1.0.0-aicube",
  "powered_by": ["Qube LCM Model", "Qube Neural Memory"]
}
```

---

## Modelos Suportados

| Modelo de Origem | Modelo de Destino | Tradutor AICUBE | Tecnologia | Dimensões |
|-----------------|------------------|-----------------|------------|-----------|
| bert_base_uncased | t5_base | aicube-bert-to-t5 | Qube LCM Model | 768→768 |
| sentence_transformers_mpnet | openai_ada002 | aicube-mpnet-to-ada002 | Qube Neural Memory | 768→1536 |
| roberta_base | gpt2_base | aicube-roberta-to-gpt2 | Qube Agentic Workflows | 768→768 |

## Códigos de Erro

### 400 - Bad Request
```json
{
  "error": "ValidationError",
  "message": "Modelo de origem não suportado: invalid_model",
  "aicube_service": "aicube-embedding2embedding",
  "timestamp": "2023-06-19T12:00:00Z",
  "request_id": "req_12345"
}
```

### 422 - Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "embedding"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 - Internal Server Error
```json
{
  "error": "Erro interno do servidor AICUBE",
  "message": "Por favor, entre em contato com o suporte técnico",
  "request_id": "req_12345"
}
```

## Rate Limiting

- 100 requisições por minuto por IP
- Headers de resposta incluem limites:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

## Exemplos de Uso

### cURL

```bash
# Health check
curl -X GET http://localhost:8000/api/v1/health

# Traduzir embedding
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "origem": "bert_base_uncased",
    "destino": "t5_base", 
    "embedding": [0.1, 0.2, -0.3, 0.4],
    "include_metadata": true
  }'

# Listar modelos
curl -X GET http://localhost:8000/api/v1/models
```

### Python

```python
import requests

# Cliente AICUBE
class AICUBEClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def translate_embedding(self, origem, destino, embedding, include_metadata=False):
        response = requests.post(
            f"{self.base_url}/api/v1/translate",
            json={
                "origem": origem,
                "destino": destino,
                "embedding": embedding,
                "include_metadata": include_metadata
            }
        )
        return response.json()
    
    def list_models(self):
        response = requests.get(f"{self.base_url}/api/v1/models")
        return response.json()

# Uso
client = AICUBEClient()
result = client.translate_embedding(
    origem="bert_base_uncased",
    destino="t5_base",
    embedding=[0.1, 0.2, -0.3, 0.4, 0.5],
    include_metadata=True
)
print(result)
```

### JavaScript

```javascript
class AICUBEClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async translateEmbedding(origem, destino, embedding, includeMetadata = false) {
    const response = await fetch(`${this.baseUrl}/api/v1/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        origem,
        destino,
        embedding,
        include_metadata: includeMetadata
      })
    });
    
    return await response.json();
  }

  async listModels() {
    const response = await fetch(`${this.baseUrl}/api/v1/models`);
    return await response.json();
  }
}

// Uso
const client = new AICUBEClient();
client.translateEmbedding(
  'bert_base_uncased',
  't5_base',
  [0.1, 0.2, -0.3, 0.4, 0.5],
  true
).then(result => console.log(result));
```

## SDKs Oficiais

Futuramente serão disponibilizados SDKs oficiais para:
- Python (`aicube-embedding2embedding-python`)
- JavaScript/Node.js (`aicube-embedding2embedding-js`)
- Java (`aicube-embedding2embedding-java`)
- C# (`aicube-embedding2embedding-dotnet`)

## Suporte

Para dúvidas e suporte técnico:
- **Email**: support@aicube.technology
- **Documentação**: http://localhost:8000/aicube-docs
- **Status**: http://localhost:8000/api/v1/health

---

**Desenvolvido por AICUBE TECHNOLOGY**  
*Powered by Qube LCM Model, Qube Neural Memory, Qube Agentic Workflows e Qube Computer Vision*