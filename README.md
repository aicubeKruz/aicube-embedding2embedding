# AICUBE Embedding2Embedding API

[![GitHub](https://img.shields.io/github/license/aicubeKruz/aicube-embedding2embedding?style=flat-square&logo=github)](https://github.com/aicubeKruz/aicube-embedding2embedding)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square)](https://docker.com)
[![AICUBE](https://img.shields.io/badge/AICUBE-TECHNOLOGY-purple?style=flat-square)](https://aicube.technology)

## Overview

The **AICUBE Embedding2Embedding API** is a service developed by **AICUBE TECHNOLOGY** that enables translation of embeddings between different vector spaces of natural language models. The API preserves semantic meaning during conversion using advanced neural network architectures and machine learning techniques.

## Technologies Used

- **FastAPI**: High-performance web framework for building APIs
- **PyTorch**: Deep learning framework for neural network implementation
- **MLP Architecture**: Multi-Layer Perceptron with residual connections
- **SiLU Activation**: Sigmoid Linear Unit for improved model performance
- **Layer Normalization**: For training stability and convergence
- **Structured Logging**: For comprehensive monitoring and debugging

## Key Features

### 🚀 Optimized Performance
- MLP-based architecture with residual connections
- SiLU activation for better convergence
- Layer normalization for stability
- Batch processing support

### 🔄 Interoperability
- Translation between popular models (BERT, T5, RoBERTa, GPT-2, OpenAI Ada-002)
- Semantic similarity preservation (>90% cosine similarity)
- Support for different embedding dimensionalities

### 📊 Advanced Monitoring
- Structured logging with detailed metrics
- Automatic health checks
- Real-time usage statistics
- Performance tracking

### 🔧 Flexibility
- Configuration via environment variables
- Dynamic model loading
- Smart caching for optimization
- RESTful API with OpenAPI documentation

## Installation and Setup

### Prerequisites
- Python 3.11+
- Docker (optional)
- CUDA (optional, for GPU acceleration)

### Installation with Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd aicube-embedding2embedding

# Build the image
docker build -f aicube-Dockerfile -t aicube-embedding2embedding .

# Run with Docker Compose
docker-compose -f aicube-docker-compose.yml up -d
```

### Manual Installation

```bash
# Create virtual environment
python -m venv aicube-venv
source aicube-venv/bin/activate  # Linux/Mac
# or
aicube-venv\Scripts\activate  # Windows

# Install dependencies
pip install -r aicube-requirements.txt

# Configure environment variables
cp aicube.env.example aicube.env
# Edit aicube.env as needed

# Run application
python aicube-app/aicube-main.py
```

## API Usage

### Main Endpoints

#### 1. Translate Embedding
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

**Response:**
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

#### 2. Batch Translation
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

#### 3. List Available Models
```http
GET /api/v1/models
```

#### 4. Health Check
```http
GET /api/v1/health
```

#### 5. Usage Statistics
```http
GET /api/v1/statistics
```

### Supported Models

| Source Model | Target Model | AICUBE Translator | Technology |
|--------------|--------------|-------------------|------------|
| bert_base_uncased | t5_base | aicube-bert-to-t5 | Qube LCM Model |
| sentence_transformers_mpnet | openai_ada002 | aicube-mpnet-to-ada002 | Qube Neural Memory |
| roberta_base | gpt2_base | aicube-roberta-to-gpt2 | Qube Agentic Workflows |

## Architecture

### Application Layers

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

### Main Components

- **aicube-main.py**: Main FastAPI application
- **aicube-model-manager.py**: Translation model manager
- **aicube-embedding-translator.py**: Neural translation models
- **aicube-endpoints.py**: REST endpoint definitions
- **aicube-schemas.py**: Data validation with Pydantic
- **aicube-logging.py**: Structured logging system

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AICUBE_API_NAME` | API name | aicube-embedding2embedding |
| `AICUBE_MODELS_PATH` | Models directory | ./aicube-models |
| `AICUBE_MAX_EMBEDDING_DIMENSION` | Maximum dimension | 4096 |
| `AICUBE_MAX_BATCH_SIZE` | Maximum batch size | 32 |
| `AICUBE_DEVICE` | Device (cpu/cuda) | cpu |

### Logging

The system uses structured logging with AICUBE context:

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

## Development

### Project Structure

```
aicube-embedding2embedding/
├── aicube-app/                    # Main code
│   ├── aicube-api/               # REST endpoints
│   ├── aicube-core/              # Core logic
│   ├── aicube-models/            # ML models
│   └── aicube-utils/             # Utilities
├── aicube-tests/                 # Tests
├── aicube-models/                # Pre-trained models
├── aicube-docs/                  # Documentation
├── aicube-configs/               # Configurations
└── aicube-scripts/               # Helper scripts
```

### Running Tests

```bash
# Unit tests
pytest aicube-tests/aicube-unit/ -v

# Integration tests
pytest aicube-tests/aicube-integration/ -v

# Coverage
pytest --cov=aicube-app aicube-tests/
```

### Adding New Model

1. Train translation model
2. Save in AICUBE format (.pt)
3. Add entry to `aicube-model-registry.json`
4. Update documentation

## Metrics and Monitoring

### Available Metrics

- **Latency**: Response time per request
- **Throughput**: Requests per second
- **Similarity**: Translation quality (cosine similarity)
- **Model Usage**: Statistics per model
- **Errors**: Error rate and types

### Dashboards

The API exposes Prometheus-format metrics at `/metrics` (port 8001):

```
# HELP aicube_translation_duration_seconds Time spent translating embeddings
# TYPE aicube_translation_duration_seconds histogram
aicube_translation_duration_seconds_bucket{model="aicube-bert-to-t5",le="0.01"} 150
```

## Performance

### Benchmarks

| Model | Dimension | Latency (ms) | Similarity | Throughput (req/s) |
|-------|-----------|--------------|------------|-------------------|
| BERT→T5 | 768→768 | 15.2 | 0.94 | 65 |
| MPNet→Ada002 | 768→1536 | 23.1 | 0.93 | 43 |
| RoBERTa→GPT2 | 768→768 | 18.7 | 0.92 | 53 |

### Optimizations

- **Batch Processing**: Process multiple embeddings together
- **Model Caching**: Keep models loaded in memory
- **GPU Acceleration**: Use CUDA when available
- **Quantization**: Quantized models for lower latency

## Use Cases

### 1. Financial Sector
- Financial news sentiment analysis
- Fraud detection in transactions
- Specialized financial chatbots

### 2. Insurance Sector
- Claims processing
- Multimodal analysis (text + image)
- Automatic complaint classification

### 3. Retail
- Product recommendations
- Multilingual semantic search
- Customer review analysis

### 4. Legal
- Case law research
- Contract analysis
- Automated e-discovery

## Support and Contribution

### Reporting Issues
To report bugs or request features, use the issue system with tags:
- `aicube-bug`: For bugs
- `aicube-enhancement`: For new features
- `aicube-question`: For questions

### Contributing
1. Fork the repository
2. Create branch with `aicube-feature/` prefix
3. Implement following AICUBE standards
4. Add tests with `aicube_test_` prefix
5. Submit pull request

### License
This project is proprietary to **AICUBE TECHNOLOGY** and uses proprietary technologies.

---

**Developed by AICUBE TECHNOLOGY**  
Powered by Qube LCM Model, Qube Neural Memory, Qube Agentic Workflows, and Qube Computer Vision