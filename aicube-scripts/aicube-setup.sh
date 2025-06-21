#!/bin/bash

# AICUBE Embedding2Embedding API - Setup Script
# Script para configuração inicial do ambiente AICUBE

set -e

echo "=========================================="
echo "AICUBE Embedding2Embedding Setup"
echo "Powered by AICUBE TECHNOLOGY"
echo "=========================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para log
log_info() {
    echo -e "${BLUE}[AICUBE INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[AICUBE SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[AICUBE WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[AICUBE ERROR]${NC} $1"
}

# Verificar se está no diretório correto
if [ ! -f "aicube-requirements.txt" ]; then
    log_error "Execute este script no diretório raiz do projeto AICUBE"
    exit 1
fi

# Verificar Python
log_info "Verificando instalação do Python..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 não encontrado. Instale Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_success "Python $PYTHON_VERSION encontrado"

# Verificar se a versão é adequada
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    log_warning "Recomenda-se Python 3.11+. Versão atual: $PYTHON_VERSION"
fi

# Criar ambiente virtual AICUBE
log_info "Criando ambiente virtual AICUBE..."
if [ -d "aicube-venv" ]; then
    log_warning "Ambiente virtual AICUBE já existe. Removendo..."
    rm -rf aicube-venv
fi

python3 -m venv aicube-venv
log_success "Ambiente virtual AICUBE criado"

# Ativar ambiente virtual
log_info "Ativando ambiente virtual AICUBE..."
source aicube-venv/bin/activate

# Atualizar pip
log_info "Atualizando pip..."
pip install --upgrade pip

# Instalar dependências AICUBE
log_info "Instalando dependências AICUBE..."
pip install -r aicube-requirements.txt
log_success "Dependências AICUBE instaladas"

# Criar diretórios necessários
log_info "Criando estrutura de diretórios AICUBE..."

mkdir -p aicube-logs
mkdir -p aicube-models/aicube-pretrained
mkdir -p aicube-models/aicube-configs
mkdir -p aicube-data/aicube-cache
mkdir -p aicube-data/aicube-metrics

log_success "Diretórios AICUBE criados"

# Configurar arquivo de ambiente
log_info "Configurando arquivo de ambiente AICUBE..."
if [ ! -f "aicube.env" ]; then
    log_warning "Arquivo aicube.env não encontrado"
else
    log_success "Arquivo aicube.env encontrado"
fi

# Criar configuração padrão de modelos se não existir
AICUBE_MODEL_CONFIG="aicube-models/aicube-model-registry.json"
if [ ! -f "$AICUBE_MODEL_CONFIG" ]; then
    log_info "Criando configuração padrão de modelos AICUBE..."
    cat > "$AICUBE_MODEL_CONFIG" << 'EOF'
{
  "aicube_version": "1.0.0-aicube",
  "created_by": "AICUBE TECHNOLOGY",
  "aicube_technologies": [
    "Qube LCM Model",
    "Qube Neural Memory", 
    "Qube Agentic Workflows",
    "Qube Computer Vision"
  ],
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
EOF
    log_success "Configuração de modelos AICUBE criada"
fi

# Verificar CUDA (opcional)
log_info "Verificando suporte CUDA..."
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU detectada"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs -I {} echo "GPU: {}"
    
    # Verificar PyTorch CUDA
    if python3 -c "import torch; print('CUDA disponível:', torch.cuda.is_available())" 2>/dev/null; then
        log_success "PyTorch com suporte CUDA instalado"
    else
        log_warning "PyTorch sem suporte CUDA. Para GPU, reinstale PyTorch com CUDA"
    fi
else
    log_info "GPU NVIDIA não detectada. Usando CPU"
fi

# Executar testes básicos
log_info "Executando testes básicos AICUBE..."
if python3 -c "
import sys
sys.path.append('.')
try:
    from aicube_app.aicube_core.aicube_config import aicube_settings
    from aicube_app.aicube_models.aicube_embedding_translator import AICUBEEmbeddingTranslator
    print('Imports AICUBE OK')
except Exception as e:
    print(f'Erro no import: {e}')
    sys.exit(1)
"; then
    log_success "Testes básicos AICUBE passaram"
else
    log_error "Falha nos testes básicos AICUBE"
    exit 1
fi

# Criar script de execução
log_info "Criando script de execução AICUBE..."
cat > aicube-run.sh << 'EOF'
#!/bin/bash
# AICUBE Embedding2Embedding Run Script

source aicube-venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Iniciando AICUBE Embedding2Embedding API..."
echo "Powered by AICUBE TECHNOLOGY"
echo "Technologies: Qube LCM Model, Qube Neural Memory, Qube Agentic Workflows, Qube Computer Vision"

python3 aicube-app/aicube-main.py
EOF

chmod +x aicube-run.sh
log_success "Script de execução criado: ./aicube-run.sh"

# Criar script de teste
log_info "Criando script de teste AICUBE..."
cat > aicube-test.sh << 'EOF'
#!/bin/bash
# AICUBE Embedding2Embedding Test Script

source aicube-venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Executando testes AICUBE..."
pytest aicube-tests/ -v --tb=short
EOF

chmod +x aicube-test.sh
log_success "Script de teste criado: ./aicube-test.sh"

# Resumo final
echo ""
echo "=========================================="
log_success "Setup AICUBE concluído com sucesso!"
echo "=========================================="
echo ""
echo -e "${BLUE}Para executar a API AICUBE:${NC}"
echo "  ./aicube-run.sh"
echo ""
echo -e "${BLUE}Para executar os testes AICUBE:${NC}"
echo "  ./aicube-test.sh"
echo ""
echo -e "${BLUE}Para usar com Docker:${NC}"
echo "  docker-compose -f aicube-docker-compose.yml up -d"
echo ""
echo -e "${BLUE}Documentação da API:${NC}"
echo "  http://localhost:8000/aicube-docs"
echo ""
echo -e "${BLUE}Health Check:${NC}"
echo "  http://localhost:8000/api/v1/health"
echo ""
echo -e "${GREEN}Powered by AICUBE TECHNOLOGY${NC}"
echo -e "${GREEN}Technologies: Qube LCM Model, Qube Neural Memory, Qube Agentic Workflows, Qube Computer Vision${NC}"