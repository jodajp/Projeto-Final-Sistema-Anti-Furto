# 📋 Manual de Teste - API de Métricas do Sistema Anti-Furto

## 📌 Visão Geral

Este manual descreve como testar os novos endpoints de métricas da API do sistema anti-furto.

A API agora oferece acesso a:
- ✅ Métricas agregadas do cluster em tempo real
- ✅ Métricas individuais de cada nó
- ✅ Histórico de métricas
- ✅ Registro de novas métricas via HTTP

---

## 🚀 Começar Rápido

### 1️⃣ Iniciar o Backend (API)

```bash
cd Backend
uvicorn main_api:app --reload
```

**Esperado:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 2️⃣ Executar o Script de Teste (Terminal Separado)

```bash
# Gerar dados de teste + executar todos os testes
python test_api.py full

# OU apenas executar testes
python test_api.py
```

### 3️⃣ Abrir o Dashboard (Terminal Separado)

```bash
cd dashboard-antifurto
npm run dev
```

O dashboard agora mostrará:
- Status de conexão à API
- Métricas agregadas do cluster
- Detalhes individuais de cada nó

---

## 📊 Endpoints Disponíveis

### 1. GET `/api/metricas/atuais`
**Obtém as métricas mais recentes de todos os nós**

```bash
curl http://127.0.0.1:8000/api/metricas/atuais
```

**Resposta esperada:**
```json
{
  "metricas": [
    {
      "node_id": "node1",
      "timestamp": 1234567890.123,
      "fps": 25.5,
      "frame_count": 1500,
      "detection_count": 45,
      "inference_calls": 750,
      "average_inference_ms": 15.3,
      "success_rate": 6.0,
      "uptime_seconds": 3600
    }
  ],
  "total_nodes": 1,
  "timestamp": "2023-05-15T14:30:22.123456"
}
```

---

### 2. GET `/api/metricas/cluster`
**Calcula e retorna métricas agregadas do cluster inteiro**

```bash
curl http://127.0.0.1:8000/api/metricas/cluster
```

**Resposta esperada:**
```json
{
  "cluster_metrics": {
    "num_nodes": 3,
    "media_fps": 24.5,
    "total_frames": 45000,
    "total_detections": 1350,
    "total_inference_calls": 22500,
    "tempo_medio_inferencia_ms": 15.2,
    "taxa_sucesso_media_pct": 6.0,
    "uptime_maximo_segundos": 10800
  },
  "nodes": [...],
  "timestamp": "2023-05-15T14:30:22.123456"
}
```

---

### 3. GET `/api/metricas/node/{node_id}`
**Obtém métricas de um nó específico**

```bash
curl http://127.0.0.1:8000/api/metricas/node/node1
```

**Resposta esperada:**
```json
{
  "node_id": "node1",
  "metricas": {
    "node_id": "node1",
    "timestamp": 1234567890.123,
    "fps": 25.5,
    "frame_count": 1500,
    ...
  },
  "timestamp": "2023-05-15T14:30:22.123456"
}
```

---

### 4. GET `/api/metricas/historico?node_id=node1&limite=50`
**Obtém histórico de métricas**

```bash
# Histórico geral (últimos 10 registos)
curl http://127.0.0.1:8000/api/metricas/historico

# Histórico de um nó específico (últimos 20 registos)
curl "http://127.0.0.1:8000/api/metricas/historico?node_id=node1&limite=20"
```

**Parâmetros:**
- `node_id` (opcional): Filtrar por nó específico
- `limite` (opcional, máximo 100): Quantos registos retornar (padrão: 50)

---

### 5. POST `/api/metricas/registar`
**Registar novas métricas (para o Edge enviar dados)**

```bash
curl -X POST http://127.0.0.1:8000/api/metricas/registar \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "node1",
    "fps": 25.5,
    "frame_count": 1500,
    "detection_count": 45,
    "inference_calls": 750,
    "average_inference_ms": 15.3,
    "success_rate": 6.0,
    "uptime_seconds": 3600
  }'
```

**Resposta esperada:**
```json
{
  "status": "sucesso",
  "mensagem": "Métricas registadas para node1",
  "ficheiro": "metricas_node1_20230515_143022.json"
}
```

---

### 6. GET `/api/alertas/recentes`
**Obtém os alertas recentes (endpoint anterior)**

```bash
curl http://127.0.0.1:8000/api/alertas/recentes
```

---

## 🧪 Script de Teste Automático

### Uso Básico

```bash
# Executar todos os testes
python test_api.py

# Gerar dados de teste (sem executar testes)
python test_api.py generate

# Gerar dados de teste + executar testes
python test_api.py full
```

### O que o Script Testa

1. ✅ **Endpoint Raiz** - Verifica se API está respondendo
2. ✅ **Alertas Recentes** - Testa endpoint de alertas
3. ✅ **Métricas Atuais** - Obtém métricas de todos os nós
4. ✅ **Métricas do Cluster** - Calcula agregações
5. ✅ **Métricas de Nó** - Obtém dados de nó específico
6. ✅ **Histórico** - Recupera registos históricos
7. ✅ **Registar Métricas** - Envia novos dados via POST

---

## 📁 Estrutura de Ficheiros de Métricas

As métricas são guardadas em `{RAIZ_PROJETO}/Metricas/`:

```
Metricas/
├── metricas_node1_20230515_143022.json
├── metricas_node1_20230515_143327.json
├── metricas_node2_20230515_143022.json
├── metricas_node3_20230515_143022.json
└── ...
```

**Formato do ficheiro:**
```json
{
  "node_id": "node1",
  "timestamp": 1234567890.123,
  "fps": 25.5,
  "frame_count": 1500,
  "detection_count": 45,
  "inference_calls": 750,
  "average_inference_ms": 15.3,
  "success_rate": 6.0,
  "uptime_seconds": 3600
}
```

---

## 🔧 Teste Manual com cURL

### Teste Rápido

```bash
# 1. Verificar se API está viva
curl http://127.0.0.1:8000/

# 2. Verificar métricas do cluster
curl http://127.0.0.1:8000/api/metricas/cluster

# 3. Registar métrica de teste
curl -X POST http://127.0.0.1:8000/api/metricas/registar \
  -H "Content-Type: application/json" \
  -d '{"node_id":"test","fps":30,"frame_count":100,"detection_count":5,"inference_calls":50,"average_inference_ms":10,"success_rate":10,"uptime_seconds":600}'
```

### Teste com Postman

1. Abrir Postman
2. Importar collection:

```json
{
  "info": {
    "name": "Anti-Furto API Tests",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Get Root",
      "request": { "method": "GET", "url": "{{base_url}}/" }
    },
    {
      "name": "Get Cluster Metrics",
      "request": { "method": "GET", "url": "{{base_url}}/api/metricas/cluster" }
    },
    {
      "name": "Get Current Metrics",
      "request": { "method": "GET", "url": "{{base_url}}/api/metricas/atuais" }
    },
    {
      "name": "Register Metrics",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/api/metricas/registar",
        "body": {
          "mode": "raw",
          "raw": "{\"node_id\":\"node1\",\"fps\":25,\"frame_count\":1000,\"detection_count\":30,\"inference_calls\":500,\"average_inference_ms\":15,\"success_rate\":6,\"uptime_seconds\":3600}"
        }
      }
    }
  ]
}
```

3. Configurar variável `{{base_url}}` = `http://127.0.0.1:8000`

---

## 🐛 Troubleshooting

### ❌ "Não conseguiu conectar à API"

**Solução:**
```bash
# Verificar se o backend está em execução
cd Backend
uvicorn main_api:app --reload
```

### ❌ "Nenhuma métrica disponível"

**Solução:**
```bash
# Gerar dados de teste
python test_api.py generate

# OU executar o Edge
cd Edge
python main.py
```

### ❌ "Dashboard mostra 'API Desconectada'"

**Checklist:**
1. ✅ Backend está em execução?
2. ✅ URL correta em `ClusterMetrics.vue`?
3. ✅ CORS está habilitado na API?
4. ✅ Não há erro no console do browser?

**Debug:**
```javascript
// Abrir console do browser (F12) e testar:
fetch('http://127.0.0.1:8000/api/metricas/cluster')
  .then(r => r.json())
  .then(data => console.log(data))
  .catch(err => console.error('Erro:', err))
```

---

## 📈 Fluxo de Dados em Produção

```
┌─────────────────────┐
│  Edge/orchestrator  │
│  (corre a cada N)   │
│  frames, guarda     │
│  JSON em Metricas/  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Backend API        │
│  - Lê JSON          │
│  - Agrega dados     │
│  - Expõe endpoints  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Dashboard Vue      │
│  - Fetch a cada 5s  │
│  - Mostra em tempo  │
│  - real os dados    │
└─────────────────────┘
```

---

## ✅ Checklist de Verificação

- [ ] Backend a correr em `http://127.0.0.1:8000`
- [ ] Pasta `Metricas/` criada com ficheiros JSON
- [ ] Teste manual com `curl` a funcionar
- [ ] Script `test_api.py` passa todos os testes
- [ ] Dashboard mostra "API Conectada"
- [ ] Métricas atualizadas em tempo real (a cada 5s)
- [ ] Detalhes dos nós aparecem corretamente
- [ ] Histórico mostra registos anteriores

---

## 📞 Suporte

Se encontrar problemas:

1. **Verificar logs da API:**
   ```bash
   # Terminal onde corre o uvicorn
   # Procurar por erros ou warnings
   ```

2. **Verificar logs do browser:**
   ```
   F12 → Console → Ver erros de rede
   ```

3. **Verificar ficheiros de métricas:**
   ```bash
   ls -la Metricas/
   cat Metricas/metricas_node1_*.json | python -m json.tool
   ```

---

**Última atualização:** 2026-05-26
**Versão da API:** 1.0
