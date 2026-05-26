# ⚡ Quick Start - Testando a API de Métricas

## 🎯 Em 3 Passos

### Passo 1: Iniciar o Backend

```bash
cd Backend
uvicorn main_api:app --reload
```

### Passo 2: Testar a API (em outro terminal)

**Windows (PowerShell):**
```powershell
# Teste rápido
.\test_api.ps1

# Teste + gerar dados de teste
.\test_api.ps1 -Command full
```

**Linux/macOS (Python):**
```bash
# Teste rápido
python test_api.py

# Teste + gerar dados de teste
python test_api.py full
```

### Passo 3: Abrir a Dashboard

```bash
cd dashboard-antifurto
npm run dev
```

---

## ✅ Verificação Rápida

### Com cURL (Linux/macOS/Windows)

```bash
# 1. API está viva?
curl http://127.0.0.1:8000/

# 2. Há métricas?
curl http://127.0.0.1:8000/api/metricas/cluster

# 3. Registar métrica de teste
curl -X POST http://127.0.0.1:8000/api/metricas/registar \
  -H "Content-Type: application/json" \
  -d '{"node_id":"test","fps":30,"frame_count":100,"detection_count":5,"inference_calls":50,"average_inference_ms":10,"success_rate":10,"uptime_seconds":600}'
```

### Com Python

```python
import requests

# Verificar cluster
resp = requests.get("http://127.0.0.1:8000/api/metricas/cluster")
print(resp.json())
```

---

## 🐛 Não Funciona?

| Problema | Solução |
|----------|---------|
| "Não consegue conectar" | Backend não está em execução? `cd Backend && uvicorn main_api:app --reload` |
| "Sem métricas" | Gere dados: `python test_api.py generate` ou `.\test_api.ps1 -Command generate` |
| "Dashboard mostra erro" | Verificar console (F12) → Ver se API responde |
| "Porta já em uso" | `uvicorn main_api:app --reload --port 8001` |

---

## 📊 O que Esperar

✅ **API Funcionando:**
```
GET http://127.0.0.1:8000/ → {"status": "A Corner API está a correr!"}
```

✅ **Métricas Disponíveis:**
```
GET /api/metricas/cluster → {
  "cluster_metrics": {
    "num_nodes": 1,
    "media_fps": 25.5,
    "total_detections": 45,
    ...
  }
}
```

✅ **Dashboard Mostrando:**
- Status "API Conectada" (verde)
- Cards com métricas: FPS, Detecções, etc.
- Lista de nós com detalhes

---

## 🚀 Próximos Passos

1. **Executar o Edge** para gerar métricas reais:
   ```bash
   cd Edge
   python main.py
   ```

2. **Integrar no seu cluster** - O orchestrator já guarda métricas automaticamente

3. **Monitorar em produção** - A dashboard atualiza a cada 5 segundos

---

## 📖 Documentação Completa

Para mais detalhes, ver: [API_TESTING_GUIDE.md](API_TESTING_GUIDE.md)
