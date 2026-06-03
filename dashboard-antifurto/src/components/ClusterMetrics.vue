<template>
  <div class="metrics-container">
    <!-- Status Conexão API -->
    <div class="connection-status" :class="{ connected: apiConnected, disconnected: !apiConnected }">
      <span class="status-indicator"></span>
      <span class="status-text">{{ apiConnected ? 'API Conectada' : 'API Desconectada' }}</span>
    </div>

    <!-- Métricas Agregadas do Cluster -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Nós Ativos</div>
        <div class="metric-value">{{ clusterMetrics.num_nodes || 0 }}</div>
        <div class="metric-subtitle">no cluster</div>
      </div>

      <div class="metric-card">
        <div class="metric-label">FPS Médio</div>
        <div class="metric-value">{{ clusterMetrics.media_fps || 0 }}</div>
        <div class="metric-subtitle">frames/segundo</div>
      </div>

      <div class="metric-card">
        <div class="metric-label">Total Detecções</div>
        <div class="metric-value">{{ clusterMetrics.total_detections || 0 }}</div>
        <div class="metric-subtitle">eventos detectados</div>
      </div>

      <div class="metric-card">
        <div class="metric-label">Taxa Sucesso</div>
        <div class="metric-value">{{ clusterMetrics.taxa_sucesso_media_pct || 0 }}%</div>
        <div class="metric-subtitle">média</div>
      </div>

      <div class="metric-card">
        <div class="metric-label">Tempo Inferência</div>
        <div class="metric-value">{{ clusterMetrics.tempo_medio_inferencia_ms || 0 }}</div>
        <div class="metric-subtitle">ms</div>
      </div>

      <div class="metric-card">
        <div class="metric-label">Uptime Máximo</div>
        <div class="metric-value">{{ formatUptime(clusterMetrics.uptime_maximo_segundos || 0) }}</div>
        <div class="metric-subtitle">tempo funcionamento</div>
      </div>
    </div>

  <div class="infra-section">
      <div class="section-header">
        <h3>☁️ Estado da Infraestrutura (Docker Swarm)</h3>
      </div>
      
      <div v-if="infraError" class="error-msg">
        ⚠️ {{ infraError }}
      </div>

      <div v-else>
        <div class="services-list">
          <div v-for="svc in infraServices" :key="svc.id" class="service-card">
            <div class="service-header">
              <div class="service-info">
                <div class="service-name">{{ svc.name }}</div>
                <span class="service-id">{{ svc.id }}</span>
              </div>
              <button class="delete-btn" @click="deleteService(svc.id)" :title="`Eliminar ${svc.name}`">
                🗑️ Apagar
              </button>
            </div>
            
            <div class="service-details">
              <div class="detail">
                <span class="detail-label">Modo</span>
                <span class="detail-value">{{ svc.mode }}</span>
              </div>
              <div class="detail">
                <span class="detail-label">Réplicas</span>
                <span class="replica-badge" :class="{ 'healthy': svc.status === 'healthy' }">
                  {{ svc.replicas_running }}/{{ svc.replicas_target }}
                </span>
              </div>
              <div class="detail">
                <span class="detail-label">Estado</span>
                <span class="status-badge" :class="{ 'healthy': svc.status === 'healthy', 'failed': svc.status !== 'healthy' }">
                  <span class="status-dot" :class="{ 'bg-green': svc.status === 'healthy', 'bg-red': svc.status !== 'healthy' }"></span>
                  {{ svc.status === 'healthy' ? 'A Correr' : 'Falha' }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Detalhes por Nó -->
    <div v-if="nodes.length > 0" class="nodes-section">
      <div class="section-header">
        <h3>Detalhes dos Nós</h3>
        <span class="refresh-btn" @click="fetchClusterMetrics" :class="{ loading: isLoading }">
          ⟳ Atualizar
        </span>
      </div>

      <div class="nodes-list">
        <div v-for="node in nodes" :key="node.node_id" class="node-card">
          <div class="node-header">
            <span class="node-id">{{ node.node_id }}</span>
            <span class="node-status" :class="getNodeStatus(node)">
              {{ getNodeStatusText(node) }}
            </span>
          </div>

          <div class="node-details-grid">
            <div class="detail-item">
              <span class="detail-label">FPS</span>
              <span class="detail-value">{{ node.fps?.toFixed(1) || 0 }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Frames</span>
              <span class="detail-value">{{ node.frame_count || 0 }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Detecções</span>
              <span class="detail-value">{{ node.detection_count || 0 }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Inferências</span>
              <span class="detail-value">{{ node.inference_calls || 0 }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Tempo Médio (ms)</span>
              <span class="detail-value">{{ node.average_inference_ms?.toFixed(2) || 0 }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Taxa Sucesso</span>
              <span class="detail-value">{{ node.success_rate?.toFixed(1) || 0 }}%</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Uptime</span>
              <span class="detail-value">{{ formatUptime(node.uptime_seconds || 0) }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Última Atualização</span>
              <span class="detail-value">{{ formatTimestamp(node.timestamp) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Mensagem quando não há dados -->
    <div v-if="!apiConnected && !isLoading" class="empty-state">
      <p>⚠️ Não conseguiu conectar à API</p>
      <p class="subtitle">Certifique-se que o backend está em execução em http://127.0.0.1:8000</p>
    </div>

    <!-- Indicador de Carregamento -->
    <div v-if="isLoading" class="loading-state">
      <p>Carregando métricas...</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const apiConnected = ref(false)
const isLoading = ref(true)
const clusterMetrics = ref({})
const nodes = ref([])
let fetchInterval = null

const API_URL = 'http://projeto-antifurto-vm1.norwayeast.cloudapp.azure.com:8000/api/metricas/cluster'

const formatUptime = (seconds) => {
  if (!seconds) return '0s'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  
  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m ${secs}s`
  return `${secs}s`
}

const formatTimestamp = (timestamp) => {
  if (!timestamp) return '-'
  const date = new Date(timestamp * 1000)
  return date.toLocaleTimeString('pt-PT')
}

const getNodeStatus = (node) => {
  if (node.fps > 20) return 'healthy'
  if (node.fps > 10) return 'warning'
  return 'unhealthy'
}

const getNodeStatusText = (node) => {
  const status = getNodeStatus(node)
  return status === 'healthy' ? '✓ Ativo' : status === 'warning' ? '⚠ Lento' : '✗ Crítico'
}

const fetchClusterMetrics = async () => {
  isLoading.value = true
  try {
    const response = await fetch(API_URL, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            // O CORS no backend já aceita tudo, mas é bom garantir
        }
    })
    
    if (!response.ok) throw new Error(`Erro na API: ${response.status}`)
    
    const data = await response.json()
    
    clusterMetrics.value = data.cluster_metrics || {}
    nodes.value = data.nodes || []
    
    apiConnected.value = true
  } catch (error) {
    console.error('Erro de conexão:', error)
    apiConnected.value = false
    clusterMetrics.value = {}
    nodes.value = []
  } finally {
    isLoading.value = false
  }
}

const infraServices = ref([])
const infraError = ref(null)

const INFRA_API_URL = 'http://20.251.152.37:8000/api/infra/services' 

const fetchInfraStatus = async () => {
  try {
    const response = await fetch(INFRA_API_URL)
    const data = await response.json()
    
    if (data.error) {
      infraError.value = data.error
      infraServices.value = []
    } else {
      infraServices.value = data
      infraError.value = null
    }
  } catch (error) {
    console.error('Erro na infraestrutura:', error)
    infraError.value = 'Falha na comunicação com a API'
  }
}

const deleteService = async (serviceId) => {
  if (!confirm(`Tem a certeza que deseja eliminar este serviço?`)) return
  
  try {
    const response = await fetch(`${INFRA_API_URL.replace('/api/infra/services', '')}/api/infra/services/${serviceId}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json'
      }
    })
    
    if (!response.ok) throw new Error(`Erro ao eliminar: ${response.status}`)
    
    await fetchInfraStatus()
  } catch (error) {
    console.error('Erro ao eliminar serviço:', error)
    infraError.value = 'Falha ao eliminar o serviço'
  }
}


onMounted(() => {
  if (fetchInterval) clearInterval(fetchInterval)
  fetchClusterMetrics()
  fetchInterval = setInterval(fetchClusterMetrics, 5000) 
  fetchInfraStatus()
  setInterval(fetchInfraStatus, 10000) // Atualizar a cada 10 segundos
})



</script>

<style scoped>
.metrics-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

/* Status Conexão */
.connection-status {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  border-radius: 6px;
  font-size: 0.95rem;
  font-weight: 600;
}

.connection-status.connected {
  background-color: #ecfdf5;
  color: #065f46;
  border: 1px solid #d1fae5;
}

.connection-status.disconnected {
  background-color: #fef2f2;
  color: #991b1b;
  border: 1px solid #fee2e2;
}

.status-indicator {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.connection-status.connected .status-indicator {
  background-color: #10b981;
}

.connection-status.disconnected .status-indicator {
  background-color: #ef4444;
}

/* Grid de Métricas */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
}

.metric-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
  transition: all 0.3s ease;
}

.metric-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
}

.metric-label {
  font-size: 0.85rem;
  color: #64748b;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 0.75rem;
}

.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.5rem;
}

.metric-subtitle {
  font-size: 0.8rem;
  color: #94a3b8;
}

/* Seção de Infraestrutura */
.infra-section {
  background: white;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  padding: 2rem;
}

.services-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.5rem;
}

.service-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem;
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.service-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.service-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #e2e8f0;
}

.service-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.service-name {
  font-size: 1.1rem;
  font-weight: 700;
  color: #0f172a;
}

.service-id {
  font-size: 0.75rem;
  color: #94a3b8;
  font-family: monospace;
  letter-spacing: 0.5px;
}

.service-details {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.detail {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.detail-label {
  font-size: 0.7rem;
  color: #64748b;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.detail-value {
  font-size: 0.95rem;
  color: #0f172a;
  font-weight: 600;
}

.replica-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.4rem 0.8rem;
  background: #fef3c7;
  color: #92400e;
  border-radius: 4px;
  font-size: 0.85rem;
  font-weight: 600;
}

.replica-badge.healthy {
  background: #d1fae5;
  color: #065f46;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.8rem;
  border-radius: 4px;
  font-size: 0.85rem;
  font-weight: 600;
}

.status-badge.healthy {
  background: #d1fae5;
  color: #065f46;
}

.status-badge.failed {
  background: #fee2e2;
  color: #991b1b;
}

.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-dot.bg-green {
  background-color: #10b981;
}

.status-dot.bg-red {
  background-color: #ef4444;
}

/* Botão Apagar */
.delete-btn {
  background-color: #ef4444;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
  flex-shrink: 0;
}

.delete-btn:hover {
  background-color: #dc2626;
  box-shadow: 0 4px 8px rgba(220, 38, 38, 0.25);
  transform: translateY(-1px);
}

.delete-btn:active {
  background-color: #b91c1c;
  transform: translateY(0);
}

/* Seção de Nós */
.nodes-section {
  background: white;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  padding: 2rem;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.section-header h3 {
  margin: 0;
  color: #0f172a;
  font-size: 1.2rem;
}

.refresh-btn {
  padding: 0.5rem 1rem;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  color: #475569;
  transition: all 0.2s ease;
}

.refresh-btn:hover {
  background: #e2e8f0;
  border-color: #94a3b8;
}

.refresh-btn.loading {
  opacity: 0.6;
}

/* Seção de Infraestrutura */
.nodes-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.node-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 1.5rem;
  overflow: hidden;
}

.node-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #e2e8f0;
}

.node-id {
  font-weight: 700;
  color: #0f172a;
  font-size: 1rem;
}

.node-status {
  font-size: 0.8rem;
  font-weight: 600;
  padding: 0.4rem 0.8rem;
  border-radius: 4px;
  white-space: nowrap;
}

.node-status.healthy {
  background: #d1fae5;
  color: #065f46;
}

.node-status.warning {
  background: #fef3c7;
  color: #92400e;
}

.node-status.unhealthy {
  background: #fee2e2;
  color: #991b1b;
}

/* Grid de Detalhes */
.node-details-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.detail-label {
  font-size: 0.75rem;
  color: #64748b;
  font-weight: 600;
  text-transform: uppercase;
}

.detail-value {
  font-size: 0.95rem;
  color: #0f172a;
  font-weight: 600;
}

/* Responsivo para tablets */
@media (max-width: 1024px) {
  .metrics-container {
    gap: 1.5rem;
  }
  .card {
    padding: 1.5rem;
  }
  .node-card {
    padding: 1rem;
  }
  .node-details-grid {
    gap: 0.75rem;
  }
  .section-header h3 {
    font-size: 1.1rem;
  }
}

/* Responsivo para mobile */
@media (max-width: 768px) {
  .metrics-container {
    gap: 1rem;
  }
  .card {
    padding: 1rem;
    border-radius: 6px;
  }
  .section-header {
    flex-direction: column;
    gap: 0.75rem;
  }
  .section-header h3 {
    font-size: 1rem;
    margin: 0;
  }
  .refresh-btn {
    width: 100%;
  }
  .nodes-grid {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  .node-card {
    padding: 0.75rem;
  }
  .node-header {
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .node-id {
    font-size: 0.95rem;
  }
  .node-status {
    font-size: 0.7rem;
    padding: 0.3rem 0.6rem;
  }
  .node-details-grid {
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  .detail-label {
    font-size: 0.7rem;
  }
  .detail-value {
    font-size: 0.85rem;
  }
  .summary-grid {
    grid-template-columns: 1fr 1fr;
  }
}

/* Mobile pequeno */
@media (max-width: 480px) {
  .metrics-container {
    gap: 0.75rem;
  }
  .card {
    padding: 0.75rem;
  }
  .card-title {
    font-size: 1rem;
  }
  .summary-grid {
    grid-template-columns: 1fr;
  }
  .summary-item-value {
    font-size: 1.3rem;
  }
  .summary-item-label {
    font-size: 0.75rem;
  }
  .nodes-grid {
    gap: 0.75rem;
  }
  .node-id {
    font-size: 0.85rem;
  }
}

/* Estados Vazios e Carregamento */
.empty-state,
.loading-state {
  text-align: center;
  padding: 3rem 2rem;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  color: #64748b;
}

.empty-state p,
.loading-state p {
  margin: 0.5rem 0;
}

.empty-state .subtitle {
  font-size: 0.9rem;
  color: #94a3b8;
}

.loading-state {
  font-weight: 600;
}
</style>