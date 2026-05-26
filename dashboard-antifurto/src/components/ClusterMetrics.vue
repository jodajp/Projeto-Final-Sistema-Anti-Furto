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

const API_URL = 'http://20.251.152.37:8000/api/metricas/cluster'

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
    const response = await fetch(API_URL)
    if (!response.ok) throw new Error('Erro na resposta da API')
    const data = await response.json()
    
    clusterMetrics.value = data.cluster_metrics || {}
    nodes.value = data.nodes || []
    apiConnected.value = true
  } catch (error) {
    console.error('Erro ao buscar métricas:', error)
    apiConnected.value = false
    clusterMetrics.value = {}
    nodes.value = []
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  fetchClusterMetrics()
  fetchInterval = setInterval(fetchClusterMetrics, 5000) // Atualizar a cada 5 segundos
})

onUnmounted(() => {
  if (fetchInterval) clearInterval(fetchInterval)
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
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Lista de Nós */
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