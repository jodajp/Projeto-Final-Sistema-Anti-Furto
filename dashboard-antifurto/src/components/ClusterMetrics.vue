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

    <!-- Máquinas Virtuais (Nós no Docker Swarm) -->
    <div class="infra-section vms-section">
      <div class="section-header">
        <h3>🖥️ Máquinas Virtuais (Swarm Nodes)</h3>
      </div>
      
      <div v-if="infraError" class="error-msg">
        ⚠️ Falha na obtenção das VMs...
      </div>

      <div v-else>
        <div class="services-list">
          <div v-for="node in infraNodes" :key="node.id" class="service-card vms-card">
            <div class="service-header vms-header">
              <div class="service-info">
                <div class="service-name">{{ node.hostname }}</div>
                <span class="service-id">{{ node.id }}</span>
              </div>
              <span class="role-badge" :class="node.role">{{ node.role }}</span>
            </div>
            
            <div class="service-details">
              <div class="detail">
                <span class="detail-label">IP (Rede)</span>
                <span class="detail-value">{{ node.ip || '---' }}</span>
              </div>
              <div class="detail">
                <span class="detail-label">Disponibilidade</span>
                <span class="detail-value">{{ node.availability }}</span>
              </div>
              <div class="detail">
                <span class="detail-label">Estado Node</span>
                <span class="status-badge" :class="{ 'healthy': node.status === 'ready', 'failed': node.status !== 'ready' }">
                  <span class="status-dot" :class="{ 'bg-green': node.status === 'ready', 'bg-red': node.status !== 'ready' }"></span>
                  {{ node.status === 'ready' ? 'Pronto' : 'Falha' }}
                </span>
              </div>
            </div>
          </div>
        </div>
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
              
              <!-- Ações de Escala -->
              <div class="service-actions" v-if="svc.mode === 'replicated'">
                <div class="scale-group" >
                  <button v-if="svc.replicas_target > 1" class="action-btn scale" @click="scaleService(svc.id, Math.max(1, svc.replicas_target - 1))" title="Reduzir Réplica (-)">
                    -
                  </button>
                  <button class="action-btn scale" @click="scaleService(svc.id, svc.replicas_target + 1)" title="Aumentar Réplica (+)">
                    +
                  </button>
                </div>
              </div>

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
import { API_BASE, API_URL } from '../utils/api.js'

const apiConnected = ref(false)
const isLoading = ref(true)
const clusterMetrics = ref({})
const nodes = ref([])
let fetchInterval = null

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
const infraNodes = ref([])
const infraError = ref(null)

const INFRA_API_URL = `${API_BASE}/infra/services`
const NODES_API_URL = `${API_BASE}/infra/nodes`

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

const fetchInfraNodes = async () => {
  try {
    const response = await fetch(NODES_API_URL)
    const data = await response.json()
    
    if (!data.error) {
      infraNodes.value = data
    }
  } catch (error) {
    console.error('Erro nos Nós:', error)
  }
}

const scaleService = async (serviceId, replicas, isStart = false) => {
  const actionText = replicas === 0 ? 'PARAR (reduzir a 0 réplicas)' : (isStart ? 'LIGAR (iniciar 1 réplica)' : `escalar para ${replicas} réplica(s)`)
  if (!confirm(`Tem a certeza que deseja ${actionText} este serviço?`)) return
  
  try {
    const response = await fetch(`${INFRA_API_URL}/${serviceId}/scale`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ replicas })
    })
    
    if (!response.ok) throw new Error(`Erro ao escalar: ${response.status}`)
    
    await fetchInfraStatus()
  } catch (error) {
    console.error('Erro ao escalar serviço:', error)
    infraError.value = 'Falha ao alterar as réplicas do serviço'
  }
}


onMounted(() => {
  if (fetchInterval) clearInterval(fetchInterval)
  fetchClusterMetrics()
  fetchInterval = setInterval(fetchClusterMetrics, 5000) 
  fetchInfraStatus()
  fetchInfraNodes()
  setInterval(() => {
    fetchInfraStatus()
    fetchInfraNodes()
  }, 10000) // Atualizar a cada 10 segundos
})



</script>

