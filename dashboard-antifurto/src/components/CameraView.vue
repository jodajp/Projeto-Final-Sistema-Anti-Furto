<template>
  <div class="camera-container">
    <div class="camera-header">
      <div class="header-left">
        <h2>Monitoramento em Direto</h2>
        <p class="subtitle">Visualização em tempo real das câmaras do Edge</p>
      </div>
      
      <div class="camera-controls">
        <!-- Seletor de Modo de Visualização (se houver mais que 1) -->
        <div v-if="activeStreams.length > 1" class="view-mode-selector">
          <button 
            :class="['btn-toggle', { active: viewMode === 'single' }]" 
            @click="viewMode = 'single'"
            title="Visualização Individual"
          >
            📺 Foco Único
          </button>
          <button 
            :class="['btn-toggle', { active: viewMode === 'grid' }]" 
            @click="viewMode = 'grid'"
            title="Visualização em Grelha (CCTV)"
          >
            🎛️ Grelha CCTV ({{ activeStreams.length }})
          </button>
        </div>

        <button class="btn-refresh" @click="refreshStream" :disabled="activeStreams.length === 0">
          🔄 Atualizar
        </button>

        <span class="stream-status" :class="{ active: activeStreams.length > 0, inactive: activeStreams.length === 0 }">
          <span class="status-dot"></span>
          {{ activeStreams.length > 0 ? `${activeStreams.length} Online` : 'Offline' }}
        </span>
      </div>
    </div>

    <!-- MODO FOCO / CAMARA UNICA -->
    <div v-if="viewMode === 'single' || activeStreams.length <= 1" class="focus-layout">
      <div class="camera-stream-wrapper main-focused">
        <!-- Barra de Seleção de Node se houver múltiplos -->
        <div v-if="activeStreams.length > 1" class="node-selector-bar">
          <label for="node-select">Selecionar Câmara:</label>
          <select id="node-select" v-model="selectedNode" class="node-dropdown">
            <option v-for="node in activeStreams" :key="node" :value="node">
              📹 {{ node }}
            </option>
          </select>
        </div>

        <!-- Stream MJPEG -->
        <div v-if="selectedNode" class="stream-container">
          <div class="stream-badge-overlay">
            <span class="live-tag">LIVE</span>
            <span class="node-tag">{{ selectedNode }}</span>
          </div>
          <img 
            :src="getStreamUrlForNode(selectedNode)" 
            :key="`${selectedNode}-${streamKey}`"
            @error="handleStreamError"
            @load="handleStreamLoad"
            alt="Feed de câmara em direto"
            class="stream-image"
          />
        </div>

        <!-- Estado: Offline -->
        <div v-else class="stream-placeholder offline">
          <div class="placeholder-content">
            <div class="radar-animation">
              <div class="ring"></div>
              <div class="ring"></div>
              <div class="ring"></div>
            </div>
            <p class="title">Nenhuma Câmara Ativa</p>
            <p class="subtitle">Os nós do Edge estão offline. Inicie o orchestrator nos nós locais.</p>
          </div>
        </div>
      </div>

      <!-- Informações da Stream em Foco -->
      <div v-if="selectedNode && activeStreams.length > 0" class="stream-info">
        <div class="info-item">
          <span class="label">Nó de Câmara:</span>
          <span class="value font-semibold text-primary">{{ selectedNode }}</span>
        </div>
        <div class="info-item">
          <span class="label">Resolução:</span>
          <span class="value">{{ streamResolution }}</span>
        </div>
        <div class="info-item">
          <span class="label">Status do Stream:</span>
          <span class="value status-badge" :class="streamStatus.toLowerCase().replace(' ', '-')">{{ streamStatus }}</span>
        </div>
        <div class="info-item">
          <span class="label">URL de Origem:</span>
          <span class="value mono text-xs">{{ getStreamUrlForNode(selectedNode) }}</span>
        </div>
      </div>
    </div>

    <!-- MODO GRELHA CCTV / MULTIPLOS NODES -->
    <div v-else class="grid-layout">
      <div class="cctv-grid">
        <div v-for="node in activeStreams" :key="node" class="cctv-cell">
          <div class="cell-header">
            <span class="cell-node-id">📹 {{ node }}</span>
            <span class="cell-live-dot"></span>
          </div>
          <div class="cell-stream-container">
            <img 
              :src="getStreamUrlForNode(node)" 
              :key="`${node}-${streamKey}`"
              alt="Feed de câmara em direto"
              class="grid-stream-image"
              @error="(e) => handleGridError(node, e)"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { API_BASE_URL, API_BASE } from '../utils/api.js'

const activeStreams = ref([])
const selectedNode = ref('')
const viewMode = ref('single') // 'single' ou 'grid'
const streamKey = ref(0)
const streamStatus = ref('Desconectado')
const streamResolution = ref('1280x720 (Auto-Scale)')

let checkInterval = null
const autoLayoutInitialized = ref(false)

const getStreamUrlForNode = (nodeId) => {
  return `${API_BASE_URL}/api/video/stream?node_id=${nodeId}&t=${streamKey.value}`
}

const handleStreamLoad = () => {
  streamStatus.value = 'Ativo'
}

const handleStreamError = () => {
  streamStatus.value = 'Erro de Ligação'
}

const handleGridError = (node, e) => {
  console.warn(`Erro no stream do nó ${node}`);
}

const refreshStream = () => {
  streamKey.value++
  streamStatus.value = 'A Recarregar...'
}

const checkActiveStreams = async () => {
  try {
    const response = await fetch(`${API_BASE}/video/active_streams`, { 
      signal: AbortSignal.timeout(3000) 
    })
    if (response.ok) {
      const data = await response.json()
      const streams = data.active_streams || []
      activeStreams.value = streams
      
      // Lógica de Layout Automático:
      // Se há mais do que 1 stream ativo e ainda não definimos o layout, muda para grelha
      if (!autoLayoutInitialized.value && streams.length > 0) {
        if (streams.length > 1) {
          viewMode.value = 'grid'
        } else {
          viewMode.value = 'single'
        }
        autoLayoutInitialized.value = true
      }

      // Garante que temos um nó selecionado para o modo Foco Único
      if (streams.length > 0) {
        if (!selectedNode.value || !streams.includes(selectedNode.value)) {
          selectedNode.value = streams[0]
        }
      } else {
        selectedNode.value = ''
      }
    }
  } catch (error) {
    console.error("Erro a obter streams ativos:", error)
  }
}

onMounted(() => {
  // Inicializa a verificação
  checkActiveStreams()
  
  // Polling de streams ativos a cada 4 segundos
  checkInterval = setInterval(checkActiveStreams, 4000)
})

onUnmounted(() => {
  if (checkInterval) clearInterval(checkInterval)
})
</script>

<style scoped>
.camera-container {
  background: #ffffff;
  border-radius: 16px;
  padding: 1.75rem;
  box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.08);
  border: 1px solid #f1f5f9;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.camera-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  border-bottom: 1px solid #f1f5f9;
  padding-bottom: 1.25rem;
}

.camera-header h2 {
  margin: 0;
  color: #0f172a;
  font-size: 1.5rem;
  font-weight: 600;
}

.camera-header .subtitle {
  margin: 0.25rem 0 0 0;
  color: #64748b;
  font-size: 0.875rem;
}

.camera-controls {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.view-mode-selector {
  display: flex;
  background: #f1f5f9;
  padding: 0.25rem;
  border-radius: 8px;
}

.btn-toggle {
  background: transparent;
  border: none;
  padding: 0.5rem 0.875rem;
  font-size: 0.85rem;
  font-weight: 500;
  border-radius: 6px;
  color: #475569;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.btn-toggle.active {
  background: #ffffff;
  color: #0f172a;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.btn-refresh {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  padding: 0.55rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-refresh:hover:not(:disabled) {
  background: #f8fafc;
  border-color: #94a3b8;
  color: #0f172a;
}

.btn-refresh:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.stream-status {
  padding: 0.4rem 0.8rem;
  font-size: 0.8rem;
  font-weight: 600;
  border-radius: 20px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.stream-status.active {
  background: #ecfdf5;
  color: #059669;
}

.stream-status.inactive {
  background: #fef2f2;
  color: #dc2626;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: currentColor;
}

.stream-status.active .status-dot {
  animation: pulse 1.8s infinite;
}

@keyframes pulse {
  0% {
    transform: scale(0.9);
    box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.7);
  }
  70% {
    transform: scale(1);
    box-shadow: 0 0 0 6px rgba(5, 150, 105, 0);
  }
  100% {
    transform: scale(0.9);
    box-shadow: 0 0 0 0 rgba(5, 150, 105, 0);
  }
}

/* Foco Layout (Câmara Única) */
.focus-layout {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.camera-stream-wrapper {
  background: #0f172a;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
  aspect-ratio: 16/9;
  display: flex;
  flex-direction: column;
  border: 1px solid #1e293b;
}

.node-selector-bar {
  background: rgba(15, 23, 42, 0.75);
  backdrop-filter: blur(8px);
  padding: 0.75rem 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  align-items: center;
  gap: 0.75rem;
  z-index: 10;
}

.node-selector-bar label {
  color: #94a3b8;
  font-size: 0.85rem;
  font-weight: 500;
}

.node-dropdown {
  background: #1e293b;
  border: 1px solid #334155;
  color: #f8fafc;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
  outline: none;
  cursor: pointer;
}

.node-dropdown:focus {
  border-color: #3b82f6;
}

.stream-container {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #020617;
  overflow: hidden;
}

.stream-badge-overlay {
  position: absolute;
  top: 1rem;
  left: 1rem;
  display: flex;
  gap: 0.5rem;
  z-index: 10;
}

.live-tag {
  background: #dc2626;
  color: #ffffff;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  letter-spacing: 0.05em;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.node-tag {
  background: rgba(15, 23, 42, 0.75);
  color: #f8fafc;
  font-size: 0.7rem;
  font-weight: 500;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,0.1);
}

.stream-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  width: 100%;
  height: 100%;
}

/* Radar Offline Placeholder */
.stream-placeholder.offline {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
  padding: 3rem;
  color: #94a3b8;
}

.placeholder-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  max-width: 320px;
}

.radar-animation {
  position: relative;
  width: 80px;
  height: 80px;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.radar-animation::before {
  content: "📡";
  font-size: 2.2rem;
  z-index: 5;
}

.ring {
  position: absolute;
  border: 2px solid rgba(148, 163, 184, 0.2);
  border-radius: 50%;
  animation: radar-pulse 3s infinite linear;
  opacity: 0;
}

.ring:nth-child(1) { animation-delay: 0s; }
.ring:nth-child(2) { animation-delay: 1s; }
.ring:nth-child(3) { animation-delay: 2s; }

@keyframes radar-pulse {
  0% {
    width: 20px;
    height: 20px;
    opacity: 0.8;
  }
  100% {
    width: 100px;
    height: 100px;
    opacity: 0;
  }
}

.stream-placeholder .title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #f8fafc;
  margin: 0 0 0.5rem 0;
}

.stream-placeholder .subtitle {
  font-size: 0.875rem;
  color: #64748b;
  margin: 0;
  line-height: 1.4;
}

/* Info do Stream */
.stream-info {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  background: #f8fafc;
  border-radius: 12px;
  padding: 1.25rem;
  border: 1px solid #e2e8f0;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.info-item .label {
  color: #64748b;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.info-item .value {
  color: #1e293b;
  font-size: 0.9rem;
  font-weight: 500;
}

.info-item .mono {
  font-family: monospace;
}

.status-badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  width: fit-content;
}

.status-badge.ativo {
  background: #d1fae5;
  color: #065f46;
}

.status-badge.erro-de-ligacao {
  background: #fee2e2;
  color: #991b1b;
}

.status-badge.desconectado {
  background: #f1f5f9;
  color: #475569;
}

/* CCTV Grid Layout */
.grid-layout {
  display: flex;
  flex-direction: column;
}

.cctv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 1.25rem;
}

@media (max-width: 900px) {
  .cctv-grid {
    grid-template-columns: 1fr;
  }
}

.cctv-cell {
  background: #0f172a;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #1e293b;
  display: flex;
  flex-direction: column;
  aspect-ratio: 16/9;
  position: relative;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: transform 0.2s ease, border-color 0.2s ease;
}

.cctv-cell:hover {
  transform: translateY(-2px);
  border-color: #3b82f6;
}

.cell-header {
  background: rgba(15, 23, 42, 0.85);
  padding: 0.6rem 0.85rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  z-index: 5;
}

.cell-node-id {
  color: #f8fafc;
  font-size: 0.8rem;
  font-weight: 600;
}

.cell-live-dot {
  width: 7px;
  height: 7px;
  background-color: #10b981;
  border-radius: 50%;
  animation: pulse 1.8s infinite;
  box-shadow: 0 0 8px #10b981;
}

.cell-stream-container {
  flex: 1;
  background: #020617;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.grid-stream-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
</style>
