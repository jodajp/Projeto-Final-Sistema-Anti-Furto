<template>
  <div class="camera-container">
    <div class="camera-header">
      <h2>Feed de Câmara em Direto</h2>
      <div class="camera-controls">
        <button class="btn-refresh" @click="refreshStream" :disabled="!streamActive">
          Atualizar
        </button>
        <span class="stream-status" :class="{ active: streamActive, inactive: !streamActive }">
          {{ streamActive ? 'Em Direto' : 'Offline' }}
        </span>
      </div>
    </div>

    <div class="camera-stream-wrapper">
      <!-- Stream MJPEG -->
      <div v-if="streamActive" class="stream-container">
        <img 
          :src="streamUrl" 
          :key="streamKey"
          @error="handleStreamError"
          @load="handleStreamLoad"
          alt="Feed de câmara em direto"
          class="stream-image"
        />
      </div>

      <!-- Estado: Offline -->
      <div v-else class="stream-placeholder offline">
        <div class="placeholder-content">
          <p class="title">Câmara Offline</p>
          <p class="subtitle">O Edge não está em execução</p>
          <button class="btn-try" @click="startServices" :disabled="isStartingServices">
            {{ isStartingServices ? 'A iniciar...' : 'Tentar Conectar' }}
          </button>
        </div>
      </div>

      <!-- Modal de Progresso -->
      <div v-if="showProgressModal" class="modal-overlay">
        <div class="modal-content progress-modal" @click.stop>
          <div class="modal-header">
            <h3>Iniciando Serviços</h3>
          </div>
          <div class="modal-body progress-body">
            <div class="progress-item" :class="{ completed: apiStarted, error: apiError, active: isStartingServices }">
              <div class="progress-step">
                <span v-if="apiStarted" class="status-icon success">✓</span>
                <span v-else-if="apiError" class="status-icon error">✗</span>
                <span v-else class="status-icon loading">⏳</span>
              </div>
              <div class="progress-text">
                <p class="status">A tentar ligar a API Backend...</p>
                <p v-if="apiError" class="error-msg">{{ apiError }}</p>
              </div>
            </div>

            <div class="progress-item" :class="{ completed: edgeStarted, error: edgeError, active: isStartingServices && apiStarted }">
              <div class="progress-step">
                <span v-if="edgeStarted" class="status-icon success">✓</span>
                <span v-else-if="edgeError" class="status-icon error">✗</span>
                <span v-else class="status-icon loading">⏳</span>
              </div>
              <div class="progress-text">
                <p class="status">A iniciar Edge Service...</p>
                <p v-if="edgeError" class="error-msg">{{ edgeError }}</p>
              </div>
            </div>

            <div class="progress-item" :class="{ completed: dashboardStarted, error: dashboardError, active: isStartingServices && edgeStarted }">
              <div class="progress-step">
                <span v-if="dashboardStarted" class="status-icon success">✓</span>
                <span v-else-if="dashboardError" class="status-icon error">✗</span>
                <span v-else class="status-icon loading">⏳</span>
              </div>
              <div class="progress-text">
                <p class="status">A iniciar Dashboard...</p>
                <p v-if="dashboardError" class="error-msg">{{ dashboardError }}</p>
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button v-if="!isStartingServices" class="btn-cancel" @click="closeProgressModal">
              {{ hasErrors ? 'Fechar' : 'Ok' }}
            </button>
            <p v-if="hasErrors" class="error-summary">Verifique as janelas PowerShell abertas para mais detalhes.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Informações da Stream -->
    <div v-if="streamActive" class="stream-info">
      <div class="info-item">
        <span class="label">Resolução:</span>
        <span class="value">{{ streamResolution }}</span>
      </div>
      <div class="info-item">
        <span class="label">Status:</span>
        <span class="value">{{ streamStatus }}</span>
      </div>
      <div class="info-item">
        <span class="label">URL:</span>
        <span class="value mono">{{ streamUrl }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'

const API_BASE_URL = 'http://127.0.0.1:8000'
const streamUrl = computed(() => `${API_BASE_URL}/api/video/stream?t=${streamKey.value}`)
const streamKey = ref(0)
const streamActive = ref(false)
const streamStatus = ref('Desconectado')
const streamResolution = ref('Desconhecido')
const showProgressModal = ref(false)
const isStartingServices = ref(false)

// Estados de progresso
const apiStarted = ref(false)
const edgeStarted = ref(false)
const dashboardStarted = ref(false)

// Erros
const apiError = ref(null)
const edgeError = ref(null)
const dashboardError = ref(null)

const hasErrors = computed(() => !!(apiError.value || edgeError.value || dashboardError.value))

let checkInterval = null

const handleStreamLoad = () => {
  streamStatus.value = 'Ativo'
  streamActive.value = true
}

const handleStreamError = () => {
  streamStatus.value = 'Erro de conexão'
  streamActive.value = false
}

const refreshStream = () => {
  streamKey.value++
  streamStatus.value = 'Recarregando...'
}

const startServices = async () => {
  showProgressModal.value = true
  isStartingServices.value = true
  
  // Resetar estados
  apiStarted.value = false
  edgeStarted.value = false
  dashboardStarted.value = false
  apiError.value = null
  edgeError.value = null
  dashboardError.value = null

  try {
    // Chamar o endpoint para iniciar serviços
    const response = await fetch(`${API_BASE_URL}/api/services/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })

    const data = await response.json()

    if (data.status === 'sucesso') {
      // Simular progresso dos serviços
      apiStarted.value = true
      await sleep(1500)
      
      edgeStarted.value = true
      await sleep(1500)
      
      dashboardStarted.value = true
      await sleep(2000)

      // Fechar modal automaticamente e tentar conectar
      showProgressModal.value = false
      tryConnect()
    } else {
      apiError.value = data.mensagem
    }
  } catch (error) {
    apiError.value = 'Erro ao contactar a API'
  } finally {
    isStartingServices.value = false
  }
}

const closeProgressModal = () => {
  showProgressModal.value = false
  tryConnect()
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const tryConnect = async () => {
  streamStatus.value = 'Tentando conectar...'
  try {
    const response = await fetch(`${API_BASE_URL}/api/video/frame`, { 
      signal: AbortSignal.timeout(5000) 
    })
    if (response.ok) {
      streamActive.value = true
      streamStatus.value = 'Conectado'
    } else {
      streamActive.value = false
      streamStatus.value = 'Erro na API'
    }
  } catch (error) {
    streamActive.value = false
    streamStatus.value = 'Sem conexão'
  }
}

const checkStreamAvailability = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/video/frame`, { 
      signal: AbortSignal.timeout(3000) 
    })
    if (response.ok) {
      if (!streamActive.value) {
        streamActive.value = true
        streamStatus.value = 'Ativo'
      }
    } else {
      streamActive.value = false
    }
  } catch (error) {
    streamActive.value = false
  }
}

onMounted(() => {
  // Tentar conectar inicialmente
  tryConnect()
  
  // Verificar disponibilidade a cada 10 segundos
  checkInterval = setInterval(checkStreamAvailability, 10000)
})
</script>

<style scoped>
.camera-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  padding: 2rem;
  overflow: hidden;
}

.camera-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 1rem;
  border-bottom: 1px solid #e2e8f0;
}

.camera-header h2 {
  margin: 0;
  color: #0f172a;
  font-size: 1.3rem;
  font-weight: 600;
}

.camera-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.btn-refresh {
  padding: 0.5rem 1rem;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  color: #475569;
  transition: all 0.2s ease;
  font-weight: 500;
}

.btn-refresh:hover:not(:disabled) {
  background: #e2e8f0;
  border-color: #94a3b8;
}

.btn-refresh:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.stream-status {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: #f8fafc;
  border-radius: 4px;
  font-size: 0.9rem;
  font-weight: 600;
  color: #64748b;
}

.stream-status.active {
  background: #ecfdf5;
  color: #065f46;
}

.stream-status.inactive {
  background: #fef2f2;
  color: #991b1b;
}

/* Stream Container */
.camera-stream-wrapper {
  position: relative;
  width: 100%;
  background: #000;
  border-radius: 6px;
  overflow: hidden;
  aspect-ratio: 16 / 9;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stream-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #1a1a1a;
}

.stream-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

/* Placeholder quando offline */
.stream-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
}

.placeholder-content {
  text-align: center;
  color: #cbd5e1;
}

.placeholder-content .title {
  margin: 0.5rem 0;
  font-size: 1.3rem;
  font-weight: 600;
  color: #e2e8f0;
}

.placeholder-content .subtitle {
  margin: 0.5rem 0 1.5rem;
  font-size: 0.9rem;
  color: #94a3b8;
}

.btn-try {
  padding: 0.7rem 1.5rem;
  background: #3b82f6;
  border: none;
  border-radius: 4px;
  color: white;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-try:hover:not(:disabled) {
  background: #2563eb;
  transform: translateY(-2px);
}

.btn-try:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

/* Info */
.stream-info {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.info-item .label {
  font-size: 0.8rem;
  color: #64748b;
  font-weight: 600;
  text-transform: uppercase;
}

.info-item .value {
  font-size: 0.95rem;
  color: #0f172a;
  font-weight: 500;
}

.info-item .value.mono {
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 0.85rem;
  background: white;
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  border: 1px solid #e2e8f0;
  word-break: break-all;
}

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.modal-content {
  background: white;
  border-radius: 8px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  animation: slideUp 0.3s ease;
}

@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-header {
  display: flex;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.modal-header h3 {
  margin: 0;
  font-size: 1.25rem;
  color: #0f172a;
  font-weight: 600;
}

.modal-body {
  padding: 1.5rem;
}

.modal-body.progress-body {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.progress-item {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 1rem;
  border-radius: 6px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  opacity: 0.6;
  transition: all 0.3s ease;
}

.progress-item.active {
  opacity: 1;
  background: #eff6ff;
  border-color: #3b82f6;
}

.progress-item.completed {
  opacity: 1;
  background: #ecfdf5;
  border-color: #10b981;
}

.progress-item.error {
  opacity: 1;
  background: #fef2f2;
  border-color: #ef4444;
}

.progress-step {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #e2e8f0;
  font-size: 1.2rem;
}

.progress-item.active .progress-step {
  background: #3b82f6;
  color: white;
}

.progress-item.completed .progress-step {
  background: #10b981;
  color: white;
}

.progress-item.error .progress-step {
  background: #ef4444;
  color: white;
}

.status-icon.loading {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.progress-text {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.progress-text .status {
  margin: 0;
  font-size: 0.95rem;
  color: #0f172a;
  font-weight: 500;
}

.progress-text .error-msg {
  margin: 0;
  font-size: 0.85rem;
  color: #dc2626;
  font-weight: 500;
}

.modal-footer {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  align-items: flex-end;
  padding: 1.5rem;
  border-top: 1px solid #e2e8f0;
}

.error-summary {
  margin: 0;
  font-size: 0.85rem;
  color: #dc2626;
  text-align: right;
  font-weight: 500;
}

.btn-cancel {
  padding: 0.6rem 1.5rem;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  color: #475569;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-cancel:hover {
  background: #e2e8f0;
  border-color: #94a3b8;
}

.progress-modal {
  max-width: 500px;
}

/* Responsivo */
@media (max-width: 768px) {
  .camera-container {
    padding: 1rem;
  }

  .camera-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  }

  .camera-controls {
    width: 100%;
    justify-content: space-between;
  }

  .stream-info {
    grid-template-columns: 1fr;
  }

  .modal-content {
    width: 95%;
    max-height: 90vh;
  }

  .progress-item {
    flex-direction: column;
    align-items: flex-start;
  }

  .modal-footer {
    align-items: stretch;
  }

  .btn-cancel {
    width: 100%;
  }
}
</style>
