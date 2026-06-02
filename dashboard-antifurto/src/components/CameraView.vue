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
