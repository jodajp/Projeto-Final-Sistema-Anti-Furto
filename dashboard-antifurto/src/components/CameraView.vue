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

