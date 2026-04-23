<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { io } from 'socket.io-client'

const isMonitoring = ref(false)
const videoRef = ref(null)
const canvasRef = ref(null)
const resultImgRef = ref(null)

let socket = null
let captureInterval = null

const startMonitoring = async () => {
  isMonitoring.value = true

  socket = io('http://localhost:5000')

  socket.on('response_frame', (data) => {
    const blob = new Blob([data], { type: 'image/jpeg' })
    const url = URL.createObjectURL(blob)
    if (resultImgRef.value) {
      resultImgRef.value.src = url
    }
  })

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    if (videoRef.value) {
      videoRef.value.srcObject = stream
      videoRef.value.play()
    }
  } catch (err) {
    console.error('Erro câmara:', err)
    alert('Não foi possível aceder à câmara.')
    isMonitoring.value = false
    return
  }

  captureInterval = setInterval(() => {
    const video = videoRef.value
    const canvas = canvasRef.value
    if (video && canvas && socket && socket.connected) {
      const ctx = canvas.getContext('2d')
      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      canvas.toBlob((blob) => {
        if (blob) {
          socket.emit('frame', blob)
        }
      }, 'image/jpeg', 0.8)
    }
  }, 100)
}

const stopMonitoring = () => {
  isMonitoring.value = false

  if (captureInterval) clearInterval(captureInterval)
  
  if (socket) {
    socket.disconnect()
    socket = null
  }

  if (videoRef.value && videoRef.value.srcObject) {
    const tracks = videoRef.value.srcObject.getTracks()
    tracks.forEach(track => track.stop())
    videoRef.value.srcObject = null
  }
}

onUnmounted(() => {
  stopMonitoring()
})
</script>

<template>
  <div class="container">
    <div v-if="!isMonitoring" class="start-container">
      <button @click="startMonitoring" class="btn primary-btn">Iniciar Monitorização</button>
    </div>

    <div v-else>
      <div class="controls">
        <button @click="stopMonitoring" class="btn danger-btn">Parar Monitorização</button>
      </div>

      <div class="video-grid">
        <div class="video-box">
          <h3>1. Câmara Web (Vue)</h3>
          <video ref="videoRef" autoplay muted playsinline></video>
          <canvas ref="canvasRef" style="display: none;"></canvas>
        </div>

        <div class="video-box">
          <h3>2. Deteção IA (Flask Python)</h3>
          <img ref="resultImgRef" alt="A aguardar dados do Servidor IA..." />
        </div>
      </div>
    </div>
  </div>
</template>

