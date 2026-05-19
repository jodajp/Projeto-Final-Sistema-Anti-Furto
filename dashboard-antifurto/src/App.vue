<template>
  <div class="app-layout">
    
    <SideBar 
      :currentView="currentView" 
      :apiConnected="apiConnected" 
      @navigate="currentView = $event" 
    />

    <main class="main-area">
      <div v-if="currentView === 'dashboard'" class="page-content">
        <header class="page-header">
          <h1>Visão Geral do Sistema</h1>
        </header>
        <div class="vertical-layout">
          <ClusterMetrics />
          <AlertList :alerts="alertList" :loading="isLoading" />
        </div>
      </div>

      <div v-if="currentView === 'alertas'" class="page-content">
        <header class="page-header">
          <h1>Histórico de Eventos</h1>
        </header>
        <div class="vertical-layout">
          <AlertList :alerts="alertList" :loading="isLoading" />
        </div>
      </div>

      <div v-if="currentView === 'cluster'" class="page-content">
        <header class="page-header">
          <h1>Gestão de Infraestrutura</h1>
        </header>
        <div class="vertical-layout">
          <ClusterMetrics />
        </div>
      </div>

      <div v-if="currentView === 'login'" class="page-content">
        <LoginView @navigate="currentView = $event" />
      </div>

      <div v-if="currentView === 'signup'" class="page-content">
        <SignUpView @navigate="currentView = $event" />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import SideBar from './components/SideBar.vue'
import ClusterMetrics from './components/ClusterMetrics.vue'
import AlertList from './components/AlertList.vue'

// IMPORTS DAS NOVAS PÁGINAS QUE ESTAVAM EM FALTA:
import LoginView from './components/LoginView.vue'
import SignUpView from './components/SignUpView.vue'

const currentView = ref('dashboard')
const apiConnected = ref(false)
const isLoading = ref(true)
const alertList = ref([])
let fetchInterval = null

const API_URL = 'http://127.0.0.1:8000/api/alertas/recentes'

const fetchAlerts = async () => {
  try {
    const response = await fetch(API_URL)
    if (!response.ok) throw new Error('Erro na resposta')
    const data = await response.json()
    alertList.value = data.alertas || []
    apiConnected.value = true
    isLoading.value = false
  } catch (error) {
    apiConnected.value = false
  }
}

onMounted(() => {
  fetchAlerts()
  fetchInterval = setInterval(fetchAlerts, 2000)
})

onUnmounted(() => {
  if (fetchInterval) clearInterval(fetchInterval)
})
</script>

<style>
html, body, #app {
  margin: 0 !important;
  padding: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  max-width: 100% !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  background-color: #f8fafc;
  color: #333;
}

.app-layout {
  display: flex;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
}

.main-area {
  flex: 1;
  overflow-y: auto;
  padding: 2.5rem 3.5rem;
}

.page-header { margin-bottom: 2rem; }
.page-header h1 { color: #0f172a; font-size: 1.7rem; font-weight: 600; }

.vertical-layout {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}
</style>