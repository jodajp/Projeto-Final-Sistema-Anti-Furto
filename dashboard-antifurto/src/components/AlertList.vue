<template>
  <div class="history-wrapper">
    <div v-if="loading" class="global-loading">A sincronizar com a Cloud...</div>

    <div class="history-grid" v-else>

      <div class="history-section">
        <h3 class="section-title">🚨 Últimos 10 Alertas</h3>

        <div class="list-container">
          <div class="list-header">
            <span class="col-type">Tipo de Evento</span>
            <span class="col-conf">Confiança</span>
            <span class="col-desc">Descrição</span>
            <span class="col-time">Hora</span>
          </div>

          <div v-if="alerts.length === 0" class="empty-state">Nenhum evento registado.</div>

          <div v-else class="list-body">
            <div v-for="alert in alerts" :key="alert.id" class="list-row">
              <span class="col-type">
                <span class="badge" :class="getBadgeClass(alert.tipo_alerta)">
                  {{ formatType(alert.tipo_alerta) }}
                </span>
              </span>

              <span class="col-conf">
                {{ alert.confianca <= 1 ? (alert.confianca * 100).toFixed(1) : alert.confianca.toFixed(1) }}% </span>

                  <span class="col-desc">Track ID: #{{ alert.track_id }}</span>
                  <span class="col-time">{{ formatTime(alert.timestamp) }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="history-section">
        <h3 class="section-title">📊 Últimas 10 Métricas</h3>

        <div class="list-container">
          <div class="list-header">
            <span class="col-node">Nó (Câmara)</span>
            <span class="col-fps">FPS</span>
            <span class="col-pessoas">Pessoas Detetadas</span>
          </div>

          <div v-if="metrics.length === 0" class="empty-state">Nenhuma métrica registada.</div>

          <div v-else class="list-body">
            <div v-for="metrica in metrics" :key="metrica.id" class="list-row">
              <span class="col-node">
                <span class="badge badge-node">{{ metrica.node_id }}</span>
              </span>

              <span class="col-fps">{{ (metrica.fps || 0).toFixed(1) }}</span>

              <span class="col-pessoas">
                👥 {{ metrica.pessoas_detetadas || 0 }}
              </span>

            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { API_BASE } from '../utils/api.js'

const alerts = ref([])
const metrics = ref([])
const loading = ref(true)
let fetchInterval = null

const fetchHistoricoCompleto = async () => {
  try {
    const [alertsRes, metricsRes] = await Promise.all([
      fetch(`${API_BASE}/alertas/recentes`),
      fetch(`${API_BASE}/metricas/historico?limite=10`)
    ])

    if (alertsRes.ok) {
      const aData = await alertsRes.json()
      alerts.value = aData.alertas || []
    }

    if (metricsRes.ok) {
      const mData = await metricsRes.json()
      metrics.value = mData.historico || []
    }
  } catch (error) {
    console.error("Erro ao sincronizar com a Base de Dados:", error)
  } finally {
    loading.value = false
  }
}

// Formatação Visual
const formatType = (type) => {
  if (!type) return 'DESCONHECIDO'
  return type.replace('_', ' ').toUpperCase()
}

const formatTime = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleTimeString('pt-PT')
}

const getBadgeClass = (type) => {
  if (!type) return 'badge-info'
  const t = type.toLowerCase()
  if (t.includes('furto') || t.includes('ocultacao')) return 'badge-danger'
  if (t.includes('velocidade') || t.includes('suspeito')) return 'badge-warning'
  return 'badge-info'
}

onMounted(() => {
  fetchHistoricoCompleto()
  fetchInterval = setInterval(fetchHistoricoCompleto, 5000) // Atualiza a cada 5 seg
})

onUnmounted(() => {
  if (fetchInterval) clearInterval(fetchInterval)
})
</script>

