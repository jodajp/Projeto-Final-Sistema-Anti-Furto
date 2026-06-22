<template>
  <div class="estatisticas-page">
    <header class="page-header">
      <h1>Estatísticas por Hora</h1>
    </header>

    <div class="controls-row">
      <label>
        Escolher dia:
        <input type="date" v-model="selectedDay" @change="fetchStats" />
      </label>
      <button @click="fetchStats">Atualizar</button>
    </div>

    <div class="card">
      <div v-if="loading">A carregar dados...</div>
      <div v-else>
        <div class="chart-legend">
          <span>Dia selecionado: {{ selectedDay }}</span>
          <span>Registros lidos: {{ rowsQueried }}</span>
        </div>

        <div class="chart-error" v-if="errorMessage">
          {{ errorMessage }}
        </div>

        <div class="chart-wrap" v-if="counts.length">
          <svg :width="chartWidth" :height="chartHeight" class="chart">
            <g v-for="line in yLines" :key="line.value">
              <line :x1="xAxisMargin" :x2="chartWidth - 10" :y1="line.y" :y2="line.y" stroke="#e2e8f0" stroke-width="1" />
              <text :x="xAxisMargin - 10" :y="line.y + 4" text-anchor="end" font-size="10" fill="#64748b">{{ line.label }}</text>
            </g>

            <line :x1="xAxisMargin" :x2="chartWidth - 10" :y1="chartHeight - xAxisMargin" :y2="chartHeight - xAxisMargin" stroke="#94a3b8" stroke-width="1.5" />
            <line :x1="xAxisMargin" :x2="xAxisMargin" :y1="chartHeight - xAxisMargin" :y2="10" stroke="#94a3b8" stroke-width="1.5" />

            <g v-for="(count, idx) in counts" :key="idx">
              <rect
                :x="xAxisMargin + idx * barW + 4"
                :y="barTop(count)"
                :width="barW - 6"
                :height="barHeight(count)"
                rx="4"
                fill="#16a34a" />
              <text
                :x="xAxisMargin + idx * barW + barW / 2"
                :y="chartHeight - xAxisMargin + 16"
                text-anchor="middle"
                font-size="10"
                fill="#111">
                {{ hourLabels[idx] }}
              </text>
              <text
                v-if="count > 0"
                :x="xAxisMargin + idx * barW + barW / 2"
                :y="Math.max(barTop(count) - 6, 14)"
                text-anchor="middle"
                font-size="10"
                fill="#0f172a">
                {{ count }}
              </text>
            </g>
          </svg>
        </div>

        <div v-else>
          Nenhum dado disponível para o dia selecionado.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const selectedDay = ref(new Date().toISOString().slice(0, 10))
const loading = ref(true)
const counts = ref([])
const rowsQueried = ref(0)
const maxCount = ref(5)
const errorMessage = ref('')

const barW = 32
const xAxisMargin = 50
const chartWidth = computed(() => xAxisMargin + 24 * barW + 30)
const chartHeight = 320
const chartInnerHeight = chartHeight - xAxisMargin - 20

const hourLabels = Array.from({ length: 24 }, (_, hour) => `${String(hour).padStart(2, '0')}:00`)

const API_BASE = 'http://20.251.152.37:8000/api'
const API_URL_BASE = `${API_BASE}/estatisticas/horas`

const yLines = computed(() => {
  const maxValue = Math.max(5, Math.ceil(maxCount.value / 5) * 5)
  const numLines = Math.floor(maxValue / 5) + 1
  return Array.from({ length: numLines }, (_, i) => {
    const value = i * 5
    const y = chartHeight - xAxisMargin - (value / maxValue) * chartInnerHeight
    return { value, label: String(value), y }
  }).reverse()
})

const barHeight = (count) => {
  const maxValue = Math.max(5, Math.ceil(maxCount.value / 5) * 5)
  const height = (count / maxValue) * chartInnerHeight
  return count > 0 ? Math.max(height, 2) : 0
}

const barTop = (count) => chartHeight - xAxisMargin - barHeight(count)

const fetchStats = async () => {
  loading.value = true
  errorMessage.value = ''
  try {
    const pathUrl = `${API_URL_BASE}/${selectedDay.value}`
    let response = await fetch(pathUrl)
    if (!response.ok) {
      const queryUrl = `${API_URL_BASE}?day=${selectedDay.value}`
      response = await fetch(queryUrl)
    }

    if (!response.ok) {
      throw new Error(`Falha ao buscar dados: ${response.status}`)
    }

    const data = await response.json()
    counts.value = Array.isArray(data.counts) ? data.counts : Array(24).fill(0)
    rowsQueried.value = data.rows_queried || 0
    maxCount.value = Math.max(5, ...counts.value)
  } catch (e) {
    counts.value = Array(24).fill(0)
    rowsQueried.value = 0
    maxCount.value = 5
    errorMessage.value = e.message || 'Erro ao carregar estatísticas.'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchStats()
})
</script>
