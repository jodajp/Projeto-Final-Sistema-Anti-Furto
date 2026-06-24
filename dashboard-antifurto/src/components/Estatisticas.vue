<template>
  <div class="estatisticas-page">
    <div class="controls-row">
      <label>
        Escolher dia:
        <input type="date" v-model="selectedDay" @change="fetchStats" />
      </label>
      <button @click="fetchStats" :disabled="loading">
        {{ loading ? 'A Atualizar...' : 'Atualizar' }}
      </button>
    </div>

    <div class="charts-grid">
      
      <div class="card">
        <h2 class="card-title">Pessoas Detetadas por Hora</h2>
        <div v-if="loadingHourly" class="loading-state">A carregar dados horários...</div>
        <div v-else>
          <div class="chart-legend">
            <span>Registos lidos: {{ rowsQueried }}</span>
          </div>

          <div class="chart-error" v-if="errorMessage">
            {{ errorMessage }}
          </div>

          <div class="chart-wrap" v-if="counts.length">
            <svg :width="chartWidth" :height="chartHeight" class="chart">
              <g v-for="line in yLines" :key="'yline-'+line.value">
                <line :x1="xAxisMargin" :x2="chartWidth - 10" :y1="line.y" :y2="line.y" stroke="#e2e8f0" stroke-width="1" />
                <text :x="xAxisMargin - 10" :y="line.y + 4" text-anchor="end" font-size="10" fill="#64748b">{{ line.label }}</text>
              </g>

              <line :x1="xAxisMargin" :x2="chartWidth - 10" :y1="chartHeight - xAxisMargin" :y2="chartHeight - xAxisMargin" stroke="#94a3b8" stroke-width="1.5" />
              <line :x1="xAxisMargin" :x2="xAxisMargin" :y1="chartHeight - xAxisMargin" :y2="10" stroke="#94a3b8" stroke-width="1.5" />

              <g v-for="(count, idx) in counts" :key="'bar-'+idx">
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
          <div v-else class="empty-state">
            Nenhum dado disponível para o dia selecionado.
          </div>
        </div>
      </div>

      <div class="card">
        <h2 class="card-title">Acessos por Zona (Grabs)</h2>
        <div v-if="loadingZones" class="loading-state">A carregar dados de zonas...</div>
        <div v-else>
          <div class="chart-legend" style="margin-bottom: 20px;">
            <span>Total de Interações: {{ zoneTotalEvents }}</span>
          </div>

          <div class="chart-error" v-if="zoneErrorMessage">
            {{ zoneErrorMessage }}
          </div>

          <div class="pie-chart-wrap" v-if="zoneTotalEvents > 0">
            <svg width="240" height="240" viewBox="0 0 300 300">
              <g v-for="(slice, idx) in pieSlices" :key="'slice-'+idx">
                <circle v-if="slice.isFull"
                  :cx="slice.cx" :cy="slice.cy" :r="slice.r"
                  :fill="slice.color" />
                <path v-else
                  :d="slice.pathData"
                  :fill="slice.color"
                  stroke="#ffffff"
                  stroke-width="2" />
              </g>
            </svg>

            <div class="legend-list">
              <div v-for="(slice, idx) in pieSlices" :key="'legend-'+idx" class="legend-item">
                <span class="color-box" :style="{ backgroundColor: slice.color }"></span>
                <span class="legend-label">{{ slice.label }}</span>
                <span class="legend-value">{{ slice.count }} ({{ ((slice.count / zoneTotalEvents) * 100).toFixed(1) }}%)</span>
              </div>
            </div>
          </div>
          <div v-else class="empty-state">
            Nenhum acesso a zonas detetado neste dia.
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const selectedDay = ref(new Date().toISOString().slice(0, 10))
const loading = ref(false)

// Estado - Gráfico Horário
const loadingHourly = ref(true)
const counts = ref([])
const rowsQueried = ref(0)
const maxCount = ref(5)
const errorMessage = ref('')

// Estado - Gráfico Zonas
const loadingZones = ref(true)
const zoneLabels = ref([])
const zoneCounts = ref([])
const zoneTotalEvents = ref(0)
const zoneErrorMessage = ref('')

const API_BASE = 'http://20.251.152.37:8000/api'
const HOURLY_URL = `${API_BASE}/estatisticas/horas`
const ZONES_URL = `${API_BASE}/estatisticas/zonas`

// ================= LÓGICA GRÁFICO BARRAS =================
const barW = 32
const xAxisMargin = 50
const chartWidth = computed(() => xAxisMargin + 24 * barW + 30)
const chartHeight = 320
const chartInnerHeight = chartHeight - xAxisMargin - 20
const hourLabels = Array.from({ length: 24 }, (_, hour) => `${String(hour).padStart(2, '0')}:00`)

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

// ================= LÓGICA GRÁFICO CIRCULAR =================
const pieColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6']

const pieSlices = computed(() => {
  let total = zoneTotalEvents.value
  if (total === 0) return []

  let cumulativeValue = 0
  const cx = 150
  const cy = 150
  const r = 140

  return zoneLabels.value.map((label, i) => {
    const count = zoneCounts.value[i]
    const percent = count / total
    
    // Matemática para desenhar fatias usando SVG Path
    const startAngle = (cumulativeValue / total) * Math.PI * 2 - Math.PI / 2
    cumulativeValue += count
    const endAngle = (cumulativeValue / total) * Math.PI * 2 - Math.PI / 2

    if (percent === 1) {
      return { label, count, color: pieColors[i % pieColors.length], isFull: true, cx, cy, r }
    }

    const x1 = cx + r * Math.cos(startAngle)
    const y1 = cy + r * Math.sin(startAngle)
    const x2 = cx + r * Math.cos(endAngle)
    const y2 = cy + r * Math.sin(endAngle)

    const largeArcFlag = percent > 0.5 ? 1 : 0
    const pathData = [
      `M ${cx} ${cy}`,
      `L ${x1} ${y1}`,
      `A ${r} ${r} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
      'Z'
    ].join(' ')

    return { label, count, color: pieColors[i % pieColors.length], pathData, isFull: false }
  })
})

// ================= BUSCA DE DADOS =================
const fetchHourlyStats = async () => {
  loadingHourly.value = true
  errorMessage.value = ''
  try {
    let response = await fetch(`${HOURLY_URL}/${selectedDay.value}`)
    if (!response.ok) response = await fetch(`${HOURLY_URL}?day=${selectedDay.value}`)
    if (!response.ok) throw new Error(`Falha API: ${response.status}`)

    const data = await response.json()
    counts.value = Array.isArray(data.counts) ? data.counts : Array(24).fill(0)
    rowsQueried.value = data.rows_queried || 0
    maxCount.value = Math.max(5, ...counts.value)
  } catch (e) {
    counts.value = Array(24).fill(0)
    rowsQueried.value = 0
    maxCount.value = 5
    errorMessage.value = e.message || 'Erro ao carregar pessoas por hora.'
  } finally {
    loadingHourly.value = false
  }
}

const fetchZoneStats = async () => {
  loadingZones.value = true
  zoneErrorMessage.value = ''
  try {
    let response = await fetch(`${ZONES_URL}/${selectedDay.value}`)
    if (!response.ok) response = await fetch(`${ZONES_URL}?day=${selectedDay.value}`)
    if (!response.ok) throw new Error(`Falha API Zonas: ${response.status}`)

    const data = await response.json()
    zoneLabels.value = data.labels || []
    zoneCounts.value = data.counts || []
    zoneTotalEvents.value = data.total_events || 0
  } catch (e) {
    zoneLabels.value = []
    zoneCounts.value = []
    zoneTotalEvents.value = 0
    zoneErrorMessage.value = e.message || 'Erro ao carregar zonas.'
  } finally {
    loadingZones.value = false
  }
}

const fetchStats = async () => {
  loading.value = true
  // O Promise.all faz as duas queries à base de dados na Cloud em simultâneo!
  await Promise.all([
    fetchHourlyStats(),
    fetchZoneStats()
  ])
  loading.value = false
}

onMounted(() => {
  fetchStats()
})
</script>

