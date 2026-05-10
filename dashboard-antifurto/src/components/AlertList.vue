<template>
  <div class="list-container">
    <div class="list-header">
      <span class="col-type">Tipo de Evento</span>
      <span class="col-conf">Confiança</span>
      <span class="col-desc">Descrição</span>
      <span class="col-time">Hora</span>
    </div>
    
    <div v-if="loading" class="empty-state">A carregar registos...</div>
    <div v-else-if="alerts.length === 0" class="empty-state">Nenhum evento registado.</div>

    <div v-else class="list-body">
      <div v-for="alert in alerts" :key="alert.timestamp" class="list-row">
        
        <span class="col-type">
          <span class="badge" :class="getBadgeClass(alert.tipo)">
            {{ formatType(alert.tipo) }}
          </span>
        </span>
        
        <span class="col-conf">{{ (alert.confianca * 100).toFixed(1) }}%</span>
        
        <span class="col-desc">{{ alert.descricao }} (Frame: {{ alert.frame_id }})</span>
        
        <span class="col-time">{{ formatTime(alert.timestamp_legivel) }}</span>
        
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  alerts: { type: Array, default: () => [] },
  loading: { type: Boolean, default: true }
})

const formatType = (type) => type.replace('_', ' ').toUpperCase()
const formatTime = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleTimeString('pt-PT')
}

const getBadgeClass = (type) => {
  if (type === 'ocultacao_produto') return 'badge-danger'
  if (type === 'velocidade') return 'badge-warning'
  return 'badge-info'
}
</script>

<style scoped>
.list-container {
  background: white;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  overflow: hidden;
}

.list-header {
  display: flex;
  background-color: #f8fafc;
  padding: 12px 20px;
  font-weight: 600;
  color: #475569;
  border-bottom: 1px solid #e2e8f0;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.list-body {
  max-height: 60vh;
  overflow-y: auto;
}

.list-row {
  display: flex;
  padding: 14px 20px;
  border-bottom: 1px solid #f1f5f9;
  align-items: center;
  transition: background-color 0.15s;
}

.list-row:hover {
  background-color: #f8fafc;
}

.list-row:last-child {
  border-bottom: none;
}

.empty-state {
  padding: 30px;
  text-align: center;
  color: #94a3b8;
  font-style: italic;
}

/* Grelha de Colunas Alinhadas */
.col-type { flex: 0 0 180px; }
.col-conf { flex: 0 0 100px; color: #0f172a; font-weight: 500; font-size: 0.95rem; }
.col-desc { flex: 1; color: #334155; font-size: 0.95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 20px; }
.col-time { flex: 0 0 100px; text-align: right; color: #64748b; font-size: 0.9rem; }

.badge {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
}

.badge-danger { background-color: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
.badge-warning { background-color: #fef3c7; color: #d97706; border: 1px solid #fcd34d; }
.badge-info { background-color: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; }
</style>