<template>
  <div class="list-container">
    <div class="list-header">Métricas do Cluster</div>
    
    <div class="list-row">
      <span class="row-label">Nós Ativos no Cluster</span>
      <span class="row-value">{{ activeNodes }} / 3</span>
    </div>
    
    <div class="list-row">
      <span class="row-label">Carga Média de CPU</span>
      <span class="row-value">{{ cpuLoad }}%</span>
    </div>
    
    <div class="list-row">
      <span class="row-label">Latência da API</span>
      <span class="row-value">{{ apiLatency }} ms</span>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const activeNodes = ref(0)
const cpuLoad = ref(0)
const apiLatency = ref(0)
</script>

<style scoped>
.list-container {
  background: white;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  overflow: hidden;
}

.list-header {
  background-color: #f8fafc;
  padding: 12px 20px;
  font-weight: 600;
  color: #475569;
  border-bottom: 1px solid #e2e8f0;
  font-size: 0.95rem;
}

.list-row {
  display: flex;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid #f1f5f9;
  align-items: center;
}

.list-row:last-child {
  border-bottom: none;
}

.row-label {
  color: #64748b;
  font-size: 0.95rem;
}

.row-value {
  font-weight: 600;
  color: #0f172a;
  font-size: 1rem;
}
</style>