<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <h2>Corner</h2>
      <span class="subtitle">Anti-Furto System</span>
    </div>

    <nav class="sidebar-nav">
      <button 
        :class="['nav-btn', { active: currentView === 'dashboard' }]" 
        @click="$emit('navigate', 'dashboard')">
        Dashboard
      </button>
      <button 
        :class="['nav-btn', { active: currentView === 'alertas' }]" 
        @click="$emit('navigate', 'alertas')">
        Histórico de Alertas
      </button>
      <button 
        :class="['nav-btn', { active: currentView === 'cluster' }]" 
        @click="$emit('navigate', 'cluster')">
        Infraestrutura
      </button>
    </nav>

    <div class="sidebar-footer">
      <div class="status-indicator">
        <span class="pulse-dot" :class="{ 'connected': apiConnected }"></span>
        <span>{{ apiConnected ? 'API Online' : 'API Offline' }}</span>
      </div>
    </div>
  </aside>
</template>

<script setup>
defineProps({
  currentView: String,
  apiConnected: Boolean
})

defineEmits(['navigate'])
</script>

<style scoped>
.sidebar {
  width: 260px;
  background: linear-gradient(180deg, #1A4031 0%, #2A664F 100%);
  color: white;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  height: 100vh;
}

.sidebar-header {
  padding: 2rem 1.5rem;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-header h2 { 
  margin: 0; 
  font-size: 1.5rem; 
  font-weight: 600; 
  letter-spacing: 0.5px; 
}

.subtitle { 
  font-size: 0.8rem; 
  color: #a8d5ba; 
  opacity: 0.8; 
}

.sidebar-nav {
  flex: 1;
  padding: 1.5rem 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-btn {
  background: transparent;
  border: none;
  color: #d1e8dd;
  text-align: left;
  padding: 14px 24px;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  width: 100%;
}

.nav-btn:hover {
  background: rgba(255,255,255,0.1);
  color: white;
}

.nav-btn.active {
  background: rgba(255,255,255,0.15);
  color: white;
  border-left: 4px solid #4ade80;
}

.sidebar-footer {
  padding: 1.5rem;
  background: rgba(0,0,0,0.15);
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.85rem;
  color: #ecfdf5;
}

.pulse-dot { 
  width: 8px; 
  height: 8px; 
  border-radius: 50%; 
  background-color: #ef4444; 
}

.pulse-dot.connected { 
  background-color: #10b981; 
  box-shadow: 0 0 8px #10b981; 
}
</style>