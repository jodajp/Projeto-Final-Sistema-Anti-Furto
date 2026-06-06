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
      <button 
        :class="['nav-btn', { active: currentView === 'camera' }]" 
        @click="$emit('navigate', 'camera')">
        Câmara
      </button>
      
    </nav>

    <div class="sidebar-auth">
      <button 
        :class="['nav-btn', { active: currentView === 'login' || currentView === 'signup' }]" 
        @click="$emit('navigate', 'login')">
        Login
      </button>
    </div>

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

