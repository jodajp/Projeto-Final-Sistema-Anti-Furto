<template>
  <div class="auth-wrapper">
    <div class="auth-card">
      <header class="auth-header">
        <h2>Iniciar Sessão</h2>
        <p>Aceda ao Sistema Anti-Furto</p>
      </header>

      <form class="auth-form" @submit.prevent>
        
        <div class="form-group">
          <label>Email Profissional</label>
          <input type="email" placeholder="Insira o seu email" class="form-input" required />
        </div>

        <div class="form-group">
          <label>Palavra-passe</label>
          <input type="password" placeholder="Insira a sua palavra-passe" class="form-input" required />
        </div>
        
        <button type="submit" class="auth-submit-btn">Entrar</button>
      </form>
      
      <div class="auth-footer">
        <span class="small-text">Não tem conta? </span>
        <a href="#" class="green-link" @click.prevent="$emit('navigate', 'signup')">Criar Conta</a>
      </div>
    </div>
  </div>
</template>

<script setup>
defineEmits(['navigate'])
</script>

