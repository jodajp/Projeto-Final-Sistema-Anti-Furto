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

<style scoped>
/* Wrapper */
.auth-wrapper { 
  display: flex; 
  justify-content: center; 
  align-items: center; 
  padding: 3rem 0; 
  min-height: 100%; 
}

/* Largura esticada como no SignUp */
.auth-card { 
  background: white; 
  border-radius: 8px; 
  border: 1px solid #e2e8f0; 
  width: 90%; 
  max-width: 480px; 
  padding: 40px; 
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
}

/* Header */
.auth-header { text-align: center; margin-bottom: 30px; }
.auth-header h2 { margin: 0 0 8px 0; color: #0f172a; font-size: 1.5rem; }
.auth-header p { margin: 0; color: #64748b; font-size: 0.95rem; }

/* Estrutura */
.auth-form { display: flex; flex-direction: column; gap: 18px; }
.form-group { display: flex; flex-direction: column; gap: 6px; }

/* Labels */
.form-group label { 
  color: #334155; 
  font-size: 0.9rem; 
  font-weight: 600; 
}

/* Placeholders claros e itálicos */
.form-input::placeholder {
  color: #cbd5e1;
  font-style: italic;
  font-size: 0.9rem;
}

.form-input { 
  padding: 12px 16px; 
  border: 1px solid #cbd5e1; 
  border-radius: 6px; 
  font-size: 0.95rem; 
  color: #334155; 
  outline: none; 
  transition: border-color 0.2s; 
  width: 100%; 
  box-sizing: border-box; 
}
.form-input:focus { border-color: #2A664F; }

/* Botão */
.auth-submit-btn { 
  background: #1A4031; 
  color: white; 
  border: none; 
  padding: 12px; 
  border-radius: 6px; 
  font-size: 1rem; 
  font-weight: 600; 
  cursor: pointer; 
  margin-top: 10px; 
  width: 100%; 
}
.auth-submit-btn:hover { background: #2A664F; }

/* Footer */
.auth-footer { text-align: center; margin-top: 25px; }
.small-text { font-size: 0.85rem; color: #94a3b8; }
.green-link { 
  font-size: 0.85rem; 
  color: #10b981 !important; 
  font-weight: 600; 
  text-decoration: none; 
}
.green-link:hover { 
  color: #059669 !important; 
  text-decoration: underline; 
}
</style>