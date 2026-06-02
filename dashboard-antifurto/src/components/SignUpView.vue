<template>
  <div class="auth-wrapper">
    <div class="auth-card">
      <header class="auth-header">
        <h2>Criar Conta</h2>
        <p>Registar novo administrador Corner</p>
      </header>

      <form class="auth-form" @submit.prevent>
        
        <div class="form-row">
          <div class="form-group">
            <label>Nome</label>
            <input type="text" placeholder="Insira o seu nome" class="form-input" required />
          </div>
          <div class="form-group">
            <label>Apelido</label>
            <input type="text" placeholder="Insira o seu apelido" class="form-input" required />
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Data de Nascimento</label>
            <input type="text" placeholder="DD/MM/AAAA" onfocus="(this.type='date')" onblur="(this.type=this.value?'date':'text')" class="form-input" required />
          </div>
          <div class="form-group">
            <label>Nº de Funcionário</label>
            <input type="text" placeholder="Insira o seu número" class="form-input" required />
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Email Profissional</label>
            <input type="email" placeholder="Insira o seu email" class="form-input" required />
          </div>
          <div class="form-group">
            <label>Contacto Telefónico</label>
            <input type="tel" placeholder="Insira o seu contacto" class="form-input" required />
          </div>
        </div>

        <div class="form-group">
          <label>ID da Loja / Filial</label>
          <input type="text" placeholder="Insira o ID da loja" class="form-input" required />
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Palavra-passe</label>
            <input type="password" placeholder="Insira a sua palavra-passe" class="form-input" required />
          </div>
          <div class="form-group">
            <label>Confirmar Palavra-passe</label>
            <input type="password" placeholder="Confirme a sua palavra-passe" class="form-input" required />
          </div>
        </div>

        <div class="form-checkbox">
          <input type="checkbox" id="terms" required />
          <label for="terms">Confirmo a veracidade dos dados e aceito a política de privacidade.</label>
        </div>
        
        <button type="submit" class="auth-submit-btn">Criar Conta</button>
      </form>
      
      <div class="auth-footer">
        <span class="small-text">Já tem uma conta? </span>
        <a href="#" class="green-link" @click.prevent="$emit('navigate', 'login')">Inicie sessão aqui</a>
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

/* Largura aumentada em ~25% (de 600px para 750px) */
.auth-card { 
  background: white; 
  border-radius: 8px; 
  border: 1px solid #e2e8f0; 
  width: 90%; 
  max-width: 750px; 
  padding: 40px; 
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
}

/* Header */
.auth-header { text-align: center; margin-bottom: 30px; }
.auth-header h2 { margin: 0 0 8px 0; color: #0f172a; font-size: 1.5rem; }
.auth-header p { margin: 0; color: #64748b; font-size: 0.95rem; }

/* Estrutura */
.auth-form { display: flex; flex-direction: column; gap: 18px; }
.form-row { display: flex; gap: 20px; }
.form-row .form-group { flex: 1; }
.form-group { display: flex; flex-direction: column; gap: 6px; }

/* Labels normais */
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

/* Checkbox */
.form-checkbox { display: flex; align-items: flex-start; gap: 10px; margin-top: 5px; }
.form-checkbox input { margin-top: 3px; accent-color: #1A4031; cursor: pointer; }
.form-checkbox label { font-size: 0.85rem; color: #64748b; cursor: pointer; }

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

/* Responsivo para tablets */
@media (max-width: 1024px) {
  .auth-card {
    padding: 30px;
  }
  .auth-header h2 {
    font-size: 1.3rem;
  }
}

/* Responsivo para mobile */
@media (max-width: 768px) {
  .auth-card {
    width: 95%;
    padding: 25px;
  }
  .auth-header h2 {
    font-size: 1.2rem;
  }
  .auth-header p {
    font-size: 0.85rem;
  }
  .form-input {
    font-size: 0.9rem;
    padding: 10px 12px;
  }
  .auth-submit-btn {
    padding: 10px;
    font-size: 0.95rem;
  }
}

/* Mobile pequeno */
@media (max-width: 480px) {
  .signup-container {
    padding: 1rem 0;
  }
  .auth-card {
    width: 95%;
    padding: 20px;
    margin: 1rem auto;
  }
  .auth-header {
    margin-bottom: 20px;
  }
  .auth-header h2 {
    font-size: 1.1rem;
  }
  .auth-header p {
    font-size: 0.8rem;
  }
  .auth-form {
    gap: 14px;
  }
  .form-group {
    gap: 4px;
  }
  .form-group label {
    font-size: 0.85rem;
  }
  .form-input {
    font-size: 0.85rem;
    padding: 8px 10px;
  }
  .form-checkbox {
    gap: 8px;
  }
  .form-checkbox label {
    font-size: 0.8rem;
  }
  .small-text {
    font-size: 0.75rem;
  }
  .green-link {
    font-size: 0.75rem;
  }
}
</style>