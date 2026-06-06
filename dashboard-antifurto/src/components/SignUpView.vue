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

