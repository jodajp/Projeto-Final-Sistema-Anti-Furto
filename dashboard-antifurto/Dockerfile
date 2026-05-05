# --- FASE 1: Construção (Build) ---
FROM node:20-alpine AS build-stage

WORKDIR /app

# Copia os ficheiros de dependências do Vue (package.json)
COPY package*.json ./

# Instala as dependências do Node
RUN npm install

# Copia o resto do código do dashboard
COPY . .

# Compila o site para produção (cria a pasta 'dist')
RUN npm run build


# --- FASE 2: Servidor de Produção (Nginx) ---
FROM nginx:alpine AS production-stage

# Copia o site compilado da FASE 1 para a pasta pública do Nginx
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Expõe a porta 80
EXPOSE 80

# Arranca o servidor web
CMD ["nginx", "-g", "daemon off;"]