# Script para iniciar todos os serviços do Sistema Anti-Furto

# Cores para output
$colors = @{
    Success = 'Green'
    Error = 'Red'
    Info = 'Cyan'
    Warning = 'Yellow'
}

# Diretório raiz do projeto
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[INFO] Iniciando Sistema Anti-Furto..." -ForegroundColor $colors.Info
Write-Host "[INFO] Diretório raiz: $projectRoot" -ForegroundColor $colors.Info

# Terminal 1: API Backend
Write-Host "[INIT] Iniciando API Backend..." -ForegroundColor $colors.Info
$apiScript = @"
cd '$projectRoot\Backend'
Write-Host 'A instalar dependências da API...' -ForegroundColor Cyan
pip install -r requirements.txt
Write-Host 'A iniciar API no porto 8000...' -ForegroundColor Green
uvicorn app.main_api:app --reload --port 8000
"@

Start-Process PowerShell -ArgumentList "-NoExit", "-Command", $apiScript -PassThru | Out-Null
Start-Sleep -Milliseconds 1000

# Terminal 2: Edge Service
Write-Host "[INIT] Iniciando Edge Service..." -ForegroundColor $colors.Info
$edgeScript = @"
cd '$projectRoot\Edge'
Write-Host 'A instalar dependências do Edge...' -ForegroundColor Cyan
pip install -r requirements.txt
Write-Host 'A iniciar Edge Service...' -ForegroundColor Green
python main.py
"@

Start-Process PowerShell -ArgumentList "-NoExit", "-Command", $edgeScript -PassThru | Out-Null
Start-Sleep -Milliseconds 1000

# Terminal 3: Dashboard (apenas se não estiver a correr)
Write-Host "[INIT] Iniciando Dashboard..." -ForegroundColor $colors.Info
$dashboardScript = @"
cd '$projectRoot\dashboard-antifurto'
Write-Host 'A instalar dependências do Dashboard...' -ForegroundColor Cyan
npm install
Write-Host 'A iniciar Dashboard...' -ForegroundColor Green
npm run dev
"@

Start-Process PowerShell -ArgumentList "-NoExit", "-Command", $dashboardScript -PassThru | Out-Null

Write-Host "" -ForegroundColor $colors.Success
Write-Host "========================================" -ForegroundColor $colors.Success
Write-Host "Todos os serviços foram iniciados!" -ForegroundColor $colors.Success
Write-Host "========================================" -ForegroundColor $colors.Success
Write-Host "" -ForegroundColor $colors.Success
Write-Host "API Backend: http://127.0.0.1:8000" -ForegroundColor $colors.Info
Write-Host "Dashboard: http://localhost:5173 (ou similar)" -ForegroundColor $colors.Info
Write-Host "" -ForegroundColor $colors.Success

# Aguardar alguns segundos antes de fechar
Start-Sleep -Seconds 5
