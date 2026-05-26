# Script de Teste da API - Sistema Anti-Furto (Windows PowerShell)
# Uso: .\test_api.ps1

param(
    [ValidateSet("test", "generate", "full", "metrics", "node")]
    [string]$Command = "test",
    [string]$NodeId = "node1"
)

$API_URL = "http://127.0.0.1:8000"
$METRICS_DIR = ".\Metricas"

# Cores para output
function Write-Success { Write-Host "$args" -ForegroundColor Green }
function Write-Error { Write-Host "$args" -ForegroundColor Red }
function Write-Warning { Write-Host "$args" -ForegroundColor Yellow }
function Write-Info { Write-Host "$args" -ForegroundColor Cyan }

function Print-Header {
    param([string]$Text)
    Write-Info "`n$('=' * 70)"
    Write-Info $Text.PadLeft(50)
    Write-Info "$('=' * 70)`n"
}

function Test-APIRoot {
    Print-Header "Teste 1: Endpoint Raiz (/)"
    try {
        $response = Invoke-WebRequest -Uri "$API_URL/" -ErrorAction Stop -UseBasicParsing
        Write-Success "✓ API respondendo"
        $response.Content | ConvertFrom-Json | Format-List
        return $true
    }
    catch {
        Write-Error "✗ Erro: $_"
        return $false
    }
}

function Test-Alertas {
    Print-Header "Teste 2: Alertas Recentes (/api/alertas/recentes)"
    try {
        $response = Invoke-WebRequest -Uri "$API_URL/api/alertas/recentes" -ErrorAction Stop -UseBasicParsing
        $data = $response.Content | ConvertFrom-Json
        Write-Success "✓ Endpoint respondendo"
        Write-Info "  Alertas encontrados: $($data.alertas.Count)"
        return $true
    }
    catch {
        Write-Error "✗ Erro: $_"
        return $false
    }
}

function Test-MetricasAtuais {
    Print-Header "Teste 3: Métricas Atuais (/api/metricas/atuais)"
    try {
        $response = Invoke-WebRequest -Uri "$API_URL/api/metricas/atuais" -ErrorAction Stop -UseBasicParsing
        $data = $response.Content | ConvertFrom-Json
        Write-Success "✓ Endpoint respondendo"
        Write-Info "  Nós encontrados: $($data.total_nodes)"
        if ($data.metricas.Count -gt 0) {
            $data.metricas[0] | Format-List
        }
        return $true
    }
    catch {
        Write-Error "✗ Erro: $_"
        return $false
    }
}

function Test-MetricasCluster {
    Print-Header "Teste 4: Métricas do Cluster (/api/metricas/cluster)"
    try {
        $response = Invoke-WebRequest -Uri "$API_URL/api/metricas/cluster" -ErrorAction Stop -UseBasicParsing
        $data = $response.Content | ConvertFrom-Json
        Write-Success "✓ Endpoint respondendo"
        Write-Info "  Métricas Agregadas:"
        $data.cluster_metrics | Format-List
        return $true
    }
    catch {
        Write-Error "✗ Erro: $_"
        return $false
    }
}

function Test-MetricasNode {
    Print-Header "Teste 5: Métricas de Nó (/api/metricas/node/{node_id})"
    try {
        $response = Invoke-WebRequest -Uri "$API_URL/api/metricas/node/$NodeId" -ErrorAction Stop -UseBasicParsing
        $data = $response.Content | ConvertFrom-Json
        Write-Success "✓ Endpoint respondendo para nó: $NodeId"
        $data.metricas | Format-List
        return $true
    }
    catch {
        Write-Error "✗ Erro: $_"
        return $false
    }
}

function Test-Historico {
    Print-Header "Teste 6: Histórico (/api/metricas/historico)"
    try {
        $response = Invoke-WebRequest -Uri "$API_URL/api/metricas/historico?limite=5" -ErrorAction Stop -UseBasicParsing
        $data = $response.Content | ConvertFrom-Json
        Write-Success "✓ Endpoint respondendo"
        Write-Info "  Registos encontrados: $($data.total)"
        return $true
    }
    catch {
        Write-Error "✗ Erro: $_"
        return $false
    }
}

function Test-RegistarMetricas {
    Print-Header "Teste 7: Registar Métricas (POST /api/metricas/registar)"
    try {
        $body = @{
            node_id = "test_node"
            fps = 24.5
            frame_count = 1234
            detection_count = 42
            inference_calls = 617
            average_inference_ms = 16.3
            success_rate = 6.8
            uptime_seconds = 3600
        } | ConvertTo-Json

        $response = Invoke-WebRequest -Uri "$API_URL/api/metricas/registar" `
            -Method POST `
            -Headers @{"Content-Type"="application/json"} `
            -Body $body `
            -ErrorAction Stop `
            -UseBasicParsing

        $data = $response.Content | ConvertFrom-Json
        Write-Success "✓ Métricas registadas"
        Write-Info "  Ficheiro: $($data.ficheiro)"
        return $true
    }
    catch {
        Write-Error "✗ Erro: $_"
        return $false
    }
}

function Generate-TestMetrics {
    Print-Header "Gerando Dados de Teste"
    
    if (-not (Test-Path $METRICS_DIR)) {
        New-Item -ItemType Directory -Path $METRICS_DIR -Force | Out-Null
        Write-Success "✓ Pasta Metricas/ criada"
    }

    for ($i = 1; $i -le 3; $i++) {
        $timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
        $filename = "$METRICS_DIR\metricas_node$i`_$timestamp.json"

        $metricas = @{
            node_id = "node$i"
            timestamp = [int](Get-Date -UFormat %s)
            fps = 20 + ($i * 3)
            frame_count = 5000 + ($i * 1000)
            detection_count = 150 + ($i * 50)
            inference_calls = 2500 + ($i * 500)
            average_inference_ms = [math]::Round(15.5 + ($i * 0.5), 2)
            success_rate = [math]::Round(5.5 + ($i * 0.5), 2)
            uptime_seconds = 7200 + ($i * 1800)
        } | ConvertTo-Json

        Set-Content -Path $filename -Value $metricas
        Write-Success "✓ Ficheiro criado: metricas_node$i`_$timestamp.json"
    }
}

function Run-AllTests {
    Print-Header "TESTE COMPLETO DA API - SISTEMA ANTI-FURTO"
    Write-Info "URL da API: $API_URL"
    Write-Info "Data: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    $tests = @(
        @{ Name = "Endpoint Raiz"; Func = ${function:Test-APIRoot} },
        @{ Name = "Alertas Recentes"; Func = ${function:Test-Alertas} },
        @{ Name = "Métricas Atuais"; Func = ${function:Test-MetricasAtuais} },
        @{ Name = "Métricas do Cluster"; Func = ${function:Test-MetricasCluster} },
        @{ Name = "Métricas de Nó"; Func = ${function:Test-MetricasNode} },
        @{ Name = "Histórico"; Func = ${function:Test-Historico} },
        @{ Name = "Registar Métricas"; Func = ${function:Test-RegistarMetricas} }
    )

    $results = @()
    foreach ($test in $tests) {
        try {
            $result = & $test.Func
            $results += @{ Name = $test.Name; Result = $result }
        }
        catch {
            $results += @{ Name = $test.Name; Result = $false }
        }
        Start-Sleep -Milliseconds 500
    }

    # Resumo
    Print-Header "Resumo dos Testes"
    $passed = ($results | Where-Object { $_.Result } | Measure-Object).Count
    $total = $results.Count

    foreach ($result in $results) {
        $status = if ($result.Result) { "PASSOU" } else { "FALHOU" }
        $color = if ($result.Result) { "Green" } else { "Red" }
        Write-Host "  $($result.Name.PadRight(30)) ... " -NoNewline
        Write-Host $status -ForegroundColor $color
    }

    Write-Info "`nResultado: $passed/$total testes passaram`n"
}

# Execução
switch ($Command) {
    "test" {
        Run-AllTests
    }
    "generate" {
        Generate-TestMetrics
    }
    "full" {
        Generate-TestMetrics
        Start-Sleep -Seconds 1
        Run-AllTests
    }
    "metrics" {
        Test-MetricasCluster
    }
    "node" {
        Test-MetricasNode
    }
}

Write-Info "Script finalizado!`n"
