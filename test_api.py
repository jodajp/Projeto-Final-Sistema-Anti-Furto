#!/usr/bin/env python3
"""
Script de teste para a API de Métricas do Sistema Anti-Furto
Verifica se os endpoints estão a funcionar corretamente
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

# Configuração
API_BASE_URL = "http://127.0.0.1:8000"
COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'END': '\033[0m'
}

def print_header(text):
    print(f"\n{COLORS['BLUE']}{'=' * 70}{COLORS['END']}")
    print(f"{COLORS['BLUE']}{text:^70}{COLORS['END']}")
    print(f"{COLORS['BLUE']}{'=' * 70}{COLORS['END']}\n")

def print_success(text):
    print(f"{COLORS['GREEN']}✓ {text}{COLORS['END']}")

def print_error(text):
    print(f"{COLORS['RED']}✗ {text}{COLORS['END']}")

def print_warning(text):
    print(f"{COLORS['YELLOW']}⚠ {text}{COLORS['END']}")

def print_info(text):
    print(f"{COLORS['BLUE']}ℹ {text}{COLORS['END']}")

def test_endpoint_root():
    """Testa o endpoint raiz"""
    print_header("Teste 1: Endpoint Raiz (/)")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print_success(f"API está respondendo: {response.json()}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Não conseguiu conectar à API. Verifique se o servidor está em execução.")
        print_info(f"Verifique: {API_BASE_URL}")
        return False
    except Exception as e:
        print_error(f"Erro: {str(e)}")
        return False

def test_endpoint_alertas():
    """Testa o endpoint de alertas recentes"""
    print_header("Teste 2: Alertas Recentes (/api/alertas/recentes)")
    try:
        response = requests.get(f"{API_BASE_URL}/api/alertas/recentes", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Endpoint respondendo")
            print(f"  Alertas encontrados: {len(data.get('alertas', []))}")
            if data.get('alertas'):
                print(f"  Últimos alertas: {json.dumps(data['alertas'][:2], indent=2)}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Erro: {str(e)}")
        return False

def test_endpoint_metricas_atuais():
    """Testa o endpoint de métricas atuais"""
    print_header("Teste 3: Métricas Atuais (/api/metricas/atuais)")
    try:
        response = requests.get(f"{API_BASE_URL}/api/metricas/atuais", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Endpoint respondendo")
            num_nodes = data.get('total_nodes', 0)
            print(f"  Nós encontrados: {num_nodes}")
            if data.get('metricas'):
                print(f"  Dados do primeiro nó:")
                print(json.dumps(data['metricas'][0], indent=4))
            else:
                print_warning("Nenhuma métrica disponível ainda. Execute o orchestrator.")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Erro: {str(e)}")
        return False

def test_endpoint_metricas_cluster():
    """Testa o endpoint de métricas agregadas do cluster"""
    print_header("Teste 4: Métricas do Cluster (/api/metricas/cluster)")
    try:
        response = requests.get(f"{API_BASE_URL}/api/metricas/cluster", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Endpoint respondendo")
            metrics = data.get('cluster_metrics', {})
            print(f"  Métricas Agregadas:")
            print(f"    • Nós ativos: {metrics.get('num_nodes', 0)}")
            print(f"    • FPS médio: {metrics.get('media_fps', 0)}")
            print(f"    • Total detecções: {metrics.get('total_detections', 0)}")
            print(f"    • Taxa sucesso: {metrics.get('taxa_sucesso_media_pct', 0)}%")
            print(f"    • Tempo médio inferência: {metrics.get('tempo_medio_inferencia_ms', 0)}ms")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Erro: {str(e)}")
        return False

def test_endpoint_metricas_node():
    """Testa o endpoint de métricas de um nó específico"""
    print_header("Teste 5: Métricas de Nó Específico (/api/metricas/node/{node_id})")
    try:
        # Primeiro, obter lista de nós
        response = requests.get(f"{API_BASE_URL}/api/metricas/atuais", timeout=5)
        if response.status_code != 200:
            print_warning("Não conseguiu obter lista de nós")
            return False
        
        data = response.json()
        if not data.get('metricas'):
            print_warning("Nenhum nó disponível para testar")
            return False
        
        node_id = data['metricas'][0].get('node_id', 'node1')
        print(f"  Testando nó: {node_id}")
        
        response = requests.get(f"{API_BASE_URL}/api/metricas/node/{node_id}", timeout=5)
        if response.status_code == 200:
            metricas = response.json()
            print_success(f"Endpoint respondendo")
            print(json.dumps(metricas, indent=4))
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Erro: {str(e)}")
        return False

def test_endpoint_historico():
    """Testa o endpoint de histórico de métricas"""
    print_header("Teste 6: Histórico de Métricas (/api/metricas/historico)")
    try:
        response = requests.get(f"{API_BASE_URL}/api/metricas/historico?limite=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Endpoint respondendo")
            print(f"  Registos encontrados: {data.get('total', 0)}")
            if data.get('historico'):
                print(f"  Últimos registos:")
                for i, record in enumerate(data['historico'][:3], 1):
                    print(f"    {i}. Node: {record.get('node_id')}, FPS: {record.get('fps')}, Frames: {record.get('frame_count')}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Erro: {str(e)}")
        return False

def test_endpoint_registar_metricas():
    """Testa o endpoint de registro de métricas"""
    print_header("Teste 7: Registar Métricas (/api/metricas/registar)")
    try:
        # Dados de teste
        metricas_teste = {
            "node_id": "test_node",
            "fps": 24.5,
            "frame_count": 1234,
            "detection_count": 42,
            "inference_calls": 617,
            "average_inference_ms": 16.3,
            "success_rate": 6.8,
            "uptime_seconds": 3600
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/metricas/registar",
            json=metricas_teste,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'sucesso':
                print_success(f"Métricas registadas: {result.get('mensagem')}")
                print(f"  Ficheiro: {result.get('ficheiro')}")
                return True
            else:
                print_error(f"Erro: {result.get('mensagem')}")
                return False
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Erro: {str(e)}")
        return False

def generate_test_metrics():
    """Gera ficheiros de métricas de teste na pasta Metricas/"""
    print_header("Teste Extra: Gerar Métricas de Teste")
    try:
        metricas_dir = Path(__file__).parent / "Metricas"
        metricas_dir.mkdir(exist_ok=True)
        
        # Gerar múltiplos nós de teste
        for node_num in range(1, 4):
            metricas = {
                "node_id": f"node{node_num}",
                "timestamp": time.time(),
                "fps": 20 + (node_num * 3),
                "frame_count": 5000 + (node_num * 1000),
                "detection_count": 150 + (node_num * 50),
                "inference_calls": 2500 + (node_num * 500),
                "average_inference_ms": 15.5 + (node_num * 0.5),
                "success_rate": 5.5 + (node_num * 0.5),
                "uptime_seconds": 7200 + (node_num * 1800)
            }
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            ficheiro = metricas_dir / f"metricas_node{node_num}_{timestamp_str}.json"
            
            with open(ficheiro, 'w') as f:
                json.dump(metricas, f, indent=2)
            
            print_success(f"Ficheiro criado: {ficheiro.name}")
        
        return True
    except Exception as e:
        print_error(f"Erro ao gerar métricas: {str(e)}")
        return False

def run_all_tests():
    """Executa todos os testes"""
    print(f"\n{COLORS['BLUE']}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     TESTE COMPLETO DA API - SISTEMA ANTI-FURTO                    ║")
    print("║     " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " * 45 + "║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{COLORS['END']}")
    
    print_info(f"URL da API: {API_BASE_URL}")
    print_info("Certifique-se que o servidor está em execução:")
    print_info("  cd Backend && uvicorn main_api:app --reload")
    
    tests = [
        ("Endpoint Raiz", test_endpoint_root),
        ("Alertas Recentes", test_endpoint_alertas),
        ("Métricas Atuais", test_endpoint_metricas_atuais),
        ("Métricas do Cluster", test_endpoint_metricas_cluster),
        ("Métricas de Nó", test_endpoint_metricas_node),
        ("Histórico", test_endpoint_historico),
        ("Registar Métricas", test_endpoint_registar_metricas),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            time.sleep(0.5)  # Pequena pausa entre testes
        except Exception as e:
            print_error(f"Erro ao executar teste: {str(e)}")
            results.append((name, False))
    
    # Resumo
    print_header("Resumo dos Testes")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{COLORS['GREEN']}PASSOU{COLORS['END']}" if result else f"{COLORS['RED']}FALHOU{COLORS['END']}"
        print(f"  {name:30} ... {status}")
    
    print(f"\n{COLORS['BLUE']}Resultado: {passed}/{total} testes passaram{COLORS['END']}\n")
    
    if passed == total:
        print_success("Todos os testes passaram! ✓")
    elif passed > total // 2:
        print_warning(f"Alguns testes falharam ({total - passed})")
    else:
        print_error("Maioria dos testes falhou. Verifique a API.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate_test_metrics()
        elif sys.argv[1] == "full":
            generate_test_metrics()
            time.sleep(1)
            run_all_tests()
        else:
            print("Uso: python test_api.py [generate|full]")
            print("  generate: Gerar ficheiros de métricas de teste")
            print("  full:     Gerar métricas e executar todos os testes")
    else:
        run_all_tests()
