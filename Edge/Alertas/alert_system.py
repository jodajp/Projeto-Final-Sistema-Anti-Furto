"""
Sistema de Alertas
Guarda eventos suspeitos em JSON e console
"""

import json
import os
from datetime import datetime
from typing import List
from pathlib import Path
from Edge.Atividades_Suspeitas.base_activity import SuspiciousEvent

class AlertSystem:
    """Sistema centralizado de alertas."""
    
    def __init__(
        self,
        pasta_alertas: str = './Alertas/history',
        save_json: bool = True,
        verbose: bool = True,
    ):
        self.pasta_alertas = Path(pasta_alertas)
        self.pasta_alertas.mkdir(parents=True, exist_ok=True)
        self.save_json = save_json
        self.verbose = verbose
        
        # Ficheiro de log de eventos
        self.ficheiro_log = self.pasta_alertas / f"eventos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.eventos = []
        
    def registra_evento(self, evento: SuspiciousEvent, verbose: bool = None):
        """Registra um evento suspeito."""
        if verbose is None:
            verbose = self.verbose
        
        # Serializar evento
        evento_dict = {
            'tipo': evento.tipo,
            'timestamp': evento.timestamp,
            'timestamp_legivel': datetime.fromtimestamp(evento.timestamp).isoformat(),
            'confianca': evento.confianca,
            'frame_id': evento.frame_id,
            'pessoa_id': evento.pessoa_id,
            'descricao': evento.descricao,
            'dados': evento.dados_adicionais
        }
        
        self.eventos.append(evento_dict)
        
        # Imprime no console se verbose
        if verbose:
            print(f"\n[ALERTA] {evento.tipo.upper()}")
            print(f"         Confiança: {evento.confianca*100:.1f}%")
            print(f"         {evento.descricao}")
            print(f"         Frame: {evento.frame_id}")
        
        # Guarda imediatamente em JSON
        if self.save_json:
            self._guarda_json()
    
    def _guarda_json(self):
        """Guarda eventos em JSON."""
        try:
            with open(self.ficheiro_log, 'w') as f:
                json.dump(self.eventos, f, indent=2)
        except Exception as e:
            print(f"[ERRO] Não consegui guardar JSON: {e}")
    
    def get_resumo(self) -> dict:
        """Retorna resumo dos eventos."""
        resumo = {
            'total_eventos': len(self.eventos),
            'por_tipo': {},
            'confianca_media': 0.0,
            'eventos_altos': 0  # Confiança > 0.8
        }
        
        if self.eventos:
            for evento in self.eventos:
                tipo = evento['tipo']
                if tipo not in resumo['por_tipo']:
                    resumo['por_tipo'][tipo] = 0
                resumo['por_tipo'][tipo] += 1
                
                if evento['confianca'] > 0.8:
                    resumo['eventos_altos'] += 1
            
            resumo['confianca_media'] = sum(e['confianca'] for e in self.eventos) / len(self.eventos)
        
        return resumo
    
    def imprime_resumo(self):
        """Imprime resumo dos eventos."""
        resumo = self.get_resumo()
        
        print("\n" + "="*60)
        print("RESUMO DE ALERTAS")
        print("="*60)
        print(f"Total de eventos: {resumo['total_eventos']}")
        print(f"Confiança média: {resumo['confianca_media']*100:.1f}%")
        print(f"Eventos de alta confiança: {resumo['eventos_altos']}")
        
        if resumo['por_tipo']:
            print("\nEventos por tipo:")
            for tipo, count in resumo['por_tipo'].items():
                print(f"  - {tipo}: {count}")
        
        print(f"\nLog guardado em: {self.ficheiro_log}")
        print("="*60 + "\n")
