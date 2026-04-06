"""Carga de handlers de alerta e roteamento de eventos."""

from typing import Any, Dict, List

from .plugins import instantiate_from_spec


def load_alert_handlers(specs: List[Dict[str, Any]]):
    """Carrega handlers habilitados com contrato registra_evento."""
    handlers = []

    for spec in specs:
        if not spec.get("enabled", True):
            continue

        handler = instantiate_from_spec(spec)
        if not hasattr(handler, "registra_evento"):
            raise TypeError(f"Handler de alerta sem metodo registra_evento: {spec.get('plugin')}")

        handlers.append(handler)

    return handlers


class AlertDispatcher:
    """Encaminha eventos para todos os handlers de alerta carregados."""

    def __init__(self, handlers: List[Any]):
        self.handlers = handlers

    def dispatch(self, event):
        for handler in self.handlers:
            handler.registra_evento(event)

    def print_summary(self):
        for handler in self.handlers:
            summary_fn = getattr(handler, "imprime_resumo", None)
            if callable(summary_fn):
                summary_fn()
