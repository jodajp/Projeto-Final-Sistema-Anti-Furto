"""Carregamento dinâmico de plugins no formato module.path:ClassName."""

from importlib import import_module
from typing import Any, Dict


class PluginError(RuntimeError):
    """Erro de carregamento/instanciação de plugin."""


def load_symbol(path: str):
    """Resolve um símbolo no formato module.path:Symbol."""
    if ":" not in path:
        raise PluginError(f"Plugin invalido '{path}'. Use formato module.path:ClassName")

    module_name, symbol_name = path.split(":", 1)

    try:
        module = import_module(module_name)
    except Exception as exc:
        raise PluginError(f"Falha ao importar modulo '{module_name}': {exc}") from exc

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise PluginError(f"Simbolo '{symbol_name}' nao encontrado em '{module_name}'") from exc


def instantiate(path: str, params: Dict[str, Any] | None = None):
    """Instancia um plugin com kwargs opcionais."""
    symbol = load_symbol(path)
    params = params or {}

    if not isinstance(params, dict):
        raise PluginError(f"Params do plugin '{path}' devem ser dict")

    try:
        return symbol(**params)
    except TypeError as exc:
        raise PluginError(f"Falha ao instanciar '{path}' com params {params}: {exc}") from exc


def instantiate_from_spec(spec: Dict[str, Any], key: str = "plugin"):
    """Instancia plugin a partir de spec {'plugin': 'a.b:C', 'params': {...}}."""
    plugin_path = spec.get(key)
    if not plugin_path:
        raise PluginError(f"Spec de plugin sem chave '{key}': {spec}")

    return instantiate(plugin_path, spec.get("params", {}))
