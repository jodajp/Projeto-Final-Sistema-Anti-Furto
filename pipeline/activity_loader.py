from typing import Any, Dict, List
from .plugins import instantiate_from_spec

def load_activities(specs: List[Dict[str, Any]]):
    """Carrega atividades habilitadas com contrato BaseActivity.detecta."""
    activities = []

    for spec in specs:
        if not spec.get("enabled", True):
            continue

        activity = instantiate_from_spec(spec)
        if not hasattr(activity, "detecta"):
            raise TypeError(f"Plugin de atividade sem metodo detecta: {spec.get('plugin')}")

        activities.append(activity)

    return activities
