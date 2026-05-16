"""Carregamento e validacao de configuracao para o pipeline modular."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import yaml

class ConfigError(ValueError):
    """Erro de configuracao invalida."""

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "backend": "CAP_DSHOW",
    },
    "runtime": {
        "frame_skip": 2,
        "cache_result": True,
        "debug": False,
    },
    "detector": {
        "type": "mmpose",
        "mmpose": {
            "model_name": "rtmpose-m_8xb256-420e_coco-256x192",
            "device": "cpu",
        },
        "onnx": {
            "model_path": "./models/rtmo-m_640x640.onnx",
            "use_gpu": True,
        },
    },
    "activities": [
        {
            "enabled": True,
            "plugin": "Atividades_Suspeitas.velocity_detector:VelocityDetector",
            "params": {"velocidade_maxima": 200.0},
        },
        {
            "enabled": True,
            "plugin": "Atividades_Suspeitas.posture_detector:PostureDetector",
            "params": {"agachamento_threshold": 0.6, "tempo_minimo": 5},
        },
    ],
    "alerts": {
        "handlers": [
            {
                "enabled": True,
                "plugin": "Alertas.alert_system:AlertSystem",
                "params": {
                    "pasta_alertas": "./Edge/Alertas/history",
                    "save_json": True,
                    "verbose": True,
                },
            }
        ]
    },
    "visualization": {
        "enabled": True,
        "show_skeleton_canvas": True,
        "confidence_threshold": 0.3,
        "window_name": "ANTI-FURTO",
        "bbox_padding": {"x": 25, "y": 35},
        "default_class_id": 0.0,
        "colors": {
            "line": [0, 255, 255],
            "point": [0, 255, 0],
            "canvas_line": [0, 0, 255],
            "canvas_point": [0, 200, 0],
            "text": [0, 255, 0],
            "warning": [0, 0, 255],
            "muted": [200, 200, 200],
        },
    },
    "temporal_filter": {
        "enabled": True,
        "smoothing_factor": 0.6,
        "smoothing_factor_fast": 0.85,
        "rapid_movement_threshold": 5.0,
        "velocity_smoothing": 0.3,
        "occlusion_confidence_threshold": 0.3,
        "max_occlusion_frames": 5,
        "velocity_damping": 0.94,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _validate_plugin_list(items: Any, section_name: str):
    if not isinstance(items, list):
        raise ConfigError(f"Secao '{section_name}' deve ser uma lista")

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ConfigError(f"{section_name}[{idx}] deve ser dict")

        plugin_path = item.get("plugin")
        if not isinstance(plugin_path, str) or ":" not in plugin_path:
            raise ConfigError(
                f"{section_name}[{idx}].plugin deve ter formato module.path:ClassName"
            )

        params = item.get("params", {})
        if params is not None and not isinstance(params, dict):
            raise ConfigError(f"{section_name}[{idx}].params deve ser dict")


class AppConfig:
    """Config normalizada do sistema."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def from_file(cls, file_path: str = "config.yaml") -> "AppConfig":
        config_path = Path(file_path)
        if not config_path.exists():
            raise ConfigError(f"Arquivo de configuracao nao encontrado: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            raw_data = yaml.safe_load(file) or {}

        if not isinstance(raw_data, dict):
            raise ConfigError("Arquivo de configuracao deve conter objeto YAML no topo")

        merged = deepcopy(DEFAULT_CONFIG)
        _deep_merge(merged, raw_data)
        cls._validate(merged)
        return cls(merged)

    @staticmethod
    def _validate(data: Dict[str, Any]):
        required_sections = [
            "camera",
            "runtime",
            "detector",
            "activities",
            "alerts",
            "visualization",
        ]
        for section in required_sections:
            if section not in data:
                raise ConfigError(f"Secao obrigatoria ausente: {section}")

        detector = data["detector"]
        detector_type = detector.get("type")
        if detector_type not in ("mmpose", "onnx"):
            raise ConfigError("detector.type deve ser 'mmpose' ou 'onnx'")

        if detector_type == "onnx":
            model_path = detector.get("onnx", {}).get("model_path")
            if not model_path:
                raise ConfigError("detector.onnx.model_path e obrigatorio quando detector.type=onnx")

        frame_skip = data["runtime"].get("frame_skip", 2)
        if not isinstance(frame_skip, int) or frame_skip <= 0:
            raise ConfigError("runtime.frame_skip deve ser inteiro > 0")

        _validate_plugin_list(data["activities"], "activities")
        _validate_plugin_list(data["alerts"].get("handlers", []), "alerts.handlers")

    def apply_cli_overrides(self, args):
        detector = self.data["detector"]

        if getattr(args, "source", None) is not None:
            self.data["camera"]["id"] = args.source
        
        if getattr(args, "backend", None):
            detector["type"] = args.backend

        if detector["type"] == "mmpose" and getattr(args, "model", None):
            detector["mmpose"]["model_name"] = args.model

        if detector["type"] == "onnx":
            if getattr(args, "model_path", None):
                detector["onnx"]["model_path"] = args.model_path
            if getattr(args, "gpu", None) is not None:
                detector["onnx"]["use_gpu"] = bool(args.gpu)

        if getattr(args, "debug", False):
            self.data["runtime"]["debug"] = True

        self._validate(self.data)

    def camera(self) -> Dict[str, Any]:
        return self.data["camera"]

    def runtime(self) -> Dict[str, Any]:
        return self.data["runtime"]

    def visualization(self) -> Dict[str, Any]:
        return self.data["visualization"]

    def frame_skip(self) -> int:
        return int(self.data["runtime"].get("frame_skip", 2))

    def detector_config(self) -> Dict[str, Any]:
        detector = self.data["detector"]
        detector_type = detector["type"]

        if detector_type == "mmpose":
            mmpose = detector["mmpose"]
            return {
                "type": "mmpose",
                "model_name": mmpose.get("model_name", "rtmpose-m_8xb256-420e_coco-256x192"),
                "device": mmpose.get("device", "cpu"),
            }

        onnx = detector["onnx"]
        return {
            "type": "onnx",
            "model_path": onnx.get("model_path"),
            "use_gpu": bool(onnx.get("use_gpu", True)),
        }

    def activity_specs(self) -> List[Dict[str, Any]]:
        return self.data["activities"]

    def alert_specs(self) -> List[Dict[str, Any]]:
        return self.data["alerts"].get("handlers", [])

    def temporal_filter_config(self) -> Dict[str, Any]:
        """Get temporal filtering configuration."""
        return self.data.get("temporal_filtering") or self.data.get("temporal_filter", {})
