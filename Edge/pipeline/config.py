"""Carregamento e validacao de configuracao para o pipeline modular."""

from pathlib import Path
from typing import Any, Dict, List
import yaml

class ConfigError(ValueError):
    """Erro de configuracao invalida."""

class AppConfig:
    """Config normalizada do sistema, simplificada para usar fallbacks amigaveis."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def from_file(cls, file_path: str = "config.yaml") -> "AppConfig":
        config_path = Path(file_path)
        if not config_path.exists():
            raise ConfigError(f"Arquivo de configuracao nao encontrado: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            raise ConfigError("Arquivo de configuracao deve conter objeto YAML no topo")

        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Permite usar self.config.get('chave') facilmente."""
        return self.data.get(key, default)

    def apply_cli_overrides(self, args):
        camera = self.data.setdefault("camera", {})
        detector = self.data.setdefault("detector", {"type": "onnx", "onnx": {}})
        
        if getattr(args, "source", None) is not None:
            camera["id"] = args.source
            
        if getattr(args, "backend", None):
            detector["type"] = args.backend

        if detector["type"] == "mmpose" and getattr(args, "model", None):
            detector.setdefault("mmpose", {})["model_name"] = args.model

        if detector["type"] == "onnx":
            onnx = detector.setdefault("onnx", {})
            if getattr(args, "model_path", None):
                onnx["model_path"] = args.model_path
            if getattr(args, "gpu", None) is not None:
                onnx["use_gpu"] = bool(args.gpu)

        if getattr(args, "debug", False):
            self.data.setdefault("runtime", {})["debug"] = True

        if getattr(args, "zones", False):
            self.data.setdefault("zone_tracking", {})["enabled"] = True

    def camera(self) -> Dict[str, Any]:
        return self.data.get("camera", {})

    def runtime(self) -> Dict[str, Any]:
        return self.data.get("runtime", {})

    def visualization(self) -> Dict[str, Any]:
        return self.data.get("visualization", {})

    def tracker(self) -> Dict[str, Any]:
        return self.data.get("tracker", {})

    def frame_skip(self) -> int:
        return int(self.data.get("runtime", {}).get("frame_skip", 2))

    def detector_config(self) -> Dict[str, Any]:
        detector = self.data.get("detector", {"type": "onnx"})
        detector_type = detector.get("type", "onnx")

        if detector_type == "mmpose":
            mmpose = detector.get("mmpose", {})
            return {
                "type": "mmpose",
                "model_name": mmpose.get("model_name", "rtmpose-m_8xb256-420e_coco-256x192"),
                "device": mmpose.get("device", "cpu"),
            }

        onnx = detector.get("onnx", {})
        return {
            "type": "onnx",
            "model_path": onnx.get("model_path", "./models/rtmo-m_640x640.onnx"),
            "use_gpu": bool(onnx.get("use_gpu", True)),
        }

    def activity_specs(self) -> List[Dict[str, Any]]:
        return self.data.get("activities", [])

    def alert_specs(self) -> List[Dict[str, Any]]:
        return self.data.get("alerts", {}).get("handlers", [])

    def temporal_filter_config(self) -> Dict[str, Any]:
        return self.data.get("temporal_filtering") or self.data.get("temporal_filter", {})
