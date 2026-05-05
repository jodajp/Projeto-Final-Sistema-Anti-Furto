"""Sistema anti-furto: entrypoint minimo e modular."""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Edge.pipeline import AppConfig, ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sistema Anti-Furto (modular)")
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo YAML")
    parser.add_argument("--source", help="Fonte de vídeo: ID da câmara (ex: 0) ou caminho do ficheiro (ex: video.mp4)")
    parser.add_argument("--backend", choices=["mmpose", "onnx"], help="Seleciona backend")
    parser.add_argument("--model", help="Modelo MMPose (quando backend=mmpose)")
    parser.add_argument("--model-path", help="Caminho do modelo ONNX (quando backend=onnx)")
    parser.add_argument("--gpu", action="store_true", dest="gpu", help="Forca GPU no ONNX")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Forca CPU no ONNX")
    parser.set_defaults(gpu=None)
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        config = AppConfig.from_file(args.config)
        config.apply_cli_overrides(args)

        from Edge.pipeline.orchestrator import AntiTheftOrchestrator

        orchestrator = AntiTheftOrchestrator(config)
        orchestrator.run()
        return 0
    except ConfigError as exc:
        print(f"[CONFIG] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n[INFO] Encerrado pelo utilizador")
        return 0
    except Exception as exc:
        print(f"[ERRO FATAL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
