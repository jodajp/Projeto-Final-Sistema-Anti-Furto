#!/usr/bin/env python3
"""Wrapper delegating to run_phase4_experiment.py."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Train.run_phase4_experiment import main

if __name__ == "__main__":
    main()