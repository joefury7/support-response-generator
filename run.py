#!/usr/bin/env python3
"""Convenient entry point for running Support Response Generator directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.cli import main

if __name__ == "__main__":
    main()
