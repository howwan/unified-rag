#!/usr/bin/env python3
"""Unified entry point for the RAG demo."""

import os
import sys
from pathlib import Path

# Ensure src is on path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging

# Configure logging before any other imports so all modules inherit it
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
setup_logging(level=getattr(__import__("logging"), log_level, 20))

from src.cli import run

if __name__ == "__main__":
    run()
