#!/usr/bin/env python
"""
Shim: delegates to webui/server.py for the actual WebUI.

Usage (same as before):
    python webui.py                          # serves metrics.jsonl from ./checkpoints
    python webui.py --file path/to/metrics.jsonl --port 8080
    python webui.py --file path/to/logs/     # auto-finds metrics.jsonl in dir
"""

import sys
from pathlib import Path

_webui_dir = Path(__file__).parent / "webui"
sys.path.insert(0, str(_webui_dir))

from server import main

if __name__ == "__main__":
    main()
