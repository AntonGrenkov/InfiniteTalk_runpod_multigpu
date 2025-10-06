#!/bin/bash
set -euo pipefail
export WAN_GPU_COUNT="${WAN_GPU_COUNT:-0}"
python -u handler.py
