#!/usr/bin/env python3
"""
Debug version of run_system.py
"""
import sys
import os

# Write debug info
with open('/tmp/debug_output.txt', 'w') as f:
    f.write("=== Debug Start ===\n")
    f.write(f"Python: {sys.version}\n")
    f.write(f"CWD: {os.getcwd()}\n")
    f.write(f"sys.path: {sys.path}\n\n")

# Try importing modules
try:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write("Importing cv2...\n")
    import cv2
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✓ cv2 imported: {cv2.__version__}\n")
except Exception as e:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✗ cv2 import error: {e}\n")

try:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write("Importing numpy...\n")
    import numpy as np
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✓ numpy imported: {np.__version__}\n")
except Exception as e:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✗ numpy import error: {e}\n")

# Add to path and try importing project modules
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write("Importing nevigation.av_system...\n")
    from nevigation.av_system import ObjectDetector, MODEL_XML, CLASSES_PATH
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✓ av_system imported\n")
        f.write(f"  MODEL_XML: {MODEL_XML}\n")
        f.write(f"  CLASSES_PATH: {CLASSES_PATH}\n")
except Exception as e:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✗ av_system import error: {e}\n")
        import traceback
        f.write(traceback.format_exc())

try:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write("Importing nevigation.av_map...\n")
    from nevigation.av_map import MapSystem, DetectedObject
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✓ av_map imported\n")
except Exception as e:
    with open('/tmp/debug_output.txt', 'a') as f:
        f.write(f"✗ av_map import error: {e}\n")
        import traceback
        f.write(traceback.format_exc())

with open('/tmp/debug_output.txt', 'a') as f:
    f.write("\n=== Debug End ===\n")

print("Debug output written to /tmp/debug_output.txt")
