#!/usr/bin/env python3
"""
run_system_headless.py — Headless Terminal Version
Runs the full pipeline without GUI display (works in servers/WSL)
"""

import os
import sys
import cv2
import numpy as np

# Force path setup
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nevigation.av_system import ObjectDetector, MODEL_XML, CLASSES_PATH
from src.nevigation.av_map import MapSystem, DetectedObject

def main():
    VIDEO_PATH = "/home/jannat/sdc_2026/data/SDC_March_Sessions/DATA/20260305_145211/202603051452114/Middle20260305145211.mp4"
    
    print("\n" + "="*60)
    print(f"🏋️  LOADING OPENVINO MODEL: {os.path.basename(MODEL_XML)}")
    print("Please wait, compiling YOLOv8 network graph for your CPU...")
    print("="*60)
    
    try:
        # Initialize detector
        detector = ObjectDetector(MODEL_XML, CLASSES_PATH)
        print("🎯 AI Model successfully compiled!")
        
        # Initialize the mapping framework - this returns a complete system
        print("🎬 Initializing mapping systems and loading track layout data...")
        map_system = MapSystem(mode="video", video_path=VIDEO_PATH)
        
        print("\n🚀 PROCESSING VIDEO...")
        print("Running integrated pipeline (object detection + mapping)...")
        
        # Run the integrated system - it handles video playback and processing
        map_system.run()
        print("\n✅ Processing complete!")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("🔒 Resources released.")

if __name__ == "__main__":
    main()
