#!/usr/bin/env python3
"""
run_system_with_video_output.py — Saves detected video with bounding boxes
Generates a new MP4 file showing object detection results
"""

import os
import sys
import cv2
import numpy as np

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nevigation.av_system import ObjectDetector, MODEL_XML, CLASSES_PATH

def main():
    VIDEO_PATH = "/home/jannat/sdc_2026/data/SDC_March_Sessions/DATA/20260305_145211/202603051452114/Middle20260305145211.mp4"
    OUTPUT_VIDEO = "/home/jannat/sdc_2026/Output/result_with_detections.mp4"
    
    print("\n" + "="*60)
    print(f"🏋️  LOADING OPENVINO MODEL: {os.path.basename(MODEL_XML)}")
    print("Please wait, compiling YOLOv8 network graph...")
    print("="*60)
    
    try:
        # Initialize detector
        detector = ObjectDetector(MODEL_XML, CLASSES_PATH)
        print("🎯 AI Model successfully compiled!")
        
        # Open input video
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Error: Video file not found at {VIDEO_PATH}")
            return
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📹 Input video: {frame_count} frames @ {fps:.1f} fps ({width}x{height})")
        print(f"💾 Output: {OUTPUT_VIDEO}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Error: Could not create video writer. Trying with H264...")
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
        
        print("\n🚀 PROCESSING AND SAVING VIDEO WITH DETECTIONS...")
        
        frame_num = 0
        total_detections = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"\n✅ Video processing complete!")
                break
            
            frame_num += 1
            
            # Run detection
            raw_detections = detector.detect(frame)
            total_detections += len(raw_detections)
            
            # Draw detections on frame
            detector.draw(frame, raw_detections)
            
            # Write frame to output video
            out.write(frame)
            
            # Progress
            if frame_num % 100 == 0:
                percent = 100 * frame_num / frame_count
                print(f"  Frame {frame_num}/{frame_count} ({percent:.1f}%) - Detections this frame: {len(raw_detections)}")
        
        # Release resources
        cap.release()
        out.release()
        
        # Calculate and display stats
        avg_detections = total_detections / frame_num if frame_num > 0 else 0
        file_size = os.path.getsize(OUTPUT_VIDEO) / (1024*1024)  # MB
        
        print(f"\n📊 Statistics:")
        print(f"   Total frames: {frame_num}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average per frame: {avg_detections:.2f}")
        print(f"   Output file size: {file_size:.1f} MB")
        print(f"\n✅ Result saved to:")
        print(f"   {OUTPUT_VIDEO}")
        print(f"   \\wsl$\\Ubuntu{OUTPUT_VIDEO}")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        print("🔒 Resources released.")

if __name__ == "__main__":
    main()
