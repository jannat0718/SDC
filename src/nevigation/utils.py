"""
utils.py — Utility functions for camera calibration and navigation config loading
"""
import json
import numpy as np
from typing import Tuple


class CameraParams:
    """Container for camera calibration parameters"""
    def __init__(self, data: dict):
        self.K = np.array(data.get('K', [[0, 0, 0], [0, 0, 0], [0, 0, 1]]))
        self.dist = np.array(data.get('dist', [0, 0, 0, 0, 0]))
        self.resolution = tuple(data.get('resolution', [1280, 720]))
        self.metadata = data.get('camera_metadata', '')
        self.reprojection_error = data.get('reprojection_error_px', 0)
        
        # Extract camera intrinsics from K matrix
        self.fx = float(self.K[0, 0])  # focal length x
        self.fy = float(self.K[1, 1])  # focal length y
        self.cx = float(self.K[0, 2])  # principal point x
        self.cy = float(self.K[1, 2])  # principal point y

        # Derived from calibration (measured values for Logitech StreamCam)
        derived = data.get('derived', {})
        self.hfov_deg = float(derived.get('hfov_deg', 67.4))   # measured HFOV
        self.vfov_deg = float(derived.get('vfov_deg', 41.1))   # measured VFOV
        self.hfov = self.hfov_deg  # alias
    
    def print_summary(self):
        """Print calibration parameters summary"""
        print(f"[CameraParams] Resolution: {self.resolution[0]}x{self.resolution[1]}")
        print(f"[CameraParams] K matrix: {self.K[0,0]:.1f}, {self.K[1,1]:.1f} (focal length)")
        print(f"[CameraParams] Principal point: ({self.K[0,2]:.1f}, {self.K[1,2]:.1f})")
        print(f"[CameraParams] Distortion coeffs: {self.dist}")
        if self.reprojection_error > 0:
            print(f"[CameraParams] Reprojection error: {self.reprojection_error:.3f} px")


class NavConfig:
    """Container for navigation configuration"""
    def __init__(self, data: dict):
        self.scale_factor = data.get('scale_factor', 0.05)
        self.camera_height_m = data.get('camera_height_m', 0.50)
        self.lidar_angle_offset = data.get('lidar_angle_offset', 0.0)
        self.map_world_range_m = data.get('map_world_range_m', 50.0)
        self.track_image_path = data.get('track_image_path', '')
        self.start_px = data.get('start_px', [0, 0])
        self.px_per_meter = data.get('px_per_meter', 1.0)
        self.world_pts = data.get('world_pts', [])
        self.image_pts = data.get('image_pts', [])
        self._raw_data = data
    
    def print_summary(self):
        """Print navigation configuration summary"""
        print(f"[NavConfig] Scale factor: {self.scale_factor}")
        print(f"[NavConfig] Camera height: {self.camera_height_m} m")
        print(f"[NavConfig] Map range: {self.map_world_range_m} m")
        print(f"[NavConfig] Pixels per meter: {self.px_per_meter:.1f}")
        print(f"[NavConfig] Track image: {self.track_image_path}")


def load_camera_params(filepath: str) -> CameraParams:
    """
    Load camera calibration parameters from JSON file
    
    Args:
        filepath: Path to camera_K.json
        
    Returns:
        CameraParams object with K matrix, distortion coefficients, and resolution
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return CameraParams(data)
    except FileNotFoundError:
        print(f"Warning: Camera calibration file not found at {filepath}")
        # Return default calibration
        return CameraParams({})
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return CameraParams({})


def load_nav_config(filepath: str) -> NavConfig:
    """
    Load navigation configuration from JSON file
    
    Args:
        filepath: Path to nav_config.json
        
    Returns:
        NavConfig object with navigation parameters
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return NavConfig(data)
    except FileNotFoundError:
        print(f"Warning: Navigation config file not found at {filepath}")
        # Return default config
        return NavConfig({})
    except Exception as e:
        print(f"Error loading navigation config: {e}")
        return NavConfig({})
