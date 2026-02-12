"""
Face detection and expression analysis utilities.
"""

import numpy as np
from typing import Optional


def detect_smile_simple(f) -> bool:
    """
    Simple smile detection based on mouth keypoints geometry.
    Uses the relative position of mouth corners to estimate smile.
    """
    if len(f.kps) < 5:
        return False
    
    # Get mouth keypoints (indices 3 and 4 are mouth corners in our 5pt system)
    left_mouth = f.kps[3]  # left mouth corner
    right_mouth = f.kps[4]  # right mouth corner
    nose_tip = f.kps[2]    # nose tip for reference
    
    # Calculate mouth width and curvature
    mouth_width = np.linalg.norm(right_mouth - left_mouth)
    
    # Simple heuristic: if mouth is relatively wide compared to nose-mouth distance
    nose_to_mouth_distance = np.linalg.norm((left_mouth + right_mouth) / 2 - nose_tip)
    
    # Smile detected if mouth is wide relative to face proportions
    if nose_to_mouth_distance > 0:
        smile_ratio = mouth_width / nose_to_mouth_distance
        return smile_ratio > 1.5  # Threshold for smile detection
    
    return False


def calculate_face_center(f) -> tuple[float, float]:
    """Calculate the center point of a face bounding box."""
    center_x = (f.x1 + f.x2) / 2
    center_y = (f.y1 + f.y2) / 2
    return center_x, center_y


def determine_movement_direction(prev_pos: tuple[float, float], current_pos: tuple[float, float], 
                           frame_width: int, frame_height: int, threshold: float = 50.0) -> Optional[str]:
    """
    Determine movement direction based on position change.
    
    Args:
        prev_pos: Previous position (x, y)
        current_pos: Current position (x, y)
        frame_width: Frame width for center detection
        frame_height: Frame height for center detection
        threshold: Movement threshold in pixels
    
    Returns:
        Movement direction or None if no significant movement
    """
    if prev_pos is None or current_pos is None:
        return None
    
    prev_x, prev_y = prev_pos
    curr_x, curr_y = current_pos
    
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    
    # Check if movement exceeds threshold
    distance = (dx**2 + dy**2)**0.5
    if distance < threshold:
        return "CENTERED"
    
    # Determine primary movement direction
    if abs(dx) > abs(dy):
        if dx > 0:
            return "MOVE_RIGHT"
        else:
            return "MOVE_LEFT"
    else:
        if dy > 0:
            return "MOVE_DOWN"
        else:
            return "MOVE_UP"


def is_face_centered(face_pos: tuple[float, float], frame_width: int, frame_height: int, 
                    tolerance: float = 0.15) -> bool:
    """
    Check if face is centered in the frame.
    
    Args:
        face_pos: Face center position (x, y)
        frame_width: Frame width
        frame_height: Frame height
        tolerance: Tolerance as fraction of frame dimensions
    
    Returns:
        True if face is centered
    """
    center_x, center_y = face_pos
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    x_tolerance = frame_width * tolerance
    y_tolerance = frame_height * tolerance
    
    return (abs(center_x - frame_center_x) < x_tolerance and 
            abs(center_y - frame_center_y) < y_tolerance)
