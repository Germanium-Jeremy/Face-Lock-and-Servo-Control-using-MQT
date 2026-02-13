import time
from pathlib import Path
from typing import Dict, Tuple, Optional

class ActivityLogger:
    """
    Logs activities of locked persons to a text file.
    Tracks movements, expressions, and presence changes.
    """
    def __init__(self, log_file_path: str = "data/locked_person_activity.txt"):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Track previous positions for movement detection
        self.previous_positions: Dict[str, Tuple[float, float]] = {}
        self.movement_threshold = 50  # pixels threshold for movement detection
        
    def log_activity(self, person_name: str, activity: str):
        """Log an activity with timestamp to the file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {person_name}: {activity}\n"
        
        # print(f"DEBUG: Attempting to log: {log_entry.strip()}")
        # print(f"DEBUG: Log file path: {self.log_file_path}")
        
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            # print(f"DEBUG: Successfully wrote to log file")
        except Exception as e:
            print(f"[ActivityLogger] Error writing to log: {e}")
        
        # Also print to console for immediate feedback
        print(f"[Activity Log] {person_name}: {activity}")
    
    def detect_movement(self, person_name: str, current_x: float, current_y: float) -> Optional[str]:
        """Detect movement direction based on position change."""
        if person_name not in self.previous_positions:
            self.previous_positions[person_name] = (current_x, current_y)
            return None
        
        prev_x, prev_y = self.previous_positions[person_name]
        dx = current_x - prev_x
        dy = current_y - prev_y
        
        # Update previous position
        self.previous_positions[person_name] = (current_x, current_y)
        
        # Check if movement is significant enough
        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
            return None
        
        # Determine primary movement direction
        if abs(dx) > abs(dy):
            if dx > 0:
                return "moved right"
            else:
                return "moved left"
        else:
            if dy > 0:
                return "moved down"
            else:
                return "moved up"
    
    def log_movement(self, person_name: str, current_x: float, current_y: float):
        """Log movement if detected."""
        # print(f"DEBUG: Checking movement for {person_name} at ({current_x:.1f}, {current_y:.1f})")
        movement = self.detect_movement(person_name, current_x, current_y)
        if movement:
            print(f"[DEBUG] Movement detected: {movement}")
            self.log_activity(person_name, movement)
        else:
            pass
            # print(f"DEBUG: No significant movement detected")
    
    def log_presence_change(self, person_name: str, present: bool):
        """Log when a person enters or leaves the frame."""
        if present:
            self.log_activity(person_name, "returned to camera")
        else:
            self.log_activity(person_name, "left camera")
    
    def log_expression(self, person_name: str, expression: str):
        """Log facial expression."""
        self.log_activity(person_name, f"detected {expression}")
    
    def clear_tracking(self, person_name: str):
        """Clear tracking data for a person."""
        if person_name in self.previous_positions:
            del self.previous_positions[person_name]
