"""
Refactored main face recognition system with MQTT integration.
Modular architecture with separate components for locking, logging, and MQTT communication.
"""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Import modular components
from .haar_5pt import align_face_5pt
from .tracker import FaceTracker, draw_tracked_face
from .face_lock import FaceLockManager
from .activity_logger import ActivityLogger
from .mqtt_client import FaceTrackingMQTTClient, MovementStatus
from .face_utils import detect_smile_simple, calculate_face_center, determine_movement_direction, is_face_centered


class FaceRecognitionSystem:
    """
    Main face recognition system with MQTT integration.
    Handles face detection, recognition, locking, and real-time tracking.
    """
    
    def __init__(self, db_path: Path, camera_id: int = 1, team_id: str = "team1"):
        # Core components
        self.db_path = db_path
        self.camera_id = camera_id
        self.team_id = team_id
        
        # Initialize detection and recognition
        self.detector = None
        self.embedder = None
        self.matcher = None
        self.tracker = None
        
        # Initialize managers
        self.lock_manager = FaceLockManager(lock_duration=300.0, match_threshold=0.3)
        self.activity_logger = ActivityLogger()
        self.mqtt_client = FaceTrackingMQTTClient(team_id=team_id)
        
        # Tracking state
        self.selected_face_name = None
        self.selected_face_was_present = False
        self.selected_face_position = None
        self.locked_face_position = None
        self.last_movement_publish = 0
        self.publish_interval = 1.0  # seconds
        
        # UI state
        self.use_tracking = True
        self.show_debug = False
        self.notification_text = ""
        self.notification_timer = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            # Load face database
            db = self._load_db()
            
            # Initialize detection and recognition
            self.detector = self._create_detector()
            self.embedder = self._create_embedder()
            self.matcher = self._create_matcher(db)
            
            # Initialize tracker
            self.tracker = self._create_tracker()
            
            # Load existing locks
            self.lock_manager._load_from_disk()
            
            # Connect to MQTT
            if not self.mqtt_client.connect():
                print("[System] Warning: MQTT connection failed, continuing without remote tracking")
            
            print("[System] All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"[System] Initialization failed: {e}")
            return False
    
    def run(self):
        """Main processing loop."""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {self.camera_id} not available")
        
        print("[System] Starting face recognition loop...")
        print("Controls: q=quit, r=reload DB, +/- threshold, d=debug, t=tracking")
        print("Locking: l=lock, u=unlock, c=clear, L=reload")
        print("Click on faces to select them")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display
                cv2.imshow("Face Recognition", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if self._handle_keyboard(key, frame):
                    break
                    
                # Update MQTT heartbeat
                self.mqtt_client.update_heartbeat()
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mqtt_client.disconnect()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the recognition pipeline."""
        # Detect faces
        faces = self.detector.detect(frame)
        
        # Update tracker
        if self.use_tracking:
            tracked_faces = self.tracker.update(faces)
        else:
            tracked_faces = []
        
        # Process each detected face
        self._process_faces(frame, faces, tracked_faces)
        
        # Draw results
        return self._draw_results(frame, faces, tracked_faces)
    
    def _process_faces(self, frame: np.ndarray, faces: List, tracked_faces: List):
        """Process detected faces for recognition and tracking."""
        # Map detections to tracked faces
        detection_to_track = self._map_detections_to_tracks(faces, tracked_faces)
        
        # Check for selected face presence
        self._update_selected_face_presence(faces, tracked_faces, detection_to_track)
        
        # Process each face
        for i, face in enumerate(faces):
            # Align and embed
            aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
            embedding = self.embedder.embed(aligned)
            
            # Recognize face
            match_result = self.matcher.match(embedding)
            
            # Check for locked face
            locked_name = self.lock_manager.check_and_lock_by_embedding(embedding)
            
            # Update tracker with recognition results
            if self.use_tracking and i in detection_to_track:
                track_id = detection_to_track[i]
                if track_id in tracked_faces:
                    tracked = tracked_faces[track_id]
                    self.tracker.update_identity(
                        track_id,
                        match_result.name if match_result.accepted else None,
                        match_result.distance,
                        match_result.similarity,
                        embedding=embedding,
                    )
                    tracked.kps = face.kps
            
            # Log activities for locked persons
            if locked_name:
                self._handle_locked_person_activity(locked_name, face, frame)
            
            # Update selection
            if (match_result.accepted and match_result.name == self.selected_face_name and 
                i in detection_to_track):
                self.selected_face_position = calculate_face_center(face)
    
    def _handle_locked_person_activity(self, locked_name: str, face, frame: np.ndarray):
        """Handle activity logging and MQTT publishing for locked persons."""
        # Calculate face position
        face_center = calculate_face_center(face)
        
        # Log movement
        self.activity_logger.log_movement(locked_name, face_center[0], face_center[1])
        
        # Check for smile
        if detect_smile_simple(face):
            self.activity_logger.log_expression(locked_name, "smile")
        
        # Publish movement via MQTT
        current_time = time.time()
        if current_time - self.last_movement_publish > self.publish_interval:
            self._publish_movement(face_center, frame.shape[1], frame.shape[0])
            self.last_movement_publish = current_time
    
    def _publish_movement(self, face_position: Tuple[float, float], frame_width: int, frame_height: int):
        """Publish movement status via MQTT."""
        if face_position is None:
            self.mqtt_client.publish_movement(MovementStatus.NO_FACE_LOCKED, 0.0)
            return
        
        # Check if face is centered
        if is_face_centered(face_position, frame_width, frame_height):
            status = MovementStatus.CENTERED
            confidence = 0.9
        else:
            # Determine movement direction
            if self.locked_face_position is not None:
                direction = determine_movement_direction(
                    self.locked_face_position, face_position, frame_width, frame_height
                )
                status = direction if direction else MovementStatus.CENTERED
                confidence = 0.8
            else:
                status = MovementStatus.CENTERED
                confidence = 0.7
        
        self.mqtt_client.publish_movement(status, confidence)
        self.locked_face_position = face_position
    
    def _update_selected_face_presence(self, faces: List, tracked_faces: List, detection_to_track: Dict):
        """Update presence tracking for selected face."""
        selected_face_present = False
        
        # Check if selected face is present
        if self.selected_face_name:
            for i, face in enumerate(faces):
                if i in detection_to_track:
                    track_id = detection_to_track[i]
                    if track_id in tracked_faces:
                        tracked = tracked_faces[track_id]
                        if tracked.identity == self.selected_face_name:
                            selected_face_present = True
                            break
        
        # Check for presence changes
        if self.selected_face_was_present and not selected_face_present:
            self._handle_selected_face_left()
        elif not self.selected_face_was_present and selected_face_present:
            self._handle_selected_face_returned()
        
        self.selected_face_was_present = selected_face_present
    
    def _handle_selected_face_left(self):
        """Handle when selected face leaves the frame."""
        self.notification_text = f"TARGET {self.selected_face_name} LEFT"
        self.notification_timer = 120
        
        if self.lock_manager.is_locked(self.selected_face_name):
            self.activity_logger.log_presence_change(self.selected_face_name, False)
        
        print(f"[Target Tracking] {self.selected_face_name} left camera view")
    
    def _handle_selected_face_returned(self):
        """Handle when selected face returns to the frame."""
        self.notification_text = f"TARGET {self.selected_face_name} RETURNED"
        self.notification_timer = 120
        
        if self.lock_manager.is_locked(self.selected_face_name):
            self.activity_logger.log_presence_change(self.selected_face_name, True)
        
        print(f"[Target Tracking] {self.selected_face_name} returned to camera view")
    
    def _draw_results(self, frame: np.ndarray, faces: List, tracked_faces: List) -> np.ndarray:
        """Draw detection and tracking results on frame."""
        result_frame = frame.copy()
        
        # Draw header info
        self._draw_header(result_frame)
        
        # Draw faces
        y_offset = 80
        for i, face in enumerate(faces):
            # Get track info if available
            track_info = None
            if self.use_tracking and i in self._map_detections_to_tracks(faces, tracked_faces):
                track_id = self._map_detections_to_tracks(faces, tracked_faces)[i]
                if track_id in tracked_faces:
                    track_info = tracked_faces[track_id]
            
            # Draw face
            result_frame = self._draw_face(result_frame, face, track_info, y_offset)
            y_offset += 100
        
        # Draw notification
        if self.notification_timer > 0:
            self._draw_notification(result_frame)
            self.notification_timer -= 1
        
        # Update FPS
        self._update_fps()
        
        return result_frame
    
    def _draw_face(self, frame: np.ndarray, face, track_info, y_offset: int) -> np.ndarray:
        """Draw a single face with tracking information."""
        # Draw bounding box
        color = (0, 255, 0)  # Green for recognized
        cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        
        # Draw keypoints
        for kp in face.kps:
            cv2.circle(frame, tuple(kp.astype(int)), 3, (0, 0, 255), -1)
        
        # Draw label
        label = "Unknown"
        if track_info and track_info.identity:
            label = track_info.identity
            if self.lock_manager.is_locked(track_info.identity):
                label += " [LOCKED]"
        
        cv2.putText(frame, label, (face.x1, face.y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def _draw_header(self, frame: np.ndarray):
        """Draw header information."""
        # Background for header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        
        # System info
        info_text = f"FPS: {self.current_fps:.1f} | Locked: {len(self.lock_manager.get_locked_names())}"
        cv2.putText(frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # MQTT status
        mqtt_status = "MQTT: Connected" if self.mqtt_client.connected else "MQTT: Disconnected"
        cv2.putText(frame, mqtt_status, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.mqtt_client.connected else (0, 0, 255), 2)
    
    def _draw_notification(self, frame: np.ndarray):
        """Draw notification text."""
        text_size = cv2.getTextSize(self.notification_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 50
        
        # Background
        cv2.rectangle(frame, (text_x - 10, text_y - 30), 
                    (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
        
        # Text
        cv2.putText(frame, self.notification_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _map_detections_to_tracks(self, faces: List, tracked_faces: List) -> Dict[int, int]:
        """Map face detections to tracked faces."""
        # Simplified mapping - in real implementation, use IoU and distance
        mapping = {}
        for i, face in enumerate(faces):
            face_center = calculate_face_center(face)
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, tracked in tracked_faces.items():
                if tracked.centroid:
                    distance = ((face_center[0] - tracked.centroid[0])**2 + 
                              (face_center[1] - tracked.centroid[1])**2)**0.5
                    if distance < best_distance and distance < 100:  # 100 pixel threshold
                        best_distance = distance
                        best_track_id = track_id
            
            if best_track_id is not None:
                mapping[i] = best_track_id
        
        return mapping
    
    def _handle_keyboard(self, key: int, frame: np.ndarray) -> bool:
        """Handle keyboard input."""
        if key == ord('q'):
            return True  # Quit
        
        elif key == ord('r'):
            # Reload database
            db = self._load_db()
            self.matcher = self._create_matcher(db)
            print("[System] Database reloaded")
        
        elif key == ord('d'):
            self.show_debug = not self.show_debug
            print(f"[System] Debug overlay: {'ON' if self.show_debug else 'OFF'}")
        
        elif key == ord('t'):
            self.use_tracking = not self.use_tracking
            if not self.use_tracking:
                self.tracker.clear()
                self.selected_face_name = None
            print(f"[System] Tracking: {'ON' if self.use_tracking else 'OFF'}")
        
        elif key == ord('l'):
            # Lock selected face
            self._lock_selected_face(frame)
        
        elif key == ord('u'):
            # Unlock selected face
            self._unlock_selected_face(frame)
        
        elif key == ord('c'):
            self.lock_manager.clear_all_locks()
            print("[LockManager] Cleared all locks")
        
        elif key == ord('L'):
            self.lock_manager.reload_from_disk()
        
        return False
    
    def _lock_selected_face(self, frame: np.ndarray):
        """Lock the currently selected face."""
        # This would need mouse click integration
        # For now, lock the first recognized face
        print("[System] Please click on a face to select it, then press 'l' to lock")
    
    def _unlock_selected_face(self, frame: np.ndarray):
        """Unlock the currently selected face."""
        if self.selected_face_name and self.lock_manager.is_locked(self.selected_face_name):
            self.lock_manager.unlock_face(self.selected_face_name, self.activity_logger)
            print(f"[LockManager] Unlocked face: {self.selected_face_name}")
        else:
            print("[LockManager] No locked face selected")
    
    def _load_db(self):
        """Load face database."""
        db = np.load(self.db_path)
        return {
            'names': db['names'],
            'embeddings': db['embeddings']
        }
    
    def _create_detector(self):
        """Create face detector."""
        from .haar_5pt import HaarFaceMesh5pt
        return HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    
    def _create_embedder(self):
        """Create face embedder."""
        from .haar_5pt import ArcFaceEmbedderONNX
        return ArcFaceEmbedderONNX(
            model_path="models/embedder_arcface.onnx", 
            input_size=(112, 112), 
            debug=False
        )
    
    def _create_matcher(self, db):
        """Create face matcher."""
        from .haar_5pt import FaceDBMatcher
        return FaceDBMatcher(db=db, dist_thresh=0.34)
    
    def _create_tracker(self):
        """Create face tracker."""
        return FaceTracker(
            max_disappeared=30,
            max_distance=100.0,
            iou_threshold=0.3,
            smooth_alpha=0.7, 
            velocity_alpha=0.5,
        )


def main():
    """Main entry point."""
    db_path = Path("data/db/face_db.npz")
    system = FaceRecognitionSystem(db_path, camera_id=1, team_id="team1")
    
    if system.initialize():
        system.run()
    else:
        print("[System] Failed to initialize face recognition system")


if __name__ == "__main__":
    main()
