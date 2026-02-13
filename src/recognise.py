
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.

Run:
python -m src.recognise

Keys:
q : quit
r : reload DB from disk (data/db/face_db.npz)
+/- : adjust threshold (distance) live
d : toggle debug overlay
t : toggle tracking
l : lock face (select face by clicking first)
u : unlock face
c : clear all locks
L : reload locks
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# Import refactored modules
from .recognize.types import FaceDet
from .recognize.utils import cosine_distance, _clip_xyxy
from .recognize.detector import HaarFaceMesh5pt
from .recognize.embedder import ArcFaceEmbedderONNX
from .recognize.matcher import FaceDBMatcher, load_db_npz, detect_smile_simple
from .recognize.lock_manager import FaceLockManager
from .recognize.logger import ActivityLogger

# Import shared modules
from .haar_5pt import align_face_5pt
from .tracker import FaceTracker
from .mqtt_manager import MQTTManager

def main():
    db_path = Path("data/db/face_db.npz")
    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(input_size=(112, 112), debug=False)
    
    db = load_db_npz(db_path)
    matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
    
    # Initialize face lock manager
    lock_manager = FaceLockManager(lock_duration=300.0, match_threshold=0.3)
    lock_manager.reload_from_disk()
    
    # Initialize activity logger
    activity_logger = ActivityLogger()
    print("[Activity Logger] Started - logging locked person activities to data/locked_person_activity.txt") 
    
    # Initialize face tracker
    tracker = FaceTracker(
         max_disappeared=30,
         max_distance=100.0,
         iou_threshold=0.3,
         smooth_alpha=0.7, 
         velocity_alpha=0.5,
    )

    # Initialize MQTT Manager
    team_id = "Phoenix_team" # Change as needed
    mqtt_manager = MQTTManager(team_id=team_id)
    last_heartbeat = 0
    heartbeat_interval = 5.0 # seconds

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("[Error] Camera not available.")
             return

    print(f"Recognize (multi-face with tracking & persistent locking) - Team: {team_id}")
    print("q=quit, r=reload DB, +/- threshold, d=debug overlay, t=toggle tracking")
    print("Locking: l=lock face, u=unlock face, c=clear all locks, L=reload locks")
    print("Click on a face to select it for locking/unlocking.")
    
    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False
    use_tracking = True
    y0 = 80
    thumb = 112
    shown = 0
    x0 = 0
    pad = 8 

    # Variables for face selection
    selected_track_id: Optional[int] = None
    selected_face_name: Optional[str] = None
    selected_embedding: Optional[np.ndarray] = None
    selected_face_was_present = False
    re_acquire_threshold = 0.25
    notification_text = ""
    notification_timer = 0
    
    # For mouse click: temporarily store the clicked detection index
    clicked_detection_index: Optional[int] = None
    
    def on_mouse_click(event, x, y, flags, param):
         nonlocal clicked_detection_index
         if event == cv2.EVENT_LBUTTONDOWN:
              # Check if click is on any detected face
              # face list 'faces' must be accessible or we need a way to pass it.
              # Since this is local function, it captures 'faces' from outer scope if defined.
              # But 'faces' changes every frame.
              # Better to store click coordinates and process in loop.
              # Or trust that 'faces' variable in outer scope is current.
              # Python closures capture variables by reference, so it should work if we access 'faces'
              # But 'faces' is defined inside loop.
              pass
    
    # Redefine mouse callback to just store click coordinates
    last_click_pos = None
    def on_mouse_click_simple(event, x, y, flags, param):
        nonlocal last_click_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            last_click_pos = (x, y)

    cv2.namedWindow("recognize_new")
    cv2.setMouseCallback("recognize_new", on_mouse_click_simple)

    while True:
         if not cap.isOpened(): break
         ok, frame = cap.read()
         if not ok: break

         h, w = frame.shape[:2]

         faces = det.detect(frame, max_faces=5)
         vis = frame.copy()

         # Check for click
         if last_click_pos:
              cx, cy = last_click_pos
              for i, f in enumerate(faces):
                   if f.x1 <= cx <= f.x2 and f.y1 <= cy <= f.y2:
                        clicked_detection_index = i
                        break
              last_click_pos = None

         # compute fps
         frames += 1
         dt = time.time() - t0
         if dt >= 1.0:
              fps = frames / dt
              frames = 0
              t0 = time.time()
         
         detection_to_track = {}
         track_to_detection = {}
         selected_face_index = None

         if use_tracking:
              detections = [(f.x1, f.y1, f.x2, f.y2) for f in faces]
              kps_list = [f.kps for f in faces]
              tracked_faces_dict = tracker.update(detections, kps_list=kps_list)

              for i, f in enumerate(faces):
                   det_bbox = (f.x1, f.y1, f.x2, f.y2)
                   det_centroid = ((f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2)
                   best_track_id = None
                   best_dist = float('inf')

                   for track_id, tracked in tracked_faces_dict.items():
                        track_centroid = tracked.centroid
                        dist = np.sqrt((det_centroid[0] - track_centroid[0])**2 + 
                                     (det_centroid[1] - track_centroid[1])**2)
                        iou = tracker._compute_iou(det_bbox, tracked.bbox)
                        score = (1.0 - iou) * 0.5 + (dist / 100.0) * 0.5
                        if score < best_dist and dist < 80:
                             best_dist = score
                             best_track_id = track_id

                   if best_track_id is not None:
                        detection_to_track[i] = best_track_id
                        track_to_detection[best_track_id] = i

         # Handle selection from click
         if clicked_detection_index is not None:
              if use_tracking and clicked_detection_index in detection_to_track:
                   selected_track_id = detection_to_track[clicked_detection_index]
                   f = faces[clicked_detection_index]
                   aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                   emb = embedder.embed(aligned)
                   mr = matcher.match(emb)
                   selected_face_name = mr.name if mr.accepted else "Unknown"
                   selected_embedding = emb
                   selected_face_was_present = True
                   print(f"Selected track {selected_track_id} ({selected_face_name})")
              clicked_detection_index = None

         # Identify index of selected face
         if selected_track_id is not None and selected_track_id in track_to_detection:
              selected_face_index = track_to_detection[selected_track_id]

         # Recognition Loop
         y0 = 80
         shown = 0
         
         has_faces = len(faces) > 0
         has_locked_face = False
         primary_locked_face_center = None

         for i, f in enumerate(faces):
              aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
              emb = embedder.embed(aligned)
              mr = matcher.match(emb)
              
              locked_name = lock_manager.check_and_lock_by_embedding(emb)
              
              if locked_name:
                   has_locked_face = True
                   if primary_locked_face_center is None:
                        primary_locked_face_center = ((f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2)
                   
                   activity_logger.log_movement(locked_name, (f.x1+f.x2)/2, (f.y1+f.y2)/2)
                   if detect_smile_simple(f):
                        activity_logger.log_expression(locked_name, "smile")

              # Update tracker identity
              if use_tracking and i in detection_to_track:
                   track_id = detection_to_track[i]
                   if track_id in tracked_faces_dict:
                        tracked_faces_dict[track_id].update_identity(
                             mr.name if mr.accepted else None, 
                             mr.distance, 
                             mr.similarity, 
                             embedding=emb
                        )

              # Draw
              color = (0, 255, 0) if mr.accepted else (0, 0, 255)
              display_name = mr.name if mr.accepted else "Unknown"
              
              if i == selected_face_index:
                   cv2.rectangle(vis, (f.x1-3, f.y1-3), (f.x2+3, f.y2+3), (255, 255, 0), 3)

              cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
              for (x, y) in f.kps.astype(int):
                   cv2.circle(vis, (int(x), int(y)), 2, color, -1)
              
              caption = f"{display_name}"
              if locked_name: caption += " [LOCKED]"
              if i == selected_face_index: caption += " [SEL]"
              cv2.putText(vis, caption, (f.x1, max(0, f.y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
              
              # Thumbnail
              if y0 + thumb <= h and shown < 4:
                   vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                   cv2.putText(vis, f"{i+1}:{display_name}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                   y0 += thumb + pad
                   shown += 1

         # MQTT & Servo Logic
         current_time = time.time()
         if current_time - last_heartbeat > heartbeat_interval:
              mqtt_manager.publish_heartbeat()
              last_heartbeat = current_time

         if has_locked_face and primary_locked_face_center:
              cx, cy = primary_locked_face_center
              center_x = w / 2
              deadzone_x = w * 0.15 
              
              status = "CENTER"
              if cx < center_x - deadzone_x:
                   status = "MOVE_LEFT"
              elif cx > center_x + deadzone_x:
                   status = "MOVE_RIGHT"
              
              mqtt_manager.publish_movement(status, confidence=1.0, face_name=locked_name)
              cv2.putText(vis, f"SERVO: {status}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
              
         elif has_faces:
              mqtt_manager.publish_movement("NO_LOCK", confidence=0.0)
         else:
              mqtt_manager.publish_movement("NO_FACE", confidence=0.0)

         cv2.imshow("recognize_new", vis)
         key = cv2.waitKey(1) & 0xFF
         
         if key == ord('q'): 
              break
         elif key == ord('r'): 
              matcher.reload_from(db_path)
              print(f"[recognize] reloaded DB")
         elif key == ord('l'): 
              # Lock selected face
              if selected_face_index is not None and selected_face_index < len(faces):
                   f = faces[selected_face_index]
                   aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                   emb = embedder.embed(aligned)
                   mr = matcher.match(emb)
                   if mr.accepted:
                        lock_manager.lock_face(mr.name, emb, activity_logger)
                        print(f"[LockManager] LOCKED: {mr.name}")
                   else:
                        print(f"[LockManager] Cannot lock unknown/unselected face.")
              else:
                   print("[LockManager] Select a detected face to lock.")
         elif key == ord('u'):
              # Unlock selected face
              if selected_face_index is not None and selected_face_index < len(faces):
                   f = faces[selected_face_index]
                   aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                   emb = embedder.embed(aligned)
                   mr = matcher.match(emb)
                   if mr.accepted and lock_manager.is_locked(mr.name):
                        lock_manager.unlock_face(mr.name, activity_logger)
                        print(f"[LockManager] UNLOCKED: {mr.name}")
         elif key == ord('c'):
              lock_manager.clear_all_locks()
              print("[LockManager] Cleared all locks")
         elif key == ord('L'):
              lock_manager.reload_from_disk()
         elif key == ord('d'):
              show_debug = not show_debug
         elif key == ord('t'):
              use_tracking = not use_tracking
              if not use_tracking: 
                   tracker.clear()
                   selected_track_id = None

    cap.release()
    cv2.destroyAllWindows()
    mqtt_manager.stop()

if __name__ == "__main__":
    main()