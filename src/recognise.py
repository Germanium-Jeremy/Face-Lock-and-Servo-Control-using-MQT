# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.
Run:
python -m src.recognize
Keys:
q : quit
r : reload DB from disk (data/db/face_db.npz)
+/- : adjust threshold (distance) live
d : toggle debug overlay
t : toggle tracking
Locking: l=lock face, u=unlock face, c=clear all locks, L=reload locks
Notes:
- We run FaceMesh on EACH Haar face ROI (not the full frame). This avoids the
"FaceMesh points not consistent with Haar box" problem and enables multi-face.
- DB is expected from enroll: data/db/face_db.npz (name -> embedding vector)
- Distance definition: cosine_distance = 1 - cosine_similarity.
Since embeddings are L2-normalized, cosine_similarity = dot(a,b).
- PERSISTENT LOCKING: Locked faces are stored to disk and automatically
re-locked when they reappear in the camera frame.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
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

# Reuse your known-good alignment method
from .haar_5pt import align_face_5pt
from .tracker import FaceTracker, draw_tracked_face


# -------------------------
# Data
# -------------------------

@dataclass
class FaceDet:
     x1: int
     y1: int
     x2: int
     y2: int
     score: float
     kps: np.ndarray # (5,2) float32 in FULL-frame coords

@dataclass
class MatchResult:
     name: Optional[str]
     distance: float
     similarity: float
     accepted: bool

@dataclass
class LockedFace:
     name: str
     embedding: np.ndarray
     timestamp: float
     lock_duration: float = 300.0  # 5 minutes default
     
     def is_expired(self, current_time: float) -> bool:
         return current_time - self.timestamp > self.lock_duration


# -------------------------
# Math helpers
# -------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
     a = a.reshape(-1).astype(np.float32)
     b = b.reshape(-1).astype(np.float32)
     return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
     return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
     x1 = int(max(0, min(W - 1, round(x1))))
     y1 = int(max(0, min(H - 1, round(y1))))
     x2 = int(max(0, min(W - 1, round(x2))))
     y2 = int(max(0, min(H - 1, round(y2))))
     if x2 < x1:
          x1, x2 = x2, x1
     if y2 < y1:
          y1, y2 = y2, y1
     return x1, y1, x2, y2

def _bbox_from_5pt(
     kps: np.ndarray,
     pad_x: float = 0.55,
     pad_y_top: float = 0.85,
     pad_y_bot: float = 1.15,
) -> np.ndarray:
     """
     Build a nicer face-like bbox from 5 points with asymmetric padding.
     kps: (5,2) in full-frame coords
     """

     k = kps.astype(np.float32)
     x_min = float(np.min(k[:, 0]))
     x_max = float(np.max(k[:, 0]))
     y_min = float(np.min(k[:, 1]))
     y_max = float(np.max(k[:, 1]))
     w = max(1.0, x_max - x_min)
     h = max(1.0, y_max - y_min)
     x1 = x_min - pad_x * w
     x2 = x_max + pad_x * w
     y1 = y_min - pad_y_top * h
     y2 = y_max + pad_y_bot * h
     return np.array([x1, y1, x2, y2], dtype=np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
     """
     Minimal geometry sanity:
     - eyes not collapsed
     - mouth generally below nose
     """
     k = kps.astype(np.float32)
     le, re, no, lm, rm = k
     eye_dist = float(np.linalg.norm(re - le))
     if eye_dist < float(min_eye_dist):
          return False
     if not (lm[1] > no[1] and rm[1] > no[1]):
          return False
     return True


# -------------------------
# DB helpers
# -------------------------

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
     if not db_path.exists():
          return {}
     data = np.load(str(db_path), allow_pickle=True)
     out: Dict[str, np.ndarray] = {}
     for k in data.files:
          out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
     return out


# -------------------------
# Embedder (same as embed_new)
# -------------------------

class ArcFaceEmbedderONNX:
     """
     ArcFace-style ONNX embedder.
     Input: 112x112 BGR -> internally RGB + (x-127.5)/128, NCHW float32.
     Output: (1,D) or (D,)
     """
     def __init__(
          self,
          model_path: str = "models/embedder_arcface.onnx",
          input_size: Tuple[int, int] = (112, 112),
          debug: bool = False,
     ):
          self.model_path = model_path
          self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
          self.debug = bool(debug)
          self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
          self.in_name = self.sess.get_inputs()[0].name
          self.out_name = self.sess.get_outputs()[0].name

          if self.debug:
               print("[embed] model:", model_path)
               print("[embed] input:", self.sess.get_inputs()[0].name, self.sess.get_inputs()[0].shape, self.sess.get_inputs()[0].type)
               print("[embed] output:", self.sess.get_outputs()[0].name, self.sess.get_outputs()[0].shape, self.sess.get_outputs()[0].type)

     def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
          img = aligned_bgr_112
          if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
               img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)

          rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
          rgb = (rgb - 127.5) / 128.0
          x = np.transpose(rgb, (2, 0, 1))[None, ...]
          return x.astype(np.float32)
     
     @staticmethod
     def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
          v = v.astype(np.float32).reshape(-1)
          n = float(np.linalg.norm(v) + eps)
          return (v / n).astype(np.float32)
     
     def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
          x = self._preprocess(aligned_bgr_112)
          y = self.sess.run([self.out_name], {self.in_name: x})[0]
          emb = np.asarray(y, dtype=np.float32).reshape(-1)
          return self._l2_normalize(emb)


# -------------------------
# Multi-face Haar + FaceMesh(ROI) 5pt
# -------------------------

class HaarFaceMesh5pt:
     def __init__(
          self,
          haar_xml: Optional[str] = None,
          min_size: Tuple[int, int] = (70, 70),
          debug: bool = False,
     ):
          self.debug = bool(debug)
          self.min_size = tuple(map(int, min_size))

          if haar_xml is None:
               haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

          self.face_cascade = cv2.CascadeClassifier(haar_xml)
          if self.face_cascade.empty():
               raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")
          
          if mp is None:
               raise RuntimeError(f"mediapipe import failed: {_MP_IMPORT_ERROR}\n Install: pip install mediapipe==0.10.21")
          
          # Use MediaPipe Tasks API (same as haar_5pt.py)
          MODEL_PATH = Path(__file__).resolve().parent.parent / "face_landmarker.task"
          
          options = vision.FaceLandmarkerOptions(
               base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
               num_faces=1,  # IMPORTANT: we run FaceMesh on ROI (one face per ROI)
               output_face_blendshapes=False,
               output_facial_transformation_matrixes=False,
          )
          
          self.landmarker = vision.FaceLandmarker.create_from_options(options)
          
          # 5pt indices (same as your working file)
          self.IDX_LEFT_EYE = 33
          self.IDX_RIGHT_EYE = 263
          self.IDX_NOSE_TIP = 1
          self.IDX_MOUTH_LEFT = 61
          self.IDX_MOUTH_RIGHT = 291

     def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
          faces = self.face_cascade.detectMultiScale(
               gray,
               scaleFactor=1.1,
               minNeighbors=5,
               flags=cv2.CASCADE_SCALE_IMAGE,
               minSize=self.min_size,
          )
          if faces is None or len(faces) == 0:
               return np.zeros((0, 4), dtype=np.int32)
          
          return faces.astype(np.int32) # (x,y,w,h)
     
     def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
          H, W = roi_bgr.shape[:2]
          if H < 20 or W < 20:
               return None

          # Convert to RGB and create MediaPipe Image
          rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
          mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
          
          # Detect landmarks using Tasks API
          res = self.landmarker.detect(mp_image)
          
          if not res.face_landmarks:
               return None
          
          # Extract 5 keypoints
          lm = res.face_landmarks[0]
          idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]

          pts = []
          for i in idxs:
               p = lm[i]
               pts.append([p.x * W, p.y * H])

          kps = np.array(pts, dtype=np.float32)

          # enforce left/right ordering
          if kps[0, 0] > kps[1, 0]:
               kps[[0, 1]] = kps[[1, 0]]
          if kps[3, 0] > kps[4, 0]:
               kps[[3, 4]] = kps[[4, 3]]
          return kps
     
     def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
          H, W = frame_bgr.shape[:2]
          gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

          faces = self._haar_faces(gray)
          if faces.shape[0] == 0:
               return []
          
          # sort by area desc, keep top max_faces
          areas = faces[:, 2] * faces[:, 3]
          order = np.argsort(areas)[::-1]
          faces = faces[order][:max_faces]

          out: List[FaceDet] = []

          for (x, y, w, h) in faces:
               # expand ROI a bit for FaceMesh stability
               mx, my = 0.25 * w, 0.35 * h
               rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
               roi = frame_bgr[ry1:ry2, rx1:rx2]

               kps_roi = self._roi_facemesh_5pt(roi)
               if kps_roi is None:
                    if self.debug:
                         print("[recognize] FaceMesh none for ROI -> skip")
                    continue

               # map ROI kps back to full-frame coords
               kps = kps_roi.copy()
               kps[:, 0] += float(rx1)
               kps[:, 1] += float(ry1)

               # sanity: eye distance relative to Haar width
               if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))):
                    if self.debug:
                         print("[recognize] 5pt geometry failed -> skip")
                    continue

               # build bbox from kps (centered)
               bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
               x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)

               out.append(
                    FaceDet(
                         x1=x1, y1=y1, x2=x2, y2=y2,
                         score=1.0,
                         kps=kps.astype(np.float32),
                    )
               )
          return out
    

# -------------------------
# Matcher
# -------------------------

class FaceDBMatcher:
     def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
          self.db = db
          self.dist_thresh = float(dist_thresh)
          # pre-stack for speed
          self._names: List[str] = []
          self._mat: Optional[np.ndarray] = None
          self._rebuild()
          
     def _rebuild(self):
          self._names = sorted(self.db.keys())
          if self._names:
               self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
               
          # (K,D)
          else:
               self._mat = None

     def reload_from(self, path: Path):
          self.db = load_db_npz(path)
          self._rebuild()

     def match(self, emb: np.ndarray) -> MatchResult:
          if self._mat is None or len(self._names) == 0:
               return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
          e = emb.reshape(1, -1).astype(np.float32) # (1,D)

          # cosine similarity since both sides are normalized: sim = dot
          sims = (self._mat @ e.T).reshape(-1) # (K,)
          best_i = int(np.argmax(sims))
          best_sim = float(sims[best_i])
          best_dist = 1.0 - best_sim
          ok = best_dist <= self.dist_thresh

          return MatchResult(
               name=self._names[best_i] if ok else None,
               distance=float(best_dist),
               similarity=float(best_sim),
               accepted=bool(ok),
          )


def detect_smile_simple(f: FaceDet) -> bool:
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

# -------------------------
# Activity Logger for Locked Persons
# -------------------------

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
          
          print(f"[DEBUG] Attempting to log: {log_entry.strip()}")
          print(f"[DEBUG] Log file path: {self.log_file_path}")
          
          try:
               with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
               print(f"[DEBUG] Successfully wrote to log file")
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
          print(f"[DEBUG] Checking movement for {person_name} at ({current_x:.1f}, {current_y:.1f})")
          movement = self.detect_movement(person_name, current_x, current_y)
          if movement:
               print(f"[DEBUG] Movement detected: {movement}")
               self.log_activity(person_name, movement)
          else:
               print(f"[DEBUG] No significant movement detected")
     
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

# -------------------------
# Face Lock Manager
# -------------------------

class FaceLockManager:
     """
     Manages persistent face locking across camera frames.
     Maintains a database of locked faces with their embeddings.
     """
     def __init__(self, lock_duration: float = 300.0, match_threshold: float = 0.3):
          self.lock_duration = lock_duration  # seconds
          self.match_threshold = match_threshold  # cosine distance threshold
          self.locked_faces: Dict[str, LockedFace] = {}  # name -> LockedFace
          self.lock_file_path = Path("data/locked_faces.json")
          
     def lock_face(self, name: str, embedding: np.ndarray, logger: Optional[ActivityLogger] = None) -> bool:
          """Lock a face by name and embedding."""
          print(f"[DEBUG] LockManager: Attempting to lock face '{name}'")
          current_time = time.time()
          self.locked_faces[name] = LockedFace(
               name=name,
               embedding=embedding.copy(),
               timestamp=current_time,
               lock_duration=self.lock_duration
          )
          self._save_to_disk()
          
          print(f"[DEBUG] LockManager: Successfully locked face '{name}'. Total locked faces: {len(self.locked_faces)}")
          
          # Log the locking action
          if logger:
               logger.log_activity(name, "was locked")
          
          return True
          
     def unlock_face(self, name: str, logger: Optional[ActivityLogger] = None) -> bool:
          """Unlock a face by name."""
          if name in self.locked_faces:
               del self.locked_faces[name]
               self._save_to_disk()
               
               # Log the unlocking action
               if logger:
                    logger.log_activity(name, "was unlocked")
               
               return True
          return False
          
     def is_locked(self, name: str) -> bool:
          """Check if a face is currently locked."""
          if name not in self.locked_faces:
               return False
          current_time = time.time()
          if self.locked_faces[name].is_expired(current_time):
               del self.locked_faces[name]
               self._save_to_disk()
               return False
          return True
          
     def check_and_lock_by_embedding(self, embedding: np.ndarray) -> Optional[str]:
          """
          Check if the embedding matches any locked face.
          If matched, returns the locked face name.
          """
          current_time = time.time()
          
          print(f"[DEBUG] LockManager: Checking {len(self.locked_faces)} locked faces")
          
          # Remove expired locks
          expired_names = []
          for name, locked_face in self.locked_faces.items():
               if locked_face.is_expired(current_time):
                    expired_names.append(name)
          
          for name in expired_names:
               del self.locked_faces[name]
          
          if expired_names:
               self._save_to_disk()
          
          # Check for matches
          for name, locked_face in self.locked_faces.items():
               distance = cosine_distance(embedding, locked_face.embedding)
               print(f"[DEBUG] LockManager: Comparing with locked '{name}', distance={distance:.3f}, threshold={self.match_threshold}")
               if distance <= self.match_threshold:
                    print(f"[DEBUG] LockManager: Match found! Returning locked name: {name}")
                    # Only return the locked face name, don't auto-lock similar faces
                    return name
          
          print(f"[DEBUG] LockManager: No matches found")
          return None
          
     def get_locked_names(self) -> Set[str]:
          """Get set of currently locked face names."""
          current_time = time.time()
          locked_names = set()
          expired_names = []
          
          for name, locked_face in self.locked_faces.items():
               if locked_face.is_expired(current_time):
                    expired_names.append(name)
               else:
                    locked_names.add(name)
          
          for name in expired_names:
               del self.locked_faces[name]
          
          if expired_names:
               self._save_to_disk()
          
          return locked_names
          
     def clear_all_locks(self):
          """Clear all face locks."""
          self.locked_faces.clear()
          self._save_to_disk()
          
     def _save_to_disk(self):
          """Save locked faces to disk."""
          try:
               self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
               data = {}
               for name, locked_face in self.locked_faces.items():
                    data[name] = {
                         'name': locked_face.name,
                         'embedding': locked_face.embedding.tolist(),
                         'timestamp': locked_face.timestamp,
                         'lock_duration': locked_face.lock_duration
                    }
               with open(self.lock_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
          except Exception as e:
               print(f"[LockManager] Error saving to disk: {e}")
               
     def _load_from_disk(self):
          """Load locked faces from disk."""
          if not self.lock_file_path.exists():
               return
               
          try:
               with open(self.lock_file_path, 'r') as f:
                    data = json.load(f)
               
               current_time = time.time()
               for name, face_data in data.items():
                    locked_face = LockedFace(
                         name=face_data['name'],
                         embedding=np.array(face_data['embedding'], dtype=np.float32),
                         timestamp=face_data['timestamp'],
                         lock_duration=face_data.get('lock_duration', self.lock_duration)
                    )
                    # Only load if not expired
                    if not locked_face.is_expired(current_time):
                         self.locked_faces[name] = locked_face
          except Exception as e:
               print(f"[LockManager] Error loading from disk: {e}")
               
     def reload_from_disk(self):
          """Reload locked faces from disk."""
          self.locked_faces.clear()
          self._load_from_disk()
          print(f"[LockManager] Reloaded {len(self.locked_faces)} locked faces")


# -------------------------
# Demo
# -------------------------

def main():
     db_path = Path("data/db/face_db.npz")
     det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
     embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)
     
     db = load_db_npz(db_path)
     matcher = FaceDBMatcher(db=db, dist_thresh=0.34) # from your evaluate_new output
     
     # Initialize face lock manager
     lock_manager = FaceLockManager(lock_duration=300.0, match_threshold=0.3)
     lock_manager._load_from_disk()  # Load existing locks
     
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

     cap = cv2.VideoCapture(2)
     if not cap.isOpened():
          raise RuntimeError("Camera not available")
     
     print("Recognize (multi-face with tracking & persistent locking). q=quit, r=reload DB, +/- threshold, d=debug overlay, t=toggle tracking")
     print("Locking: l=lock face, u=unlock face, c=clear all locks, L=reload locks")
     print("Click on a face to select it, then press 'l' to lock it")

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
     re_acquire_threshold = 0.25  # Tighter threshold for re-acquiring selected face on return
     notification_text = ""
     notification_timer = 0
     
     # For mouse click: temporarily store the clicked detection index
     clicked_detection_index: Optional[int] = None
     
     def on_mouse_click(event, x, y, flags, param):
          nonlocal clicked_detection_index
          if event == cv2.EVENT_LBUTTONDOWN:
               # Check if click is on any detected face
               for i, f in enumerate(faces):
                    if f.x1 <= x <= f.x2 and f.y1 <= y <= f.y2:
                         clicked_detection_index = i
                         break

     cv2.namedWindow("recognize_new")
     cv2.setMouseCallback("recognize_new", on_mouse_click)

     while True:
          ok, frame = cap.read()
          if not ok:
               break

          h, w = frame.shape[:2]  # Initialize frame dimensions immediately after reading the frame

          faces = det.detect(frame, max_faces=5)
          vis = frame.copy()

          # compute fps
          frames += 1
          dt = time.time() - t0
          if dt >= 1.0:
               fps = frames / dt
               frames = 0
               t0 = time.time()

          detection_to_track = {}  # det_index -> track_id
          track_to_detection = {}  # track_id -> det_index
          selected_face_index = None  # Current frame's det index for selected

          # Update tracker with detections
          if use_tracking:
               # Prepare detections for tracker
               detections = [(f.x1, f.y1, f.x2, f.y2) for f in faces]
               kps_list = [f.kps for f in faces]

               # Update tracker
               tracked_faces_dict = tracker.update(detections, kps_list=kps_list)

               # Map detections to tracked faces
               for i, f in enumerate(faces):
                    det_bbox = (f.x1, f.y1, f.x2, f.y2)
                    det_centroid = ((f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2)

                    # Find closest tracked face
                    best_track_id = None
                    best_dist = float('inf')

                    for track_id, tracked in tracked_faces_dict.items():
                         # Check distance and IoU
                         track_centroid = tracked.centroid
                         dist = np.sqrt((det_centroid[0] - track_centroid[0])**2 + 
                                      (det_centroid[1] - track_centroid[1])**2)

                         iou = tracker._compute_iou(det_bbox, tracked.bbox)

                         # Combined score
                         score = (1.0 - iou) * 0.5 + (dist / 100.0) * 0.5

                         if score < best_dist and dist < 80:
                              best_dist = score
                              best_track_id = track_id

                    if best_track_id is not None:
                         detection_to_track[i] = best_track_id
                         track_to_detection[best_track_id] = i

          else:
               # If no tracking, can't reliably select by track_id, fallback or disable selection
               print("[Warning] Selection requires tracking enabled")
               selected_track_id = None

          # Handle mouse click: assign selected_track_id if clicked
          if clicked_detection_index is not None:
               if use_tracking and clicked_detection_index in detection_to_track:
                    selected_track_id = detection_to_track[clicked_detection_index]
                    # Compute name and embedding for selected
                    f = faces[clicked_detection_index]
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    mr = matcher.match(emb)
                    selected_face_name = mr.name if mr.accepted else "Unknown"
                    selected_embedding = emb
                    selected_face_was_present = True
                    print(f"Selected track {selected_track_id} ({selected_face_name})")
               clicked_detection_index = None

          # If selected_track_id, find current det index
          if selected_track_id is not None and selected_track_id in track_to_detection:
               selected_face_index = track_to_detection[selected_track_id]

          # Handle re-acquire if selected disappeared
          selected_face_present = selected_track_id is not None and selected_track_id in tracked_faces_dict

          if not selected_face_present and selected_embedding is not None:
               # Try to re-acquire by embedding similarity
               for i, f in enumerate(faces):
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    dist = cosine_distance(emb, selected_embedding)
                    if dist <= re_acquire_threshold:
                         if use_tracking and i in detection_to_track:
                              selected_track_id = detection_to_track[i]
                              selected_face_present = True
                              selected_face_index = i
                              print(f"Re-acquired selected face on track {selected_track_id} (dist={dist:.3f})")
                              break

          # Check if selected face left or returned
          if selected_face_was_present and not selected_face_present:
               # Face just left
               notification_text = f"TARGET {selected_face_name} LEFT"
               notification_timer = 120  # Show for 2 seconds at 60fps
               frames_since_selected_left = 0
               print(f"[Target Tracking] {selected_face_name} left camera view")
               # Log activity if this person is locked
               if lock_manager.is_locked(selected_face_name):
                    activity_logger.log_presence_change(selected_face_name, False)
          elif not selected_face_was_present and selected_face_present:
               # Face just returned
               notification_text = f"TARGET {selected_face_name} LOCKED"
               notification_timer = 120  # Show for 2 seconds at 60fps
               print(f"[Target Tracking] {selected_face_name} returned to camera view")
               # Log activity if this person is locked
               if lock_manager.is_locked(selected_face_name):
                    activity_logger.log_presence_change(selected_face_name, True)

          selected_face_was_present = selected_face_present

          # Update notification timer
          if notification_timer > 0:
               notification_timer -= 1

          # Recognition loop
          y0 = 80  # Reset y0 each frame
          shown = 0
          for i, f in enumerate(faces):
               # align -> embed -> match
               aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
               emb = embedder.embed(aligned)
               mr = matcher.match(emb)
               
               # Check if this face matches any locked face (persistent locking)
               locked_name = lock_manager.check_and_lock_by_embedding(emb)
               
               print(f"[DEBUG] Face {i}: mr.name={mr.name if mr.accepted else 'Unknown'}, locked_name={locked_name}")
               
               # Log activities for locked persons
               if locked_name:
                    print(f"[DEBUG] Locked person detected: {locked_name}")
                    # Get face center position for movement tracking
                    face_center_x = (f.x1 + f.x2) / 2
                    face_center_y = (f.y1 + f.y2) / 2
                    
                    # Log movement
                    activity_logger.log_movement(locked_name, face_center_x, face_center_y)
                    
                    # Check for smile
                    if detect_smile_simple(f):
                         activity_logger.log_expression(locked_name, "smile")
               else:
                    print(f"[DEBUG] No locked person detected for face {i}")
               
               # Update tracked face with recognition results
               if use_tracking and i in detection_to_track:
                    track_id = detection_to_track[i]
                    if track_id in tracked_faces_dict:
                         tracked = tracked_faces_dict[track_id]
                         tracker.update_identity(
                              track_id,
                              mr.name if mr.accepted else None,
                              mr.distance,
                              mr.similarity,
                              embedding=emb,
                         )
                         tracked.kps = f.kps

               # Color
               if mr.accepted:
                    color = (0, 255, 0)
                    display_name = mr.name
               else:
                    color = (0, 0, 255)
                    display_name = "Unknown"

               # Highlight selected
               if i == selected_face_index:
                    cv2.rectangle(vis, (f.x1-3, f.y1-3), (f.x2+3, f.y2+3), (255, 255, 0), 3)

               # Draw bbox and kps
               cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
               for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, color, -1)

               # Label
               line1 = f"{display_name}"
               if locked_name:
                    line1 += " [LOCKED]"
               if i == selected_face_index:
                    line1 += " [SELECTED]"
               line2 = f"dist={mr.distance:.3f} sim={mr.similarity:.3f}"
               cv2.putText(vis, line1, (f.x1, max(0, f.y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
               cv2.putText(vis, line2, (f.x1, max(0, f.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

               # Thumbnail
               if y0 + thumb <= h and shown < 4:
                    vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                    cv2.putText(vis, f"{i+1}:{display_name}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    y0 += thumb + pad
                    shown += 1

          # Header
          locked_names = lock_manager.get_locked_names()
          header = f"IDs={len(matcher._names)} thr(dist)={matcher.dist_thresh:.2f}"
          if fps is not None:
               header += f" fps={fps:.1f}"
          if use_tracking:
               header += f" tracks={len(tracked_faces_dict)}"
          if locked_names:
               header += f" locked={len(locked_names)}"

          cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

          # Notification
          if notification_timer > 0:
               text_size = cv2.getTextSize(notification_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
               text_x = (w - text_size[0]) // 2
               text_y = h - 50
               cv2.rectangle(vis, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
               cv2.putText(vis, notification_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

          cv2.imshow("recognize_new", vis)
          key = cv2.waitKey(1) & 0xFF

          if key == ord("q"):
               break
          elif key == ord("r"):
               matcher.reload_from(db_path)
               print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
          elif key in (ord("+"), ord("=")):
               matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
               print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
          elif key == ord("-"):
               matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
               print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
          elif key == ord("d"):
               show_debug = not show_debug
               print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")
          elif key == ord("t"):
               use_tracking = not use_tracking
               if not use_tracking:
                    tracker.clear()
                    selected_track_id = None
               print(f"[recognize] tracking: {'ON' if use_tracking else 'OFF'}")
          elif key == ord("l"):
               # Lock selected face
               print(f"[DEBUG] Lock key 'l' pressed!")
               if selected_face_index is not None and selected_face_index < len(faces):
                    f = faces[selected_face_index]
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    mr = matcher.match(emb)
                    if mr.accepted:
                         lock_manager.lock_face(mr.name, emb, activity_logger)
                         print(f"[LockManager] Locked face: {mr.name}")
                    else:
                         print(f"[LockManager] Cannot lock unknown face. Face must be recognized first.")
               else:
                    print("[LockManager] Please select a recognized face first (click on it)")
          elif key == ord("1"):
               print(f"[DEBUG] Number '1' key pressed - this is not the lock key!")
          elif key == ord("u"):
               # Unlock selected face
               if selected_face_index is not None and selected_face_index < len(faces):
                    f = faces[selected_face_index]
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    mr = matcher.match(emb)
                    if mr.accepted and lock_manager.is_locked(mr.name):
                         lock_manager.unlock_face(mr.name, activity_logger)
                         print(f"[LockManager] Unlocked face: {mr.name}")
                    else:
                         print(f"[LockManager] Face {mr.name if mr.accepted else 'Unknown'} is not locked")
               else:
                    print("[LockManager] Please select a face first (click on it)")
          elif key == ord("c"):
               lock_manager.clear_all_locks()
               print("[LockManager] Cleared all locks")
          elif key == ord("L"):
               lock_manager.reload_from_disk()

          # If selected disappeared and not re-acquired, clear selection
          if not selected_face_present:
               selected_track_id = None

     cap.release()
     cv2.destroyAllWindows()


if __name__ == "__main__":
     main()