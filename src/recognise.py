
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
<<<<<<< HEAD
from .tracker import FaceTracker, draw_tracked_face
from .mqtt_manager import MQTTManager

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
               print("[embed] input:", self.sess.get.inputs()[0].shape, self.sess.get.inputs()[0].type)
               print("[embed] output:", self.sess.get.outputs()[0].shape, self.sess.get.outputs()[0].type)

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


# -------------------------
# Demo
# -------------------------
=======
from .tracker import FaceTracker
from .mqtt_manager import MQTTManager
>>>>>>> parent of 8662f0b (auto rotate for face search)

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

<<<<<<< HEAD
     cap = cv2.VideoCapture(1)
     if not cap.isOpened():
          raise RuntimeError("Camera not available")
     
     print("Recognize (multi-face with tracking). q=quit, r=reload DB, +/- threshold, d=debug overlay, t=toggle tracking")
=======
    # Initialize MQTT Manager
    team_id = "Phoenix_Team" # Change as needed
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
>>>>>>> parent of 8662f0b (auto rotate for face search)

    cv2.namedWindow("FaceLock")
    cv2.setMouseCallback("FaceLock", on_mouse_click_simple)

    while True:
         if not cap.isOpened(): break
         ok, frame = cap.read()
         if not ok: break

         h, w = frame.shape[:2]

         faces = det.detect(frame, max_faces=5)
         vis = frame.copy()

<<<<<<< HEAD
     mqtt_manager = MQTTManager()
     servo_angle = 0.0  # Current angle (can be any value, not limited to 180)
     servo_target_angle = 0.0
     servo_speed = 2.0  # degrees per update (smoothness)
     searching = False
     search_direction = 1  # 1 for right, -1 for left
     search_pause = 0
     search_pause_frames = 20  # Pause for a moment when changing direction

     while True:
          ok, frame = cap.read()
          if not ok:
               break
=======
         # Check for click
         if last_click_pos:
              cx, cy = last_click_pos
              for i, f in enumerate(faces):
                   if f.x1 <= cx <= f.x2 and f.y1 <= cy <= f.y2:
                        clicked_detection_index = i
                        break
              last_click_pos = None
>>>>>>> parent of 8662f0b (auto rotate for face search)

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

<<<<<<< HEAD
          # Update tracker with detections
          detection_to_track = {}  # Ensure this is initialized every frame
          if use_tracking:
               # Prepare detections for tracker
               detections = [(f.x1, f.y1, f.x2, f.y2) for f in faces]
               kps_list = [f.kps for f in faces]
               # Update tracker
               tracked_faces_dict = tracker.update(detections, kps_list=kps_list)
               # Map detections to tracked faces for recognition
               for i, f in enumerate(faces):
                    det_bbox = (f.x1, f.y1, f.x2, f.y2)
                    det_centroid = ((f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2)
                    best_track_id = None
                    best_dist = float('inf')
                    for track_id, tracked in tracked_faces_dict.items():
                         track_centroid = tracked.centroid
                         dist = np.sqrt((det_centroid[0] - track_centroid[0])**2 + (det_centroid[1] - track_centroid[1])**2)
                         iou = tracker._compute_iou(det_bbox, tracked.bbox)
                         score = (1.0 - iou) * 0.5 + (dist / 100.0) * 0.5
                         if score < best_dist and dist < 80:
                              best_dist = score
                              best_track_id = track_id
                    if best_track_id is not None:
                         detection_to_track[i] = best_track_id
          else:
               tracked_faces_dict = {}
=======
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
>>>>>>> parent of 8662f0b (auto rotate for face search)

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

         cv2.imshow("FaceLock", vis)
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