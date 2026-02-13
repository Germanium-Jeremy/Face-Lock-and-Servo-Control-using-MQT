import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .types import FaceDet, MatchResult

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out

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
