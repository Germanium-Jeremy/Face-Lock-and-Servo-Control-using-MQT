import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional
from .types import LockedFace
from .utils import cosine_distance
from .logger import ActivityLogger

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
        
        # print(f"[DEBUG] LockManager: Checking {len(self.locked_faces)} locked faces")
        
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
            # print(f"[DEBUG] LockManager: Comparing with locked '{name}', distance={distance:.3f}, threshold={self.match_threshold}")
            if distance <= self.match_threshold:
                print(f"[DEBUG] LockManager: Match found! Returning locked name: {name}")
                # Only return the locked face name, don't auto-lock similar faces
                return name
        
        # print(f"[DEBUG] LockManager: No matches found")
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
