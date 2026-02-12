"""
Face locking system for persistent face tracking.
Maintains locked faces across camera frames and sessions.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set
import numpy as np


@dataclass
class LockedFace:
    """Represents a locked face with its embedding and metadata."""
    name: str
    embedding: np.ndarray
    timestamp: float
    lock_duration: float  # seconds
    
    def is_expired(self, current_time: float) -> bool:
        """Check if the lock has expired."""
        return (current_time - self.timestamp) > self.lock_duration


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
        
    def lock_face(self, name: str, embedding: np.ndarray, logger=None) -> bool:
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
        
    def unlock_face(self, name: str, logger=None) -> bool:
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
        """Clear all locked faces."""
        self.locked_faces.clear()
        self._save_to_disk()
        
    def _save_to_disk(self):
        """Save locked faces to disk."""
        try:
            data = {}
            for name, locked_face in self.locked_faces.items():
                data[name] = {
                    'name': locked_face.name,
                    'embedding': locked_face.embedding.tolist(),
                    'timestamp': locked_face.timestamp,
                    'lock_duration': locked_face.lock_duration
                }
            
            self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[LockManager] Error saving locks: {e}")
            
    def _load_from_disk(self):
        """Load locked faces from disk."""
        try:
            if self.lock_file_path.exists():
                with open(self.lock_file_path, 'r') as f:
                    data = json.load(f)
                
                for name, face_data in data.items():
                    self.locked_faces[name] = LockedFace(
                        name=face_data['name'],
                        embedding=np.array(face_data['embedding']),
                        timestamp=face_data['timestamp'],
                        lock_duration=face_data['lock_duration']
                    )
                
                print(f"[LockManager] Loaded {len(self.locked_faces)} locked faces from disk")
        except Exception as e:
            print(f"[LockManager] Error loading locks: {e}")
            
    def reload_from_disk(self):
        """Reload locked faces from disk."""
        self.locked_faces.clear()
        self._load_from_disk()
        print(f"[LockManager] Reloaded {len(self.locked_faces)} locked faces")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance between two vectors."""
    return 1.0 - np.dot(a, b)
