from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords

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
