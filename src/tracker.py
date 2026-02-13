"""
Face Tracker Module
Implements centroid-based tracking to follow faces across frames.
Tracks bounding boxes and maintains identity associations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class TrackedFace:
    """Represents a tracked face with its history."""
    track_id: int
    bbox: Tuple[int, int, it, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]  # (cx, cy)
    age: int  # frames since first detection
    hits: int  # number of successful matches
    time_since_update: int  # frames since last update
    confidence: float  # tracking confidence
    velocity: Tuple[float, float]  # (vx, vy) estimated velocity
    kps: Optional[np.ndarray] = None  # (5, 2) keypoints if available
    embedding: Optional[np.ndarray] = None  # face embedding if available
    identity: Optional[str] = None  # recognized identity name
    match_distance: float = 1.0  # last match distance
    match_similarity: float = 0.0  # last match similarity

    def update_identity(self, identity: Optional[str], distance: float, similarity: float, embedding: Optional[np.ndarray] = None):
        self.identity = identity
        self.match_distance = distance
        self.match_similarity = similarity
        if embedding is not None:
            self.embedding = embedding


class FaceTracker:
    """
    Centroid-based face tracker.
    Associates detected faces across frames using IoU and centroid distance.
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,  # frames before removing track
        max_distance: float = 100.0,  # max centroid distance for matching
        iou_threshold: float = 0.3,  # IoU threshold for matching
        smooth_alpha: float = 0.7,  # smoothing factor for bbox updates
        velocity_alpha: float = 0.5,  # smoothing factor for velocity
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.smooth_alpha = smooth_alpha
        self.velocity_alpha = velocity_alpha
        
        self.next_id = 0
        self.tracked_faces: Dict[int, TrackedFace] = {}
        
    def _compute_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Compute centroid from bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _compute_distance(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def _predict_position(self, tracked: TrackedFace) -> Tuple[float, float]:
        """Predict next position using velocity."""
        cx, cy = tracked.centroid
        vx, vy = tracked.velocity
        return (cx + vx, cy + vy)
    
    def update(
        self,
        detections: List[Tuple[int, int, int, int]],  # List of (x1, y1, x2, y2) bboxes
        kps_list: Optional[List[np.ndarray]] = None,  # Optional keypoints for each detection
    ) -> Dict[int, TrackedFace]:
        """
        Update tracker with new detections.
        Returns dictionary mapping track_id to TrackedFace.
        """
        # If no detections, increment time_since_update for all tracks
        if len(detections) == 0:
            for track_id in list(self.tracked_faces.keys()):
                self.tracked_faces[track_id].time_since_update += 1
                if self.tracked_faces[track_id].time_since_update > self.max_disappeared:
                    del self.tracked_faces[track_id]
            return self.tracked_faces.copy()
        
        # Compute centroids for all detections
        detection_centroids = [self._compute_centroid(bbox) for bbox in detections]
        
        # If no existing tracks, create new ones
        if len(self.tracked_faces) == 0:
            for i, (bbox, centroid) in enumerate(zip(detections, detection_centroids)):
                track_id = self.next_id
                self.next_id += 1
                kps = kps_list[i] if kps_list and i < len(kps_list) else None
                self.tracked_faces[track_id] = TrackedFace(
                    track_id=track_id,
                    bbox=bbox,
                    centroid=centroid,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    confidence=1.0,
                    velocity=(0.0, 0.0),
                    kps=kps,
                )
            return self.tracked_faces.copy()
        
        # Match detections to existing tracks
        # Build cost matrix: rows = tracks, cols = detections
        track_ids = list(self.tracked_faces.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            tracked = self.tracked_faces[track_id]
            predicted_centroid = self._predict_position(tracked)
            
            for j, (det_bbox, det_centroid) in enumerate(zip(detections, detection_centroids)):
                # Combined cost: IoU + distance
                iou = self._compute_iou(tracked.bbox, det_bbox)
                distance = self._compute_distance(predicted_centroid, det_centroid)
                
                # Cost: lower is better
                # Use negative IoU (higher IoU = lower cost) + normalized distance
                iou_cost = 1.0 - iou  # 0 (perfect match) to 1 (no overlap)
                dist_cost = min(distance / self.max_distance, 1.0)  # normalized to [0, 1]
                
                # Combined cost (weighted)
                cost = 0.4 * iou_cost + 0.6 * dist_cost
                cost_matrix[i, j] = cost
        
        # Greedy matching (simple Hungarian-like)
        matched_tracks = set()
        matched_detections = set()
        matches = []
        
        # Sort by cost and match greedily
        match_candidates = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                match_candidates.append((cost_matrix[i, j], i, j))
        
        match_candidates.sort(key=lambda x: x[0])
        
        for cost, i, j in match_candidates:
            if i not in matched_tracks and j not in matched_detections:
                # Additional check: cost must be reasonable
                if cost < 0.8:  # threshold for acceptable match
                    matches.append((track_ids[i], j))
                    matched_tracks.add(i)
                    matched_detections.add(j)
        
        # Update matched tracks
        for track_id, det_idx in matches:
            tracked = self.tracked_faces[track_id]
            new_bbox = detections[det_idx]
            new_centroid = detection_centroids[det_idx]
            
            # Smooth bbox update
            old_bbox = tracked.bbox
            smoothed_bbox = (
                int(self.smooth_alpha * old_bbox[0] + (1 - self.smooth_alpha) * new_bbox[0]),
                int(self.smooth_alpha * old_bbox[1] + (1 - self.smooth_alpha) * new_bbox[1]),
                int(self.smooth_alpha * old_bbox[2] + (1 - self.smooth_alpha) * new_bbox[2]),
                int(self.smooth_alpha * old_bbox[3] + (1 - self.smooth_alpha) * new_bbox[3]),
            )
            
            # Update velocity
            old_centroid = tracked.centroid
            dx = new_centroid[0] - old_centroid[0]
            dy = new_centroid[1] - old_centroid[1]
            new_velocity = (
                self.velocity_alpha * tracked.velocity[0] + (1 - self.velocity_alpha) * dx,
                self.velocity_alpha * tracked.velocity[1] + (1 - self.velocity_alpha) * dy,
            )
            
            # Update track
            tracked.bbox = smoothed_bbox
            tracked.centroid = new_centroid
            tracked.velocity = new_velocity
            tracked.age += 1
            tracked.hits += 1
            tracked.time_since_update = 0
            tracked.confidence = min(1.0, tracked.hits / max(1, tracked.age))
            
            # Update keypoints if provided
            if kps_list and det_idx < len(kps_list):
                tracked.kps = kps_list[det_idx]
        
        # Increment time_since_update for unmatched tracks
        for track_id in track_ids:
            if track_id not in [tid for tid, _ in matches]:
                self.tracked_faces[track_id].time_since_update += 1
                if self.tracked_faces[track_id].time_since_update > self.max_disappeared:
                    del self.tracked_faces[track_id]
        
        # Create new tracks for unmatched detections
        for j in range(len(detections)):
            if j not in [idx for _, idx in matches]:
                track_id = self.next_id
                self.next_id += 1
                bbox = detections[j]
                centroid = detection_centroids[j]
                kps = kps_list[j] if kps_list and j < len(kps_list) else None
                self.tracked_faces[track_id] = TrackedFace(
                    track_id=track_id,
                    bbox=bbox,
                    centroid=centroid,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    confidence=1.0,
                    velocity=(0.0, 0.0),
                    kps=kps,
                )
        
        return self.tracked_faces.copy()
    

    
    def get_track(self, track_id: int) -> Optional[TrackedFace]:
        """Get tracked face by ID."""
        return self.tracked_faces.get(track_id)
    
    def clear(self):
        """Clear all tracks."""
        self.tracked_faces.clear()
        self.next_id = 0


def draw_tracked_face(
    img: np.ndarray,
    tracked: TrackedFace,
    show_id: bool = True,
    show_identity: bool = True,
    show_stats: bool = False,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a tracked face on the image.
    Returns the modified image.
    """
    x1, y1, x2, y2 = tracked.bbox
    
    # Color based on identity
    if tracked.identity:
        color = (0, 255, 0)  # Green for recognized
    else:
        color = (0, 165, 255)  # Orange for unknown
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw track ID
    if show_id:
        id_text = f"ID:{tracked.track_id}"
        cv2.putText(
            img, id_text, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    
    # Draw identity
    if show_identity and tracked.identity:
        identity_text = tracked.identity
        cv2.putText(
            img, identity_text, (x1, max(0, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    
    # Draw stats
    if show_stats:
        stats_text = f"dist={tracked.match_distance:.3f} sim={tracked.match_similarity:.3f}"
        cv2.putText(
            img, stats_text, (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    
    # Draw keypoints if available
    if tracked.kps is not None:
        for (x, y) in tracked.kps.astype(int):
            cv2.circle(img, (int(x), int(y)), 2, color, -1)
    
    # Draw centroid
    cx, cy = int(tracked.centroid[0]), int(tracked.centroid[1])
    cv2.circle(img, (cx, cy), 3, color, -1)
    
    return img
