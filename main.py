import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Placeholder imports for external deep learning models
# YOLOv8 segmentation model from ultralytics
from ultralytics import YOLO

# Depth estimation model - Depth Anything V2
# This is a placeholder import; in a real environment, install the correct package
# and adjust the import accordingly.
try:
    from depth_anything_v2 import DepthAnythingV2
except ImportError:  # pragma: no cover - model library is not available in this environment
    DepthAnythingV2 = None  # type: ignore

# BotSort tracker from ultralytics
try:
    from ultralytics.tracker.bot_sort import BOTSORT
except Exception:  # pragma: no cover - tracker may not be installed
    BOTSORT = None  # type: ignore


@dataclass
class NodeTrack:
    """Stores tracking information for a single node."""
    id: int
    init_pos: Tuple[float, float]
    init_depth: float
    positions: List[Tuple[float, float]] = field(default_factory=list)
    depths: List[float] = field(default_factory=list)

    def update(self, pos: Tuple[float, float], depth: float) -> None:
        self.positions.append(pos)
        self.depths.append(depth)

    def displacement(self, scale: float) -> Tuple[float, float, float]:
        """Compute scaled X, Y, Z displacement from the initial state."""
        if not self.positions:
            return 0.0, 0.0, 0.0
        last_x, last_y = self.positions[-1]
        init_x, init_y = self.init_pos
        dz = (self.depths[-1] - self.init_depth) * scale
        dx = (last_x - init_x) * scale
        dy = (last_y - init_y) * scale
        return dx, dy, dz


def segment_frame(model: YOLO, frame: np.ndarray) -> np.ndarray:
    """Run YOLOv8 segmentation and mask non-steel regions."""
    results = model(frame)[0]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if results.masks:
        # Assume the first mask corresponds to the steel frame
        mask = results.masks.data[0].cpu().numpy().astype(np.uint8)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame


def estimate_depth(depth_model, frame: np.ndarray) -> np.ndarray:
    """Estimate relative depth for the masked frame."""
    if depth_model is None:
        raise RuntimeError("DepthAnythingV2 model is not available.")
    depth = depth_model.predict(frame)  # type: ignore[attr-defined]
    return depth


def track_nodes(tracker, detections: np.ndarray) -> List[Tuple[int, Tuple[float, float]]]:
    """Track node centroids across frames using BotSort."""
    if tracker is None:
        raise RuntimeError("BotSort tracker is not available.")
    tracks = tracker.update(detections)
    output = []
    for t in tracks:
        track_id = int(t[4])
        x1, y1, x2, y2 = t[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        output.append((track_id, (cx, cy)))
    return output


def process_video(
    video_path: str,
    yolo_steel_weights: str,
    yolo_node_weights: str,
    depth_model_weights: str,
    scale_factor: float,
) -> Dict[int, NodeTrack]:
    """Main processing pipeline for 3D displacement measurement.

    Args:
        video_path: Path to the input video.
        yolo_steel_weights: Path to YOLOv8-seg-steel weights (.pt) used for masking.
        yolo_node_weights: Path to YOLOv8-seg-node weights (.pt) used for node detection.
        depth_model_weights: Path to Depth Anything V2 weights (.pt) for depth estimation.
        scale_factor: Real-world scale factor derived from camera distance.

    Returns:
        Dictionary mapping node IDs to NodeTrack instances.
    """
    # Load models from local weight files
    steel_model = YOLO(yolo_steel_weights)
    node_model = YOLO(yolo_node_weights)
    depth_model = DepthAnythingV2(depth_model_weights) if DepthAnythingV2 else None
    tracker = BOTSORT() if BOTSORT else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    node_tracks: Dict[int, NodeTrack] = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        masked = segment_frame(steel_model, frame)
        depth_map = estimate_depth(depth_model, masked)

        # Detect and track nodes using the dedicated YOLO model
        node_results = node_model(frame)[0]
        detections = (
            node_results.boxes.xyxy.cpu().numpy() if node_results.boxes else np.empty((0, 4))
        )
        tracked = track_nodes(tracker, detections)

        for track_id, (cx, cy) in tracked:
            depth = depth_map[int(cy), int(cx)]
            if track_id not in node_tracks:
                node_tracks[track_id] = NodeTrack(track_id, (cx, cy), depth)
            node_tracks[track_id].update((cx, cy), depth)

        frame_idx += 1

    cap.release()
    return node_tracks


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Monocular 3D displacement measurement")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "yolo_steel", help="Path to YOLOv8-seg-steel weights (.pt) for masking"
    )
    parser.add_argument(
        "yolo_node", help="Path to YOLOv8-seg-node weights (.pt) for node detection"
    )
    parser.add_argument(
        "depth", help="Path to Depth Anything V2 weights (.pt) for depth estimation"
    )
    parser.add_argument(
        "scale", type=float, help="Scale factor from camera to structure distance"
    )
    args = parser.parse_args()

    tracks = process_video(
        args.video, args.yolo_steel, args.yolo_node, args.depth, args.scale
    )
    for node_id, track in tracks.items():
        dx, dy, dz = track.displacement(args.scale)
        print(f"Node {node_id}: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")




