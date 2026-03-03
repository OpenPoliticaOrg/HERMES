"""
Automatic entity observation adapters for streaming.
"""


def _clamp_bbox_xyxy(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), int(width) - 1))
    y1 = max(0, min(int(y1), int(height) - 1))
    x2 = max(0, min(int(x2), int(width) - 1))
    y2 = max(0, min(int(y2), int(height) - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = float(inter_w * inter_h)

    area_a = float(max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1))
    area_b = float(max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


class MotionEntityObservationAdapter:
    """
    Lightweight motion-blob tracker for automatic entity observation generation.

    It emits `entity_id` and bbox metadata per window. Event scores are left empty,
    allowing downstream classifier fallback to provide observation likelihoods.
    """

    def __init__(
        self,
        min_area=500,
        iou_threshold=0.25,
        max_tracks=12,
        max_missed=3,
        bg_history=120,
        bg_var_threshold=16,
    ):
        self.min_area = max(1, int(min_area))
        self.iou_threshold = float(max(0.0, iou_threshold))
        self.max_tracks = max(1, int(max_tracks))
        self.max_missed = max(0, int(max_missed))
        self.bg_history = max(20, int(bg_history))
        self.bg_var_threshold = max(1, int(bg_var_threshold))

        self._bg_subtractor = None
        self._next_track_idx = 1
        self._tracks = {}

    def reset(self):
        self._bg_subtractor = None
        self._next_track_idx = 1
        self._tracks = {}

    def _ensure_bg(self, cv2_module):
        if self._bg_subtractor is None:
            self._bg_subtractor = cv2_module.createBackgroundSubtractorMOG2(
                history=self.bg_history,
                varThreshold=self.bg_var_threshold,
                detectShadows=False,
            )
        return self._bg_subtractor

    def _detect(self, frame_bgr, cv2_module):
        bg = self._ensure_bg(cv2_module)
        fg = bg.apply(frame_bgr)
        _, mask = cv2_module.threshold(fg, 200, 255, cv2_module.THRESH_BINARY)
        kernel = cv2_module.getStructuringElement(cv2_module.MORPH_ELLIPSE, (3, 3))
        mask = cv2_module.morphologyEx(mask, cv2_module.MORPH_OPEN, kernel, iterations=1)
        mask = cv2_module.dilate(mask, kernel, iterations=2)

        contours, _ = cv2_module.findContours(
            mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE
        )
        h, w = frame_bgr.shape[:2]
        detections = []
        for contour in contours:
            area = float(cv2_module.contourArea(contour))
            if area < float(self.min_area):
                continue
            x, y, bw, bh = cv2_module.boundingRect(contour)
            x1, y1, x2, y2 = _clamp_bbox_xyxy(x, y, x + bw - 1, y + bh - 1, w, h)
            detections.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)],
                    "area_px": int(area),
                }
            )

        detections.sort(key=lambda x: float(x["area_px"]), reverse=True)
        return detections

    def _prune_tracks(self):
        keep = {}
        for track_id, track in self._tracks.items():
            if int(track.get("missed", 0)) <= self.max_missed:
                keep[track_id] = track
        self._tracks = keep

    def _match_detections(self, detections):
        track_ids = list(self._tracks.keys())
        candidates = []
        for det_idx, det in enumerate(detections):
            det_box = det["bbox_xyxy"]
            for track_id in track_ids:
                track_box = self._tracks[track_id]["bbox_xyxy"]
                iou = _iou_xyxy(det_box, track_box)
                if iou >= self.iou_threshold:
                    candidates.append((float(iou), det_idx, track_id))
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_det = set()
        matched_track = set()
        matches = []
        for iou, det_idx, track_id in candidates:
            if det_idx in matched_det or track_id in matched_track:
                continue
            matched_det.add(det_idx)
            matched_track.add(track_id)
            matches.append((det_idx, track_id))
        return matches, matched_det, matched_track

    def _new_track_id(self):
        track_id = f"motion_{self._next_track_idx:04d}"
        self._next_track_idx += 1
        return track_id

    def observe_window(self, window_frames_bgr, window_idx, cv2_module):
        if not isinstance(window_frames_bgr, (list, tuple)) or len(window_frames_bgr) == 0:
            for track in self._tracks.values():
                track["missed"] = int(track.get("missed", 0)) + 1
            self._prune_tracks()
            return []

        for frame in window_frames_bgr[:-1]:
            self._ensure_bg(cv2_module).apply(frame)

        ref_frame = window_frames_bgr[-1]
        detections = self._detect(ref_frame, cv2_module)
        if len(detections) > self.max_tracks * 2:
            detections = detections[: self.max_tracks * 2]

        matches, matched_det, matched_track = self._match_detections(detections)

        for det_idx, track_id in matches:
            det = detections[det_idx]
            track = self._tracks[track_id]
            track["bbox_xyxy"] = list(det["bbox_xyxy"])
            track["bbox_xywh"] = list(det["bbox_xywh"])
            track["area_px"] = int(det["area_px"])
            track["missed"] = 0
            track["seen_windows"] = int(track.get("seen_windows", 0)) + 1
            track["last_window"] = int(window_idx)

        for track_id, track in self._tracks.items():
            if track_id in matched_track:
                continue
            track["missed"] = int(track.get("missed", 0)) + 1

        unmatched_det = [idx for idx in range(len(detections)) if idx not in matched_det]
        for det_idx in unmatched_det:
            if len(self._tracks) >= self.max_tracks:
                break
            det = detections[det_idx]
            track_id = self._new_track_id()
            self._tracks[track_id] = {
                "entity_id": track_id,
                "bbox_xyxy": list(det["bbox_xyxy"]),
                "bbox_xywh": list(det["bbox_xywh"]),
                "area_px": int(det["area_px"]),
                "missed": 0,
                "seen_windows": 1,
                "first_window": int(window_idx),
                "last_window": int(window_idx),
            }

        self._prune_tracks()

        h, w = ref_frame.shape[:2]
        observed = []
        for track_id, track in self._tracks.items():
            if int(track.get("last_window", -1)) != int(window_idx):
                continue
            x1, y1, x2, y2 = track["bbox_xyxy"]
            bbox_norm = [
                float(x1) / float(max(1, w)),
                float(y1) / float(max(1, h)),
                float(x2) / float(max(1, w)),
                float(y2) / float(max(1, h)),
            ]
            observed.append(
                {
                    "entity_id": str(track_id),
                    "metadata": {
                        "source": "auto_motion",
                        "window_index": int(window_idx),
                        "bbox_xyxy_px": [int(x1), int(y1), int(x2), int(y2)],
                        "bbox_xywh_px": [int(v) for v in track["bbox_xywh"]],
                        "bbox_xyxy_norm": bbox_norm,
                        "area_px": int(track.get("area_px", 0)),
                    },
                }
            )

        observed.sort(key=lambda x: x["entity_id"])
        return observed
