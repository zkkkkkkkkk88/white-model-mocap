"""Reject sudden wrist reacquisition when local image evidence contradicts it."""

from dataclasses import dataclass
from itertools import groupby

import cv2
import numpy as np

from image_motion import _scope, _track_points


_ARMS = {"L": (13, 15), "R": (14, 16)}


def _point(sample, joint, size, confidence):
    if sample is None:
        return None
    x, y, score = sample[joint][3:6]
    if score < confidence or not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return np.array((x, y)) * size


def _shoulders(sample, size):
    left = _point(sample, 11, size, .68)
    right = _point(sample, 12, size, .68)
    if left is None or right is None:
        return None
    width = np.linalg.norm(left - right)
    if width < .07 * size[1]:
        return None
    return (left + right) / 2, width


def _local_flow(before, after, center, width):
    """Use bidirectional feature correspondences near the previous wrist."""
    h, w = before.shape
    x, y = center
    padding = max(36, .38 * width)
    x0, y0 = max(0, int(x - padding)), max(0, int(y - padding))
    x1, y1 = min(w, int(x + padding + 1)), min(h, int(y + padding + 1))
    before, after = before[y0:y1, x0:x1], after[y0:y1, x0:x1]
    mask = np.zeros_like(before)
    x, y = x - x0, y - y0
    radius = max(8, .18 * width)
    _scope(mask, x - radius, y - radius, x + radius, y + radius, 255)
    inputs, outputs = _track_points(before, after, mask, 36, max(2, .012 * width))
    if len(inputs) < 6:
        return None
    offsets = outputs - inputs
    motion = np.median(offsets, axis=0)
    if np.median(np.linalg.norm(offsets - motion, axis=1)) > max(1.5, .024 * width):
        return None
    return motion


@dataclass
class _SuspectPath:
    wrist: np.ndarray
    elbow: np.ndarray
    starts_at: int
    recovered: int = 0
    elbow_conflict: bool = False


class WristReacquisitionGuard:
    """Mark contradictory wrist observations missing without inventing pose data."""

    def __init__(self, fps):
        self.fps = float(fps)
        self.previous = None
        self.paths = {}
        self.rejections = {side: {} for side in _ARMS}

    def _new_path(self, before, after, previous, current, speed, elbow, wrist, size, width, index):
        old_wrist = _point(previous, wrist, size, .45)
        new_wrist = _point(current, wrist, size, .50)
        old_elbow = _point(previous, elbow, size, .63)
        new_elbow = _point(current, elbow, size, .63)
        if any(point is None for point in (old_wrist, new_wrist, old_elbow, new_elbow)):
            return None
        if previous[wrist][5] >= .68:
            return None
        old_height = old_wrist[1] - old_elbow[1]
        new_height = new_wrist[1] - new_elbow[1]
        shift = new_wrist - old_wrist
        if (old_height * new_height >= 0 or min(abs(old_height), abs(new_height)) < .18 * width
                or np.linalg.norm(shift - speed) < max(.63, 18 / self.fps) * width
                or np.linalg.norm(new_elbow - old_elbow - speed) > .23 * width):
            return None
        traced = _local_flow(before, after, old_wrist, width)
        if (traced is None or np.linalg.norm(traced - speed) > .21 * width
                or np.linalg.norm(shift - traced) < max(.42, 12 / self.fps) * width):
            return None
        return _SuspectPath(old_wrist, old_elbow, index)

    def _check_path(self, path, sample, elbow, wrist, speed, size, width, index, before, after):
        if index - path.starts_at >= max(2, round(.80 * self.fps)):
            return ()
        old_elbow = path.elbow
        path.wrist += speed
        path.elbow = old_elbow + speed
        wrist_point = _point(sample, wrist, size, .58)
        if wrist_point is not None and np.linalg.norm(wrist_point - path.wrist) < .24 * width:
            path.recovered += 1
            if path.recovered >= 2:
                return ()
        else:
            path.recovered = 0
        elbow_point = _point(sample, elbow, size, .50)
        if elbow_point is not None:
            discrepancy = np.linalg.norm(elbow_point - path.elbow)
            if path.elbow_conflict and discrepancy < .17 * width:
                path.elbow_conflict = False
            elif not path.elbow_conflict and discrepancy > .26 * width:
                traced = _local_flow(before, after, old_elbow, width)
                path.elbow_conflict = bool(traced is not None and np.linalg.norm(traced - speed) < .18 * width
                                           and np.linalg.norm(elbow_point - old_elbow - traced) > .24 * width)
        # Lower confidence rather than substituting a projected 2D position for a 3D landmark.
        return (elbow, wrist) if path.elbow_conflict else (wrist,)

    def filter(self, frame, sample, index):
        h, w = frame.shape[:2]
        scale = min(1, 960 / max(h, w))
        size = np.array((round(w * scale), round(h * scale)))
        if scale < 1:
            frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shoulder = _shoulders(sample, size)
        previous = self.previous
        self.previous = (gray, sample, shoulder)
        if previous is None or shoulder is None or previous[2] is None:
            self.paths.clear()
            return sample
        before, prior_sample, (old_center, old_width) = previous
        center, width = shoulder
        speed = center - old_center
        if np.linalg.norm(speed) > .18 * width or not .75 <= width / old_width <= 1.25:
            self.paths.clear()
            return sample

        filtered = list(sample)
        for side, (elbow, wrist) in _ARMS.items():
            path = self.paths.get(side)
            if path is None:
                path = self._new_path(before, gray, prior_sample, sample, speed, elbow, wrist, size, width, index)
                if path is None:
                    continue
            joints = self._check_path(path, sample, elbow, wrist, speed, size, width, index, before, gray)
            if not joints:
                self.paths.pop(side, None)
                continue
            self.paths[side] = path
            self.rejections[side][index] = tuple(joints)
            for joint in joints:
                filtered[joint] = [*sample[joint][:5], 0.0]
        return filtered

    def report(self):
        rows = []
        for side, rejected in self.rejections.items():
            for joints, group in groupby(range(min(rejected, default=0), max(rejected, default=-1) + 1),
                                         rejected.get):
                if joints:
                    indices = list(group)
                    rows.append({"side": side, "source_range": [indices[0], indices[-1]],
                                 "joints": ["wrist" if joint == _ARMS[side][1] else "elbow" for joint in joints],
                                 "reason": "optical_flow_contradicts_wrist_reacquisition"})
        return rows
