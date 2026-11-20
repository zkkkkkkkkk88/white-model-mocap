"""Stabilize posterior body angles and optionally honor source-reviewed yaw anchors."""

from bisect import bisect_right
import math
from statistics import median

from mathutils import Quaternion, Vector


UP = Vector((0, 0, 1))
FRONT = Vector((0, -1, 0))


def parse_yaw_anchors(value, first, last):
    """Read zero-based source-frame:degrees anchors, positive toward screen right."""
    if not value.strip():
        return ()
    try:
        anchors = tuple((int(frame), float(angle))
                        for frame, angle in (entry.split(":") for entry in value.split(",")))
    except ValueError as exc:
        raise ValueError("背向角度锚点使用 源帧:角度，例如 24:20,72:0") from exc
    if (len(anchors) < 2 or any(not first <= frame <= last or not math.isfinite(angle)
                                or abs(angle) > 60 for frame, angle in anchors)
            or any(left[0] >= right[0] for left, right in zip(anchors, anchors[1:]))):
        raise ValueError("背向角度需有至少两个按顺序排列的源帧，角度在 -60° 至 60°")
    return anchors


def _smoothed(values, radius):
    filtered = [median(values[max(0, i - radius):i + radius + 1]) for i in range(len(values))]
    for _ in range(2):
        output = []
        for i in range(len(filtered)):
            start, end = max(0, i - radius), min(len(filtered), i + radius + 1)
            weights = [radius + 1 - abs(i - j) for j in range(start, end)]
            output.append(sum(filtered[j] * weight for j, weight in zip(range(start, end), weights))
                          / sum(weights))
        filtered = output
    return filtered


def _anchored_yaw(anchors, frame):
    frames = [value[0] for value in anchors]
    after = bisect_right(frames, frame)
    if after == 0 or after == len(anchors):
        return math.radians(anchors[0 if after == 0 else -1][1])
    start, a = anchors[after - 1]
    stop, b = anchors[after]
    t = (frame - start) / (stop - start)
    return math.radians(a + (b - a) * (t * t * (3 - 2 * t)))


def stabilize_rear_orientation(rotations, *, source_start, fps, anchors=()):
    """Keep torso lean and posterior heading from amplifying short landmark errors."""
    rate = float(fps)
    yaw, roll = [], []
    for rotation in rotations:
        forward, up = rotation @ FRONT, rotation @ UP
        yaw.append(math.atan2(forward.x, forward.y))
        roll.append(math.atan2(up.x, up.z))

    if anchors:
        yaw = [_anchored_yaw(anchors, source_start + index) for index in range(len(yaw))]
    else:
        yaw = _smoothed(yaw, max(1, round(0.28 * rate)))
    roll = _smoothed(roll, max(1, round(0.16 * rate)))
    limit = math.radians(5)
    return [(Quaternion((0, 1, 0), max(-limit, min(limit, lean * 0.60)))
             @ Quaternion(UP, math.pi - heading)).normalized()
            for heading, lean in zip(yaw, roll, strict=True)]
