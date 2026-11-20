"""Stabilize one camera-space orientation track without moving its frame times."""

from itertools import groupby
import math

from mathutils import Quaternion, Vector


def angle(a, b):
    dot = a.dot(b)
    if isinstance(a, Quaternion):
        return 2.0 * math.acos(min(1.0, abs(dot)))
    return math.acos(max(-1.0, min(1.0, dot)))


def blend(a, b, amount):
    if isinstance(a, Quaternion):
        return a.slerp(b, amount).normalized()
    if a.dot(b) <= -0.9999:
        # Blender's vector slerp is undefined at antipodes. Use a consistent arc.
        turn = a.rotation_difference(b)
        return (Quaternion().slerp(turn, amount) @ a).normalized()
    return a.slerp(b, amount).normalized()


def short_excursions(values, fps):
    """Reject a brief departure only when the observed poses around it agree."""
    rejected = set()
    horizon = max(1, round(0.085 * fps))
    for count in range(horizon, 0, -1):
        agreement = math.radians(min(26, max(12, 230 * (count + 1) / fps)))
        for start in range(1, len(values) - count):
            end = start + count
            if any(i in rejected or values[i] is None for i in range(start - 1, end + 1)):
                continue
            before, after = values[start - 1], values[end]
            if angle(before, after) > agreement:
                continue
            if all(angle(values[i], blend(before, after, (i - start + 1) / (count + 1)))
                   > math.radians(34) for i in range(start, end)):
                rejected.update(range(start, end))
    return rejected


def oscillating_bursts(values, fps):
    """Drop the interior of brief off-path bursts, even when the endpoints differ."""
    limit = math.radians(min(110, max(35, 20 + 1500 / fps)))
    max_edges = max(2, round(0.18 * fps))
    steps = [angle(a, b) if a is not None and b is not None else 0.0
             for a, b in zip(values, values[1:])]
    rejected = set()
    for fast, run in groupby(range(len(steps)), lambda i: steps[i] > limit):
        if not fast:
            continue
        edges = list(run)
        if not 2 <= len(edges) <= max_edges:
            continue
        start, end = edges[0], edges[-1] + 1
        detour = sum(steps[i] for i in edges) - angle(values[start], values[end])
        if detour > math.radians(95):
            rejected.update(range(start + 1, end))
    return rejected


def fill_gaps(values, fallback, fps):
    treatments = [None] * len(values)
    # A brief limb occlusion can cross a whole swing. Bridge when both endpoints are
    # available; prefer a shorter inference span for ambiguous head and torso turns.
    bridge_duration = 0.20 if isinstance(fallback, Quaternion) else 0.55
    max_bridge = max(1, round(bridge_duration * fps))
    cursor = 0
    while cursor < len(values):
        if values[cursor] is not None:
            cursor += 1
            continue
        start = cursor
        while cursor < len(values) and values[cursor] is None:
            cursor += 1
        count = cursor - start
        before = values[start - 1] if start else None
        after = values[cursor] if cursor < len(values) else None
        if before is None or after is None:
            available = before if before is not None else after
            values[start:cursor] = [available if available is not None else fallback] * count
            treatments[start:cursor] = ["held" if available is not None else "rest"] * count
            continue

        if count <= max_bridge:
            transition = count
        else:
            # Hold an unrecoverable long gap, with a bounded blend before re-entry.
            # Extend the blend when the recovered direction differs substantially.
            transition = min(count, max(1, round(0.22 * fps),
                                        math.ceil(angle(before, after) * fps / math.radians(450))))
        held = count - transition
        values[start:start + held] = [before] * held
        treatments[start:start + held] = ["held"] * held
        for step in range(1, transition + 1):
            t = step / (transition + 1)
            eased = t * t * (3.0 - 2.0 * t)
            amount = eased if held else 0.75 * t + 0.25 * eased
            i = start + held + step - 1
            values[i] = blend(before, after, amount)
            treatments[i] = "interpolated"
    return treatments


def smooth(values, radius):
    """Centered smoothing uses more weight when neighbors indicate a reversal."""
    output = values.copy()
    for i in range(1, len(values) - 1):
        center = values[i]
        radius_here = min(radius, i, len(values) - i - 1)
        predicted = center
        total = 0.0
        for offset in range(1, radius_here + 1):
            before, after = values[i - offset], values[i + offset]
            detour = max(0.0, angle(before, center) + angle(center, after) - angle(before, after))
            strength = min(0.78, 0.20 + detour / math.radians(85))
            smoothed = blend(center, blend(before, after, 0.5), strength)
            weight = 1 / offset
            total += weight
            predicted = blend(predicted, smoothed, weight / total)
        output[i] = predicted
    return output


def centered_rotations(rotations, fps, seconds):
    """Remove short angular noise on the original time grid, with no phase lag."""
    radius = round(float(fps) * seconds)
    if radius < 1 or len(rotations) < 3:
        return rotations
    weights = [0.5 + 0.5 * math.cos(math.pi * offset / (radius + 1))
               for offset in range(radius + 1)]
    output = []
    for index, reference in enumerate(rotations):
        total = Vector((0.0, 0.0, 0.0, 0.0))
        for sample in range(max(0, index - radius), min(len(rotations), index + radius + 1)):
            rotation = rotations[sample]
            weight = weights[abs(sample - index)]
            if reference.dot(rotation) < 0:
                weight = -weight
            total += weight * Vector(rotation)
        output.append(Quaternion(total).normalized())
    return output


def stabilize_track(values, *, part, source_start, fps, fallback):
    """Fill invisible samples, suppress abrupt excursions and preserve frame alignment."""
    fps = float(fps)
    values = [value.normalized() if value is not None else None for value in values]
    rejected = short_excursions(values, fps) | oscillating_bursts(values, fps)
    for i in rejected:
        values[i] = None
    treatments = fill_gaps(values, fallback.normalized(), fps)
    classifications = [
        (treatment, "angular_outlier" if i in rejected else "missing_observation")
        for i, treatment in enumerate(treatments)
    ]
    ranges = []
    for (treatment, reason), indexes in groupby(range(len(values)), classifications.__getitem__):
        indices = list(indexes)
        if treatment is not None:
            ranges.append({"control": part, "source_range": [source_start + indices[0], source_start + indices[-1]],
                           "treatment": treatment, "reason": reason})
    radius = max(1, round(0.07 * fps))
    return smooth(smooth(values, radius), radius), ranges
