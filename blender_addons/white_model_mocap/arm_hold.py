"""Stabilize prolonged, visibly held arm poses after mesh clearance."""

import math
from statistics import median

from mathutils import Vector

from .arm_clearance import _forward_flexion, _local_chain
from .joint_limits import flex_axis
from .stabilize import angle


_LONG_AXIS = Vector((0, 1, 0))
_MIN_CONFIDENCE = 0.54


def _image_point(row, index, size, aperture, minimum=_MIN_CONFIDENCE):
    _, _, _, x, y, confidence = row[index]
    left, top, right, bottom = aperture
    if confidence >= minimum and left <= x <= right and top <= y <= bottom:
        return Vector((size[0] * x, size[1] * y))
    return None


def _image_arm(row, side, size, aperture):
    if row is None:
        return None
    near = 0 if side == "L" else 1

    def point(index):
        return _image_point(row, index, size, aperture)

    shoulder_l, shoulder_r, hip_l, hip_r = map(point, (11, 12, 23, 24))
    shoulder = shoulder_l if side == "L" else shoulder_r
    elbow, wrist = point(13 + near), point(15 + near)
    if any(marker is None for marker in (shoulder_l, shoulder_r, elbow, wrist)):
        return None
    shoulder_span = (shoulder_l - shoulder_r).length
    scale = ((hip_l + hip_r - shoulder_l - shoulder_r).length / 2
             if hip_l is not None and hip_r is not None else shoulder_span)
    if (scale < size[1] * 0.055
            or min((elbow - shoulder).length, (wrist - elbow).length) < max(3, scale * 0.10)):
        return None
    return (elbow - shoulder) / scale, (wrist - shoulder) / scale


def _image_legs(row, size, aperture):
    if row is None:
        return None, None
    shoulder_l, shoulder_r, hip_l, hip_r = (
        _image_point(row, index, size, aperture) for index in (11, 12, 23, 24)
    )
    if any(point is None for point in (shoulder_l, shoulder_r, hip_l, hip_r)):
        return None, None
    hip = (hip_l + hip_r) / 2
    scale = (hip - (shoulder_l + shoulder_r) / 2).length
    if scale < size[1] * 0.055:
        return None, None
    legs = []
    for knee_index in (25, 26):
        knee = _image_point(row, knee_index, size, aperture, minimum=0.52)
        ankle = _image_point(row, knee_index + 2, size, aperture, minimum=0.55)
        legs.append(((knee - hip) / scale, (ankle - hip) / scale)
                    if knee is not None and ankle is not None else None)
    return legs


def _center(samples):
    return tuple(Vector(tuple(median(point[i][axis] for point in samples) for axis in range(2)))
                 for i in range(2))


def _stationary_legs(legs, start, count):
    usable = 0
    for side in (0, 1):
        entries = [pair[side] for pair in legs[start:start + count] if pair[side] is not None]
        if len(entries) < math.ceil(count * 0.65):
            continue
        usable += 1
        center = _center(entries)
        percentile = math.ceil(len(entries) * 0.90) - 1
        for index, maximum in enumerate((0.15, 0.19)):
            if sorted((entry[index] - center[index]).length for entry in entries)[percentile] > maximum:
                return False
    return usable > 0


def _hold_intervals(observations, legs, chest, rate):
    """Merge neighboring one-second windows only while they describe the same hold."""
    count = max(6, math.ceil(float(rate) * 0.95))
    required = math.ceil(count * 0.88)
    maximum_gap = max(1, round(float(rate) * 0.10))
    intervals = []
    for start in range(len(observations) - count + 1):
        indices = [i for i in range(start, start + count) if observations[i] is not None]
        if len(indices) < required or any(b - a > maximum_gap + 1 for a, b in zip(indices, indices[1:])):
            continue
        if any(angle(chest[indices[len(indices) // 2]], chest[i]) > math.radians(23) for i in indices):
            continue
        entries = [observations[i] for i in indices]
        center = _center(entries)
        deviations = [sorted((entry[i] - center[i]).length for entry in entries) for i in range(2)]
        percentile = math.ceil(len(entries) * 0.90) - 1
        if any(d[percentile] > bound or d[-1] > 0.15
               for d, bound in zip(deviations, (0.080, 0.105), strict=True)):
            continue
        if not _stationary_legs(legs, start, count):
            continue
        first, last = indices[0], indices[-1]
        if intervals and first <= intervals[-1][1] + 1:
            previous_first, previous_last, reference = intervals[-1]
            if all((a - b).length <= 0.12 for a, b in zip(center, reference, strict=True)):
                intervals[-1] = previous_first, max(previous_last, last), reference
            continue
        intervals.append((first, last, center))
    return [(first, last) for first, last, _ in intervals]


def _pose_distance(left, right):
    return max(angle(a, b) for a, b in zip(left, right, strict=True))


def _projected_error(pose, chest, elbow, wrist):
    shoulder, hinge = pose
    upper = chest @ shoulder
    forearm = upper @ hinge
    errors = []
    for rotation, observed in ((upper, elbow), (forearm, wrist - elbow)):
        direction = rotation @ _LONG_AXIS
        screen = Vector((direction.x, -direction.z))
        if screen.length < 0.13 or observed.length < 0.06:
            return math.inf
        errors.append(screen.angle(observed))
    return sum(errors) if max(errors) <= math.radians(48) else math.inf


def _steady_poses(current, raw, observations, chest, first, last):
    """Favor an observed bend from the same sequence over a synthetic rest arm."""
    observed = [i for i in range(first, last + 1) if observations[i] is not None]
    center = _center([observations[i] for i in observed])
    middle = observed[len(observed) // 2]
    reference = raw[middle]
    stride = max(1, len(observed) // 18)
    candidates = [track[i] for track in (current, raw) for i in observed[::stride]]
    scores = [(_projected_error(pose, chest[middle], *center) + 0.18 * _pose_distance(pose, reference), pose)
              for pose in candidates]
    chosen = []
    for cost, pose in sorted(scores, key=lambda item: item[0]):
        if math.isfinite(cost) and all(_pose_distance(pose, prior) > math.radians(2.5) for prior in chosen):
            chosen.append(pose)
            if len(chosen) == 12:
                break
    for pose in (current[first], current[last]):
        if math.isfinite(_projected_error(pose, chest[middle], *center)):
            chosen.append(pose)
    return chosen


def _steady_interval(current, raw, observations, chest, first, last, rate, clear, bend_axis):
    baseline = [_pose_distance(current[i], current[i + 1]) for i in range(first, last)]
    triggers = [i for i, step in enumerate(baseline) if step >= math.radians(280 / float(rate))]
    if not triggers:
        return None
    fade = max(2, round(float(rate) * 0.22))
    # Keep the intact lead-in, including any entrance or settling into the hold.
    lead_in = max(0, triggers[0] - 2 * fade)
    first += lead_in
    baseline = baseline[lead_in:]
    baseline_peak = max(baseline)
    starts_clip, ends_clip = first == 0, last == len(current) - 1
    changed_start = first if starts_clip else first + 1
    changed_end = last + 1 if ends_clip else last
    changed_indices = range(changed_start, changed_end)
    for shoulder, hinge in _steady_poses(current, raw, observations, chest, first, last):
        local = {}
        for i in changed_indices:
            entering = 1.0 if starts_clip else (i - first) / fade
            leaving = 1.0 if ends_clip else (last - i) / fade
            blend = min(1.0, entering, leaving)
            blend = blend * blend * (3 - 2 * blend)
            boundary = current[first if entering <= leaving else last]
            local[i] = boundary[0].slerp(shoulder, blend), boundary[1].slerp(hinge, blend)
        steps = [_pose_distance(local.get(i, current[i]), local.get(i + 1, current[i + 1]))
                 for i in range(first, last)]
        if (max(steps) > baseline_peak - math.radians(0.75)
                or sum(step ** 2 for step in steps) >= 0.9 * sum(step ** 2 for step in baseline)):
            continue
        replacements = {}
        for i, (local_upper, local_hinge) in local.items():
            upper = (chest[i] @ local_upper).normalized()
            replacements[i] = upper, (upper @ local_hinge).normalized()
        endpoints = {first: _world_chain(chest[first], current[first]),
                     last: _world_chain(chest[last], current[last])}
        if not all(_forward_flexion(pose, bend_axis) and clear(i, i, replacements, include_other=True)
                   for i, pose in replacements.items()):
            continue
        track = endpoints | replacements
        if not all(_forward_flexion(tuple(a.slerp(b, 0.5) for a, b in zip(track[i], track[i + 1], strict=True)),
                                    bend_axis)
                   and clear(i, i + 1, replacements, include_other=True) for i in range(first, last)):
            continue
        return replacements, math.degrees(baseline_peak), math.degrees(max(steps))
    return None


def _world_chain(chest, pose):
    upper = chest @ pose[0]
    return upper, upper @ pose[1]


def stabilize_observed_holds(arms, raw_arms, rest, chest, head, capture, trusted_torso, rate,
                             clearance, leg_clearance, legs):
    """Keep an observed hold steady only if its revised keys and midpoints are clear."""
    tracks = {part: values.copy() for part, values in arms.items()}
    check_for = clearance._collision_checker(tracks, rest, chest, head, leg_clearance, legs)
    source_start = capture["source_range"][0]
    aperture = capture.get("source_aperture", (0.0, 0.0, 1.0, 1.0))
    image_legs = [_image_legs(row, capture["source_size"], aperture) for row in capture["samples"]]
    ranges = []
    for side in ("L", "R"):
        observations = [_image_arm(row, side, capture["source_size"], aperture) if reliable else None
                        for row, reliable in zip(capture["samples"], trusted_torso, strict=True)]
        upper_part, lower_part = f"upper_arm.{side}", f"forearm.{side}"
        upper, lower = tracks[upper_part], tracks[lower_part]
        current = [_local_chain(body, (u, l)) for body, u, l in zip(chest, upper, lower, strict=True)]
        raw = [_local_chain(body, (u, l)) for body, u, l in
               zip(chest, raw_arms[upper_part], raw_arms[lower_part], strict=True)]
        pose_rest = [rest[part].matrix_local.to_quaternion() for part in (upper_part, lower_part)]
        bend_axis = flex_axis(*pose_rest)
        clear = check_for(side)
        for first, last in _hold_intervals(observations, image_legs, chest, rate):
            result = _steady_interval(current, raw, observations, chest, first, last, rate, clear, bend_axis)
            if result is None:
                continue
            replacements, before, after = result
            for i, pose in replacements.items():
                upper[i], lower[i] = pose
            ranges.append({
                "control": f"arm.{side}", "reason": "observed_arm_hold", "treatment": "corrected",
                "source_range": [source_start + min(replacements), source_start + max(replacements)],
                "peak_local_step_before_degrees": round(before, 2),
                "peak_local_step_after_degrees": round(after, 2),
                "mesh_key_and_midpoint_check": "passed",
            })
    return tracks, ranges
