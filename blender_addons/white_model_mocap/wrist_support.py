"""Preserve a user-confirmed wrist support on a fixed object."""

import math
from statistics import median

from mathutils import Quaternion, Vector

from .arm_clearance import _forward_flexion, _interpolate_chain
from .arm_stability import _align
from .joint_limits import LENGTH_AXIS, flex_axis
from .observations import LIMBS, _is_visible, _screen_point, rear_sample


def validate_support_interval(hand, frames, first, last):
    if hand not in ("none", "L", "R"):
        raise ValueError("支撑手须为 none、L 或 R")
    if hand == "none":
        if frames is not None:
            raise ValueError("填写支撑帧段时请同时指定支撑手")
        return None
    if (not isinstance(frames, (tuple, list)) or len(frames) != 2
            or any(type(value) is not int for value in frames)
            or not first <= frames[0] < frames[1] <= last):
        raise ValueError(f"支撑起止帧须递增，并位于捕捉的源帧 {first}–{last} 内")
    return tuple(frames)


def _median(points):
    return Vector(tuple(median(axis) for axis in zip(*points, strict=True)))


def _contact_samples(samples, size, hand, lo, hi, rate, rear):
    elbow, wrist = LIMBS[f"forearm.{hand}"]
    evidence = []
    for i in range(lo, hi + 1):
        row = rear_sample(samples[i], size) if rear else samples[i]
        if row is None or not all(_is_visible(row[j], 0.50) for j in (11, 12, elbow, wrist)):
            continue
        span = (_screen_point(row[11], size) - _screen_point(row[12], size)).length
        if span >= 0.045 * size[1]:
            evidence.append((i, _screen_point(row[wrist], size), span))
    minimum = max(3, math.ceil(.10 * rate))
    if len(evidence) < minimum:
        return []
    radius = .16 * median(item[2] for item in evidence)
    # Sample possible centers across the interval. The largest compact cluster
    # can survive gaps and an isolated, high-confidence false reacquisition.
    stride = max(1, len(evidence) // 64)
    groups = [[i for i, wrist_pos, _ in evidence if (wrist_pos - center).length <= radius]
              for _, center, _ in evidence[::stride]]
    inliers = max(groups, key=len)
    if (len(inliers) < max(minimum, math.ceil(.65 * len(evidence)))
            or inliers[-1] - inliers[0] < min(math.ceil(.20 * rate), (hi - lo) / 2)):
        return []
    return inliers


def _contact_pose(shoulder, target, pole, chest, lengths, rotations, axis, solve):
    a, b = lengths
    distance = (target - shoulder).length
    minimum = math.sqrt(a * a + b * b + 2 * a * b * math.cos(math.radians(142)))
    if not minimum < distance < .995 * (a + b):
        return None
    direction = (target - shoulder).normalized()
    if pole.cross(direction).length < .04:
        pole = Vector((0, 0, -1))
    if pole.cross(direction).length < .04:
        pole = chest @ Vector((0, -1, 0))
    if pole.cross(direction).length < .04:
        pole = direction.orthogonal()
    joint = solve(shoulder, target, a, b, pole)["joint"]
    upper_direction = (joint - shoulder).normalized()
    fore_direction = (target - joint).normalized()
    upper = _align(chest @ rotations[0], upper_direction)
    old_hinge = upper @ axis
    new_hinge = (fore_direction - upper_direction * fore_direction.dot(upper_direction)).normalized()
    twist = math.atan2(upper_direction.dot(old_hinge.cross(new_hinge)), old_hinge.dot(new_hinge))
    upper = (Quaternion(upper_direction, twist) @ upper).normalized()
    roll = _align(rotations[0], rotations[1] @ LENGTH_AXIS).inverted() @ rotations[1]
    return upper, (_align(upper, fore_direction) @ roll).normalized()


def stabilize_supported_wrist(arms, shoulders, chests, rest, capture, *,
                              hand, frames, fps, rear, collision_check, solve):
    """Return a replacement only when the fixed-point transition is feasible."""
    report = {"hand": hand, "source_range": list(frames), "status": "skipped"}
    if collision_check is None:
        return arms, dict(report, reason="mesh_clearance_unavailable")
    size = capture.get("source_size")
    if not size:
        return arms, dict(report, reason="source_size_unavailable")
    first = capture["source_range"][0]
    lo, hi = (i - first for i in frames)
    rate = float(fps)
    reliable = _contact_samples(capture["samples"], size, hand, lo, hi, rate, rear)
    if not reliable:
        return arms, dict(report, reason="insufficient_stable_wrist_observations")
    report["supporting_observations"] = len(reliable)

    upper_name, fore_name = f"upper_arm.{hand}", f"forearm.{hand}"
    upper, fore = arms[upper_name], arms[fore_name]
    lengths = rest[upper_name].length, rest[fore_name].length
    rotations = tuple(rest[name].matrix_local.to_quaternion() for name in (upper_name, fore_name))
    axis = flex_axis(*rotations)
    elbows = [shoulders[i] + (upper[i] @ LENGTH_AXIS) * lengths[0] for i in reliable]
    wrists = [elbow + (fore[i] @ LENGTH_AXIS) * lengths[1] for i, elbow in zip(reliable, elbows, strict=True)]
    anchor = _median(wrists)
    shoulder_mid = _median([shoulders[i] for i in reliable])
    reach = (anchor - shoulder_mid).normalized()
    elbow_guide = _median(elbows) - shoulder_mid
    pole = (elbow_guide - reach * elbow_guide.dot(reach)).normalized()
    fade = max(2, round(.26 * rate))
    start, end = max(0, lo - fade + 1), min(len(upper) - 1, hi + fade - 1)
    for inward in (0.0, .055, .11):
        point = anchor.lerp(shoulder_mid, inward)
        replacement = {}
        for i in range(start, end + 1):
            target = _contact_pose(shoulders[i], point, pole, chests[i], lengths, rotations, axis, solve)
            if target is None:
                break
            weight = min(1.0, (i - lo + fade) / fade, (hi + fade - i) / fade)
            weight = weight * weight * (3 - 2 * weight)
            replacement[i] = _interpolate_chain((upper[i], fore[i]), target, weight)
        else:
            if not all(_forward_flexion(pose, axis) and collision_check(i, i, replacement, include_other=True)
                       for i, pose in replacement.items()):
                continue
            for i in range(max(0, start - 1), min(len(upper) - 1, end + 1)):
                left = replacement.get(i, (upper[i], fore[i]))
                right = replacement.get(i + 1, (upper[i + 1], fore[i + 1]))
                midpoint = tuple(a.slerp(b, .5).normalized() for a, b in zip(left, right, strict=True))
                if not _forward_flexion(midpoint, axis) or not collision_check(i, i + 1, replacement, include_other=True):
                    break
            else:
                stable = {**arms, upper_name: upper.copy(), fore_name: fore.copy()}
                for i, (u, v) in replacement.items():
                    stable[upper_name][i], stable[fore_name][i] = u, v
                return stable, dict(report, status="applied", anchor_rig_space=list(point),
                                    transition_source_range=[first + start, first + end])
    return arms, dict(report, reason="support_pose_not_feasible")
