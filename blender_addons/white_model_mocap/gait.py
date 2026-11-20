"""Complete occluded walking chains from visible torso rhythm and the rig geometry."""

from bisect import bisect_right
from itertools import groupby
import math
from statistics import correlation, mean, pstdev

from mathutils import Vector

from .stabilize import blend
from .torso_translation import smooth_maximum, stabilize_walk_lift


SIDES = ("L", "R")
UP = Vector((0, 0, 1))
FRONT = Vector((0, -1, 0))
STANCE = 0.62


def _visible_signal(capture, count):
    values = (capture.get("image_torso_motion") or {}).get("up", ())
    if len(values) != count:
        return [None] * count
    return [value if isinstance(value, (float, int)) and math.isfinite(value) else None
            for value in values]


def estimate_step_period(signal, fps):
    """Find the first strong repeating step period in camera-compensated image height."""
    valid = [value for value in signal if value is not None]
    if len(valid) < max(10, math.ceil(1.45 * fps)) or pstdev(valid) < 0.0016:
        return None, 0.0
    first = max(2, math.ceil(0.36 * fps))
    last = min(math.floor(0.82 * fps), len(signal) // 3)
    if first > last:
        return None, 0.0
    scores = {}
    for lag in range(max(1, first - 1), last + 2):
        samples = [(a, b) for a, b in zip(signal, signal[lag:]) if a is not None and b is not None]
        if len(samples) < max(9, 2 * lag):
            continue
        before, after = zip(*samples, strict=True)
        if pstdev(before) > 1e-7 and pstdev(after) > 1e-7:
            scores[lag] = correlation(before, after)
    best = max((scores.get(lag, -1) for lag in range(first, last + 1)), default=-1)
    if best < 0.44:
        return None, max(0.0, best)
    peaks = [
        lag for lag in range(first, last + 1) if scores.get(lag, -1) >= max(0.44, best * 0.78)
        and scores[lag] >= max(scores.get(lag - 1, -1), scores.get(lag + 1, -1))
    ]
    if not peaks:
        return None, best
    peak = peaks[0]
    mid = scores[peak]
    left, right = scores.get(peak - 1, mid), scores.get(peak + 1, mid)
    denominator = left - 2 * mid + right
    shift = 0.5 * (left - right) / denominator if denominator < -1e-7 else 0.0
    return peak + max(-0.5, min(0.5, shift)), mid


def _step_clock(signal, period):
    """Use torso lows as approximate double-support events, with a stable gap clock."""
    usable = [(index, value) for index, value in enumerate(signal) if value is not None]
    if not usable:
        return [index / period for index in range(len(signal))]
    centered = mean(value for _, value in usable)
    frequency = math.tau / period
    cosine = sum((centered - value) * math.cos(index * frequency) for index, value in usable)
    sine = sum((centered - value) * math.sin(index * frequency) for index, value in usable)
    start = math.atan2(sine, cosine) / frequency

    contacts = []
    first_step = math.floor(-start / period) - 1
    last_step = math.ceil((len(signal) - start) / period) + 1
    for step in range(first_step, last_step + 1):
        center = start + step * period
        reach = 0.21 * period
        first = max(1, math.ceil(center - reach))
        last = min(len(signal) - 2, math.floor(center + reach))
        candidates = []
        for index in range(first, last + 1):
            left, value, right = signal[index - 1:index + 2]
            if left is not None and value is not None and right is not None and value <= min(left, right):
                candidates.append((value, index))
        if candidates:
            _, low = min(candidates)
            center += (low - center) * 0.53
        contacts.append(center)

    clock = []
    for index in range(len(signal)):
        left = bisect_right(contacts, index) - 1
        fraction = (index - contacts[left]) / (contacts[left + 1] - contacts[left])
        clock.append(first_step + left + fraction)
    return clock


def _missing_weights(available, run_length, transition):
    weights = [0.0] * len(available)
    for valid, run in groupby(range(len(available)), available.__getitem__):
        indices = list(run)
        if valid or len(indices) < run_length:
            continue
        start, end = indices[0], indices[-1]
        for index in indices:
            before = index - start + 1 if start else math.inf
            after = end - index + 1 if end + 1 < len(weights) else math.inf
            fraction = min(1.0, before / transition, after / transition)
            weights[index] = fraction * fraction * (3 - 2 * fraction)
    return weights


def _trusted_chain(trusted, side, fps):
    upper = trusted[f"thigh.{side}"]
    lower = trusted[f"shin.{side}"]
    pairs = (a and b for a, b in zip(upper, lower, strict=True))
    result = []
    minimum = max(2, round(0.13 * fps))
    for reliable, run in groupby(pairs):
        length = sum(1 for _ in run)
        result.extend([reliable and length >= minimum] * length)
    return result


def _foot_motion(phase, stride, clearance):
    if phase < STANCE:
        return stride * math.cos(math.pi * phase / STANCE), 0.0
    swing = (phase - STANCE) / (1 - STANCE)
    return -stride * math.cos(math.pi * swing), clearance * math.sin(math.pi * swing) ** 2


def _directions(chest):
    forward = chest @ FRONT
    forward.z = 0
    forward = forward.normalized() if forward.length > 1e-5 else FRONT.copy()
    return forward, UP.cross(forward).normalized()


def settle_unobserved_legs(tracks, trusted, inferred, source_start, fps):
    """Relax long unseen legs if neither observations nor gait supply a pose."""
    count = len(tracks["torso"])
    rate = float(fps)
    for side, sign in (("L", 1), ("R", -1)):
        available = _trusted_chain(trusted, side, rate)
        for row in inferred:
            if row["control"] == f"knee.{side}":
                first, last = (frame - source_start for frame in row["source_range"])
                available[first:last + 1] = [True] * (last - first + 1)
        weights = _missing_weights(available, max(2, round(0.19 * rate)), max(1, round(0.20 * rate)))
        for index, weight in enumerate(weights):
            if weight <= 0:
                continue
            forward, left = _directions(tracks["torso"][index])
            thigh = (-UP + sign * 0.11 * left + 0.055 * forward).normalized()
            shin = (-UP + sign * 0.10 * left - 0.055 * forward).normalized()
            for name, relaxed in (("thigh", thigh), ("shin", shin)):
                track = tracks[f"{name}.{side}"]
                track[index] = blend(track[index], relaxed, weight)


def _match_support_side(clock, visible, tracks, length, stride):
    """Visible chains can choose between the two otherwise ambiguous foot phases."""
    evidence = []
    for side in SIDES:
        thigh, shin = tracks[f"thigh.{side}"], tracks[f"shin.{side}"]
        a, b = length[side]
        for index, valid in enumerate(visible[side]):
            if valid:
                forward, _ = _directions(tracks["torso"][index])
                observed = (thigh[index] * a + shin[index] * b).dot(forward)
                phase = (clock[index] / 2 + (0.5 if side == "R" else 0.0)) % 1
                expected, _ = _foot_motion(phase, stride, 0)
                swapped, _ = _foot_motion((phase + 0.5) % 1, stride, 0)
                evidence.append((observed, expected, swapped))
    if len(evidence) < 4:
        return 0.0, "first_side_unobserved"
    observed, first, second = zip(*evidence, strict=True)
    if pstdev(observed) < 0.045 or pstdev(first) < 0.045 or pstdev(second) < 0.045:
        return 0.0, "first_side_unobserved"
    normal, mirrored = correlation(observed, first), correlation(observed, second)
    if max(normal, mirrored) < 0.32 or abs(normal - mirrored) < 0.18:
        return 0.0, "first_side_unobserved"
    return (0.5 if mirrored > normal else 0.0), "observed_leg_alignment"


def _leg_plans(phase, center, chest, lift, offsets, body_weight, rest, lengths, stride, clearance):
    forward, lateral = _directions(chest)
    sway = lateral * (0.022 * sum(lengths["L"]) * math.sin(math.tau * phase) * body_weight)
    offset = sway + UP * offsets
    plans = {}
    excess = 0.0
    for side in SIDES:
        thigh = rest[f"thigh.{side}"]
        shin = rest[f"shin.{side}"]
        side_phase = (phase + (0.5 if side == "R" else 0.0)) % 1
        travel, elevation = _foot_motion(side_phase, stride, clearance)
        side_sign = 1.0 if side == "L" else -1.0
        hip_in_rig = thigh.head_local - center
        foot_width = hip_in_rig.x + side_sign * 0.02 * sum(lengths[side])
        goal = center + forward * travel + lateral * foot_width
        goal.z = shin.tail_local.z + elevation
        hip = center + chest @ hip_in_rig + UP * lift + offset
        delta = goal - hip
        maximum = sum(lengths[side]) * 0.996
        max_depth = math.sqrt(max(1e-8, maximum * maximum - delta.xy.length_squared))
        excess = smooth_maximum(excess, -delta.z - max_depth, 0.004 * sum(lengths[side]))
        plans[side] = hip, goal
    return plans, offset, forward, excess


def complete_walk(tracks, trusted, rest, capture, rate, *, two_bone, mode="auto", lift=None):
    """Fill sustained obscured leg tracks and return additive body offsets in rig space."""
    count = len(tracks["torso"])
    offsets = [Vector((0, 0, 0)) for _ in range(count)]
    diagnostics = {"mode": mode, "inferred_frames": {"L": 0, "R": 0}, "cadence_source": "none"}
    if mode == "none":
        return [], diagnostics, offsets
    if mode not in ("auto", "walk"):
        raise ValueError(f"Unsupported gait completion mode: {mode}")
    fps = float(rate)
    lift = [0.0] * count if lift is None else lift
    visibility = {side: _trusted_chain(trusted, side, fps) for side in SIDES}
    weights = {side: _missing_weights(visibility[side], max(3, round(0.25 * fps)), max(1, round(0.16 * fps)))
               for side in SIDES}
    if not any(weight > 0 for side in SIDES for weight in weights[side]):
        return [], diagnostics, offsets

    signal = _visible_signal(capture, count)
    period, score = estimate_step_period(signal, fps)
    if period is None and mode == "auto":
        diagnostics["cadence_confidence"] = round(score, 3)
        return [], diagnostics, offsets
    measured = period is not None
    period = period if measured else 0.52 * fps
    clock = _step_clock(signal if measured else [None] * count, period)
    lengths = {side: (rest[f"thigh.{side}"].length, rest[f"shin.{side}"].length) for side in SIDES}
    leg_span = mean(sum(pair) for pair in lengths.values())
    cadence = fps / period
    stride = leg_span * max(0.16, min(0.26, 0.16 + 0.064 * (cadence - 1.3)))
    clearance = leg_span * 0.10
    phase_offset, phase_source = _match_support_side(clock, visibility, tracks, lengths, stride)
    center = rest["torso"].head_local.copy()
    fitted_lift = stabilize_walk_lift(lift, clock, [value is not None for value in signal], leg_span, measured)

    for index, chest in enumerate(tracks["torso"]):
        left_weight, right_weight = weights["L"][index], weights["R"][index]
        body_weight = min(left_weight, right_weight)
        if max(left_weight, right_weight) == 0:
            continue
        phase = (clock[index] / 2 + phase_offset) % 1
        vertical = (-0.035 * leg_span + fitted_lift[index] - lift[index]) * body_weight
        plans, offset, forward, excess = _leg_plans(phase, center, chest, lift[index], vertical,
                                                     body_weight, rest, lengths, stride, clearance)
        correction = UP * (excess * body_weight)
        offsets[index] = offset - correction
        for side in SIDES:
            weight = weights[side][index]
            if weight == 0:
                continue
            hip, goal = plans[side]
            hip -= correction
            solution = two_bone(hip, goal, *lengths[side], forward)
            upper = (solution["joint"] - hip).normalized()
            lower = (solution["end"] - solution["joint"]).normalized()
            thigh, shin = tracks[f"thigh.{side}"], tracks[f"shin.{side}"]
            thigh[index] = blend(thigh[index], upper, weight)
            shin[index] = blend(shin[index], lower, weight)

    start = capture["source_range"][0]
    ranges = []
    for side in SIDES:
        for inferred, run in groupby(range(count), lambda index: weights[side][index] > 0):
            indices = list(run)
            if inferred:
                ranges.append({"control": f"knee.{side}", "source_range": [start + indices[0], start + indices[-1]],
                               "treatment": "inferred", "reason": "occluded_walking_leg"})
                diagnostics["inferred_frames"][side] += len(indices)
    diagnostics.update(
        cadence_source="torso_image_rhythm" if measured else "walking_mode_prior",
        step_period_frames=round(period, 3),
        steps_per_minute=round(cadence * 60, 1),
        cadence_confidence=round(score, 3),
        phase_source=phase_source,
    )
    return ranges, diagnostics, offsets
