"""Reconstruct elbows and knees from a shared reach, flex angle and bend plane."""

from itertools import groupby
import math

from mathutils import Quaternion, Vector

from .stabilize import angle, blend, stabilize_track


def _perpendicular(vector, axis):
    return vector - axis * vector.dot(axis)


def _carry_pole(pole, before, after):
    if before.dot(after) < -0.999999:
        rotation = Quaternion(pole, math.pi)
    else:
        rotation = before.rotation_difference(after)
    return _perpendicular(rotation @ pole, after).normalized()


def _reference_poles(reaches, body, kind):
    first = reaches[0]
    body_axis = body @ Vector((0, 1 if kind == "arm" else -1, 0))
    pole = _perpendicular(body_axis, first)
    if pole.length < 0.15:
        pole = _perpendicular(body @ Vector((0, 0, 1)), first)
    references = [pole.normalized()]
    for before, after in zip(reaches, reaches[1:]):
        references.append(_carry_pole(references[-1], before, after))
    return references


def _trusted_leg_samples(supported, rate):
    minimum = min(len(supported), max(2, round(0.10 * rate)))
    filtered = []
    for trusted, run in groupby(supported):
        count = sum(1 for _ in run)
        filtered.extend([trusted and count >= minimum] * count)
    return filtered


def _reach_samples(upper, lower, supported, lengths, kind, rate):
    a, b = lengths
    spans = [a * u + b * v for u, v in zip(upper, lower, strict=True)]
    reaches = [span.normalized() if span.length > 0.07 * (a + b) else None for span in spans]
    if kind == "leg" and any(supported):
        # During a sustained knee occlusion, the thigh alone cannot locate the ankle.
        for trusted, run in groupby(range(len(supported)), supported.__getitem__):
            indexes = list(run)
            if not trusted and len(indexes) >= max(1, round(0.25 * rate)):
                reaches[indexes[0]:indexes[-1] + 1] = [None] * len(indexes)
    return reaches


def limit_steps(values, limit):
    def sweep(sequence):
        track = [sequence[0]]
        for target in sequence[1:]:
            previous = track[-1]
            change = angle(previous, target)
            track.append(blend(previous, target, limit / change) if change > limit else target)
        return track

    # Bidirectional limiting gives the re-entry time support on both sides.
    forward, backward = sweep(values), sweep(values[::-1])[::-1]
    return [blend(a, b, 0.5) for a, b in zip(forward, backward, strict=True)]


def solve_chain(upper, lower, supported, *, upper_length, lower_length, torso, kind, part,
                source_start, fps, two_bone, max_angular_speed=None):
    """Return two fixed-length direction tracks and ranges needing joint review."""
    rate = float(fps)
    if kind == "leg":
        supported = _trusted_leg_samples(supported, rate)
    lengths = upper_length, lower_length
    sample_reaches = _reach_samples(upper, lower, supported, lengths, kind, rate)
    reaches, _ = stabilize_track(sample_reaches, part=part, source_start=source_start,
                                 fps=rate, fallback=upper[0])
    if max_angular_speed is not None:
        reaches = limit_steps(reaches, math.radians(max_angular_speed / rate))
    references = _reference_poles(reaches, torso[0], kind)

    limit = math.radians(150 if kind == "arm" else 155)
    plane_axis = Vector((0, 0, 1))
    bend_values, pole_values = [], []
    for u, v, trusted, reference, reach, observed_reach in zip(
        upper, lower, supported, references, reaches, sample_reaches, strict=True,
    ):
        bend = angle(u, v)
        if trusted:
            flex = min(bend, limit)
            bend_values.append(Vector((math.cos(flex), math.sin(flex), 0)))
        else:
            bend_values.append(None)
        pole = None
        if trusted and observed_reach is not None and bend > math.radians(16):
            chord = _perpendicular(u, observed_reach)
            if chord.length > 0.12:
                p = _carry_pole(chord.normalized(), observed_reach, reach)
                offset = math.atan2(p.dot(reach.cross(reference)), p.dot(reference))
                pole = Quaternion(plane_axis, offset)
        pole_values.append(pole)

    flat = Vector((1, 0, 0))
    bends, ranges = stabilize_track(bend_values, part=part, source_start=source_start,
                                    fps=rate, fallback=flat)
    poles, _ = stabilize_track(pole_values, part=part, source_start=source_start,
                               fps=rate, fallback=Quaternion())
    # A coherent source keypoint can still jump when a covered joint is re-estimated.
    # Distribute steep flex and plane changes around that transition.
    speed = max_angular_speed if max_angular_speed is not None else (720 if kind == "arm" else 540)
    step = math.radians(speed / rate)
    bends = limit_steps(bends, step)
    poles = limit_steps(poles, step)

    result_upper, result_lower = [], []
    start = Vector((0, 0, 0))
    a, b = lengths
    for direction, reference, bend, pole in zip(reaches, references, bends, poles, strict=True):
        distance = math.sqrt(a * a + b * b + 2 * a * b * bend.x)
        planar = pole @ flat
        lateral = reference * planar.x + direction.cross(reference) * planar.y
        solution = two_bone(start, direction * distance, a, b, lateral)
        middle, end = solution["joint"], solution["end"]
        result_upper.append(middle.normalized())
        result_lower.append((end - middle).normalized())
    for row in ranges:
        if row["reason"] == "missing_observation":
            row["reason"] = "joint_chain_uncertain"
    return result_upper, result_lower, ranges
