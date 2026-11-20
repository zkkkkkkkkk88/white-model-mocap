"""Separate anatomical arm twist and hidden-forearm completion from landmark tracking."""

from itertools import groupby
import math
from statistics import median

from mathutils import Quaternion, Vector

from .joint_limits import flex_axis, hinge_direction
from .stabilize import blend


LONG = Vector((0, 1, 0))
SIDE = Vector((1, 0, 0))


def _align(rotation, direction):
    """Swing a bone onto a direction without adding twist about its length."""
    previous_direction = rotation @ LONG
    if previous_direction.dot(direction) < -0.9999:
        rotation = Quaternion(rotation @ SIDE, math.pi) @ rotation
        previous_direction = rotation @ LONG
    return (previous_direction.rotation_difference(direction) @ rotation).normalized()


def _twist_delta(previous, target, axis):
    before = previous @ SIDE
    after = target @ SIDE
    return math.atan2(axis.dot(before.cross(after)), before.dot(after))


def _bounded_twist(previous, target, direction, max_twist, reliability):
    if previous is None:
        return target
    aligned = _align(previous, direction)
    correction = _twist_delta(aligned, target, direction)
    correction = max(-max_twist, min(max_twist, correction))
    return (Quaternion(direction, reliability * correction) @ aligned).normalized()


def _alignment_support(orientation, direction):
    cosine = (orientation @ LONG).dot(direction)
    return min(1.0, max(0.0, (cosine + 0.985) / 0.15))


def arm_rotations(chest, upper_track, lower_track, upper_rest_rotation, lower_rest_rotation, fps):
    """Anchor arm roll to the moving torso and keep the elbow twist bounded."""
    upper_rotations, lower_rotations = [], []
    max_twist = math.radians(270 / float(fps))
    hinge_limit = math.radians(30)
    bend_axis = flex_axis(upper_rest_rotation, lower_rest_rotation)
    hinge_twist = (_align(upper_rest_rotation, lower_rest_rotation @ LONG).inverted()
                   @ lower_rest_rotation)
    previous_upper = previous_lower = None
    for body, upper_world, lower_world in zip(chest, upper_track, lower_track, strict=True):
        # Use torso space so changes in the estimated body heading do not
        # consume the arm's twist allowance or appear as an extra wrist spin.
        inverse = body.inverted()
        upper_dir = (inverse @ upper_world).normalized()
        lower_dir = (inverse @ lower_world).normalized()
        # Near the antipode the shortest arc cannot identify twist. Rely more
        # on continuity there and re-anchor when the reference is well conditioned.
        target_upper = _align(upper_rest_rotation, upper_dir)
        bend = lower_dir - upper_dir * lower_dir.dot(upper_dir)
        if bend.length > 1e-6:
            neutral = target_upper @ bend_axis
            twist = math.atan2(upper_dir.dot(neutral.cross(bend)), neutral.dot(bend))
            twist = max(-math.radians(105), min(math.radians(105), twist))
            # Direction near full extension provides little evidence for humeral roll.
            strength = min(1.0, max(0.0, (bend.length - 0.08) / 0.30))
            target_upper = (Quaternion(upper_dir, twist * strength) @ target_upper).normalized()
        upper = _bounded_twist(previous_upper, target_upper, upper_dir,
                               max_twist, _alignment_support(upper_rest_rotation, upper_dir))
        lower_dir = hinge_direction(upper, lower_dir, bend_axis, math.radians(145), math.radians(23))
        stable_lower = _align(lower_rest_rotation, lower_dir)
        hinge_lower = (_align(upper, lower_dir) @ hinge_twist).normalized()
        hinge_offset = _twist_delta(stable_lower, hinge_lower, lower_dir)
        bounded_offset = max(-hinge_limit, min(hinge_limit, hinge_offset))
        reliability = _alignment_support(lower_rest_rotation, lower_dir)
        offset = reliability * bounded_offset + (1 - reliability) * hinge_offset
        target = (Quaternion(lower_dir, offset) @ stable_lower).normalized()
        lower = _bounded_twist(previous_lower, target, lower_dir, max_twist,
                               max(reliability, _alignment_support(upper, lower_dir)))
        upper_rotations.append((body @ upper).normalized())
        lower_rotations.append((body @ lower).normalized())
        previous_upper, previous_lower = upper, lower
    return upper_rotations, lower_rotations


def limit_arm_swing(track, chest, *, part, swing, forward_bent=False, low_carry=False):
    """Reduce arm movement in torso space and apply an optional held posture."""
    if swing == 1.0 and not forward_bent:
        return track
    local = [(body.inverted() @ direction).normalized() for body, direction in zip(chest, track, strict=True)]
    sign = 1 if part.endswith(".L") else -1
    is_forearm = part.startswith("forearm")

    if forward_bent:
        center = Vector((0, -1, 0.34) if is_forearm else (sign * 0.32, -0.15, -1))
        limit = math.radians(18 if is_forearm else 10)
        if is_forearm:
            local = [Vector((vec.x, -abs(vec.y), vec.z)).normalized() for vec in local]
    elif low_carry:
        center = Vector((sign * (0.09 if is_forearm else 0.18), -0.06, -1))
        limit = math.radians(17)
    else:
        center = Vector([median(vec[axis] for vec in local) for axis in range(3)])
        if center.length < 1e-5:
            center = local[len(local) // 2]
        limit = math.pi

    center.normalize()
    result = []
    for body, direction in zip(chest, local, strict=True):
        angle = center.angle(direction)
        weight = swing if angle <= limit else min(swing, limit / angle)
        restrained = blend(center, direction, weight)
        if forward_bent:
            # Keep the elbow outside the torso and the wrist from cutting
            # inward before the forearm has passed the front of the body.
            minimum = -0.12 if is_forearm else 0.26
            restrained.x = sign * max(minimum, sign * restrained.x)
            restrained.normalize()
        result.append(body @ restrained)
    return result


def _distances_to_observation(supported):
    length = len(supported)
    before = -length
    distances = [0] * length
    for i, known in enumerate(supported):
        if known:
            before = i
        distances[i] = i - before
    after = 2 * length
    for i in range(length - 1, -1, -1):
        if supported[i]:
            after = i
        distances[i] = min(distances[i], after - i)
    return distances


def _settle_weight(frames, fps):
    amount = min(1.0, max(0.0, (frames / fps - 0.075) / 0.24))
    return amount * amount * (3 - 2 * amount)


def settle_unobserved_arm(upper, lower, trusted_upper, trusted_lower, chest, *, side, fps):
    """Coast through brief gaps and relax arms that were out of frame longer."""
    sign = 1 if side == "L" else -1
    relaxed_upper = Vector((sign * 0.23, -0.05, -1)).normalized()
    relaxed_forearm = Vector((sign * 0.13, -0.20, -1)).normalized()
    upper_age = _distances_to_observation(trusted_upper)
    lower_age = _distances_to_observation(trusted_lower)
    result_upper, result_lower = [], []
    for u, v, body, age_upper, age_lower in zip(upper, lower, chest, upper_age, lower_age, strict=True):
        u = blend(u, body @ relaxed_upper, _settle_weight(age_upper, fps))
        guide = blend(u, body @ relaxed_forearm, 0.64)
        v = blend(v, guide, _settle_weight(age_lower, fps))
        result_upper.append(u)
        result_lower.append(v)
    return result_upper, result_lower


def _crosses_back(shoulder_offset, upper, lower, lengths):
    shoulder_width = abs(shoulder_offset.x)
    upper_length, forearm_length = lengths
    elbow = shoulder_offset + upper * upper_length
    wrist = elbow + lower * forearm_length
    # In the template basis, front is -Y. Hidden pose completion should keep
    # the forearm out of the central back instead of copying an ambiguous depth.
    left, right = -0.74 * shoulder_width, 0.74 * shoulder_width
    bottom, top = shoulder_offset.z - 1.9 * upper_length, shoulder_offset.z + 0.04 * upper_length
    for fraction in (0.25, 0.50, 0.75, 1.0):
        point = elbow.lerp(wrist, fraction)
        if (left < point.x < right and bottom < point.z < top
                and point.y > shoulder_offset.y - 0.29 * upper_length):
            return True
    return False


def stabilize_rear_arm(upper, lower, trusted_upper, trusted_lower, chest, *, side, fps,
                       source_start, shoulder_offset, lengths):
    """Complete unsupported arm observations in the torso frame, keeping supported poses."""
    output_upper, output_lower, reasons = [], [], []
    rate = float(fps)
    upper_age = _distances_to_observation(trusted_upper)
    lower_age = _distances_to_observation(trusted_lower)
    sign = 1 if side == "L" else -1
    resting_upper = Vector((sign * 0.15, -0.055, -1)).normalized()
    resting_fore = Vector((sign * 0.12, -0.23, -1)).normalized()
    for index, (u, v, body) in enumerate(zip(upper, lower, chest, strict=True)):
        inverse = body.inverted()
        local_upper, local_fore = inverse @ u, inverse @ v
        upper_weight = _settle_weight(upper_age[index], rate)
        fore_weight = _settle_weight(lower_age[index], rate)
        reasons.append(None)
        if upper_weight:
            local_upper = blend(local_upper, resting_upper, upper_weight)
            if not fore_weight:
                reasons[-1] = "upper_arm_temporarily_missing"
        if not trusted_lower[index]:
            # If an elbow remains visible, let its direction drive the hidden
            # wrist until a nearby wrist observation can corroborate a bend.
            guide = blend(local_upper, resting_fore, 0.67)
            ambiguous_back = _crosses_back(shoulder_offset, local_upper, local_fore, lengths)
            if ambiguous_back:
                local_fore = guide
                reasons[-1] = "hidden_forearm_crosses_back"
            elif fore_weight:
                local_fore = blend(local_fore, guide, fore_weight)
                reasons[-1] = "forearm_temporarily_missing"
        output_upper.append(body @ local_upper)
        output_lower.append(body @ local_fore)
    ranges = []
    for reason, run in groupby(range(len(reasons)), reasons.__getitem__):
        indices = list(run)
        if reason:
            ranges.append({"control": f"elbow.{side}",
                           "source_range": [source_start + indices[0], source_start + indices[-1]],
                           "treatment": "settled", "reason": reason})
    return output_upper, output_lower, ranges
