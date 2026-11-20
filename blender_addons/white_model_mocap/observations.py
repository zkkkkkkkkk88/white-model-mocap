"""Convert MediaPipe landmarks into independent camera-aligned pose observations."""

from itertools import groupby
import math

from mathutils import Matrix, Vector

from .stabilize import angle, blend


LIMBS = {
    "upper_arm.L": (11, 13), "forearm.L": (13, 15),
    "upper_arm.R": (12, 14), "forearm.R": (14, 16),
    "thigh.L": (23, 25), "shin.L": (25, 27),
    "thigh.R": (24, 26), "shin.R": (26, 28),
}
PARTS = ("torso", "head", *LIMBS)
MIN_CONFIDENCE = 0.5
TORSO_POINTS = (11, 12, 23, 24)
ARM_POINTS = {part: ids for part, ids in LIMBS.items() if part.startswith(("upper_arm", "forearm"))}
LEFT_RIGHT_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
                    *((i, i + 1) for i in range(11, 33, 2)))


def _is_visible(point, threshold):
    return (point[5] >= threshold and 0 <= point[3] <= 1 and 0 <= point[4] <= 1
            and all(math.isfinite(value) for value in point))


def _world_point(point):
    x, y, z = point[:3]
    return Vector((x, z, -y))


def _screen_point(point, source_size):
    return Vector((point[3] * source_size[0], -point[4] * source_size[1]))


def _torso_vector(points):
    shoulder_l, shoulder_r, hip_l, hip_r = (points[i] for i in TORSO_POINTS)
    return (shoulder_l + shoulder_r - hip_l - hip_r) / 2


def _perpendicular(vector, axis):
    return vector - axis * vector.dot(axis)


def rear_sample(sample, source_size):
    """Use the shoulders to resolve image-side labels in a confirmed rear shot."""
    if sample is None or not all(_is_visible(sample[i], MIN_CONFIDENCE) for i in (11, 12)):
        return None
    left = _screen_point(sample[11], source_size)
    right = _screen_point(sample[12], source_size)
    shoulders = left - right
    if abs(shoulders.x) < 0.035 * source_size[1]:
        # A compressed shoulder span does not establish the side assignment.
        return None
    if all(_is_visible(sample[i], MIN_CONFIDENCE) for i in (23, 24)):
        hips = (_screen_point(sample[23], source_size) + _screen_point(sample[24], source_size)) / 2
        if hips.y > ((left + right) / 2).y + 0.15 * shoulders.length:
            # Even a highly confident upside-down fit is incompatible with this mode.
            return None
    if shoulders.x < 0:
        return sample
    sample = list(sample)
    for left_id, right_id in LEFT_RIGHT_PAIRS:
        sample[left_id], sample[right_id] = sample[right_id], sample[left_id]
    return sample


def _rear_torso_rotation(sample, source_size):
    """Estimate shoulder roll and corroborated modest yaw around the rear view."""
    shoulders = _screen_point(sample[11], source_size) - _screen_point(sample[12], source_size)
    roll_limit = math.radians(14)
    roll = max(-roll_limit, min(roll_limit, math.atan2(shoulders.y, -shoulders.x)))
    yaw = 0.0
    if all(_is_visible(sample[i], MIN_CONFIDENCE) for i in (23, 24)):
        hip_width = _screen_point(sample[23], source_size).x - _screen_point(sample[24], source_size).x
        shoulder_axis = _world_point(sample[11]) - _world_point(sample[12])
        hip_axis = _world_point(sample[23]) - _world_point(sample[24])
        if (hip_width < -0.16 * abs(shoulders.x) and shoulder_axis.x < -0.05
                and hip_axis.x < -0.05 and shoulder_axis.y * hip_axis.y > 0):
            shoulder_yaw = math.atan2(shoulder_axis.y, -shoulder_axis.x)
            hip_yaw = math.atan2(hip_axis.y, -hip_axis.x)
            limit = math.radians(20)
            yaw = max(-limit, min(limit, 0.55 * shoulder_yaw + 0.25 * hip_yaw))
    up = Vector((math.sin(roll), 0, math.cos(roll)))
    left = Vector((-math.cos(roll) * math.cos(yaw), math.sin(yaw), math.sin(roll) * math.cos(yaw)))
    return Matrix((left, up.cross(left), up)).transposed().to_quaternion()


def _spine_projection_agrees(spine, screen_spine, shoulder_width):
    planar = Vector((spine.x, spine.z))
    # Very short projections occur in real foreshortening. They give no directional check.
    if planar.length < 0.25 * spine.length or screen_spine.length < 0.25 * shoulder_width:
        return True
    return planar.normalized().dot(screen_spine.normalized()) > 0.35


def _corroborated_spine(sample, spine, projected, shoulder_width, image_height):
    return (
        projected.length >= max(0.056 * image_height, 0.40 * shoulder_width)
        and _spine_projection_agrees(spine, projected, shoulder_width)
        and (spine.z >= 0.52 * spine.length or (
            projected.y >= 0.30 * projected.length
            and min(sample[i][5] for i in TORSO_POINTS) >= 0.75
        ))
    )


def _torso_rotation(points, spine):
    up = spine.normalized()
    shoulders = _perpendicular(points[11] - points[12], up)
    hips = _perpendicular(points[23] - points[24], up)
    # Hip direction steadies a compressed shoulder baseline. Conflicting baselines
    # can indicate a bad reconstruction, so only combine those in the same hemisphere.
    if shoulders.dot(hips) >= 0:
        lateral = shoulders + 0.65 * hips
    elif shoulders.length > 1.3 * hips.length:
        lateral = shoulders
    else:
        return None
    if lateral.length < 0.14 * spine.length:
        return None
    left = lateral.normalized()
    back = up.cross(left).normalized()
    return Matrix((back.cross(up).normalized(), back, up)).transposed().to_quaternion()


def _shoulder_rotation(points, screen, image_height):
    """Use a near-upright shoulder frame when pelvis depth is unavailable."""
    if 11 not in points or 12 not in points:
        return None
    lateral = points[11] - points[12]
    separation = Vector((lateral.x, lateral.y, 0))
    if separation.length < 0.06:
        return None
    up = Vector((0, 0, 1))
    if screen is not None:
        ears = [screen[index] for index in (7, 8) if index in points]
        if ears:
            above = sum(ears, Vector((0, 0))) / len(ears) - (screen[11] + screen[12]) / 2
            # A collapsed landmark cloud in a close-up can put both hips and
            # ears on the shoulder line despite high per-point confidence.
            if above.y < 0.02 * image_height:
                return None
            roll = max(-0.32, min(0.32, above.x / max(above.y, 0.08 * image_height) * 0.5))
            up = Vector((roll, 0, 1)).normalized()
    left = _perpendicular(separation, up).normalized()
    back = up.cross(left).normalized()
    return Matrix((back.cross(up).normalized(), back, up)).transposed().to_quaternion()


def _partial_hip_rotation(sample, source_size):
    """Estimate a deep lean from one visible hip in a cropped shot."""
    if sample is None or not all(_is_visible(sample[i], 0.68) for i in (11, 12)):
        return None
    if any(_is_visible(sample[i], MIN_CONFIDENCE) for i in (25, 26)):
        return None
    visible_hip, weak_hip = sorted((23, 24), key=lambda i: sample[i][5], reverse=True)
    if not (_is_visible(sample[visible_hip], 0.66) and _is_visible(sample[weak_hip], 0.25)):
        return None

    points = {i: _world_point(sample[i]) for i in TORSO_POINTS}
    screen = {i: _screen_point(sample[i], source_size) for i in TORSO_POINTS}
    lateral = points[11] - points[12]
    if lateral.length < 0.10:
        return None
    left = lateral.normalized()
    spine = _perpendicular((points[11] + points[12]) / 2 - points[visible_hip], left)
    if spine.length < 0.60 * lateral.length:
        return None
    up = spine.normalized()
    if not 0.37 <= up.z <= 0.92:
        return None
    projected = _torso_vector(screen)
    shoulder_width = (screen[11] - screen[12]).length
    planar = Vector((up.x, up.z))
    if (projected.length < max(0.055 * source_size[1], 0.50 * shoulder_width)
            or projected.y < 0.46 * projected.length
            or planar.normalized().dot(projected.normalized()) < 0.84):
        return None
    # The farther hip corroborates the estimate without contributing to its rotation.
    rough_spine = _torso_vector(points)
    if rough_spine.length < 0.10 or up.dot(rough_spine.normalized()) < math.cos(math.radians(16)):
        return None
    bilateral = (sample[weak_hip][5] >= MIN_CONFIDENCE
                 and _corroborated_spine(sample, rough_spine, projected, shoulder_width, source_size[1])
                 and _torso_rotation(points, rough_spine) is not None)
    return Matrix((left, up.cross(left), up)).transposed().to_quaternion(), bilateral


def guide_partial_hips(frames, samples, source_size, *, fps, source_start):
    """Replace the upright shoulder fallback during a corroborated, continuous lean."""
    fps = float(fps)
    proposals = [_partial_hip_rotation(sample, source_size) for sample in samples]
    recovered = set()
    min_samples = max(3, round(0.15 * fps))
    blend_samples = max(2, round(0.13 * fps))
    step_limit = math.radians(520 / fps)
    for supported, run in groupby(range(len(proposals)), lambda i: proposals[i] is not None):
        if not supported:
            continue
        indices = list(run)
        if len(indices) < min_samples or any(angle(proposals[a][0], proposals[b][0]) > step_limit
                                             for a, b in zip(indices, indices[1:])):
            continue
        for offset, index in enumerate(indices):
            candidate, bilateral = proposals[index]
            torso = frames[index]["torso"]
            if bilateral or (torso is not None and angle(torso, candidate) < math.radians(12)):
                continue
            # Ease away from the shoulder estimate at the beginning of the visible run.
            strength = min(1.0, (offset + 2) / blend_samples)
            frames[index]["torso"] = blend(torso, candidate, strength) if torso is not None else candidate
            recovered.add(index)
    ranges = []
    for applied, run in groupby(range(len(proposals)), recovered.__contains__):
        if applied:
            indices = list(run)
            ranges.append({"control": "torso", "source_range": [source_start + indices[0], source_start + indices[-1]],
                           "treatment": "guided", "reason": "partially_occluded_pelvis"})
    return ranges


def _head_rotation(points):
    left = points[7] - points[8]
    if left.length < 0.025:
        return None
    ear_midpoint = (points[7] + points[8]) / 2
    nose_to_ears = _perpendicular(ear_midpoint - points[0], left.normalized())
    if nose_to_ears.length < 0.2 * left.length:
        return None
    left.normalize()
    back = nose_to_ears.normalized()
    return Matrix((left, back, left.cross(back).normalized())).transposed().to_quaternion()


def _pixel_geometry(sample, source_size):
    screen = [_screen_point(point, source_size) for point in sample]
    body_span = (_torso_vector(screen).length if all(_is_visible(sample[i], MIN_CONFIDENCE)
                                                       for i in TORSO_POINTS) else 0.0)
    return screen, body_span


def _fuse_screen_direction(direction, projected, minimum):
    plane = Vector((direction.x, direction.z))
    plane_length = plane.length
    projected_length = projected.length
    if projected_length < minimum or plane_length < 0.35:
        return direction

    plane /= plane_length
    projected = projected.normalized()
    agreement = max(-1.0, min(1.0, plane.dot(projected)))
    if agreement < -0.5 and projected_length >= 1.5 * minimum and plane_length >= 0.55:
        # Two well-defined projections in opposing directions are ambiguous.
        return None
    signed_angle = math.atan2(plane.x * projected.y - plane.y * projected.x, agreement)
    support = min(1.0, (projected_length - minimum) / minimum)
    angle = signed_angle * (0.6 * support * min(1.0, (plane_length - 0.35) / 0.35))
    cos, sin = math.cos(angle), math.sin(angle)
    return Vector(((plane.x * cos - plane.y * sin) * plane_length, direction.y,
                   (plane.x * sin + plane.y * cos) * plane_length)).normalized()


def observations(sample, source_size=None, *, rear_view=False):
    """Return world-space head/torso rotations and unit limb directions, or None per part."""
    result = dict.fromkeys(PARTS)
    if sample is None:
        return result

    points = {i: _world_point(point) for i, point in enumerate(sample) if _is_visible(point, MIN_CONFIDENCE)}
    screen, body_span = _pixel_geometry(sample, source_size) if source_size else (None, 0.0)
    torso_span = 0.0
    if all(i in points for i in TORSO_POINTS):
        spine = _torso_vector(points)
        torso_span = spine.length
        projected = _torso_vector(screen) if screen is not None else None
        corroborated = projected is None or _corroborated_spine(
            sample, spine, projected, (screen[11] - screen[12]).length, source_size[1],
        )
        if not rear_view and not corroborated and projected is not None and (
            projected.length < 0.056 * source_size[1]
            and _shoulder_rotation(points, screen, source_size[1]) is None
        ):
            return result
        if torso_span >= 1e-5 and not rear_view and corroborated:
            result["torso"] = _torso_rotation(points, spine)
    if rear_view:
        result["torso"] = _rear_torso_rotation(sample, source_size)
    elif result["torso"] is None:
        result["torso"] = _shoulder_rotation(points, screen, source_size[1] if source_size else 0)
    if not rear_view and all(i in points for i in (0, 7, 8)):
        result["head"] = _head_rotation(points)

    min_pixels = max(0.012 * source_size[1], 0.12 * body_span) if source_size else 0.0
    for part, (start, end) in LIMBS.items():
        if start not in points or end not in points:
            continue
        segment = points[end] - points[start]
        if segment.length < max(1e-5, 0.12 * torso_span):
            continue
        direction = segment.normalized()
        if screen is not None:
            direction = _fuse_screen_direction(direction, screen[end] - screen[start], min_pixels)
        result[part] = direction
    return result


def weak_arm_directions(sample, source_size):
    """Return provisional arm candidates with useful pixel evidence for temporal validation."""
    if sample is None or not source_size:
        return {}

    screen, body_span = _pixel_geometry(sample, source_size)
    minimum_span = max(0.022 * source_size[1], 0.2 * body_span)
    candidates = {}
    for part, (start, end) in ARM_POINTS.items():
        a, b = sample[start], sample[end]
        if not (_is_visible(a, 0.28) and _is_visible(b, 0.28) and min(a[5], b[5]) < MIN_CONFIDENCE):
            continue
        segment = _world_point(b) - _world_point(a)
        projected = screen[end] - screen[start]
        if segment.length < 1e-5 or projected.length < minimum_span:
            continue
        direction = segment.normalized()
        plane = Vector((direction.x, direction.z))
        projected.normalize()
        if plane.length >= 0.65 and plane.normalized().dot(projected) >= 0.82:
            candidates[part] = Vector((projected.x * plane.length, direction.y,
                                       projected.y * plane.length)).normalized()
    return candidates
