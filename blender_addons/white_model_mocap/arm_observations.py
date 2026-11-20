"""Check posterior arm observations before filling their missing intervals."""

from itertools import groupby
import math
from statistics import median

from mathutils import Vector

from .observations import MIN_CONFIDENCE, _is_visible, _screen_point


ARM_JOINTS = {"L": (11, 13, 15), "R": (12, 14, 16)}


def _image_tracks(samples, source_size, joints):
    tracks = []
    for sample in samples:
        if sample is None or not all(_is_visible(sample[i], MIN_CONFIDENCE) for i in (11, 12)):
            tracks.append(None)
            continue
        shoulder, elbow, wrist = (_screen_point(sample[i], source_size) for i in joints)
        other = 12 if joints[0] == 11 else 11
        opposite_shoulder = _screen_point(sample[other], source_size)
        width = max(0.045 * source_size[1], (shoulder - opposite_shoulder).length)
        center_x = (shoulder.x + opposite_shoulder.x) / 2
        # A covered wrist reported near the head needs continuous image support,
        # since detector confidence by itself is often high for occluded points.
        raised_elbow = _is_visible(sample[joints[1]], MIN_CONFIDENCE) and elbow.y > shoulder.y + 0.08 * width
        wrist_near_head = (_is_visible(sample[joints[2]], MIN_CONFIDENCE)
                           and wrist.y > shoulder.y + 0.08 * width and abs(wrist.x - center_x) < 0.72 * width)
        near_head = raised_elbow or wrist_near_head
        tracks.append((elbow - shoulder, wrist - shoulder, width, near_head))
    return tracks


def _connected(a, b, fps):
    if a is None or b is None:
        return False
    max_step = max(a[2], b[2]) * min(0.75, 12.0 / fps)
    return max((a[i] - b[i]).length for i in (0, 1)) <= max_step


def _supported_raised(tracks, samples, frames, parts, joints, fps):
    confirmed = set()
    minimum_run = max(2, math.ceil(0.14 * fps))
    run = []
    for i, track in enumerate(tracks):
        eligible = (track is not None and track[3] and all(frames[i][part] is not None for part in parts)
                    and samples[i][joints[1]][5] >= 0.75 and samples[i][joints[2]][5] >= 0.73)
        if run and (not eligible or not _connected(tracks[run[-1]], track, fps)):
            if len(run) >= minimum_run:
                confirmed.update(run)
            run = []
        if eligible:
            run.append(i)
    if len(run) >= minimum_run:
        confirmed.update(run)
    return confirmed


def _projects_inside_torso(sample, source_size, wrist):
    if not all(_is_visible(sample[i], MIN_CONFIDENCE) for i in (23, 24)):
        return False
    shoulder_l, shoulder_r, hip_l, hip_r = (_screen_point(sample[i], source_size)
                                             for i in (11, 12, 23, 24))
    shoulders = (shoulder_l + shoulder_r) / 2
    hips = (hip_l + hip_r) / 2
    height = shoulders.y - hips.y
    shoulder_width = abs(shoulder_r.x - shoulder_l.x)
    if height <= 0.35 * shoulder_width:
        return False
    fraction = (shoulders.y - wrist.y) / height
    if not 0.1 <= fraction <= 1:
        return False
    center = shoulders.lerp(hips, fraction)
    half_width = 0.42 * ((1 - fraction) * shoulder_width + fraction * abs(hip_r.x - hip_l.x))
    return abs(wrist.x - center.x) < half_width


def _supported_depth(directions, index, radius):
    depths = [directions[index].y]
    for step in (-1, 1):
        for distance in range(1, radius + 1):
            neighbor = index + step * distance
            if not 0 <= neighbor < len(directions) or directions[neighbor] is None:
                break
            depths.append(directions[neighbor].y)
    return median(depths) * min(1, (len(depths) - 1) / 2)


def _projection(track, segment):
    elbow, wrist, _, _ = track
    return elbow if segment == 0 else wrist - elbow


def filter_rear_arms(samples, frames, source_size, *, fps, source_start):
    """Retain image-plane arm direction and use nearby frames to corroborate depth."""
    if not source_size:
        return []

    ranges = []
    radius = max(1, round(0.065 * float(fps)))
    for side, joints in ARM_JOINTS.items():
        parts = (f"upper_arm.{side}", f"forearm.{side}")
        tracks = _image_tracks(samples, source_size, joints)
        supported_raised = _supported_raised(tracks, samples, frames, parts, joints, float(fps))
        reasons = {part: [None] * len(frames) for part in parts}
        for i, (track, frame) in enumerate(zip(tracks, frames, strict=True)):
            if track is None:
                continue
            _, _, width, near_head = track
            for segment, part in enumerate(parts):
                direction = frame[part]
                if direction is None:
                    continue
                projected = _projection(track, segment)
                plane = Vector((direction.x, direction.z))
                if near_head and i not in supported_raised:
                    reasons[part][i] = "unconfirmed_head_level_arm"
                elif (projected.length >= 0.17 * width and plane.length >= 0.38
                      and plane.normalized().dot(projected.normalized()) < -0.3):
                    reasons[part][i] = "arm_projection_conflict"
                elif projected.length < 0.13 * width:
                    reasons[part][i] = "arm_projection_too_short"
                elif (segment == 1 and direction.y < -0.32
                      and _projects_inside_torso(samples[i], source_size,
                                                 _screen_point(samples[i][joints[2]], source_size))):
                    # This image cannot distinguish an occluded hand in front of
                    # the body from a hand against the back. Leave the ambiguous
                    # out-of-plane reconstruction for the gap solver.
                    reasons[part][i] = "arm_depth_ambiguous_behind_torso"
                if reasons[part][i] is not None:
                    frame[part] = None

        for segment, part in enumerate(parts):
            directions = [frame[part] for frame in frames]
            for i, direction in enumerate(directions):
                if direction is None:
                    continue
                track = tracks[i]
                projected = _projection(track, segment)
                # A long projection warrants less depth. A genuinely foreshortened
                # segment can retain depth if its neighbors corroborate it.
                projected_width = projected.length / track[2]
                limit = 0.32 + 0.40 * max(0, min(1, (0.75 - projected_width) / 0.55))
                depth = max(-limit, min(limit, _supported_depth(directions, i, radius)))
                projected = projected.normalized() * math.sqrt(1 - depth * depth)
                frames[i][part] = Vector((projected.x, depth, projected.y))
            classified = reasons[part]
            for reason, run in groupby(range(len(classified)), classified.__getitem__):
                if reason is not None:
                    indices = list(run)
                    ranges.append({"control": part,
                                   "source_range": [source_start + indices[0], source_start + indices[-1]],
                                   "treatment": "rejected", "reason": reason})
    return ranges


