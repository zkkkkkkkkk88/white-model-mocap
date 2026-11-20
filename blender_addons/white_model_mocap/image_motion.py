"""Measure short-term torso image displacement against scene motion, using local video pixels."""

import math

import cv2
import numpy as np


def _anchor(sample, width, height):
    if sample is None:
        return None
    points = np.asarray([sample[i][3:6] for i in (11, 12, 23, 24)], dtype=float)
    if (not np.isfinite(points).all() or np.min(points[:2, 2]) < .58
            or np.min(points[2:, 2]) < .28 or not np.all((points[:, :2] >= 0) & (points[:, :2] <= 1))):
        return None
    shoulders, hips = points[:2, :2], points[2:, :2]
    shoulders *= (width, height)
    hips *= (width, height)
    shoulder = shoulders.mean(axis=0)
    hip = hips.mean(axis=0)
    span = abs(shoulders[0, 0] - shoulders[1, 0])
    rise = hip[1] - shoulder[1]
    if not (span > .038 * height and .105 * height < rise < .78 * height
            and .46 * span < rise < 4 * span
            and abs(hip[0] - shoulder[0]) < .50 * rise
            and abs(shoulders[0, 1] - shoulders[1, 1]) < .47 * span):
        return None
    center = shoulder + .49 * (hip - shoulder)
    return np.array([*center, span, rise])


def _regions(samples, width, height, fps):
    anchors = [_anchor(row, width, height) for row in samples]
    valid = [i for i, row in enumerate(anchors) if row is not None]
    if len(valid) < 2:
        return [None] * len(samples)
    # Reject a spatially inconsistent landmark fit before it can move the tracking
    # window to the hair or background. Base the check on local, not whole-shot scale.
    radius = max(3, round(.4 * fps))
    retained = []
    for index in valid:
        nearby = [anchors[j] for j in range(max(0, index - radius), min(len(samples), index + radius + 1))
                  if anchors[j] is not None]
        current = anchors[index]
        expected = np.median(nearby, axis=0)
        if (len(nearby) < 3 or (.55 < current[3] / expected[3] < 1.65
                              and abs(current[0] - expected[0]) < .70 * expected[3]
                              and abs(current[1] - expected[1]) < .39 * expected[3])):
            retained.append(index)
    if len(retained) < 2:
        return [None] * len(samples)

    limits = np.asarray(retained)
    reference = np.stack([anchors[i] for i in retained])
    frames = np.arange(len(samples))
    fitted = np.column_stack([np.interp(frames, limits, reference[:, j]) for j in range(4)])
    wing = max(1, round(.11 * fps))
    smooth = cv2.GaussianBlur(fitted, (1, 2 * wing + 1), sigmaX=1, sigmaY=max(.65, wing / 2),
                             borderType=cv2.BORDER_REFLECT_101)
    regions = []
    for index, row in enumerate(smooth):
        slot = np.searchsorted(limits, index)
        if slot < len(limits) and limits[slot] == index:
            usable = True
        elif slot == 0:
            usable = limits[0] - index <= .32 * fps
        elif slot == len(limits):
            usable = index - limits[-1] <= .32 * fps
        else:
            usable = limits[slot] - limits[slot - 1] <= 1.15 * fps
        regions.append(row if usable else None)
    return regions


def _scope(mask, left, top, right, bottom, value):
    height, width = mask.shape
    x0, x1 = np.clip((math.floor(left), math.ceil(right)), 0, width)
    y0, y1 = np.clip((math.floor(top), math.ceil(bottom)), 0, height)
    mask[y0:y1, x0:x1] = value


def _track_points(before, after, mask, capacity, spacing):
    points = cv2.goodFeaturesToTrack(before, capacity, .013, spacing, mask=mask, blockSize=5)
    if points is None:
        return np.empty((0, 2)), np.empty((0, 2))
    options = {"winSize": (19, 19), "maxLevel": 3}
    moved, forward, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, **options)
    if moved is None:
        return np.empty((0, 2)), np.empty((0, 2))
    recovered, backward, _ = cv2.calcOpticalFlowPyrLK(after, before, moved, None, **options)
    if recovered is None:
        return np.empty((0, 2)), np.empty((0, 2))
    keep = ((forward[:, 0] != 0) & (backward[:, 0] != 0)
            & (np.linalg.norm((recovered - points)[:, 0, :], axis=1) < 1.25))
    return points[keep, 0, :], moved[keep, 0, :]


def _measure(before, after, region):
    height, width = before.shape
    x, y, span, rise = region
    mask = np.zeros_like(before)
    _scope(mask, x - .33 * span, y - .16 * rise, x + .33 * span, y + .20 * rise, 255)
    torso_in, torso_out = _track_points(before, after, mask, 65, max(3, .006 * height))
    if len(torso_in) < 6:
        return None
    mask.fill(0)
    _scope(mask, .045 * width, .06 * height, .955 * width, .94 * height, 255)
    exclusion = max(1.6 * span, .67 * rise)
    _scope(mask, x - exclusion, y - 1.45 * rise, x + exclusion, y + 2.0 * rise, 0)
    scene_in, scene_out = _track_points(before, after, mask, 285, max(5, .012 * height))
    if len(scene_in) < 30:
        return None
    tolerance = max(1.35, .004 * height)
    camera, inliers = cv2.estimateAffinePartial2D(scene_in, scene_out, method=cv2.RANSAC,
                                                 ransacReprojThreshold=tolerance)
    if camera is None or np.count_nonzero(inliers) < max(26, .56 * len(scene_in)):
        return None
    camera_points = torso_in @ camera[:, :2].T + camera[:, 2]
    residuals = torso_out - camera_points
    center = np.median(residuals, axis=0)
    distances = np.linalg.norm(residuals - center, axis=1)
    spread = np.median(distances)
    stable = residuals[distances <= max(tolerance, 2.6 * spread)]
    if len(stable) < 6 or spread > max(1.6, .009 * height):
        return None
    return np.median(stable, axis=0)


def _lowpass(values, seconds, fps):
    span = max(1, round(seconds * fps))
    return cv2.GaussianBlur(values, (1, 2 * span + 1), sigmaX=1, sigmaY=max(.55, span / 2),
                            borderType=cv2.BORDER_REFLECT_101)


def _filtered_motion(displacements, height, fps):
    measured = np.asarray([row is not None for row in displacements])
    frames = np.arange(len(displacements))
    result = np.zeros((len(displacements), 2))
    if measured.sum() < max(3, round(.25 * fps)):
        return result, np.zeros(len(frames), dtype=bool)
    observed = np.stack([row for row in displacements if row is not None])
    speed_limit = .24 * height / fps
    observed = np.clip(observed, -speed_limit, speed_limit)
    interpreted = np.column_stack([np.interp(frames, frames[measured], observed[:, axis])
                                   for axis in range(2)])
    # A missing interval contributes neither a root trajectory nor a walk cycle.
    # Short gaps only stabilize adjacent measured motion.
    indices = frames[measured]
    for left, right in zip(indices, indices[1:]):
        if right - left > .34 * fps:
            interpreted[left + 1:right] = 0
    interpreted[:indices[0]] = 0
    interpreted[indices[-1] + 1:] = 0
    displacement = np.cumsum(interpreted, axis=0)
    residual = _lowpass(displacement, .15, fps) - _lowpass(displacement, .80, fps)
    support = _lowpass(measured.astype(float)[:, None], .24, fps)
    fade = np.clip((support[:, 0] - .40) / .42, 0, 1)
    return residual * fade[:, None] / height, fade > 0


def track_torso_motion(source, samples, first, fps):
    """Return source-frame-aligned image residuals and shoulder widths as fractions of frame height.

    `up` and `right` hold only local, camera-relative displacement; an unsupported
    frame has None in both fields. This does not estimate absolute root motion or depth.
    """
    rate = float(fps)
    if not math.isfinite(rate) or rate <= 0 or first < 0:
        raise ValueError("Invalid torso-motion source frame range")
    displacements = [None] * len(samples)
    widths = [None] * len(samples)
    video = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    try:
        skipped = all(video.grab() for _ in range(first))
        ok, frame = video.read() if skipped and samples else (False, None)
        if ok:
            original_height, original_width = frame.shape[:2]
            scale = min(1.0, 960 / max(original_height, original_width))
            width, height = max(1, round(original_width * scale)), max(1, round(original_height * scale))
            if min(width, height) >= 48:
                regions = _regions(samples, width, height, rate)
                widths = [None if region is None else round(float(region[2] / height), 5)
                          for region in regions]
                before = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA),
                                      cv2.COLOR_BGR2GRAY)
                for index in range(1, len(samples)):
                    ok, frame = video.read()
                    if not ok:
                        break
                    after = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2GRAY)
                    if regions[index - 1] is not None and regions[index] is not None:
                        displacements[index] = _measure(before, after, regions[index - 1])
                    before = after
    finally:
        video.release()
    motion, support = _filtered_motion(displacements, height if any(widths) else 1, rate)
    return {
        "up": [round(float(-row[1]), 6) if valid else None for row, valid in zip(motion, support, strict=True)],
        "right": [round(float(row[0]), 6) if valid else None for row, valid in zip(motion, support, strict=True)],
        "width": widths,
        "measured_pairs": sum(row is not None for row in displacements),
        "space": "camera_xy_over_frame_height",
    }
