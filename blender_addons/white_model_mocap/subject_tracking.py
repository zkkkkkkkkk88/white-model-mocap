"""Select a pose using a user-provided torso region and in-shot continuity."""

from dataclasses import dataclass
from itertools import combinations, groupby, product
import math

import cv2
import numpy as np


def checked_region(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or not all(type(bound) in (int, float) and math.isfinite(bound) for bound in value)):
        raise ValueError("目标人物范围需要四个有限坐标 [左, 上, 右, 下]")
    left, top, right, bottom = value
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("目标人物范围须位于画面内，右边须大于左边，下边须大于上边")
    return np.array(value, dtype=float)


def _visible_xy(sample, index, minimum):
    x, y, confidence = sample[index][3:6]
    return (x, y) if confidence >= minimum and 0 <= x <= 1 and 0 <= y <= 1 else None


def _head_points(sample, minimum=.65):
    return [point for i in range(11) if (point := _visible_xy(sample, i, minimum)) is not None]


@dataclass
class _Torso:
    sample: list
    center: np.ndarray
    span: float
    color: np.ndarray
    confidence: float
    profile: bool = False


class SubjectTracker:
    def __init__(self, region, width, height, fps):
        self.region = checked_region(region)
        self.aspect = np.array((width / height, 1))
        self.fps = float(fps)
        self.last = None
        self.last_index = None
        self.velocity = np.zeros(2)
        self.anchor_color = None
        self.history = []

    def _torso(self, sample, frame, *, allow_profile=False):
        left = _visible_xy(sample, 11, .45)
        right = _visible_xy(sample, 12, .45)
        if left is None or right is None:
            return None
        a, b = np.asarray((left, right)) * self.aspect
        span = float(np.linalg.norm(a - b))
        center = (a + b) * .5
        profile = span < .060
        head = _head_points(sample, .8)
        if len(head) >= 5:
            head_offset = np.median(head, axis=0) * self.aspect - center
            profile = profile or abs(head_offset[0]) > max(.12, .69 * span)
        if profile:
            # A side-on or leaning subject can project very narrow shoulders and
            # an offset head. Confirm both against the pelvis of a recent match.
            if not allow_profile or len(head) < 5 or min(sample[11][5], sample[12][5]) < .74:
                return None
            hips = [_visible_xy(sample, index, .74) for index in (23, 24)]
            if any(hip is None for hip in hips):
                return None
            spine = center - (np.mean(hips, axis=0) * self.aspect)
            spine_length = np.linalg.norm(spine)
            head_distance = np.linalg.norm(head_offset)
            if (not .10 <= spine_length <= .65
                    or not .16 * spine_length <= head_distance <= .70 * spine_length
                    or head_offset.dot(spine) < .59 * head_distance * spine_length):
                return None
            span = max(span, .32 * spine_length)
        height, width = frame.shape[:2]
        x, y = center / self.aspect
        half = .15 * span * height
        x, y = round(x * width), round((y + .32 * span) * height)
        left, top = max(0, round(x - half)), max(0, round(y - half))
        right, bottom = min(width, round(x + half + 1)), min(height, round(y + half + 1))
        pixels = frame[top:bottom:2, left:right:2]
        color = np.median(pixels.reshape(-1, 3), axis=0) / 255 if pixels.size else None
        if profile and (color is None or self.anchor_color is None
                        or np.mean(np.abs(color - self.anchor_color)) > .16):
            return None
        return _Torso(sample, center, span, color, min(sample[11][5], sample[12][5]), profile)

    def _score(self, torso, index):
        if self.last is None:
            center = torso.center / self.aspect
            if not np.all(self.region[:2] <= center) or not np.all(center <= self.region[2:]):
                return None
            distance = np.linalg.norm((center - .5 * (self.region[:2] + self.region[2:]))
                                      / (self.region[2:] - self.region[:2]))
            return float(distance + .10 * (1 - torso.confidence))
        gap = index - self.last_index
        predicted = self.last.center + self.velocity * min(gap, max(1, round(.12 * self.fps)))
        seed_span = min((self.region[2:] - self.region[:2]) * self.aspect)
        distance = float(np.linalg.norm(torso.center - predicted) / max(self.last.span, .24 * seed_span))
        scale = torso.span / self.last.span
        if distance > min(1.05, .53 + .07 * (gap - 1)):
            return None
        difference = 0.0
        if torso.color is not None and self.anchor_color is not None:
            difference = float(np.mean(np.abs(torso.color - self.anchor_color)))
            if difference > .27:
                return None
        if not .46 <= scale <= 1.95:
            # A rear or side reconstruction can badly compress shoulder width.
            # Rely on position and clothing only while the subject is still within the marked area.
            center = torso.center / self.aspect
            margin = .07 * (self.region[2:] - self.region[:2])
            if (not .22 <= scale <= 4.55 or distance > .52 or difference > .17
                    or torso.color is None or self.anchor_color is None
                    or not np.all(self.region[:2] + margin <= center)
                    or not np.all(center <= self.region[2:] - margin)):
                return None
        return distance + .30 * abs(math.log(scale)) + 1.20 * difference + .10 * (1 - torso.confidence)

    def _options(self, candidates, frame, index):
        allow_profile = self.last is not None and 0 < index - self.last_index <= max(1, round(.13 * self.fps))
        for candidate, sample in enumerate(candidates):
            torso = self._torso(sample, frame, allow_profile=allow_profile)
            if torso is not None:
                score = self._score(torso, index)
                if score is not None:
                    if torso.profile:
                        score += .12
                    yield candidate, score, torso

    def select(self, candidates, frame, index):
        ranked = [(score, torso) for _, score, torso in self._options(candidates, frame, index)]
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        if len(ranked) > 1:
            best, alternative = ranked[:2]
            span = self.last.span if self.last is not None else min(best[1].span, alternative[1].span)
            if (alternative[0] - best[0] < .11
                    and np.linalg.norm(best[1].center - alternative[1].center) > .24 * span):
                return None
        return ranked[0][1]

    def suppressed_frame(self, frame, candidates):
        """Hide only a rival's recognizable head when it is outside the target area."""
        height, width = frame.shape[:2]
        protected = self.region.copy()
        if self.last is not None:
            center = self.last.center / self.aspect
            if not np.all(self.region[:2] <= center) or not np.all(center <= self.region[2:]):
                half = np.array((.4, .75)) * self.last.span / self.aspect
                protected = np.concatenate((center - half, center + half))

        matte = np.zeros((height, width), dtype=np.uint8)
        for sample in candidates:
            head = _head_points(sample)
            if len(head) < 5:
                continue
            a, b = np.min(head, axis=0), np.max(head, axis=0)
            x, y = .5 * (a + b)
            span = max(.023, b[0] - a[0], (b[1] - a[1]) / self.aspect[0] * .8)
            vertical = max(.032, b[1] - a[1], span * self.aspect[0] * .72)
            left, right = a[0] - 1.1 * span, b[0] + 1.3 * span
            top, bottom = a[1] - 2.5 * vertical, b[1] + 1.6 * vertical
            if x < protected[0] - .008:
                right = min(right, protected[0])
            elif x > protected[2] + .008:
                left = max(left, protected[2])
            elif y < protected[1] - .008:
                bottom = min(bottom, protected[1])
            elif y > protected[3] + .008:
                top = max(top, protected[3])
            else:
                continue
            if not (left <= a[0] <= b[0] <= right and top <= a[1] <= b[1] <= bottom):
                continue
            x0, x1 = int(max(0, left) * width), min(width, math.ceil(right * width))
            y0, y1 = int(max(0, top) * height), min(height, math.ceil(bottom * height))
            matte[y0:y1, x0:x1] = 255
        if not matte.any():
            return None
        sigma = max(1, .0095 * height)
        alpha = cv2.GaussianBlur(matte, (0, 0), sigma).astype(np.float32)[..., None] / 255
        return np.uint8(np.rint(frame * (1 - alpha) + 90 * alpha))

    def finish_frame(self, selected, index, *, recovered=False):
        self.history.append((index, "lost" if selected is None else "masked_recovery" if recovered else "matched"))
        if selected is None:
            return None
        if self.last is not None:
            instantaneous = (selected.center - self.last.center) / (index - self.last_index)
            self.velocity = .58 * self.velocity + .42 * instantaneous
        if selected.color is not None:
            if self.anchor_color is None:
                self.anchor_color = selected.color
            else:
                self.anchor_color = .94 * self.anchor_color + .06 * selected.color
        self.last = selected
        self.last_index = index
        return selected.sample

    def report(self):
        ranges, lost, masked = [], 0, 0
        for status, records in groupby(self.history, lambda record: record[1]):
            frames = [index for index, _ in records]
            ranges.append({"source_range": [frames[0], frames[-1]], "status": status})
            lost += len(frames) if status == "lost" else 0
            masked += len(frames) if status == "masked_recovery" else 0
        return {"region": self.region.tolist(), "matched_frames": len(self.history) - lost,
                "lost_frames": lost, "masked_recovery_frames": masked, "ranges": ranges}


def _pose_signature(sample, aspect):
    def center(indices):
        points = [_visible_xy(sample, index, .45) for index in indices]
        if any(point is None for point in points):
            return None
        return np.mean(points, axis=0) * aspect

    shoulders = center((11, 12))
    hips = center((23, 24))
    head = _head_points(sample, .50)
    face = np.median(head, axis=0) * aspect if len(head) >= 3 else None
    scale = 0.0
    if shoulders is not None:
        scale = 2 * np.linalg.norm(np.asarray(sample[11][3:5]) * aspect - shoulders)
        if hips is not None:
            scale = max(scale, np.linalg.norm(shoulders - hips))
    return (shoulders, hips, face), scale


def _duplicate_groups(candidates, aspect):
    """Treat near-identical detector rows as a single assignable person."""
    signatures = [_pose_signature(sample, aspect) for sample in candidates]
    groups = list(range(len(signatures)))
    for first, second in combinations(range(len(signatures)), 2):
        (a, a_scale), (b, b_scale) = signatures[first], signatures[second]
        tolerance = min(.05, max(.012, .17 * min(a_scale, b_scale)))
        if (a[0] is None or b[0] is None or np.linalg.norm(a[0] - b[0]) > tolerance):
            continue
        secondary = [np.linalg.norm(x - y) for x, y in zip(a[1:], b[1:], strict=True)
                     if x is not None and y is not None]
        if secondary:
            if min(secondary) > tolerance:
                continue
        else:
            # Near-identical shoulders are enough to merge ambiguous detections
            # when neither fit has reliable hips or face to distinguish them.
            left = np.asarray(candidates[first][11:13])[:, 3:5] * aspect
            right = np.asarray(candidates[second][11:13])[:, 3:5] * aspect
            if min(np.max(np.linalg.norm(left - other, axis=1)) for other in (right, right[::-1])) > .5 * tolerance:
                continue
        groups = [groups[first] if group == groups[second] else group for group in groups]
    return groups


class DualSubjectTracker:
    """Select two identities jointly from a single set of pose detections."""

    def __init__(self, regions, width, height, fps):
        if not isinstance(regions, (tuple, list)) or len(regions) != 2:
            raise ValueError("双人动捕需要按 A、B 顺序提供两个目标人物范围")
        self.tracks = tuple(SubjectTracker(region, width, height, fps) for region in regions)
        self.aspect = self.tracks[0].aspect
        left, right = (track.region for track in self.tracks)
        self.overlapping_seeds = bool(np.all(np.maximum(left[:2], right[:2]) <= np.minimum(left[2:], right[2:])))

    def select(self, candidates, frame, index):
        groups = _duplicate_groups(candidates, self.aspect)
        lost_cost = 1.25
        choices = []
        for track in self.tracks:
            best_by_group = {None: (lost_cost, None)}
            for candidate, score, torso in track._options(candidates, frame, index):
                group = groups[candidate]
                if score < best_by_group.get(group, (lost_cost, None))[0]:
                    best_by_group[group] = (score, torso)
            choices.append(best_by_group)

        ranked = [
            (choices[0][a][0] + choices[1][b][0], (a, b))
            for a, b in product(*choices)
            if a is None or b is None or a != b
        ]
        ranked.sort(key=lambda option: option[0])
        best_cost, best = ranked[0]
        plausible = [assignment for cost, assignment in ranked if cost - best_cost < .14]
        selected = tuple(
            options[group][1] if all(alternative[track] == group for alternative in plausible) else None
            for track, (group, options) in enumerate(zip(best, choices, strict=True))
        )
        if (self.overlapping_seeds and any(track.last is None for track in self.tracks)
                and any(torso is None for torso in selected)):
            # A lone detection can drift into the other person's marked region.
            # Overlapping seeds need separate, unambiguous observations together.
            return None, None
        return selected
