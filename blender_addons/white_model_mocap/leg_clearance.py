"""Separate intersecting left/right leg meshes while preserving the knee hinge."""

from itertools import groupby
import math

from mathutils import Quaternion, Vector

from .arm_clearance import (
    _capture_shape, _chain_motion_cost, _correction_candidates, _forward_flexion,
    _interpolate_chain, _intersects,
)
from .joint_limits import flex_axis


_SIDES = ("L", "R")
_ANGLES = tuple(map(math.radians, (3, 6, 9, 13, 19, 27, 38)))
_LONG = Vector((0, 1, 0))
_CLEARANCE = 0.007


def calibrated_leg_parts(rig, controls):
    by_bone = {obj.parent_bone: obj for obj in rig.children
               if obj.type == "MESH" and obj.parent_type == "BONE"}
    parts = {part: by_bone.get(controls[part]) for side in _SIDES
             for part in (f"thigh.{side}", f"shin.{side}")}
    return parts if all(parts.values()) else None


class LegClearance:
    """Correct opposite-leg intersections using measured calibrated mesh shapes."""

    def __init__(self, context, rig, parts, controls):
        from collision_scan import inside_candidate

        self._contains = inside_candidate
        self._shapes = {}
        self._knees = {}
        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        rig_inverse = rig.matrix_world.inverted()
        for side in _SIDES:
            thigh, shin = f"thigh.{side}", f"shin.{side}"
            upper = rig.pose.bones[controls[thigh]]
            lower = rig.pose.bones[controls[shin]]
            upper_inverse = upper.matrix.to_quaternion().inverted()
            upper_pivot = rig_inverse @ parts[thigh].evaluated_get(depsgraph).matrix_world.translation
            lower_pivot = rig_inverse @ parts[shin].evaluated_get(depsgraph).matrix_world.translation
            self._knees[side] = upper_inverse @ (upper_pivot - upper.head)
            self._shapes[thigh] = _capture_shape(
                parts[thigh], depsgraph, rig_inverse, upper_inverse, upper.head,
            )
            self._shapes[shin] = _capture_shape(
                parts[shin], depsgraph, rig_inverse, lower.matrix.to_quaternion().inverted(), lower_pivot,
            )

    def _volumes(self, rotations, hips):
        volumes = {}
        for side, (upper, lower) in zip(_SIDES, rotations, strict=True):
            hip = hips[side]
            knee = hip + upper @ self._knees[side]
            volumes[side] = (self._shapes[f"thigh.{side}"].placed(upper, hip, padding=_CLEARANCE),
                             self._shapes[f"shin.{side}"].placed(lower, knee, padding=_CLEARANCE))
        return volumes

    def _separate(self, rotations, hips):
        volumes = self._volumes(rotations, hips)
        return not any(_intersects(left, right, self._contains)
                       for left in volumes["L"] for right in volumes["R"])

    def obstacles_at(self, legs, rest, chest, index, *, next_index=None):
        """Return evaluated leg geometry for the same frame's upper-body solve."""
        origin = rest["torso"].head_local
        hips = {side: origin + chest @ (rest[f"thigh.{side}"].head_local - origin) for side in _SIDES}
        def sample(part):
            track = legs[part]
            return track[index] if next_index is None else track[index].slerp(track[next_index], 0.5)
        rotations = [(sample(f"thigh.{side}"), sample(f"shin.{side}")) for side in _SIDES]
        return tuple(shape for pair in self._volumes(rotations, hips).values() for shape in pair)

    @staticmethod
    def _neutral_leg(side, rest, chest):
        thigh = rest[f"thigh.{side}"].matrix_local.to_quaternion()
        shin = rest[f"shin.{side}"].matrix_local.to_quaternion()
        outward = 0.25 if side == "L" else -0.25
        axis = Vector((outward, 0, -1)).normalized()
        neutral = chest @ (thigh @ _LONG).rotation_difference(axis)
        return neutral @ thigh, neutral @ shin

    @staticmethod
    def _blend_legs(before, after, weight):
        return [_interpolate_chain(a, b, weight) for a, b in zip(before, after, strict=True)]

    def _solve(self, rotations, hips, chest, neutral, previous, decay):
        def spread(left, right):
            axis = chest @ Vector((0, 1, 0))
            return [tuple(Quaternion(axis, angle) @ rotation for rotation in leg)
                    for angle, leg in zip((-left, right), rotations, strict=True)]

        if self._separate(rotations, hips):
            easing = tuple(value * (1 - decay) for value in previous)
            adjusted = spread(*easing)
            if sum(easing) > _ANGLES[0] and self._separate(adjusted, hips):
                return adjusted, easing, "opposite_leg_clearance"
            return rotations, (0, 0), None

        for angle in _ANGLES:
            for extension in ((angle, angle), (angle, 0), (0, angle)):
                adjusted = spread(*extension)
                if self._separate(adjusted, hips):
                    return adjusted, extension, "opposite_leg_clearance"
        for weight in (0.35, 0.60, 0.80, 1.0):
            adjusted = self._blend_legs(rotations, neutral, weight)
            if self._separate(adjusted, hips):
                return adjusted, (0, 0), "opposite_leg_clearance_neutralized"
        return rotations, (0, 0), "opposite_leg_clearance_unresolved"

    def _smooth_transitions(self, raw_legs, result, rest, chest_track, hips, fps, reasons):
        pairs = [list(zip(result[f"thigh.{side}"], result[f"shin.{side}"], strict=True)) for side in _SIDES]

        def clear(left, right, changed_side, replacements):
            keyed = [[replacements.get(index, chain[index]) if side_index == changed_side else chain[index]
                      for index in (left, right)] for side_index, chain in enumerate(pairs)]
            rotations = [tuple(a.slerp(b, 0.5) for a, b in zip(first, second, strict=True))
                         for first, second in keyed]
            pivots = {side: hips[left][side].lerp(hips[right][side], 0.5) for side in _SIDES}
            return self._separate(rotations, pivots)

        for side_index, side in enumerate(_SIDES):
            raw = list(zip(raw_legs[f"thigh.{side}"], raw_legs[f"shin.{side}"], strict=True))
            safe = pairs[side_index]
            axis = flex_axis(rest[f"thigh.{side}"].matrix_local.to_quaternion(),
                             rest[f"shin.{side}"].matrix_local.to_quaternion())
            for i, target in enumerate(_correction_candidates(raw, safe, chest_track, fps)):
                if target is None:
                    continue
                adjacent = [j for j in (i - 1, i + 1) if 0 <= j < len(safe)]
                surrounding = [safe[j] for j in adjacent]
                protected = [(min(i, j), max(i, j)) for j in adjacent
                             if clear(min(i, j), max(i, j), side_index, {})]
                current = safe[i]
                cost = _chain_motion_cost(current, surrounding)
                for amount in (1.0, 0.65, 0.35):
                    candidate = _interpolate_chain(current, target, amount)
                    if (not _forward_flexion(candidate, axis)
                            or _chain_motion_cost(candidate, surrounding) >= cost - 1e-8):
                        continue
                    replacements = {i: candidate}
                    if clear(i, i, side_index, replacements) and all(
                        clear(a, b, side_index, replacements) for a, b in protected
                    ):
                        safe[i] = candidate
                        reasons[i] = reasons[i] or "opposite_leg_clearance_transition"
                        break
            for i, (upper, lower) in enumerate(safe):
                result[f"thigh.{side}"][i] = upper
                result[f"shin.{side}"][i] = lower

    def apply(self, legs, rest, chest_track, *, source_start, fps):
        result = {part: rotations.copy() for part, rotations in legs.items()}
        torso_origin = rest["torso"].head_local
        offsets = {side: rest[f"thigh.{side}"].head_local - torso_origin for side in _SIDES}
        reasons = []
        previous = (0, 0)
        decay = min(1, 6 / float(fps))
        hip_tracks = [{side: torso_origin + chest @ offsets[side] for side in _SIDES} for chest in chest_track]
        for index, (chest, hips) in enumerate(zip(chest_track, hip_tracks, strict=True)):
            rotations = [(legs[f"thigh.{side}"][index], legs[f"shin.{side}"][index]) for side in _SIDES]
            neutral = [self._neutral_leg(side, rest, chest) for side in _SIDES]
            output, previous, reason = self._solve(rotations, hips, chest, neutral, previous, decay)
            reasons.append(reason)
            for side, (upper, lower) in zip(_SIDES, output, strict=True):
                result[f"thigh.{side}"][index] = upper.normalized()
                result[f"shin.{side}"][index] = lower.normalized()
        self._smooth_transitions(legs, result, rest, chest_track, hip_tracks, fps, reasons)
        ranges = []
        for reason, run in groupby(range(len(reasons)), reasons.__getitem__):
            indices = list(run)
            if reason:
                ranges.append({"control": "legs", "reason": reason,
                               "treatment": "unresolved" if reason.endswith("unresolved") else "corrected",
                               "source_range": [source_start + indices[0], source_start + indices[-1]]})
        return result, ranges
