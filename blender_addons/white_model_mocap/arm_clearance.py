"""Resolve arm penetration using the evaluated geometry of the calibrated rig."""

from functools import lru_cache
from itertools import groupby
import math

from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

from .joint_limits import flex_axis
from .stabilize import angle, centered_rotations


_SIDES = ("L", "R")
_IDENTITY = Quaternion()
_CLEARANCE = 0.008
_ANGLES = tuple(map(math.radians, (4, 8, 13, 19, 27, 38)))
_LONG = Vector((0, 1, 0))


def calibrated_arm_parts(rig, part_names, controls):
    """Recover the calibrated meshes from an existing action target."""
    identity = rig.get("production_identity")
    meshes = [obj for obj in rig.children if obj.type == "MESH"]
    by_bone = {obj.parent_bone: obj for obj in meshes if obj.parent_type == "BONE"}
    parts = {}
    for part in ("torso", "head", *(f"{name}.{side}" for side in _SIDES
                                   for name in ("upper_arm", "forearm"))):
        if part not in ("torso", "head"):
            parts[part] = by_bone.get(controls[part])
        else:
            name = f"{identity}_{part_names[part]}"
            parts[part] = next((obj for obj in meshes if obj.name == name), None)
    return parts if all(parts.values()) else None


def _edges(faces):
    counts = {}
    for face in faces:
        for current, following in zip(face, face[1:] + face[:1], strict=True):
            edge = tuple(sorted((current, following)))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


class _Shape:
    def __init__(self, points, faces):
        self.points = points
        self.faces = faces
        self.closed = all(count == 2 for count in _edges(faces).values())

    def without_shoulder_seam(self):
        # The proximal cap meets the torso intentionally. Collision checking
        # still covers the remainder of the upper arm and all of the forearm.
        bottom = min(point.y for point in self.points)
        span = max(point.y for point in self.points) - bottom
        vertices = tuple(Vector((point.x, point.y + 0.30 * (bottom + span - point.y), point.z))
                         for point in self.points)
        return _Shape(vertices, self.faces)

    def placed(self, rotation, origin, *, padding=0):
        points = tuple(origin + rotation @ point for point in self.points)
        if padding:
            center = sum(points, Vector()) / len(points)
            points = tuple(point + padding * (point - center).normalized() for point in points)
        bounds = [Vector([limit(point[axis] for point in points) for axis in range(3)])
                  for limit in (min, max)]
        return {"vertices": points, "bounds": bounds, "closed": self.closed,
                "tree": BVHTree.FromPolygons(points, self.faces)}


def _capture_shape(mesh, depsgraph, rig_inverse, rotation_inverse, origin):
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        points = tuple(
            rotation_inverse @ ((rig_inverse @ (evaluated.matrix_world @ vertex.co)) - origin)
            for vertex in evaluated_mesh.vertices
        )
        return _Shape(points, tuple(tuple(face.vertices) for face in evaluated_mesh.polygons))
    finally:
        evaluated.to_mesh_clear()


def _intersects(left, right, containment):
    a_min, a_max = left["bounds"]
    b_min, b_max = right["bounds"]
    if any(a_max[i] < b_min[i] or b_max[i] < a_min[i] for i in range(3)):
        return False
    if left["tree"].overlap(right["tree"]):
        return True
    # With disjoint surfaces a closed solid can still fully contain the other.
    return bool(
        (all(a_min[i] <= b_min[i] <= b_max[i] <= a_max[i] for i in range(3))
         and containment(right["vertices"][0], left))
        or (all(b_min[i] <= a_min[i] <= a_max[i] <= b_max[i] for i in range(3))
        and containment(left["vertices"][0], right))
    )


def _search_axes(side):
    outward = -1 if side == "L" else 1
    return tuple(Vector(axis).normalized() for axis in (
        (0, outward, 0), (-1, 0, 0), (-1, outward, 0),
        (0, 0, outward), (0, 0, -outward), (1, 0, 0), (0, -outward, 0),
        (1, outward, 0), (-1, -outward, 0), (1, -outward, 0),
        (-1, outward, 1), (-1, outward, -1), (1, outward, 1), (1, outward, -1),
        (1, -outward, 1), (1, -outward, -1), (-1, -outward, 1), (-1, -outward, -1),
    ))


def _correction_candidates(raw, safe, chest_track, fps):
    """Smooth only collision corrections, as torso-relative rotations."""
    upper_offsets, hinge_offsets = [], []
    for (upper, lower), (new_upper, new_lower), chest in zip(raw, safe, chest_track, strict=True):
        delta = chest.inverted() @ new_upper @ upper.inverted() @ chest
        hinge_delta = (upper.inverted() @ lower).inverted() @ (new_upper.inverted() @ new_lower)
        upper_offsets.append(delta.normalized())
        hinge_offsets.append(hinge_delta.normalized())

    radius = max(1, min(4, round(0.10 * float(fps))))

    def average(values, start, end, center):
        if all(abs(values[i].w) >= 1 - 1e-7 for i in range(start, end)):
            return None
        result, total = values[start], radius + 1 - abs(start - center)
        for i in range(start + 1, end):
            weight = radius + 1 - abs(i - center)
            total += weight
            result = result.slerp(values[i], weight / total)
        return result.normalized()

    for i, ((upper, lower), chest) in enumerate(zip(raw, chest_track, strict=True)):
        start, end = max(0, i - radius), min(len(raw), i + radius + 1)
        delta = average(upper_offsets, start, end, i)
        hinge_delta = average(hinge_offsets, start, end, i)
        if delta is None and hinge_delta is None:
            yield None
            continue
        corrected = chest @ (delta or _IDENTITY) @ chest.inverted() @ upper
        hinge = (upper.inverted() @ lower) @ (hinge_delta or _IDENTITY)
        yield corrected.normalized(), (corrected @ hinge).normalized()


def _interpolate_chain(before, after, amount):
    upper, lower = before
    target_upper, target_lower = after
    upper_result = upper.slerp(target_upper, amount).normalized()
    hinge = upper.inverted() @ lower
    target_hinge = target_upper.inverted() @ target_lower
    return upper_result, (upper_result @ hinge.slerp(target_hinge, amount)).normalized()


def _chain_motion_cost(chain, neighbors):
    return sum(2 * math.acos(min(1.0, abs(a.dot(b)))) ** 2
               for neighbor in neighbors for a, b in zip(chain, neighbor, strict=True))


def _forward_flexion(chain, bend_axis):
    upper, lower = chain
    direction = upper.inverted() @ lower @ _LONG
    return direction.dot(bend_axis) >= -1e-5 and direction.y >= math.cos(math.radians(145.1))


def _local_chain(chest, chain):
    upper, forearm = chain
    return (chest.inverted() @ upper).normalized(), (upper.inverted() @ forearm).normalized()


def _steps(chains):
    return [max(angle(u, v) for u, v in zip(before, after, strict=True))
            for before, after in zip(chains, chains[1:])]


def _smooth_clearance_jumps(raw, safe, chests, fps, bend_axis, clear):
    """Replace a sharp correction transition only if the full new interval is safe."""
    rate = float(fps)
    original = [_local_chain(chest, chain) for chest, chain in zip(chests, raw, strict=True)]
    local = [_local_chain(chest, chain) for chest, chain in zip(chests, safe, strict=True)]
    minimum_step = math.radians(440 / rate)
    extra_step = math.radians(165 / rate)
    triggers = [index for index, (before, after) in enumerate(zip(_steps(original), _steps(local), strict=True), 1)
                if after >= minimum_step and after - before >= extra_step]
    if not triggers:
        return {}

    radius = max(2, round(0.14 * rate))
    affected = {i for center in triggers
                for i in range(max(0, center - radius), min(len(safe), center + radius + 1))}
    shoulder, hinge = map(list, zip(*local, strict=True))
    smoothed_shoulder = centered_rotations(shoulder, rate, 0.17)
    smoothed_hinge = centered_rotations(hinge, rate, 0.17)
    changed = {}
    for needs_check, group in groupby(range(len(safe)), affected.__contains__):
        if not needs_check:
            continue
        run = list(group)
        first, last = run[0], run[-1]
        start, stop = max(0, first - 1), min(len(safe), last + 2)
        baseline = _steps(local[start:stop])
        for amount in (1.0, 0.65, 0.35):
            poses = {}
            for i in run:
                fade = min(1.0, (i - first + 1) / 3, (last - i + 1) / 3)
                upper = (chests[i] @ shoulder[i].slerp(smoothed_shoulder[i], amount * fade)).normalized()
                lower = (upper @ hinge[i].slerp(smoothed_hinge[i], amount * fade)).normalized()
                poses[i] = (upper, lower)
            steps = _steps([_local_chain(chests[i], poses[i]) if i in poses else local[i]
                            for i in range(start, stop)])
            if (max(steps) >= max(baseline) - math.radians(0.5)
                    or sum(step * step for step in steps) >= sum(step * step for step in baseline)):
                continue
            if not all(_forward_flexion(chain, bend_axis) for chain in poses.values()):
                continue
            if not all(clear(i, i, poses, include_other=True) for i in run):
                continue
            for i in range(start, stop - 1):
                left = poses.get(i, safe[i])
                right = poses.get(i + 1, safe[i + 1])
                midpoint = tuple(a.slerp(b, 0.5).normalized() for a, b in zip(left, right, strict=True))
                if not _forward_flexion(midpoint, bend_axis) or not clear(i, i + 1, poses, include_other=True):
                    break
            else:
                changed.update(poses)
                break
    return changed


class ArmClearance:
    """Apply a shoulder correction without modifying the solved elbow hinge."""

    def __init__(self, context, rig, parts, controls):
        from collision_scan import inside_candidate

        self._contains = inside_candidate
        self._shapes = {}
        self._elbows = {}
        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        rig_inverse = rig.matrix_world.inverted()
        for part in ("torso", "head"):
            bone = rig.pose.bones[controls[part]]
            self._shapes[part] = _capture_shape(
                parts[part], depsgraph, rig_inverse, bone.matrix.to_quaternion().inverted(), bone.head,
            )
        for side in _SIDES:
            upper, forearm = f"upper_arm.{side}", f"forearm.{side}"
            upper_bone = rig.pose.bones[controls[upper]]
            forearm_bone = rig.pose.bones[controls[forearm]]
            upper_rotation_inverse = upper_bone.matrix.to_quaternion().inverted()
            upper_pivot = rig_inverse @ parts[upper].evaluated_get(depsgraph).matrix_world.translation
            forearm_pivot = rig_inverse @ parts[forearm].evaluated_get(depsgraph).matrix_world.translation
            self._elbows[side] = upper_rotation_inverse @ (upper_pivot - upper_bone.head)
            self._shapes[upper] = _capture_shape(
                parts[upper], depsgraph, rig_inverse, upper_rotation_inverse, upper_bone.head,
            )
            self._shapes[forearm] = _capture_shape(
                parts[forearm], depsgraph, rig_inverse,
                forearm_bone.matrix.to_quaternion().inverted(), forearm_pivot,
            )
        self._distal_upper = {side: self._shapes[f"upper_arm.{side}"].without_shoulder_seam()
                              for side in _SIDES}
        self._axes = {side: _search_axes(side) for side in _SIDES}

    def _safe(self, side, shoulder, upper, forearm, torso, head, obstacles):
        upper_part, forearm_part = f"upper_arm.{side}", f"forearm.{side}"
        elbow = shoulder + upper @ self._elbows[side]
        forearm_shape = self._shapes[forearm_part].placed(forearm, elbow)
        if any(_intersects(body, forearm_shape, self._contains) for body in (torso, head, *obstacles)):
            return False
        upper_distal = self._distal_upper[side].placed(upper, shoulder)
        if _intersects(torso, upper_distal, self._contains):
            return False
        upper_shape = self._shapes[upper_part].placed(upper, shoulder)
        return not any(_intersects(body, upper_shape, self._contains) for body in (head, *obstacles))

    def _pick(self, side, clear, previous, decay):
        if clear(_IDENTITY):
            prior = previous.slerp(_IDENTITY, decay)
            return prior if prior.angle > _ANGLES[0] and clear(prior) else _IDENTITY

        for angle in _ANGLES:
            # Prefer the preceding correction's direction when it is still
            # feasible so consecutive poses don't alternate around the torso.
            axes = self._axes[side]
            if previous.angle > _ANGLES[0]:
                axes = (previous.axis, *axes)
            for axis in axes:
                correction = Quaternion(axis, angle)
                if clear(correction):
                    return correction
        if previous.angle > 0 and clear(previous):
            return previous
        return None

    @staticmethod
    def _resting_arm(side, rest):
        upper = rest[f"upper_arm.{side}"].matrix_local.to_quaternion()
        forearm = rest[f"forearm.{side}"].matrix_local.to_quaternion()
        outward = 0.29 if side == "L" else -0.29
        lowered = Vector((outward, -0.12, -1)).normalized()
        delta = (upper @ _LONG).rotation_difference(lowered)
        return delta @ upper, delta @ forearm

    def _collision_checker(self, arm_tracks, rest, chest_track, head_track, leg_clearance, leg_tracks):
        origin = rest["torso"].head_local
        head_offset = rest["head"].head_local - origin
        torso_rest = rest["torso"].matrix_local.to_quaternion()
        head_rest = rest["head"].matrix_local.to_quaternion()
        shoulders = {side: rest[f"upper_arm.{side}"].head_local - origin for side in _SIDES}

        @lru_cache(maxsize=10)
        def view(left, right):
            first, second = chest_track[left], chest_track[right]
            chest = first.slerp(second, 0.5)
            head_first, head_second = first @ head_track[left], second @ head_track[right]
            head = head_first.slerp(head_second, 0.5) @ head_rest
            head_origin = origin + (first @ head_offset).lerp(second @ head_offset, 0.5)
            obstacles = (leg_clearance.obstacles_at(leg_tracks, rest, chest, left, next_index=right)
                         if leg_clearance is not None and leg_tracks is not None else ())
            return (origin, first, second,
                    self._shapes["torso"].placed(chest @ torso_rest, origin, padding=_CLEARANCE),
                    self._shapes["head"].placed(head, head_origin, padding=_CLEARANCE), obstacles)

        def check_for(side):
            other = "R" if side == "L" else "L"
            upper, forearm = (arm_tracks[f"{part}.{side}"] for part in ("upper_arm", "forearm"))
            other_upper, other_forearm = (arm_tracks[f"{part}.{other}"] for part in ("upper_arm", "forearm"))

            def clear(left, right, replacements, *, include_other=False):
                anchor, first, second, torso, head, obstacles = view(left, right)
                shoulder = anchor + (first @ shoulders[side]).lerp(second @ shoulders[side], 0.5)
                upper_left, fore_left = replacements.get(left, (upper[left], forearm[left]))
                upper_right, fore_right = replacements.get(right, (upper[right], forearm[right]))
                if include_other:
                    other_shoulder = anchor + (first @ shoulders[other]).lerp(second @ shoulders[other], 0.5)
                    other_pitch = other_upper[left].slerp(other_upper[right], 0.5)
                    other_elbow = other_shoulder + other_pitch @ self._elbows[other]
                    obstacles = (*obstacles,
                                 self._shapes[f"upper_arm.{other}"].placed(other_pitch, other_shoulder),
                                 self._shapes[f"forearm.{other}"].placed(
                                     other_forearm[left].slerp(other_forearm[right], 0.5), other_elbow,
                                 ))
                return self._safe(side, shoulder, upper_left.slerp(upper_right, 0.5),
                                  fore_left.slerp(fore_right, 0.5), torso, head, obstacles)

            return clear

        return check_for

    def _guard_interpolation(self, arm_tracks, rest, chest_track, head_track, reasons,
                             leg_clearance, leg_tracks, raw_arms, fps):
        check_for = self._collision_checker(arm_tracks, rest, chest_track, head_track, leg_clearance, leg_tracks)
        for side in _SIDES:
            upper, forearm = (arm_tracks[f"{part}.{side}"] for part in ("upper_arm", "forearm"))
            bend_axis = flex_axis(rest[f"upper_arm.{side}"].matrix_local.to_quaternion(),
                                  rest[f"forearm.{side}"].matrix_local.to_quaternion())
            clear = check_for(side)
            raw = list(zip(raw_arms[f"upper_arm.{side}"], raw_arms[f"forearm.{side}"], strict=True))
            safe = list(zip(upper, forearm, strict=True))
            for i, target in enumerate(_correction_candidates(raw, safe, chest_track, fps)):
                if target is None:
                    continue
                neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(safe)]
                surrounding = [(upper[j], forearm[j]) for j in neighbors]
                protected = [(min(i, j), max(i, j)) for j in neighbors
                             if clear(min(i, j), max(i, j), {}, include_other=True)]
                current = upper[i], forearm[i]
                cost = _chain_motion_cost(current, surrounding)
                for amount in (1.0, 0.65, 0.35):
                    candidate = _interpolate_chain(current, target, amount)
                    if (not _forward_flexion(candidate, bend_axis)
                            or _chain_motion_cost(candidate, surrounding) >= cost - 1e-8):
                        continue
                    replacements = {i: candidate}
                    if clear(i, i, replacements, include_other=True) and all(
                        clear(a, b, replacements, include_other=True) for a, b in protected
                    ):
                        upper[i], forearm[i] = candidate
                        reasons[side][i] = reasons[side][i] or "mesh_clearance_transition"
                        break

            for left in range(len(chest_track) - 1):
                right = left + 1
                if clear(left, right, {}):
                    continue
                protected = [(left, left), (right, right), (left, right)]
                for a, b in ((left - 1, left), (right, right + 1)):
                    if 0 <= a < b < len(chest_track) and clear(a, b, {}):
                        protected.append((a, b))
                selected = None
                for angle in _ANGLES:
                    for axis in self._axes[side]:
                        for indices in ((left,), (right,), (left, right)):
                            replacements = {}
                            for index in indices:
                                correction = Quaternion(chest_track[index] @ axis, angle)
                                replacements[index] = (correction @ upper[index], correction @ forearm[index])
                            if all(clear(a, b, replacements) for a, b in protected):
                                selected = replacements
                                break
                        if selected is not None:
                            break
                    if selected is not None:
                        break
                if selected is None:
                    reasons[side][left] = "mesh_clearance_interpolation_unresolved"
                    continue
                for index, (adjusted_upper, adjusted_forearm) in selected.items():
                    upper[index], forearm[index] = adjusted_upper.normalized(), adjusted_forearm.normalized()
                    reasons[side][index] = reasons[side][index] or "mesh_clearance_interpolation"

        for side in _SIDES:
            upper, forearm = (arm_tracks[f"{part}.{side}"] for part in ("upper_arm", "forearm"))
            bend_axis = flex_axis(rest[f"upper_arm.{side}"].matrix_local.to_quaternion(),
                                  rest[f"forearm.{side}"].matrix_local.to_quaternion())
            raw = list(zip(raw_arms[f"upper_arm.{side}"], raw_arms[f"forearm.{side}"], strict=True))
            safe = list(zip(upper, forearm, strict=True))
            changed = _smooth_clearance_jumps(raw, safe, chest_track, fps, bend_axis, check_for(side))
            for index, (new_upper, new_forearm) in changed.items():
                upper[index], forearm[index] = new_upper, new_forearm
                reasons[side][index] = "mesh_clearance_jump_smoothed"

    def apply(self, arms, rest, chest_track, head_track, *, source_start, fps,
              leg_clearance=None, leg_tracks=None):
        """Correct colliding frame keys and report the affected source intervals."""
        arm_tracks = {part: rotations.copy() for part, rotations in arms.items()}
        rest_origin = rest["torso"].head_local
        head_offset = rest["head"].head_local - rest_origin
        torso_rotation = rest["torso"].matrix_local.to_quaternion()
        head_rotation = rest["head"].matrix_local.to_quaternion()
        shoulders = {side: rest[f"upper_arm.{side}"].head_local - rest_origin for side in _SIDES}
        resting = {side: self._resting_arm(side, rest) for side in _SIDES}
        corrections = {side: _IDENTITY for side in _SIDES}
        decay = min(1, 7 / float(fps))
        reasons = {side: [] for side in _SIDES}
        for index, (chest, relative_head) in enumerate(zip(chest_track, head_track, strict=True)):
            torso = self._shapes["torso"].placed(chest @ torso_rotation, rest_origin, padding=_CLEARANCE)
            head = self._shapes["head"].placed(chest @ relative_head @ head_rotation,
                                               rest_origin + chest @ head_offset, padding=_CLEARANCE)
            obstacles = (leg_clearance.obstacles_at(leg_tracks, rest, chest, index)
                         if leg_clearance is not None and leg_tracks is not None else ())
            inverse = chest.inverted()
            for side in _SIDES:
                shoulder = rest_origin + chest @ shoulders[side]
                upper_part, forearm_part = f"upper_arm.{side}", f"forearm.{side}"
                upper, forearm = arms[upper_part][index], arms[forearm_part][index]

                def clear(correction):
                    delta = chest @ correction @ inverse
                    return self._safe(side, shoulder, delta @ upper, delta @ forearm, torso, head, obstacles)

                correction = self._pick(side, clear, corrections[side], decay)
                corrections[side] = correction if correction is not None else _IDENTITY
                reason = None
                if correction is None:
                    # Keep the shoulder down and open when rotating the
                    # original folded pose would require an extreme arc.
                    calm_upper = chest @ resting[side][0]
                    calm_hinge = resting[side][0].inverted() @ resting[side][1]
                    hinge = upper.inverted() @ forearm
                    for weight in (0.30, 0.50, 0.70, 0.85, 1.0):
                        candidate_upper = upper.slerp(calm_upper, weight)
                        candidate_forearm = candidate_upper @ hinge.slerp(calm_hinge, weight)
                        if self._safe(side, shoulder, candidate_upper, candidate_forearm, torso, head, obstacles):
                            upper, forearm = candidate_upper, candidate_forearm
                            reason = "mesh_clearance_neutralized"
                            break
                    else:
                        reason = "mesh_clearance_unresolved"
                elif correction.angle > 0:
                    delta = chest @ correction @ inverse
                    upper, forearm = delta @ upper, delta @ forearm
                    reason = "mesh_clearance"
                reasons[side].append(reason)
                arm_tracks[upper_part][index] = upper.normalized()
                arm_tracks[forearm_part][index] = forearm.normalized()

        self._guard_interpolation(arm_tracks, rest, chest_track, head_track, reasons,
                                  leg_clearance, leg_tracks, arms, fps)

        ranges = []
        for side, rows in reasons.items():
            for reason, run in groupby(range(len(rows)), rows.__getitem__):
                indices = list(run)
                if reason:
                    ranges.append({"control": f"arm.{side}", "reason": reason,
                                   "treatment": "unresolved" if reason.endswith("unresolved") else "corrected",
                                   "source_range": [source_start + indices[0], source_start + indices[-1]]})
        return arm_tracks, ranges
