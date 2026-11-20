"""Turn local pose estimates into an editable action on the verified white-model rig."""

from fractions import Fraction
import json
import math
from statistics import median
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

from .arm_clearance import ArmClearance, calibrated_arm_parts
from .arm_hold import stabilize_observed_holds
from .arm_observations import filter_rear_arms
from .arm_stability import arm_rotations, limit_arm_swing, settle_unobserved_arm, stabilize_rear_arm
from .chains import limit_steps, solve_chain
from .gait import complete_walk, settle_unobserved_legs
from .joint_mesh import register_joint_meshes
from .joint_limits import flex_axis, hinge_direction
from .leg_clearance import LegClearance, calibrated_leg_parts
from .observations import (
    LIMBS, MIN_CONFIDENCE, PARTS, guide_partial_hips, observations, rear_sample, weak_arm_directions,
)
from .rear_orientation import parse_yaw_anchors, stabilize_rear_orientation
from .stabilize import angle, blend, centered_rotations, stabilize_track
from .torso_translation import stable_translation
from .wrist_support import stabilize_supported_wrist, validate_support_interval

PARENTS = {
    "forearm.L": "upper_arm.L", "forearm.R": "upper_arm.R",
    "shin.L": "thigh.L", "shin.R": "thigh.R",
}
LONG_AXIS = Vector((0, 1, 0))
ROLL_AXIS = Vector((1, 0, 0))
BODY_VIEWS = ("auto", "rear")
CARRY_HANDS = ("none", "L", "R")
GAIT_MODES = ("auto", "walk", "observed")


def transport_rotation(previous, direction):
    """Align the long axis while carrying roll forward continuously within the shot."""
    current = previous @ LONG_AXIS
    if current.dot(direction) < -0.999999:
        previous = Quaternion(previous @ ROLL_AXIS, math.pi) @ previous
        current = previous @ LONG_AXIS
    turn = current.rotation_difference(direction)
    return (turn @ previous).normalized()


def calibrate_limb_deform_lengths(rig):
    """Match the appended rig's deform constraints to the normalized bone lengths."""
    for part in LIMBS:
        for suffix in ("", ".001"):
            bone = rig.pose.bones[f"DEF-{part}{suffix}"]
            stretches = [constraint for constraint in bone.constraints
                         if constraint.type == "STRETCH_TO" and constraint.target == rig]
            if len(stretches) != 1:
                raise ValueError(f"白模变形骨 {bone.name} 的长度约束与已验证绑定不符")
            stretches[0].rest_length = bone.bone.length


def guide_arm_gaps(values, weak, part, source_start, rate):
    """Use only coherent low-confidence runs beside a reliable observation."""
    frame_limit = max(1, round(float(rate) * 0.40))
    minimum_run = max(2, math.ceil(float(rate) * 0.09))
    step_limit = math.radians(min(50, 10 + 850 / float(rate)))
    accepted = set()
    reliable = [index for index, value in enumerate(values) if value is not None]
    for left, right in zip(reliable, reliable[1:]):
        for anchor, step in ((left, 1), (right, -1)):
            previous = values[anchor]
            run = []
            for offset in range(1, min(frame_limit, right - left - 1) + 1):
                index = anchor + offset * step
                direction = weak[index].get(part)
                if direction is None or previous.angle(direction) > step_limit:
                    break
                run.append(index)
                previous = direction
            if len(run) >= minimum_run:
                accepted.update(run)
    guides = []
    for index in sorted(accepted):
        values[index] = weak[index][part]
        frame = source_start + index
        if guides and guides[-1]["source_range"][1] == frame - 1:
            guides[-1]["source_range"][1] = frame
        else:
            guides.append({"control": part, "source_range": [frame, frame], "treatment": "guided_2d"})
    return guides


def low_carry_track(track, trusted, chest, side, *, forearm):
    """Constrain a declared carrying arm while retaining plausible observed motion."""
    sign = 1 if side == "L" else -1
    neutral = Vector((sign * (0.09 if forearm else 0.18), -0.06, -1)).normalized()
    limit = math.radians(14 if forearm else 16)
    result = []
    for direction, observed, body in zip(track, trusted, chest, strict=True):
        forward = body @ Vector((0, -1, 0))
        heading = Quaternion((0, 0, 1), math.atan2(forward.x, -forward.y))
        rest_direction = heading @ neutral
        angle = rest_direction.angle(direction)
        if not observed or angle > math.radians(55):
            direction = rest_direction
        elif angle > limit:
            direction = blend(rest_direction, direction, limit / angle)
        result.append(direction)
    return result


def source_torso_lift(capture, rest):
    """Convert camera-compensated image displacement to a bounded rig-local lift."""
    samples = capture["samples"]
    motion = capture.get("image_torso_motion")
    if not motion:
        return [0.0] * len(samples)
    up = motion.get("up")
    if not isinstance(up, list) or len(up) != len(samples):
        raise ValueError("躯干像素跟踪长度与源帧范围不一致")
    width, height = capture["source_size"]
    spans = [
        math.hypot((row[11][3] - row[12][3]) * width / height, row[11][4] - row[12][4])
        for row in samples if row is not None and min(row[11][5], row[12][5]) >= MIN_CONFIDENCE
    ]
    spans = [span for span in spans if math.isfinite(span) and span > 0.035]
    if not spans:
        return [0.0] * len(samples)
    rig_width = (rest["upper_arm.L"].head_local - rest["upper_arm.R"].head_local).length
    scale = rig_width / median(spans)
    cap = rig_width * 0.075
    return [max(-cap, min(cap, scale * value)) if value is not None and math.isfinite(value) else 0.0
            for value in up]


def _visible_gait_leg(row, side, aperture):
    if row is None:
        return False
    start = 23 if side == "L" else 24
    left, top, right, bottom = aperture
    return all(
        math.isfinite(x) and math.isfinite(y) and confidence >= minimum
        and left + 0.006 <= x <= right - 0.006 and top + 0.006 <= y <= bottom - 0.006
        for (_, _, _, x, y, confidence), minimum in zip(
            (row[start], row[start + 2], row[start + 4]), (0.58, 0.62, 0.65), strict=True,
        )
    )


def solve_tracks(samples, rest, rest_rotations, source_start, rate, source_size, two_bone,
                 body_view="auto", carry_hand="none", aperture=(0.0, 0.0, 1.0, 1.0), yaw_anchors=()):
    rear_view = body_view == "rear"
    if rear_view:
        if not source_size:
            raise ValueError("持续背向模式需要源片画面尺寸，请重新识别源视频")
        samples = [rear_sample(row, source_size) for row in samples]
    frames = [observations(row, source_size, rear_view=rear_view) for row in samples]
    review_ranges = []
    if not rear_view and source_size:
        review_ranges.extend(guide_partial_hips(frames, samples, source_size, fps=rate, source_start=source_start))
    if sum(row["torso"] is not None for row in frames) < 2:
        raise ValueError("躯干可靠帧少于 2 帧，请选择人物清晰可见的单人片段")

    tracks = {}
    trusted = {}
    if rear_view:
        review_ranges.extend(filter_rear_arms(samples, frames, source_size, fps=rate, source_start=source_start))

    def solve(part, values, fallback):
        track, ranges = stabilize_track(values, part=part, source_start=source_start, fps=rate, fallback=fallback)
        review_ranges.extend(ranges)
        tracks[part] = track
        if part in LIMBS or part == "torso":
            trusted[part] = [row[part] is not None for row in frames]
            for row in ranges:
                first, last = (frame - source_start for frame in row["source_range"])
                trusted[part][first:last + 1] = [False] * (last - first + 1)
        return track

    chest = solve("torso", [row["torso"] for row in frames], Quaternion())
    if rear_view:
        chest = stabilize_rear_orientation(centered_rotations(chest, rate, 0.09),
                                           source_start=source_start, fps=rate, anchors=yaw_anchors)
    else:
        chest = limit_steps(chest, math.radians(540 / float(rate)))
        chest = centered_rotations(chest, rate, 0.11)
    tracks["torso"] = chest
    # During missing head observations, hold the last head angle relative to the chest.
    head = [base.inverted() @ row["head"] if row["head"] is not None else None
            for base, row in zip(chest, frames, strict=True)]
    neutral_head = Quaternion()
    head_limit = math.radians(95)
    head = [blend(neutral_head, value, head_limit / max(head_limit, angle(neutral_head, value)))
            if value is not None else None for value in head]
    head = solve("head", head, neutral_head)
    tracks["head"] = centered_rotations(head, rate, 0.12)
    # Filter limb directions in camera-oriented space. Holding a missing wrist or
    # ankle relative to the chest would rotate it whenever the torso moves.
    weak = [weak_arm_directions(row, source_size) for row in samples] if not rear_view else None
    for part in LIMBS:
        axis = rest_rotations[part] @ LONG_AXIS
        values = [row[part] for row in frames]
        if weak is not None and part.startswith(("upper_arm", "forearm")):
            review_ranges.extend(guide_arm_gaps(values, weak, part, source_start, rate))
        solve(part, values, chest[0] @ axis)
        if not any(value is not None for value in values):
            tracks[part] = [rotation @ axis for rotation in chest]

    for lower, upper in PARENTS.items():
        kind = "arm" if lower.startswith("forearm") else "leg"
        joint = "elbow" if kind == "arm" else "knee"
        side = lower.rsplit(".", 1)[-1]
        supported = [a and b for a, b in zip(trusted[upper], trusted[lower], strict=True)]
        tracks[upper], tracks[lower], ranges = solve_chain(
            tracks[upper], tracks[lower], supported, upper_length=rest[upper].length,
            lower_length=rest[lower].length, torso=chest, kind=kind, part=f"{joint}.{side}",
            source_start=source_start, fps=rate, two_bone=two_bone,
            max_angular_speed=420 if rear_view and kind == "arm" else None,
        )
        review_ranges.extend(ranges)
        if rear_view and kind == "arm":
            tracks[upper], tracks[lower], ranges = stabilize_rear_arm(
                tracks[upper], tracks[lower], trusted[upper], trusted[lower], chest, side=side,
                fps=rate, source_start=source_start,
                shoulder_offset=rest[upper].head_local - rest["torso"].head_local,
                lengths=(rest[upper].length, rest[lower].length),
            )
            review_ranges.extend(ranges)
        if kind == "arm" and not rear_view:
            tracks[upper], tracks[lower] = settle_unobserved_arm(
                tracks[upper], tracks[lower], trusted[upper], trusted[lower], chest, side=side, fps=rate,
            )
        if kind == "arm" and side == carry_hand:
            tracks[upper] = low_carry_track(tracks[upper], trusted[upper], chest, side, forearm=False)
            tracks[lower] = low_carry_track(tracks[lower], trusted[lower], chest, side, forearm=True)
        if kind == "arm" and (rear_view or side == carry_hand):
            # Solving the chain and blending into its fallback can amplify motion
            # at the distal segment even after limiting its reach and bend plane.
            max_step = math.radians(300 / float(rate))
            tracks[upper] = limit_steps(tracks[upper], max_step)
            tracks[lower] = limit_steps(tracks[lower], max_step)
        if kind == "leg":
            visible = [_visible_gait_leg(row, side, aperture) for row in samples]
            for part in (upper, lower):
                trusted[part] = [a and b for a, b in zip(trusted[part], visible, strict=True)]
    return tracks, review_ranges, trusted


def smooth_chain(chest_track, uppers, lowers, rest_upper, rest_lower, rate, *, arm):
    """Filter in the moving body/hinge frames and keep the signed bend valid."""
    seconds = 0.12 if arm else 0.075
    relative_upper = [chest.inverted() @ upper for chest, upper in zip(chest_track, uppers, strict=True)]
    relative_lower = [upper.inverted() @ lower for upper, lower in zip(uppers, lowers, strict=True)]
    relative_upper = centered_rotations(relative_upper, rate, seconds)
    relative_lower = centered_rotations(relative_lower, rate, seconds)
    axis = flex_axis(rest_upper, rest_lower)
    side_limit = math.radians(23 if arm else 10)
    stable_upper, stable_lower = [], []
    for chest, upper, lower in zip(chest_track, relative_upper, relative_lower, strict=True):
        upper = (chest @ upper).normalized()
        lower = (upper @ lower).normalized()
        direction = hinge_direction(upper, lower @ LONG_AXIS, axis, math.radians(145), side_limit)
        stable_upper.append(upper)
        stable_lower.append(transport_rotation(lower, direction))
    return stable_upper, stable_lower


def write_action(context, rig, controls, capture, rate, body_view="auto", carry_hand="none",
                 gait_mode="auto", yaw_anchors=(), arm_swing=1.0, forward_bent_hand="none",
                 clearance=None, leg_clearance=None, *, support_hand="none", support_frames=None,
                 stabilize_arm_holds=False):
    from production_pose import two_bone

    scene = context.scene
    source_start = capture["source_range"][0]
    bones = {part: rig.pose.bones[controls[part]] for part in PARTS}
    joint_deformers = {part: rig.pose.bones[f"DEF-{part}"] for part in PARENTS}
    rest = {part: rig.data.bones[bone.name] for part, bone in bones.items()}
    rest_rotations = {part: bone.matrix_local.to_quaternion() for part, bone in rest.items()}
    torso_origin = rest["torso"].head_local.copy()
    tracks, review_ranges, trusted = solve_tracks(
        capture["samples"], rest, rest_rotations, source_start, rate, capture.get("source_size"),
        two_bone, body_view, carry_hand, capture.get("source_aperture", (0.0, 0.0, 1.0, 1.0)), yaw_anchors,
    )
    lift = source_torso_lift(capture, rest)
    ranges, gait_report, gait_offsets = complete_walk(
        tracks, trusted, rest, capture, rate, two_bone=two_bone,
        mode="none" if gait_mode == "observed" else gait_mode, lift=lift,
    )
    settle_unobserved_legs(tracks, trusted, ranges, source_start, rate)
    review_ranges.extend(ranges)
    offsets = stable_translation([offset + Vector((0, 0, rise))
                                  for offset, rise in zip(gait_offsets, lift, strict=True)], rate)
    arms = {}
    for side in ("L", "R"):
        upper, lower = f"upper_arm.{side}", f"forearm.{side}"
        if arm_swing != 1.0 or side == forward_bent_hand:
            for part in (upper, lower):
                tracks[part] = limit_arm_swing(
                    tracks[part], tracks["torso"], part=part, swing=arm_swing,
                    forward_bent=side == forward_bent_hand, low_carry=side == carry_hand,
                )
        rotations = arm_rotations(
            tracks["torso"], tracks[upper], tracks[lower], rest_rotations[upper], rest_rotations[lower], rate,
        )
        arms[upper], arms[lower] = smooth_chain(
            tracks["torso"], *rotations, rest_rotations[upper], rest_rotations[lower], rate, arm=True,
        )
    legs = {}
    for side in ("L", "R"):
        thigh, shin = f"thigh.{side}", f"shin.{side}"
        thigh_rest, shin_rest = rest_rotations[thigh], rest_rotations[shin]
        knee_axis = flex_axis(thigh_rest, shin_rest)
        twist = transport_rotation(thigh_rest, shin_rest @ LONG_AXIS).inverted() @ shin_rest
        legs[thigh], legs[shin] = [], []
        for chest, upper_direction, lower_direction in zip(
            tracks["torso"], tracks[thigh], tracks[shin], strict=True,
        ):
            upper = transport_rotation(chest @ thigh_rest, upper_direction)
            direction = hinge_direction(upper, lower_direction, knee_axis,
                                        math.radians(145), math.radians(10))
            legs[thigh].append(upper)
            legs[shin].append(transport_rotation(upper, direction) @ twist)
        legs[thigh], legs[shin] = smooth_chain(
            tracks["torso"], legs[thigh], legs[shin], thigh_rest, shin_rest, rate, arm=False,
        )
    if leg_clearance is not None:
        legs, ranges = leg_clearance.apply(legs, rest, tracks["torso"], source_start=source_start, fps=rate)
        review_ranges.extend(ranges)
    if clearance is not None:
        raw_arms = arms
        arms, ranges = clearance.apply(
            arms, rest, tracks["torso"], tracks["head"], source_start=source_start, fps=rate,
            leg_clearance=leg_clearance, leg_tracks=legs,
        )
        review_ranges.extend(ranges)
        if stabilize_arm_holds:
            arms, ranges = stabilize_observed_holds(
                arms, raw_arms, rest, tracks["torso"], tracks["head"], capture, trusted["torso"], rate,
                clearance, leg_clearance, legs,
            )
            review_ranges.extend(ranges)
    support_report = None
    if support_frames is not None:
        shoulder_offset = rest[f"upper_arm.{support_hand}"].head_local - torso_origin
        shoulders = [torso_origin + chest @ shoulder_offset + shift
                     for chest, shift in zip(tracks["torso"], offsets, strict=True)]
        collision_check = (clearance._collision_checker(
            arms, rest, tracks["torso"], tracks["head"], leg_clearance, legs,
        )(support_hand) if clearance is not None and leg_clearance is not None else None)
        arms, support_report = stabilize_supported_wrist(
            arms, shoulders, tracks["torso"], rest, capture, hand=support_hand,
            frames=support_frames, fps=rate, rear=body_view == "rear", collision_check=collision_check, solve=two_bone,
        )
        applied = support_report["status"] == "applied"
        review_ranges.append({
            "control": f"arm.{support_hand}", "source_range": list(support_frames),
            "treatment": "guided" if applied else "unresolved",
            "reason": "fixed_wrist_support" if applied else support_report["reason"],
        })
    limb_rotations = arms | legs
    rig.animation_data_clear()
    for bone in bones.values():
        bone.rotation_mode = "QUATERNION"
    previous = {}
    max_gap = max_length_error = max_deform_gap = 0.0
    unit_scale = Vector((1, 1, 1))
    for index, chest in enumerate(tracks["torso"]):
        frame = source_start + index + 1
        scene.frame_set(frame)
        for part, bone in bones.items():
            parent = PARENTS.get(part)
            if part in limb_rotations:
                rotation = limb_rotations[part][index]
            elif part == "head":
                rotation = chest @ tracks[part][index] @ rest_rotations[part]
            else:
                rotation = chest @ rest_rotations[part]
            position = (bones[parent].tail.copy() if parent else torso_origin
                        + chest @ (rest[part].head_local - torso_origin)
                        + offsets[index])
            bone.matrix = Matrix.LocRotScale(position, rotation, unit_scale)
            quaternion = bone.rotation_quaternion
            if part in previous and quaternion.dot(previous[part]) < 0:
                quaternion.negate()
            previous[part] = quaternion.copy()
            bone.keyframe_insert("location", frame=frame, group=bone.name)
            bone.keyframe_insert("rotation_quaternion", frame=frame, group=bone.name)
            context.view_layer.update()
        for part, bone in bones.items():
            max_length_error = max(max_length_error, abs((bone.tail - bone.head).length - rest[part].length))
        for part, parent in PARENTS.items():
            joint = bones[parent].tail
            max_gap = max(max_gap, (joint - bones[part].head).length)
            max_deform_gap = max(max_deform_gap, (joint - joint_deformers[part].head).length)
    if not all(math.isfinite(value) and value <= 1e-4 for value in (max_gap, max_length_error, max_deform_gap)):
        raise ValueError(f"白模绑定检测失败：骨段误差 {max_length_error:.5g}，控制与变形关节误差 {max_gap:.5g}/{max_deform_gap:.5g}")

    action = rig.animation_data.action
    action.use_fake_user = True
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for curve in channelbag.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = "LINEAR"
    return (action, review_ranges, max_gap, max_length_error, max_deform_gap,
            sum(value != 0 for value in lift), gait_report, support_report)


def action_target_clone(context, target, toolkit, controls):
    """Stage pose solving on a temporary copy of a previously calibrated rig."""
    if target is None or target.type != "ARMATURE" or target.name not in context.scene.objects:
        raise ValueError("请选择已校准的白模骨架作为动作目标")
    library = json.loads((toolkit / "assets" / "rig_library" / "library.json").read_text(encoding="utf8"))
    record = bpy.data.texts.get(target.get("white_model_mocap_report", ""))
    calibrated = target.get("white_model_mocap_binding") == "shared_fk_pivot"
    if record and not calibrated:
        calibrated = json.loads(record.as_string()).get("joint_mesh_binding") == "shared_fk_pivot"
    if (not calibrated or target.get("production_library_sha256") != library["asset_sha256"]
            or target.data.pose_position != "POSE"
            or any(target.pose.bones.get(name) is None or target.pose.bones[name].rotation_mode != "QUATERNION"
                   for name in controls.values())):
        raise ValueError("动作目标须为同模板、已校准且使用四元数 FK 控制的白模骨架")
    clone = target.copy()
    clone.animation_data_clear()
    clone.parent = None
    clone.matrix_world = Matrix.Identity(4)
    clone.hide_viewport = False
    clone.hide_render = True
    context.scene.collection.objects.link(clone)
    clone.hide_set(False)
    for owner in (clone, *clone.pose.bones):
        for constraint in owner.constraints:
            for field in ("target", "pole_target", "space_object"):
                if getattr(constraint, field, None) == target:
                    setattr(constraint, field, clone)
            for item in getattr(constraint, "targets", ()):
                if item.target == target:
                    item.target = clone
    context.view_layer.update()
    return clone


def bake(context, *, toolkit, capture, identity, role, sync_timeline, body_view="auto",
         carry_hand="none", gait_mode="auto", target_rig=None, rear_yaw_keys="",
         arm_swing=1.0, forward_bent_hand="none", support_hand="none", support_frames=None,
         stabilize_arm_holds=False):
    scene = context.scene
    first, last = capture["source_range"]
    frame_count = capture["source_frame_count"]
    rate = Fraction(capture["fps"])
    if not (0 <= first < last < frame_count and len(capture["samples"]) == last - first + 1 and rate > 0):
        raise ValueError("识别结果的帧范围与源片时间轴不一致")
    support_frames = validate_support_interval(support_hand, support_frames, first, last)
    if not identity.strip():
        raise ValueError("需要填写角色身份")
    if body_view not in BODY_VIEWS:
        raise ValueError(f"不支持的躯干朝向模式 {body_view}")
    if carry_hand not in CARRY_HANDS:
        raise ValueError(f"不支持的持物手设置 {carry_hand}")
    if forward_bent_hand not in CARRY_HANDS:
        raise ValueError(f"不支持的前屈手设置 {forward_bent_hand}")
    if forward_bent_hand != "none" and forward_bent_hand == carry_hand:
        raise ValueError("持续低位持物和贴身前屈需要设置在不同的手")
    if not 0.0 <= arm_swing <= 1.0:
        raise ValueError("手臂摆幅须在 0 到 1 之间")
    if gait_mode not in GAIT_MODES:
        raise ValueError(f"不支持的下肢补全模式 {gait_mode}")
    yaw_anchors = parse_yaw_anchors(rear_yaw_keys, first, last)
    if yaw_anchors and body_view != "rear":
        raise ValueError("背向角度锚点须与持续背向模式一起使用")
    if target_rig is not None and sync_timeline:
        raise ValueError("现有骨架使用工程时间轴，请关闭同步源片时间轴")
    if not sync_timeline and (abs(scene.render.fps / scene.render.fps_base - float(rate)) > 0.001
                              or scene.frame_start > first + 1 or scene.frame_end < last + 1):
        raise ValueError("场景帧率或范围和源片不一致。请在制作工程中校准时间轴，或在新工程中启用同步")

    scripts = str(toolkit / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from rig_library import append_character, CONTROLS, PARTS as TEMPLATE_PARTS

    scene_frame = scene.frame_current, scene.frame_subframe
    tracked_data = (bpy.data.objects, bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                    bpy.data.curves, bpy.data.actions, bpy.data.texts)
    existing = [set(items) for items in tracked_data]
    try:
        joint_mesh_calibration = None
        if target_rig is not None:
            rig = action_target_clone(context, target_rig, toolkit, CONTROLS)
            geometry_rig = target_rig
            parts = calibrated_arm_parts(target_rig, TEMPLATE_PARTS, CONTROLS)
            leg_parts = calibrated_leg_parts(target_rig, CONTROLS)
        else:
            rig, parts = append_character(toolkit / "assets" / "rig_library", identity.strip(), role,
                                          source_template=toolkit / "人.blend")
            # Preserve the approved white-model body's visible neutral proportions.
            joint_mesh_calibration = register_joint_meshes(context, rig, parts, PARENTS, CONTROLS)
            calibrate_limb_deform_lengths(rig)
            geometry_rig = rig
            leg_parts = parts
        clearance = ArmClearance(context, geometry_rig, parts, CONTROLS) if parts else None
        leg_clearance = LegClearance(context, geometry_rig, leg_parts, CONTROLS) if leg_parts else None
        action, ranges, gap, length_error, deform_gap, lift_frames, gait_report, support_report = write_action(
            context, rig, CONTROLS, capture, rate, body_view, carry_hand, gait_mode, yaw_anchors,
            arm_swing, forward_bent_hand, clearance, leg_clearance,
            support_hand=support_hand, support_frames=support_frames,
            stabilize_arm_holds=stabilize_arm_holds,
        )
        action.name = f"白模动捕_{identity.strip()}_{first:05d}"
        image_motion = capture.get("image_torso_motion") or {}
        report = {
            "schema_version": 1, **{k: v for k, v in capture.items()
                                    if k not in ("samples", "image_torso_motion")},
            "pose_processing": "shot_local_motion_stability_v18" if stabilize_arm_holds
            else "shot_local_motion_stability_v15",
            "arm_mesh_clearance_checked": clearance is not None,
            "leg_mesh_clearance_checked": leg_clearance is not None,
            "body_view": body_view,
            "rear_yaw_keys": yaw_anchors,
            "rear_heading_source": ("guided" if yaw_anchors else "filtered_landmarks")
            if body_view == "rear" else None,
            "carry_hand": carry_hand,
            "arm_swing": arm_swing,
            "forward_bent_hand": forward_bent_hand,
            "gait_mode": gait_mode,
            "gait_completion": gait_report,
            "image_motion_pairs": image_motion.get("measured_pairs", 0),
            "image_lift_frames": lift_frames,
            "head_orientation": "follows_torso" if body_view == "rear" else "tracked",
            "joint_mesh_binding": "shared_fk_pivot",
            "joint_mesh_calibration": joint_mesh_calibration,
            "identity": identity.strip(), "rig": (target_rig if target_rig is not None else rig).name,
            "action": action.name, "output_mode": "detached_action" if target_rig is not None else "capture_rig",
            "scene_range": [first + 1, last + 1],
            "model_rig_orientation": "camera level, facing -Y in Blender at the template front",
            "root_translation": "manual", "physical_contacts": "manual",
            "detected_frames": sum(row is not None for row in capture["samples"]),
            "confidence_threshold": MIN_CONFIDENCE, "review_ranges": ranges,
            "max_bone_gap": gap, "max_bone_length_error": length_error,
            "max_deform_joint_offset": deform_gap,
            "visual_acceptance": "not_determined",
        }
        if support_report is not None:
            report["wrist_support"] = support_report
        note = bpy.data.texts.new(f"白模动捕诊断_{identity.strip()}.json")
        report["report_text"] = note.name
        note.write(json.dumps(report, ensure_ascii=False, indent=2))
        action["white_model_mocap_report"] = note.name
        if target_rig is None:
            rig["white_model_mocap_report"] = note.name
            rig["white_model_mocap_binding"] = "shared_fk_pivot"
        else:
            rig.animation_data.action_slot.name_display = target_rig.name
            bpy.data.objects.remove(rig, do_unlink=True)
            rig = None
        if sync_timeline:
            scene.render.fps = round(float(rate))
            scene.render.fps_base = scene.render.fps / float(rate)
            scene.frame_start, scene.frame_end = 1, frame_count
        return rig, report
    except BaseException:
        created = [item for items, before in zip(tracked_data, existing, strict=True) for item in items if item not in before]
        bpy.data.batch_remove(ids=created)
        raise
    finally:
        scene.frame_set(scene_frame[0], subframe=scene_frame[1])
