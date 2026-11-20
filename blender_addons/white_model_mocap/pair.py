"""Bake two independently tracked subjects onto a shared white-model stage."""

from fractions import Fraction
import json
import math
from statistics import median

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from . import motion
from .observations import MIN_CONFIDENCE


ROLES = ("primary", "secondary", "third", "fourth")
SHOULDERS = ("upper_arm_fk.L", "upper_arm_fk.R")
UP = Vector((0, 0, 1))


def _clip(capture):
    try:
        first, last = capture["source_range"]
        count = capture["source_frame_count"]
        width, height = capture["source_size"]
        rate = Fraction(capture["fps"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("双人动捕的源片元数据无效") from error
    if (any(type(item) is not int for item in (first, last, count, width, height))
            or not (0 <= first < last < count and width > 0 and height > 0 and rate > 0)):
        raise ValueError("双人动捕的源片范围、尺寸或帧率无效")
    return first, last, count, width, height, rate


def _torso_centers(samples, aspect):
    """Use the shoulders when visible and the hips as a fallback."""
    centers = {}
    for index, row in enumerate(samples):
        if row is None:
            continue
        if len(row) != 33 or any(len(point) != 6 for point in row):
            raise ValueError("双人动捕的关键点格式无效")
        for left, right in ((11, 12), (23, 24)):
            lx, ly, lc = row[left][3:]
            rx, ry, rc = row[right][3:]
            if (min(lc, rc) >= MIN_CONFIDENCE and 0 <= lx <= 1 and 0 <= rx <= 1
                    and 0 <= ly <= 1 and 0 <= ry <= 1):
                centers[index] = ((lx + rx) / 2, math.hypot(lx - rx, (ly - ry) / aspect))
                break
    return centers


def _opening_layout(capture, clip):
    first, last, _, width, height, rate = clip
    subjects = capture.get("subjects")
    if (not isinstance(subjects, (list, tuple)) or len(subjects) != 2
            or any(not isinstance(subject, dict) for subject in subjects)):
        raise ValueError("双人动捕必须提供两份独立的识别结果")
    observed = []
    for subject in subjects:
        if (_clip(subject) != clip
                or ("source" in capture and subject.get("source") != capture["source"])):
            raise ValueError("两份识别结果必须来自同一源片及帧范围")
        samples = subject.get("samples")
        if not isinstance(samples, list) or len(samples) != last - first + 1:
            raise ValueError("双人动捕的姿态采样帧数不一致")
        try:
            centers = _torso_centers(samples, width / height)
        except TypeError as error:
            raise ValueError("双人动捕的关键点格式无效") from error
        if not centers:
            raise ValueError("有一人缺少可确认的躯干位置")
        observed.append(centers)

    if subjects[0]["samples"] == subjects[1]["samples"]:
        raise ValueError("两份识别结果包含同一条人物姿态轨迹")
    overlap = sorted(observed[0].keys() & observed[1].keys())
    window = max(1, round(float(rate) * 0.25))
    layouts = []
    for centers in observed:
        frames = overlap or sorted(centers)
        frames = [index for index in frames if index < frames[0] + window]
        layouts.append({
            "source_x": median(centers[index][0] for index in frames),
            "source_torso_span": median(centers[index][1] for index in frames),
            "anchor_source_frames": [index + first for index in frames],
        })
    return subjects, layouts, bool(overlap)


def _camera_offsets(scene, rigs, layout):
    camera = scene.camera
    if camera is None or camera.data.type not in {"ORTHO", "PERSP"}:
        return None
    rotation = camera.matrix_world.to_quaternion()
    # The view-plane direction that stays horizontal on the ground preserves the camera depth.
    right = (rotation @ Vector((0, 0, -1))).cross(UP)
    if right.length < 1e-5:
        right = rotation @ Vector((1, 0, 0))
        right.z = 0
    right.normalize()
    offsets = []
    for rig, entry in zip(rigs, layout, strict=True):
        center = (rig.pose.bones[SHOULDERS[0]].head + rig.pose.bones[SHOULDERS[1]].head) / 2
        center = rig.matrix_world @ center
        screen = world_to_camera_view(scene, camera, center)
        delta = world_to_camera_view(scene, camera, center + right).x - screen.x
        if screen.z <= 1e-5 or not math.isfinite(delta) or abs(delta) < 1e-5:
            return None
        offsets.append(right * ((entry["source_x"] - screen.x) / delta))
    return offsets


def _template_offsets(rigs, layout):
    """Keep the pair centered in a camera-less scene using template proportions."""
    span = median(entry["source_torso_span"] for entry in layout)
    shoulder_l, shoulder_r = (rigs[0].data.bones[name].head_local for name in SHOULDERS)
    scale = (shoulder_l - shoulder_r).length / max(span, 0.04)
    center_x = median(entry["source_x"] for entry in layout)
    return [Vector(((entry["source_x"] - center_x) * scale, 0, 0)) for entry in layout]


def _stage_pair(context, rigs, reports, layout, simultaneous):
    scene = context.scene
    scene.frame_set(min(entry["anchor_source_frames"][0] for entry in layout) + 1)
    offsets = _camera_offsets(scene, rigs, layout)
    camera = scene.camera.name if offsets is not None else None
    method = "scene_camera_horizontal" if camera is not None else "template_x_from_torso_span"
    if offsets is None:
        offsets = _template_offsets(rigs, layout)
    for rig, report, entry, offset in zip(rigs, reports, layout, offsets, strict=True):
        rig.location += offset
        report["semantic_role"] = rig["production_role"]
        report["root_translation"] = "static_horizontal_pair_placement"
        report["pair_layout"] = {
            **entry,
            "method": method,
            "camera": camera,
            "simultaneous_anchor": simultaneous,
            "world_offset": list(offset),
            "other_identity": rigs[1 if rig == rigs[0] else 0]["production_identity"],
            "depth": "unchanged_from_template",
            "horizontal_motion": "not_reconstructed",
            "inter_character_contacts": "manual",
        }
        note = bpy.data.texts[report["report_text"]]
        note.clear()
        note.write(json.dumps(report, ensure_ascii=False, indent=2))
    context.view_layer.update()


def bake_pair(context, *, toolkit, capture, identities, roles, sync_timeline, gait_mode="auto"):
    """Bake two new template rigs, rolling back both if either fails."""
    if (not isinstance(identities, (tuple, list)) or len(identities) != 2
            or any(not isinstance(name, str) or not name.strip() for name in identities)):
        raise ValueError("需要填写两个不同的角色身份")
    identities = tuple(name.strip() for name in identities)
    if identities[0] == identities[1]:
        raise ValueError("需要填写两个不同的角色身份")
    if (not isinstance(roles, (tuple, list)) or len(roles) != 2
            or roles[0] == roles[1] or any(role not in ROLES for role in roles)):
        raise ValueError("双人动捕需要两个不同的白模颜色角色")
    if any(rig.type == "ARMATURE" and (rig.get("production_identity") in identities
                                       or rig.get("production_role") in roles)
           for rig in bpy.data.objects):
        raise ValueError("已有白模使用了其中一个角色身份或颜色角色")
    if gait_mode not in motion.GAIT_MODES:
        raise ValueError(f"不支持的下肢补全模式 {gait_mode}")

    clip = _clip(capture)
    subjects, layout, simultaneous = _opening_layout(capture, clip)
    scene = context.scene
    frame = scene.frame_current, scene.frame_subframe
    start, end = scene.frame_start, scene.frame_end
    fps, fps_base = scene.render.fps, scene.render.fps_base
    existing = set(bpy.data.user_map())
    try:
        baked = [
            motion.bake(context, toolkit=toolkit, capture=subject, identity=identity, role=role,
                        sync_timeline=sync_timeline and index == 0, body_view="auto", gait_mode=gait_mode,
                        stabilize_arm_holds=True)
            for index, (subject, identity, role) in enumerate(zip(subjects, identities, roles, strict=True))
        ]
        rigs, reports = zip(*baked, strict=True)
        _stage_pair(context, rigs, reports, layout, simultaneous)
        return rigs, reports
    except BaseException:
        try:
            created = set(bpy.data.user_map()) - existing
            if created:
                bpy.data.batch_remove(ids=created)
        finally:
            scene.render.fps, scene.render.fps_base = fps, fps_base
            scene.frame_start = min(scene.frame_start, start)
            scene.frame_end = end
            scene.frame_start = start
        raise
    finally:
        scene.frame_set(frame[0], subframe=frame[1])
