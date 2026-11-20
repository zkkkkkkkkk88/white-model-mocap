"""Capture a shot into a new Blender working snapshot."""

import argparse
import json
from pathlib import Path
import sys

import bpy

ADDONS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDONS))
from white_model_mocap import motion, pair, runner


def save_snapshot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output so another capture cannot race to overwrite it.
    with path.open("xb"):
        pass
    preferences = bpy.context.preferences.filepaths
    save_versions = preferences.save_version
    try:
        preferences.save_version = 0
        result = bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
        if "FINISHED" not in result or path.stat().st_size == 0:
            raise RuntimeError(f"无法保存工作快照 {path}")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        preferences.save_version = save_versions


def _parse_support_frames(value):
    try:
        first, last = value.split(":")
        return int(first), int(last)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected FIRST:LAST with two source-frame integers") from error


def _summary(report):
    fields = ("rig", "action", "report_text", "source_range", "scene_range", "detected_frames", "body_view",
              "rear_yaw_keys", "rear_heading_source", "carry_hand", "forward_bent_hand", "arm_swing",
              "gait_mode", "gait_completion", "arm_mesh_clearance_checked", "leg_mesh_clearance_checked",
              "output_mode", "image_motion_pairs", "image_lift_frames", "max_bone_gap",
              "max_bone_length_error", "visual_acceptance")
    result = {key: report[key] for key in fields}
    result.update(review_range_count=len(report["review_ranges"]),
                  detector_preprocess=report.get("detector_preprocess", "none"))
    for key in ("subject_tracking", "wrist_support", "pair_layout"):
        if key in report:
            result[key] = report[key]
    holds = [item for item in report["review_ranges"] if item.get("reason") == "observed_arm_hold"]
    if holds:
        result["observed_arm_holds"] = holds
    return result


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, default=ADDONS.parent)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--first", required=True, type=int, help="First zero-based source frame, inclusive")
    parser.add_argument("--last", required=True, type=int, help="Last zero-based source frame, inclusive")
    parser.add_argument("--identity", required=True)
    parser.add_argument("--role", required=True, choices=("primary", "secondary", "third", "fourth"))
    parser.add_argument("--subject-region", metavar="XMIN,YMIN,XMAX,YMAX",
                        help="Optional first-frame target torso rectangle, normalized 0–1 from upper left")
    parser.add_argument("--second-identity", help="Distinct second character ID for a shared capture")
    parser.add_argument("--second-role", choices=("primary", "secondary", "third", "fourth"),
                        help="Second character palette, defaults to secondary")
    parser.add_argument("--second-subject-region", metavar="XMIN,YMIN,XMAX,YMAX",
                        help="Second character's initial torso rectangle in the same image")
    parser.add_argument("--body-view", choices=motion.BODY_VIEWS, default="auto",
                        help="Use rear only for a shot that stays back-facing")
    parser.add_argument("--rear-yaw-keys", default="",
                        help="Optional source-frame:degree rear-view anchors, e.g. 24:20,72:0")
    parser.add_argument("--carry-hand", choices=motion.CARRY_HANDS, default="none",
                        help="Constrain a known carrying arm to a low posture")
    parser.add_argument("--support-hand", choices=motion.CARRY_HANDS, default="none",
                        help="Select the hand supporting itself on a fixed object")
    parser.add_argument("--support-frames", type=_parse_support_frames, metavar="FIRST:LAST",
                        help="Inclusive zero-based source frames for the fixed support interval")
    parser.add_argument("--forward-bent-hand", choices=motion.CARRY_HANDS, default="none",
                        help="Guide the chosen forearm to bend forward near the torso")
    parser.add_argument("--arm-swing", type=float, default=1.0,
                        help="Fraction of arm motion to retain in the torso frame (0 to 1)")
    parser.add_argument("--gait-mode", choices=motion.GAIT_MODES, default="auto",
                        help="Complete off-screen legs when a walking rhythm can be established")
    parser.add_argument("--target-rig", help="Generate a detached Action for an existing calibrated rig")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--packages", type=Path)
    parser.add_argument("--sync-timeline", action="store_true", help="Set full scene frame range and FPS from the source")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    use_pair = any(value is not None for value in (
        args.second_identity, args.second_role, args.second_subject_region,
    ))
    subject_regions = None
    try:
        subject_region = runner.parse_subject_region(args.subject_region)
        if use_pair:
            subject_regions = runner.parse_subject_regions((subject_region, args.second_subject_region))
            subject_region = None
    except ValueError as error:
        parser.error(f"--subject-region: {error}")
    if use_pair:
        if not args.second_identity or not args.second_identity.strip() or (
            args.second_identity.strip() == args.identity.strip()
        ):
            parser.error("--second-identity 必须是不同的角色 ID")
        args.second_role = args.second_role or "secondary"
        if args.second_role == args.role:
            parser.error("--second-role 必须使用不同配色")
        if (args.target_rig or args.body_view != "auto" or args.rear_yaw_keys
                or args.carry_hand != "none" or args.forward_bent_hand != "none"
                or args.arm_swing != 1.0 or args.support_hand != "none" or args.support_frames):
            parser.error("双人捕捉使用自动朝向和两套新骨架。单人手臂与支撑设置仅用于单人模式。")
    try:
        support_frames = motion.validate_support_interval(args.support_hand, args.support_frames, args.first, args.last)
    except ValueError as error:
        parser.error(f"--support-frames: {error}")

    output = args.output.resolve()
    if output.suffix.lower() != ".blend":
        parser.error("--output 必须是 .blend 文件路径")
    if output.exists():
        raise FileExistsError(f"输出文件已存在，请选择新路径 {output}")
    target_rig = bpy.context.scene.objects.get(args.target_rig) if args.target_rig else None
    if args.target_rig and (target_rig is None or target_rig.type != "ARMATURE" or args.sync_timeline):
        parser.error("--target-rig 须指向当前工程中已校准的骨架，并使用工程已有时间轴")
    python = args.python or runner.default_python()
    if not python:
        parser.error("未找到 Blender 自带的 Python，请通过 --python 指定可执行文件")
    toolkit = args.toolkit.resolve()
    with runner.start(toolkit=toolkit, source=args.source, python=Path(python), packages=args.packages,
                      model=args.model, first=args.first, last=args.last, body_view=args.body_view,
                      gait_mode=args.gait_mode, subject_region=subject_region,
                      subject_regions=subject_regions) as job:
        job.process.wait()
        capture = job.result()

    if use_pair:
        _, reports = pair.bake_pair(
            bpy.context, toolkit=toolkit, capture=capture, identities=(args.identity, args.second_identity),
            roles=(args.role, args.second_role), sync_timeline=args.sync_timeline, gait_mode=args.gait_mode,
        )
        summary = {"subjects": [_summary(report) for report in reports]}
    else:
        _, report = motion.bake(bpy.context, toolkit=toolkit, capture=capture, identity=args.identity,
                                role=args.role, sync_timeline=args.sync_timeline, body_view=args.body_view,
                                carry_hand=args.carry_hand, gait_mode=args.gait_mode, target_rig=target_rig,
                                rear_yaw_keys=args.rear_yaw_keys, arm_swing=args.arm_swing,
                                forward_bent_hand=args.forward_bent_hand,
                                support_hand=args.support_hand, support_frames=support_frames)
        summary = _summary(report)
    save_snapshot(output)
    summary["output"] = str(output)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
