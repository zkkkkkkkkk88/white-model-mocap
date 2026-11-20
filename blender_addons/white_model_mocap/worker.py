"""Run single-camera pose detection in an isolated local Python process."""

import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import sys


def picture_aperture(frames, np):
    """Find persistent paired black borders without treating a dark shot as a crop."""
    if len(frames) < 3:
        return [0.0, 0.0, 1.0, 1.0]
    visible = np.maximum.reduce(frames) > 24

    def edges(activity):
        lit = np.flatnonzero(activity > 0.055)
        if not len(lit):
            return 0.0, 1.0
        size = len(activity)
        start, end = float(lit[0] / size), float((lit[-1] + 1) / size)
        if 0.025 < min(start, 1 - end) and max(start, 1 - end) < 0.28 and abs(start + end - 1) < 0.045:
            return start, end
        return 0.0, 1.0

    left, right = edges(visible.mean(axis=0))
    top, bottom = edges(visible.mean(axis=1))
    return [left, top, right, bottom]


def lower_body_mostly_hidden(samples):
    visible = sum(
        row is not None and all(row[i][5] >= 0.64 and 0 <= row[i][3] <= 1 and 0 <= row[i][4] <= 1
                                for i in (23, 24, 25, 26, 27, 28))
        for row in samples
    )
    return visible < 0.7 * len(samples)


def pose_samples(result):
    return [
        [[xyz.x, xyz.y, xyz.z, uv.x, uv.y, min(uv.visibility, uv.presence)]
         for xyz, uv in zip(world, picture, strict=True)]
        for world, picture in zip(result.pose_world_landmarks, result.pose_landmarks, strict=True)
    ]


def capture_track(metadata, samples, subject, wrist_guard, request, rate):
    result = dict(metadata, samples=samples)
    if wrist_guard is not None:
        result["wrist_reacquisition_rejections"] = wrist_guard.report()
    if subject is not None:
        result["subject_tracking"] = subject.report()
    if request.get("track_image_motion") or (request.get("gait_mode") == "auto" and lower_body_mostly_hidden(samples)):
        from image_motion import track_torso_motion

        result["image_torso_motion"] = track_torso_motion(metadata["source"], samples, metadata["source_range"][0], rate)
    return result


def capture(request):
    if request["packages"]:
        sys.path.insert(0, request["packages"])
    sys.path.insert(0, str(Path(request["toolkit"]) / "scripts"))
    # Isolated Python excludes the directory of the worker script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision
    from video_ops import guard, probe

    source = request["source"]
    info = probe(source)
    guard(info)
    first, last = request["first"], request["last"]
    if not 0 <= first < last < info["frame_count"]:
        raise ValueError(f"源帧范围须包含至少两帧，且位于 0–{info['frame_count'] - 1} 内")
    rate = Fraction(info["video"]["avg_frame_rate"])
    width, height = (info["video"][key] for key in ("width", "height"))
    subject = None
    dual = None
    regions = request.get("subject_regions")
    if regions is not None:
        if request.get("subject_region") is not None:
            raise ValueError("请分别使用单人范围或双人范围")
        from subject_tracking import DualSubjectTracker

        dual = DualSubjectTracker(regions, width, height, rate)
    elif request.get("subject_region") is not None:
        from subject_tracking import SubjectTracker

        subject = SubjectTracker(request["subject_region"], width, height, rate)
    subjects = dual.tracks if dual is not None else (subject,)
    wrist_guards = [None] * len(subjects)
    if request.get("body_view", "auto") == "auto":
        from wrist_tracking import WristReacquisitionGuard

        wrist_guards = [WristReacquisitionGuard(rate) for _ in subjects]
    model = Path(request["model"]).read_bytes()

    def detector(mode, count):
        # Passing bytes also supports Windows model paths containing non-ASCII characters.
        return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model), running_mode=mode, num_poses=count,
        ))

    video = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not video.isOpened():
        video.release()
        raise ValueError("OpenCV 无法打开源视频")

    tracks = [[] for _ in subjects]
    picture_samples = []
    sample_interval = max(1, (last - first + 1) // 16)
    lookup = None
    try:
        # Sequential decoding keeps source indices exact even when a shot starts on a non-keyframe.
        for index in range(first):
            if not video.grab():
                raise ValueError(f"无法读取源帧 {index}")
        with ExitStack() as detectors:
            # Leave room for bystanders without making a second detection pass.
            detection_slots = 4 if dual is not None else 2 if subject is not None else 1
            solver = detectors.enter_context(detector(vision.RunningMode.VIDEO, detection_slots))
            recovery = None
            previous_ms = -1
            for index in range(first, last + 1):
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f"无法解码源帧 {index}")
                if index == first:
                    reduced = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
                    median, upper = np.percentile(gray[9:81, 8:152], (50, 90))
                    if median < 42 and upper < 100:
                        # Keep a single transformation for the shot so the tracker
                        # sees no artificial exposure jumps. Rendering uses the source.
                        lookup = np.uint8(((np.arange(256) / 255.0) ** (1.0 / 2.2)) * 255.0)
                if (index - first) % sample_interval == 0:
                    reduced = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                    picture_samples.append(cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY))
                timestamp_ms = max(previous_ms + 1, round((index - first) * 1000 / rate))
                previous_ms = timestamp_ms
                detection_frame = cv2.LUT(frame, lookup) if lookup is not None else frame
                rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                result = solver.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                candidates = pose_samples(result)
                if dual is not None:
                    samples = tuple(track.finish_frame(torso, index) for track, torso in
                                    zip(subjects, dual.select(candidates, detection_frame, index), strict=True))
                elif subject is None:
                    samples = (candidates[0] if candidates else None,)
                else:
                    selected = subject.select(candidates, detection_frame, index)
                    recovered = False
                    if selected is None:
                        masked = subject.suppressed_frame(detection_frame, candidates)
                        if masked is not None:
                            if recovery is None:
                                recovery = detectors.enter_context(detector(vision.RunningMode.VIDEO, 3))
                            masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
                            found = recovery.detect_for_video(
                                mp.Image(image_format=mp.ImageFormat.SRGB, data=masked_rgb), timestamp_ms,
                            )
                            selected = subject.select(pose_samples(found), detection_frame, index)
                            recovered = selected is not None
                    samples = (subject.finish_frame(selected, index, recovered=recovered),)
                for track, wrist_guard, sample in zip(tracks, wrist_guards, samples, strict=True):
                    if wrist_guard is not None:
                        sample = wrist_guard.filter(detection_frame, sample, index)
                    track.append(sample)
    finally:
        video.release()
    metadata = {
        "source": source, "fps": str(rate), "source_frame_count": info["frame_count"],
        "source_size": [width, height],
        "source_range": [first, last], "source_aperture": picture_aperture(picture_samples, np),
        "model": Path(request["model"]).name,
        "detector_preprocess": "gamma_2_2" if lookup is not None else "none",
        "mediapipe_version": mp.__version__,
    }
    captures = [capture_track(metadata, samples, track, wrist_guard, request, rate)
                for samples, track, wrist_guard in zip(tracks, subjects, wrist_guards, strict=True)]
    return dict(metadata, subjects=captures) if dual is not None else captures[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    args.result.write_text(json.dumps(capture(request), allow_nan=False), encoding="utf-8")
