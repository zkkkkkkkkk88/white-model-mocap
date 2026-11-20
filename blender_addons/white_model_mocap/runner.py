"""Run a capture without loading inference dependencies into Blender."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def default_python():
    name = "python.exe" if os.name == "nt" else "python3"
    bundled = Path(sys.exec_prefix) / "bin" / name
    return str(bundled) if bundled.is_file() else ""


def parse_subject_region(value):
    """Validate the optional first-frame target rectangle in normalized image coordinates."""
    if value is None:
        return None
    coords = value.split(",") if isinstance(value, str) else value
    if not isinstance(coords, (list, tuple)) or len(coords) != 4 or any(isinstance(v, bool) for v in coords):
        raise ValueError("目标人物范围须按左、上、右、下填写四个 0–1 坐标")
    try:
        left, top, right, bottom = map(float, coords)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("目标人物范围须按左、上、右、下填写四个 0–1 坐标") from error
    region = [left, top, right, bottom]
    if not all(math.isfinite(v) for v in region) or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("目标人物范围须位于画面内，左缘小于右缘、上缘小于下缘")
    return region


def parse_subject_regions(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("双人捕捉需要填写两个人的躯干范围")
    regions = [parse_subject_region(region) for region in value]
    if None in regions or regions[0] == regions[1]:
        raise ValueError("请为两人分别填写不同的首帧躯干范围")
    return regions


class CaptureJob:
    def __init__(self, python, request):
        # The toolkit path is shared by Blender and its worker, including in sandboxes
        # that virtualize each process's system temp directory.
        try:
            self.scratch = TemporaryDirectory(prefix=".white_model_mocap_", dir=request["toolkit"])
        except PermissionError:
            self.scratch = TemporaryDirectory(prefix="white_model_mocap_")
        self.process = None
        folder = Path(self.scratch.name)
        request_path = folder / "request.json"
        self.result_path = folder / "result.json"
        self.log_path = folder / "worker.log"
        try:
            request_path.write_text(json.dumps(request), encoding="utf-8")
            env = os.environ.copy()
            env["MPLCONFIGDIR"] = str(folder)
            with self.log_path.open("wb") as log:
                self.process = subprocess.Popen(
                    [str(python), "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("worker.py")),
                     str(request_path), str(self.result_path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=env,
                )
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def result(self):
        if self.process.poll() != 0:
            if self.process.returncode is None:
                raise RuntimeError("姿态识别仍在运行")
            log = self.log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError("姿态识别失败：" + next(iter(reversed(log.splitlines())), "无错误日志"))
        return json.loads(self.result_path.read_text(encoding="utf-8"))

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.scratch.cleanup()


def start(*, toolkit, source, python, packages, model, first, last, body_view="auto", gait_mode="auto",
          subject_region=None, subject_regions=None):
    region = parse_subject_region(subject_region)
    regions = parse_subject_regions(subject_regions)
    if regions is not None and region is not None:
        raise ValueError("单人和双人目标范围不能同时使用")
    if not (toolkit / "assets" / "rig_library" / "library.json").is_file():
        raise ValueError("请选择包含已验证骨架库的白模复刻工具包目录")
    for label, path in (("源视频", source), ("Python 解释器", python), ("姿态模型", model)):
        if not path.is_file():
            raise ValueError(f"找不到{label}文件：{path}")
    if packages is not None and not packages.is_dir():
        raise ValueError(f"找不到依赖目录：{packages}")
    if not 0 <= first < last:
        raise ValueError("请选择至少两帧，源片首帧编号为 0")
    request = {"toolkit": str(toolkit.resolve()), "source": str(source.resolve()),
               "packages": str(packages.resolve()) if packages is not None else "",
               "model": str(model.resolve()), "first": first, "last": last,
               "body_view": body_view,
               "track_image_motion": body_view == "rear" or gait_mode == "walk", "gait_mode": gait_mode}
    if region is not None:
        request["subject_region"] = region
    if regions is not None:
        request["subject_regions"] = regions
    return CaptureJob(python.resolve(), request)
