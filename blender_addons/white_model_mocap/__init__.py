"""Blender interface for single-camera white-model motion capture."""

from pathlib import Path
import traceback

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import motion, pair, runner


bl_info = {
    "name": "白模动捕",
    "description": "从单目视频本地识别人体动作，并生成白模骨骼动作",
    "version": (0, 18, 0),
    "blender": (5, 1, 0),
    "location": "3D 视图 > 侧栏 > 白模动捕",
    "category": "Animation",
}

_active = None
_status = None
_ROLES = (
    ("primary", "第一角色 · 红", ""), ("secondary", "第二角色 · 蓝", ""),
    ("third", "第三角色 · 绿", ""), ("fourth", "第四角色 · 黄", ""),
)


def _preferences(context):
    entry = context.preferences.addons.get(__package__)
    return entry.preferences if entry is not None else None


def _path(value):
    return Path(bpy.path.abspath(value)).expanduser().resolve()


def _redraw(window_manager):
    for window in window_manager.windows:
        if window.screen is None:
            continue
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class WMOCAP_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    toolkit: StringProperty(name="白模工具包", subtype="DIR_PATH",
                            description="包含 scripts 和 assets/rig_library 的白模复刻工具包目录")
    python: StringProperty(name="Python 可执行文件", subtype="FILE_PATH",
                           description="留空使用 Blender 自带的 Python")
    packages: StringProperty(name="本地依赖目录", subtype="DIR_PATH",
                             description="使用单独存放的 MediaPipe 和 OpenCV 时填写")
    model: StringProperty(name="姿态模型", subtype="FILE_PATH",
                          description="本地 MediaPipe Pose Landmarker .task 模型")

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.prop(self, "toolkit")
        self.layout.prop(self, "python")
        self.layout.prop(self, "packages")
        self.layout.prop(self, "model")


class WMOCAP_Settings(bpy.types.PropertyGroup):
    source: StringProperty(name="源视频", subtype="FILE_PATH")
    first: IntProperty(name="源起始帧", min=0,
                       description="镜头起始帧，源片首帧编号为 0")
    last: IntProperty(name="源结束帧", min=0,
                      description="包含结束帧。每次生成同一镜头内的一段动作")
    capture_pair: BoolProperty(
        name="同时捕捉两人", default=False,
        description="指定两个人在首帧的躯干范围，分别生成两套白模与动作",
    )
    use_subject_region: BoolProperty(
        name="指定目标人物", default=False,
        description="多人镜头中，用所选源片首帧的目标躯干范围指定待跟踪的人物",
    )
    subject_left: FloatProperty(name="左缘", description="离画面左侧的百分比",
                                default=25, min=0, max=100, subtype="PERCENTAGE", precision=1)
    subject_top: FloatProperty(name="上缘", description="离画面顶部的百分比",
                               default=20, min=0, max=100, subtype="PERCENTAGE", precision=1)
    subject_right: FloatProperty(name="右缘", description="离画面左侧的百分比",
                                 default=75, min=0, max=100, subtype="PERCENTAGE", precision=1)
    subject_bottom: FloatProperty(name="下缘", description="离画面顶部的百分比",
                                  default=90, min=0, max=100, subtype="PERCENTAGE", precision=1)
    second_left: FloatProperty(name="左缘", default=55, min=0, max=100, subtype="PERCENTAGE", precision=1)
    second_top: FloatProperty(name="上缘", default=20, min=0, max=100, subtype="PERCENTAGE", precision=1)
    second_right: FloatProperty(name="右缘", default=95, min=0, max=100, subtype="PERCENTAGE", precision=1)
    second_bottom: FloatProperty(name="下缘", default=90, min=0, max=100, subtype="PERCENTAGE", precision=1)
    identity: StringProperty(name="角色 ID", default="P01",
                             description="动作名称中的角色标识。生成预览角色时须使用未占用的标识")
    role: EnumProperty(name="角色配色", items=_ROLES, default="primary")
    second_identity: StringProperty(name="角色 2 ID", default="P02")
    second_role: EnumProperty(name="角色 2 配色", items=_ROLES, default="secondary")
    body_view: EnumProperty(
        name="躯干朝向",
        items=(
            ("auto", "自动跟踪", "根据关键点估计正面、侧面、背面及转身"),
            ("rear", "持续背向", "用于整段背对镜头的镜头，允许轻微侧摆"),
        ),
        default="auto",
    )
    rear_yaw_keys: StringProperty(
        name="背向角度锚点",
        description="可选。填写源帧:角度，例如 24:20,72:0。0° 为背身，正角露出人物右侧",
    )
    carry_hand: EnumProperty(
        name="低位持物手",
        description="已知人物低垂持物时选择实际持物手",
        items=(("none", "无约束", ""), ("L", "左手", ""), ("R", "右手", "")),
        default="none",
    )
    support_hand: EnumProperty(
        name="支撑手",
        description="手扶固定物体时，稳定指定源帧区间内的腕部位置",
        items=(("none", "关闭", ""), ("L", "左手", ""), ("R", "右手", "")),
        default="none",
    )
    support_first: IntProperty(
        name="接触起始帧", min=0,
        description="手开始扶住固定物体的源帧，源片首帧编号为 0",
    )
    support_last: IntProperty(
        name="接触结束帧", min=0,
        description="手保持接触的最后一个源帧，包含该帧",
    )
    forward_bent_hand: EnumProperty(
        name="贴身前屈手",
        description="已知前臂折向身体前方并贴身时选择对应手臂",
        items=(("none", "跟随视频", ""), ("L", "左手", ""), ("R", "右手", "")),
        default="none",
    )
    arm_swing: FloatProperty(
        name="手臂摆幅",
        description="保留多少手臂相对躯干的运动幅度",
        default=1.0, min=0.0, max=1.0, subtype="FACTOR",
    )
    gait_mode: EnumProperty(
        name="画外下肢",
        items=(
            ("auto", "自动推断", "发现有规律的步行节奏时，补全缺少观测的腿部动作"),
            ("walk", "步行", "已知人物在走路，缺少足够节奏证据时采用参考步频"),
            ("observed", "仅按观测", "画外腿部沿用原有观测及缺口处理"),
        ),
        default="auto",
    )
    existing_rig: BoolProperty(name="为所选白模生成独立动作", default=False,
                               description="在动作编辑器里为已校准的白模选择生成的动作")
    sync_timeline: BoolProperty(name="同步源片时间轴", default=False,
                                description="用于新工程。将场景帧率和起止帧设为整条源视频的参数")


class WMOCAP_OT_capture(bpy.types.Operator):
    bl_idname = "white_model_mocap.capture"
    bl_label = "捕捉并生成白模动作"
    bl_description = "在本机从参考视频捕捉动作，生成可编辑的 Blender Action"
    bl_options = {"UNDO"}

    _job = None
    _timer = None
    _window_manager = None

    @classmethod
    def poll(cls, context):
        if context.mode != "OBJECT":
            cls.poll_message_set("请切换到物体模式")
            return False
        return _active is None and context.window is not None and context.scene is not None

    def execute(self, context):
        global _active, _status
        _status = None
        settings = context.scene.white_model_mocap
        prefs = _preferences(context)
        try:
            if prefs is None or not (prefs.toolkit.strip() and prefs.model.strip()):
                raise ValueError("请在插件偏好设置中选择白模工具包和姿态模型")
            if not settings.source.strip() or not settings.identity.strip():
                raise ValueError("请选择源视频并填写角色 ID")
            python = prefs.python.strip() or runner.default_python()
            if not python:
                raise ValueError("请在插件偏好设置中选择 Python 可执行文件")

            self._scene = context.scene
            self._toolkit = _path(prefs.toolkit)
            self._identity = settings.identity.strip()
            self._role = settings.role
            self._capture_pair = settings.capture_pair
            self._second_identity = settings.second_identity.strip()
            self._second_role = settings.second_role
            self._body_view = "auto" if self._capture_pair else settings.body_view
            self._rear_yaw_keys = settings.rear_yaw_keys if self._body_view == "rear" else ""
            self._carry_hand = settings.carry_hand
            self._support_hand = "none" if self._capture_pair else settings.support_hand
            support_frames = (settings.support_first, settings.support_last) if self._support_hand != "none" else None
            self._support_frames = motion.validate_support_interval(
                self._support_hand, support_frames, settings.first, settings.last,
            )
            self._forward_bent_hand = settings.forward_bent_hand
            self._arm_swing = settings.arm_swing
            self._gait_mode = settings.gait_mode
            use_existing = settings.existing_rig and not self._capture_pair
            self._target_rig = context.active_object if use_existing else None
            if use_existing and (self._target_rig is None or self._target_rig.type != "ARMATURE"):
                raise ValueError("请先选中已校准的白模骨架")
            self._sync_timeline = settings.sync_timeline and not use_existing
            subject_region = None
            subject_regions = None
            if self._capture_pair:
                if not self._second_identity or self._second_identity == self._identity:
                    raise ValueError("请填写两个不同的角色 ID")
                if self._second_role == self._role:
                    raise ValueError("请为两个角色分别选择不同配色")
                subject_regions = [_region(settings, prefix) for prefix in ("subject", "second")]
            elif settings.use_subject_region:
                subject_region = _region(settings, "subject")
            self._job = runner.start(
                toolkit=self._toolkit, source=_path(settings.source), python=_path(python),
                packages=_path(prefs.packages) if prefs.packages.strip() else None,
                model=_path(prefs.model), first=settings.first, last=settings.last, body_view=self._body_view,
                gait_mode=self._gait_mode, subject_region=subject_region, subject_regions=subject_regions,
            )
            self._window_manager = context.window_manager
            self._timer = self._window_manager.event_timer_add(0.25, window=context.window)
            self._window_manager.modal_handler_add(self)
        except (OSError, RuntimeError, ValueError) as exc:
            self._close()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _active = self
        _redraw(context.window_manager)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _status
        if self._job is None:
            return {"CANCELLED"}
        if event.type == "ESC" and event.value == "PRESS":
            self.cancel(context)
            return {"CANCELLED"}
        if event.type != "TIMER" or self._job.process.poll() is None:
            return {"PASS_THROUGH"}

        try:
            if context.scene != self._scene or context.mode != "OBJECT":
                raise ValueError("工作场景或编辑模式已切换，请回到目标场景的物体模式重新捕捉")
            capture = self._job.result()
            if self._capture_pair:
                rigs, reports = pair.bake_pair(
                    context, toolkit=self._toolkit, capture=capture, sync_timeline=self._sync_timeline,
                    identities=(self._identity, self._second_identity), roles=(self._role, self._second_role),
                    gait_mode=self._gait_mode,
                )
                rig = rigs[0]
            else:
                rig, report = motion.bake(
                    context, toolkit=self._toolkit, capture=capture, identity=self._identity,
                    role=self._role, sync_timeline=self._sync_timeline, body_view=self._body_view,
                    carry_hand=self._carry_hand, gait_mode=self._gait_mode, target_rig=self._target_rig,
                    rear_yaw_keys=self._rear_yaw_keys, arm_swing=self._arm_swing,
                    forward_bent_hand=self._forward_bent_hand,
                    support_hand=self._support_hand, support_frames=self._support_frames,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            _status = ("生成动作失败，错误见 Blender 提示", "ERROR")
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception:
            traceback.print_exc()
            _status = ("生成动作失败，详情见系统控制台", "ERROR")
            self.report({"ERROR"}, _status[0])
            return {"CANCELLED"}
        finally:
            self._close()

        if rig is not None and context.view_layer.objects.get(rig.name) == rig and rig.visible_get(view_layer=context.view_layer):
            for obj in list(context.view_layer.objects.selected):
                obj.select_set(False)
            rig.select_set(True)
            context.view_layer.objects.active = rig
        if self._capture_pair:
            details = " · ".join(
                f"{item['identity']} 缺失 {(item.get('subject_tracking') or {}).get('lost_frames', 0)} 帧"
                for item in reports
            )
            _status = ("已生成两套动作", "INFO")
            self.report({"INFO"}, f"已生成两套独立动作。{details}。待复核帧段见各角色诊断文本。")
            _redraw(context.window_manager)
            return {"FINISHED"}
        review_count = len(report["review_ranges"])
        _status = (f"已生成动作，{review_count} 段待复核", "INFO" if review_count else "CHECKMARK")
        lost_frames = (report.get("subject_tracking") or {}).get("lost_frames", 0)
        if lost_frames:
            _status = (f"已生成动作，目标丢失 {lost_frames} 帧", "INFO")
        lost_note = f"目标丢失 {lost_frames} 帧。" if lost_frames else ""
        self.report({"INFO"}, f"已生成 {report['action']}，可在动作编辑器选择。{lost_note}文本《{report['report_text']}》记录待复核帧段。")
        _redraw(context.window_manager)
        return {"FINISHED"}

    def _close(self):
        global _active
        if _active is self:
            _active = None
        job, self._job = self._job, None
        manager, self._window_manager = self._window_manager, None
        timer, self._timer = self._timer, None
        try:
            if job is not None:
                job.close()
        finally:
            if manager is not None and timer is not None:
                manager.event_timer_remove(timer)
                _redraw(manager)

    def cancel(self, context):
        global _status
        _status = ("已取消识别", "INFO")
        self._close()


class WMOCAP_OT_cancel(bpy.types.Operator):
    bl_idname = "white_model_mocap.cancel"
    bl_label = "取消识别"
    bl_description = "停止正在运行的本地识别"

    @classmethod
    def poll(cls, context):
        return _active is not None

    def execute(self, context):
        _active.cancel(context)
        return {"FINISHED"}


def _region(settings, prefix):
    return [getattr(settings, f"{prefix}_{side}") / 100 for side in ("left", "top", "right", "bottom")]


def _draw_region(layout, settings, prefix, label):
    region = layout.box()
    region.use_property_split = False
    region.label(text=label)
    horizontal = region.row(align=True)
    vertical = region.row(align=True)
    for side in ("left", "right"):
        horizontal.prop(settings, f"{prefix}_{side}")
    for side in ("top", "bottom"):
        vertical.prop(settings, f"{prefix}_{side}")


class WMOCAP_PT_capture(bpy.types.Panel):
    bl_label = "单目视频动捕"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "白模动捕"

    def draw(self, context):
        settings = context.scene.white_model_mocap
        prefs = _preferences(context)
        layout = self.layout
        inputs = layout.column(align=True)
        inputs.enabled = _active is None
        inputs.use_property_split = True
        inputs.prop(settings, "source")
        inputs.prop(settings, "first")
        inputs.prop(settings, "last")
        inputs.prop(settings, "capture_pair")
        if settings.capture_pair:
            inputs.prop(settings, "identity")
            inputs.prop(settings, "role")
            _draw_region(inputs, settings, "subject", "角色 1 · 首帧躯干范围")
            inputs.prop(settings, "second_identity")
            inputs.prop(settings, "second_role")
            _draw_region(inputs, settings, "second", "角色 2 · 首帧躯干范围")
        else:
            inputs.prop(settings, "use_subject_region")
            if settings.use_subject_region:
                _draw_region(inputs, settings, "subject", "首帧的目标躯干范围")
            inputs.prop(settings, "existing_rig")
            inputs.prop(settings, "identity")
            if not settings.existing_rig:
                inputs.prop(settings, "role")
            inputs.prop(settings, "body_view")
            if settings.body_view == "rear":
                inputs.prop(settings, "rear_yaw_keys")
            inputs.prop(settings, "carry_hand")
            inputs.prop(settings, "support_hand")
            if settings.support_hand != "none":
                inputs.prop(settings, "support_first")
                inputs.prop(settings, "support_last")
            inputs.prop(settings, "forward_bent_hand")
            inputs.prop(settings, "arm_swing")
        inputs.prop(settings, "gait_mode")
        if settings.capture_pair or not settings.existing_rig:
            inputs.prop(settings, "sync_timeline")
        if prefs is None or not (prefs.toolkit.strip() and prefs.model.strip()):
            layout.label(text="请先配置工具包和姿态模型", icon="INFO")
        if _active is not None:
            layout.label(text="正在本地识别动作", icon="TIME")
            layout.operator(WMOCAP_OT_cancel.bl_idname, icon="CANCEL")
        else:
            layout.operator(WMOCAP_OT_capture.bl_idname, icon="ARMATURE_DATA")
            if _status:
                layout.label(text=_status[0], icon=_status[1])


@persistent
def _close_on_load(_unused):
    global _status
    if _active is not None:
        _active._close()
    _status = None


_CLASSES = (WMOCAP_Preferences, WMOCAP_Settings, WMOCAP_OT_capture, WMOCAP_OT_cancel, WMOCAP_PT_capture)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.white_model_mocap = PointerProperty(type=WMOCAP_Settings)
    bpy.app.handlers.load_pre.append(_close_on_load)
    bpy.app.handlers.undo_pre.append(_close_on_load)
    bpy.app.handlers.redo_pre.append(_close_on_load)


def unregister():
    _close_on_load(None)
    bpy.app.handlers.load_pre.remove(_close_on_load)
    bpy.app.handlers.undo_pre.remove(_close_on_load)
    bpy.app.handlers.redo_pre.remove(_close_on_load)
    del bpy.types.Scene.white_model_mocap
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
