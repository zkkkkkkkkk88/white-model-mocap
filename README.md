# 白模动捕 · White Model MoCap

Blender 插件。从单机位视频提取一人或两人的身体动作，将结果烘焙成可编辑的白模骨骼 Action。视频和姿态识别在本机处理。

A Blender add-on for local, single-camera body motion capture. It turns video into editable Blender Actions on a compatible mannequin rig, with separate tracks for up to two people. The interface is in Chinese. Running this release requires an existing compatible White Model toolkit.

## 示例 · Demos

打开[同步对照播放器](https://zkkkkkkkkk88.github.io/white-model-mocap/)，左边是参考视频，右边是白模。

| 片段 | 白模输出 | 参考片 | 版本 |
| --- | --- | --- | --- |
| 双人舞蹈，5 秒 | [查看白模](docs/duo-mocap.mp4) | [RGB MoCap · Character retargeting](https://github.com/toyxyz/toyxyz_RGBMocap/blob/237e1f3b51d4d2add4f3654f49259e0afc70522e/docs/samples/character-retargeting.mp4) 的 2–7 秒 | v0.18 测试，局部重检、骨盆位置和两人交错时的纵深经过调整 |
| 单人舞蹈，5 秒 | [查看白模](docs/solo-mocap.mp4) | [RGB MoCap · Motion capture](https://github.com/toyxyz/toyxyz_RGBMocap/blob/237e1f3b51d4d2add4f3654f49259e0afc70522e/docs/samples/motion-capture.mp4) 的 6–11 秒 | v0.13 测试，约 3.1 秒的右膝朝向有待复核 |

下载上游参考片后，可用[对照命令](examples/README.md)生成同步的左右对比。上游真人素材的再发布许可尚未确认，本仓库中的视频为白模画面。

## 安装

1. 从 [v0.18.0 Release](https://github.com/zkkkkkkkkk88/white-model-mocap/releases/tag/v0.18.0) 下载 `white_model_mocap.zip`。运行需要现有的兼容白模复刻工具包，包含 `scripts/`、已校准的 `assets/rig_library/` 和其清单要求的原始模板。此次公开发布提供插件源码、安装 ZIP 和两条视频示例。
2. 使用 Blender **5.1.2**。现有白模骨架库校验此版本及工具包内相关文件的指纹。将 [FFmpeg 中的 `ffprobe`](https://ffmpeg.org/download.html) 加入 `PATH`，在启动 Blender 前运行 `ffprobe -version` 确认可用。
3. 用 Python 3.13 建立虚拟环境，安装 MediaPipe。MediaPipe 会安装 OpenCV 和 NumPy。模型在 [Google Pose Landmarker 官方页面](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#models)选择 **Pose Landmarker (Full)**，保存为 `pose_landmarker_full.task`。

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install mediapipe==0.10.33
   ```

4. 在 Blender 打开 **编辑 → 偏好设置 → 插件 → 从磁盘安装**，选 `white_model_mocap.zip` 并启用「白模动捕」。在插件偏好设置中填写「白模工具包」（现有工具包的根目录）、「Python 可执行文件」（虚拟环境中的 `python.exe`）和「姿态模型」。通过虚拟环境安装依赖时，「本地依赖目录」留空。
5. 在 3D 视图按 `N`，打开「白模动捕」，选择源视频和起止帧。首帧编号为 0，起止帧均包含。确认后点击捕捉，在 Blender 时间轴和动作编辑器中检查结果。

## 捕捉两个人

选择一帧能看到两人躯干的画面作为片段起点，启用「同时捕捉两人」。为两人分别设置 ID、不同配色，以及起始帧里的躯干框。框的边界按左、上、右、下填写，画面左上为 0%，右下为 100%。捕捉结束后，两个角色可以在同一时间轴上分别编辑。

单人模式还支持锁定指定人物、后视方向关键帧、低位持物和固定支撑手。长时间遮挡、两个人交叠、画外腿部动作以及手和物体的精确接触需要结合源片复核。当前输出为身体骨骼，面部表情和手指关节超出捕捉范围。

## 后台运行

从本仓库的根目录运行。将工具包、解释器、权重、视频和输出位置替换为本机路径，输出 `.blend` 使用新文件名。

```powershell
blender --background --factory-startup --disable-autoexec --python-exit-code 1 `
  --python '.\blender_addons\white_model_mocap\cli.py' -- `
  --toolkit 'C:\path\to\white-model-toolkit' --source '.\input.mp4' `
  --first 0 --last 149 --identity P01 --role primary `
  --python '.\.venv\Scripts\python.exe' --model '.\pose_landmarker_full.task' `
  --sync-timeline --output '.\capture.blend'
```

双人模式补充 `--subject-region '0.10,0.20,0.40,0.70' --second-identity P02 --second-role secondary --second-subject-region '0.55,0.20,0.85,0.70'`。CLI 的框坐标使用 0 至 1。

## 许可

插件源码采用 [GPL-3.0-or-later](LICENSE)。MediaPipe、模型权重和 FFmpeg 从各自项目获取，适用各自许可。视频出处见示例部分。
