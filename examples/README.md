# 还原左右对比

两条示例都使用 toyxyz 的 RGB MoCap 项目提供的参考视频。分别下载[双人参考片](https://github.com/toyxyz/toyxyz_RGBMocap/blob/237e1f3b51d4d2add4f3654f49259e0afc70522e/docs/samples/character-retargeting.mp4)与[单人参考片](https://github.com/toyxyz/toyxyz_RGBMocap/blob/237e1f3b51d4d2add4f3654f49259e0afc70522e/docs/samples/motion-capture.mp4)，保留原文件名，与 `docs/` 里的两条白模视频放进同一个临时目录，在该目录运行以下命令。FFmpeg 会裁出参考片的真人画面，左侧显示原片，右侧显示本仓库发布的白模，两个面板共用时间轴。

双人示例截取源片第 120–419 帧，按 60 fps 对照。

```text
ffmpeg -i character-retargeting.mp4 -i duo-mocap.mp4 -filter_complex "[0:v]trim=start_frame=120:end_frame=420,setpts=PTS-STARTPTS,crop=414:742:206:58,setsar=1[left];[1:v]setsar=1[right];[left][right]hstack=shortest=1[out]" -map "[out]" -an -c:v libx264 -pix_fmt yuv420p duo-comparison.mp4
```

单人示例截取源片第 360–659 帧，从 60 fps 转至 30 fps 对照。

```text
ffmpeg -i motion-capture.mp4 -i solo-mocap.mp4 -filter_complex "[0:v]trim=start_frame=360:end_frame=660,setpts=PTS-STARTPTS,crop=414:736:222:60,scale=360:640,setsar=1,fps=30[left];[1:v]setsar=1[right];[left][right]hstack=shortest=1[out]" -map "[out]" -an -c:v libx264 -pix_fmt yuv420p solo-comparison.mp4
```
