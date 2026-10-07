# RoboMaster Armor Detection

基于大恒工业相机、OpenCV 与 YOLOv8-Pose 的 RoboMaster 装甲板实时识别项目。

## 功能

- 大恒 MER2 系列工业相机实时采图
- YOLOv8n-Pose 装甲板检测
- 输出装甲板 Bounding Box
- 输出四个灯条关键点
- 红/蓝装甲颜色识别
- 数字 1~5 分类
- OpenCV 传统视觉 fallback
- 实时 FPS 显示
- spdlog 调试信息输出

## 系统架构

```text
Galaxy Camera
      |
      v
  cv::Mat frame
      |
      v
YOLOv8-Pose
      |
      +---- detected ----> bbox + 4 keypoints
      |
      +---- missed ------> OpenCV traditional fallback
                              |
                              v
                       bbox + 4 keypoints
                              |
                              v
                    Perspective Transform
                              |
                +-------------+-------------+
                |                           |
                v                           v
          Color Recognition          Number Classifier
           RED / BLUE                   1 ~ 5
                |                           |
                +-------------+-------------+
                              |
                              v
                     Draw Results + FPS
[200~模型
Armor Pose
models/armor_pose.onnx
Input:
1 x 3 x 640 x 640

Output:
1 x 17 x 8400

每个候选目标包含：
cx cy w h
armor_confidence
kp0_x kp0_y kp0_conf
kp1_x kp1_y kp1_conf
kp2_x kp2_y kp2_conf
kp3_x kp3_y kp3_conf

关键点顺序：
kp0 = left-top
kp1 = left-bottom
kp2 = right-bottom
kp3 = right-top

Number Classifier
models/number_classifier.onnx
Input:
1 x 3 x 64 x 64

Output:
1 x 5

对应数字类别：
1 2 3 4 5

Hybrid Detection
正常情况下使用 YOLO Pose 检测装甲板。
如果 YOLO 当前帧完全未检测到装甲板，则启用传统 OpenCV 方法作为 fallback：
Color threshold
-> Morphology
-> findContours
-> minAreaRect
-> Light filtering
-> Light pairing
-> Armor

随后两种检测方式共用：
4 keypoints
-> perspective transform
-> color recognition
-> number classification~
