# 功能介绍

基于深度学习的方法识别图像中赛道的中点并发布消息，使用模型为resnet18

# 使用方法

## 准备工作

具备真实的机器人或机器人仿真模块，包含运动底盘、相机及RDK套件，并且能够正常运行。

## 安装功能包

**1.安装功能包**

启动机器人后，通过终端SSH或者VNC连接机器人，点击本页面右上方的“一键部署”按钮，复制如下命令在RDK的系统上运行，完成相关Node的安装。

```bash
sudo apt update
sudo apt install -y tros-racing-track-detection-resnet
```

**2.运行巡线感知功能**

```shell
source /opt/tros/local_setup.bash

#仿真（使用仿真模型）
ros2 launch racing_track_detection_resnet racing_track_detection_resnet_simulation.launch.py

# 实际场景（使用实际场景中的模型）
ros2 launch racing_track_detection_resnet racing_track_detection_resnet.launch.py
```


# 原理简介

地平线RDK通过摄像头获取小车前方环境数据，图像数据通过训练好的CNN模型进行推理得到引导线的坐标值并发布。

# 接口说明

## 话题

### Pub话题

| 名称                          | 消息类型                                                     | 说明                                                   |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| /racing_track_center_detection                      | geometry_msgs/msg/PointStamped                | 发布赛道中点的图像坐标                 |

### Sub话题
| 名称                          | 消息类型                                                     | 说明                                                   |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| /hbmem_img                     | hbm_img_msgs/msg/HbmMsg1080P                                    | 接收相机发布的图片消息（640x480）                   |

## 参数

| 参数名                | 类型        | 说明   |
| --------------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| model_path       | string | 推理使用的模型文件，请根据实际模型路径配置，默认值为/opt/nodehub_model/race_detection/race_track_detection_simulation.bin |
| sub_img_topic       | string |  接收的图片话题名称，请根据实际接收到的话题名称配置，默认值为/hbmem_img |

# 注意
该功能包提供gazebo仿真环境中可使用的模型以及特定的实际场景中可使用的模型，若自行采集数据集进行训练，请注意替换。