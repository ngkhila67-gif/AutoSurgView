# AutoSurgView

[English](README.md) | [简体中文](README_CN.md)

AutoSurgView 是面向开放手术的遮挡感知机器人相机。术者指定需要观察的区域；当手术团队成员遮挡相机时，机器人自主移动相机，使该区域保持在视野中，从而实现术野录制，并把画面实时传送给远程专家。

机器人在三维空间中评估指定目标的遮挡：从任务相机指向目标的视野棱锥把深度相机点云分成棱锥内、外两部分，再由棱锥内的点计算遮挡分数和规避方向。多模态强化学习策略（MM-PPO）融合这两组点和紧凑的任务状态，输出相机在平面内的位移。本仓库包含机器人软件、仿真与评估代码以及训练好的策略，配套一篇正在撰写的论文；引用信息将在论文发表后补充。


> **科研软件与安全声明：**本仓库包含物理机器人控制代码，不属于医疗器械，不得直接用于临床。启用执行器前，必须先在仿真环境和隔离测试平台中验证运动范围、急停机制、标定结果与碰撞检测功能。

## 获取仓库

本仓库为私有仓库。使用已获得访问权限的 GitHub 账号登录后，可克隆仓库：

```sh
git clone https://github.com/ngkhila67-gif/AutoSurgView.git
cd AutoSurgView
```

也可打开[仓库页面](https://github.com/ngkhila67-gif/AutoSurgView)，选择 **Code > Download ZIP**，解压后进入仓库目录。模型和记录范围见 `PACKAGE CONTENTS.txt`。

本仓库基准分析与复现使用训练种子 **29、47、83**，三种输入配置共九个策略。当前分析规范及所需输入见 `simulation/benchmark_three_seed/results/analysis_protocol.json`。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `robot-controller/` | Windows C++ 控制端：深度感知、视野棱锥遮挡模型、MM-PPO 推理、机械臂与云台控制。基于 2026 年 1 月 23 日控制器整理，保留 MM-PPO 恢复和手动接管。 |
| `operator-console/` | Windows WPF 操作界面。 |
| `simulation/` | Python/C++ 仿真与评估：环境、运动学后端、MM-PPO 网络、三种子分析、控制屏障函数（CBF）安全滤波器、遮挡分数与图像遮挡率的标定。见 [`simulation/README.md`](simulation/README.md)。 |
| `models/` | 训练好的策略：9 个报告的基准策略（3 种输入配置 × 3 个随机种子）及机器人上部署的策略。见 [`models/README.md`](models/README.md)。 |
| `docs/` | 依赖说明及不随仓库分发的组件。 |

## 机器人控制端

1. 安装带 **Desktop development with C++** 的 Visual Studio 2022、CMake，以及 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) 中列出的依赖。
2. 按说明把不可再分发的头文件和库放到 `robot-controller/vendor/`。
3. 将 `robot-controller/config/parameters.example.txt` 复制为 `parameters.txt`，并逐项核对参数。
4. 在仓库根目录构建：

```powershell
cmake -S robot-controller -B build/robot-controller
cmake --build build/robot-controller --config Release
```

控制端加载 `robot-controller/models/ppo_AnzhenRobot_20.onnx`（输入：棱锥内点 256 × 3、棱锥外点 512 × 3、任务状态 6 维）。可用 `AUTOSURGVIEW_CONFIG`、`AUTOSURGVIEW_MODEL`、`AUTOSURGVIEW_LOG_DIR` 修改默认路径；遮挡规避事件记录在 `<日志目录>/RunLogs`。部署版持续监测遮挡分数，分数达到 20 时自动开始恢复，低于 20 时停止。

## 操作界面

安装带 **.NET desktop development** 的 Visual Studio 2022 和 .NET Framework 4.8 开发包：

```powershell
dotnet restore operator-console/AutoSurgView.OperatorConsole.csproj
dotnet build operator-console/AutoSurgView.OperatorConsole.csproj -c Release
```

相机功能还需要把获得授权的海康威视 SDK 运行库放到 `operator-console/native/`。

部分控制端和操作界面功能保留了原测试平台的固定路径，包括图片、视频输出及可选资源路径。运行硬件软件前，请根据实际部署环境检查和配置这些路径，详见 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md)。

## 仿真与评估

环境配置、数据目录、C++ 后端编译，以及复现三种子分析的步骤，见 [`simulation/README.md`](simulation/README.md)。

## 数据

本仓库不包含手术室测试平台的实测点云（1,990 个场景及目标坐标）。[原 Zenodo 记录](https://doi.org/10.5281/zenodo.23456180)中的附件目前为受限访问。重新训练时，请使用已获授权的本地点云副本，并按 [`simulation/README.md`](simulation/README.md) 设置 `NC_POINTDATA`。对本仓库内数值记录的重新分析无需下载这些点云。

三种子逐回合仿真记录和分析结果位于 `simulation/benchmark_three_seed/results/`；各图数值数据已包含在仓库根目录的 [Source Data.xlsx](Source%20Data.xlsx) 中，该文件为正式投稿使用的 19 工作表版本。本文报告的输入模态分析使用种子 29、47、83，每种输入配置包含 3 个策略，共 9 个策略；检查点由验证集选择。仓库提供当前分析规范、模型训练清单及数值记录。患者数据、手术录像、账号凭据、设备序列号和厂商 SDK 二进制文件均不随本仓库分发。

## 第三方软件

Orocos KDL（LGPL-2.1）、Eigen（MPL-2.0）及 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) 中的其他依赖需单独安装，并保留各自的许可证。

## 引用

论文发表后补充。

## 许可证

本仓库中的代码和训练好的策略采用 MIT 许可证；第三方和厂商组件保留各自的许可证。
