# AutoSurgView

[English](README.md) | [简体中文](README_CN.md)

AutoSurgView 是面向开放手术的遮挡感知机器人相机。术者指定需要观察的区域；当手术团队成员遮挡相机时，机器人自主移动相机，使该区域保持在视野中，从而实现术野录制，并把画面实时传送给远程专家。

机器人在三维空间中评估指定目标的遮挡：从任务相机指向目标的视野棱锥把深度相机点云分成棱锥内、外两部分，再由棱锥内的点计算遮挡分数和规避方向。多模态强化学习策略（MM-PPO）融合这两组点和紧凑的任务状态，输出相机在平面内的位移。本仓库包含机器人软件、仿真与评估代码以及训练好的策略，配套一篇正在审稿的论文；引用信息将在论文发表后补充。


> **科研软件与安全声明：**本仓库包含物理机器人控制代码，不属于医疗器械，不得直接用于临床。启用执行器前，必须先在仿真环境和隔离测试平台中验证运动范围、急停机制、标定结果与碰撞检测功能。

## 获取仓库

本仓库已公开，无需登录 GitHub 即可克隆：

```sh
git clone https://github.com/ngkhila67-gif/AutoSurgView.git
cd AutoSurgView
```

也可打开[仓库页面](https://github.com/ngkhila67-gif/AutoSurgView)，选择 **Code > Download ZIP**，解压后进入仓库目录。模型和记录范围见 `PACKAGE CONTENTS.txt`。

本仓库基准分析与复现使用训练种子 **29、47、83**，三种输入配置共九个策略。当前分析规范及所需输入见 `simulation/benchmark_three_seed/results/analysis_protocol.json`。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `robot-controller/` | Windows C++ 控制端：深度感知、视野棱锥遮挡模型、MM-PPO 推理、机械臂与云台控制。基于 2026 年 1 月 23 日控制器适配 MM-PPO 恢复和手动接管；打包与可移植性调整已记录。 |
| `operator-console/` | Windows WPF 操作界面。 |
| `simulation/` | Python/C++ 仿真与评估：环境、运动学后端、MM-PPO 网络、三种子分析、控制屏障函数（CBF）安全滤波器、遮挡分数与图像遮挡率的标定。见 [`simulation/README.md`](simulation/README.md)。 |
| `models/` | 训练好的策略：9 个报告的基准策略（3 种输入配置 × 3 个随机种子）及机器人上部署的策略。见 [`models/README.md`](models/README.md)。 |
| `docs/` | 依赖说明及不随仓库分发的组件。 |

## 机器人控制端

1. 安装带 **Desktop development with C++** 的 Visual Studio 2022、CMake，以及 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) 中列出的依赖。
2. 硬件控制器编译还需要 `docs/DEPENDENCIES.md` 列出的 13 个项目规划头文件，可向通讯作者 Yueri Cai（caiyueri@buaa.edu.cn）索取后放入 `robot-controller/vendor/include/`。另按说明配置厂商头文件和运行库。
3. 将 `robot-controller/config/parameters.example.txt` 复制为 `parameters.txt`，并逐项核对参数。
4. 在仓库根目录构建：

```powershell
cmake -S robot-controller -B build/robot-controller
cmake --build build/robot-controller --config Release
```

控制端加载 `robot-controller/models/ppo_AnzhenRobot_20.onnx`（输入：棱锥内点 256 × 3、棱锥外点 512 × 3、任务状态 6 维）。可用 `AUTOSURGVIEW_CONFIG`、`AUTOSURGVIEW_MODEL` 修改配置和模型路径。部署版持续监测遮挡分数，分数达到 20 时自动开始恢复，低于 20 时停止。

## 操作界面

安装带 **.NET desktop development** 的 Visual Studio 2022 和 .NET Framework 4.8 开发包：

```powershell
dotnet restore operator-console/AutoSurgView.OperatorConsole.csproj
dotnet build operator-console/AutoSurgView.OperatorConsole.csproj -c Release
```

相机功能还需要把获得授权的海康威视 SDK 运行库放到 `operator-console/native/`。

部分控制端和操作界面功能保留了原测试平台的固定路径，包括图片、视频输出及可选资源路径。运行硬件软件前，请根据实际部署环境检查和配置这些路径，详见 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md)。

## 仿真与评估

仓库已包含 Windows 与 Linux 原生后端、对应清单和几何源码。先运行 `python simulation/native/check_native.py` 检查本机原生环境。设置 `NC_POINTDATA` 后，运行 `python simulation/benchmark_three_seed/replay_reported_policies.py` 可直接重跑九个已提供策略，无需重新训练；添加 `--scene-limit 1` 可进行小规模安装检查。数据目录、源码重建和论文配置的重新训练步骤见 [`simulation/README.md`](simulation/README.md)。

## 论文结果与仓库文件对照

`Source Data.xlsx` 是按图整理的正式投稿源数据工作簿。下表中的 `results/` 指 `simulation/benchmark_three_seed/results/`。

| 论文结果 | 文件及字段或工作表 |
|---|---|
| 表 1 的输入模态结果、补充表 1 | `results/benchmark_retrained.json` 的 `collision_free`、`summary`、`pooled_rates`；九个策略的单独数值见 `results/seed_level.csv`；图像区域终点见 `results/analysis_v3.json`。 |
| 图 3a：无碰撞恢复率 | 工作簿 `Figure 3` 的 a 面板；`results/seed_level.csv` 的 `macro_strict_success`；`results/benchmark_retrained.json` 的 `collision_free`。 |
| 图 3b：终止遮挡评分 | 工作簿 `Figure 3` 的 b 面板；`results/seed_level.csv` 的 `mean_final`。 |
| 图 3c：最终图像区域恢复率 | 工作簿 `Figure 3` 的 c 面板；`results/analysis_v3.json` 的 `secondary_O_le_20`；原始终态测量见 `results/final_state_O.csv.gz`。 |
| 图 3d、补充图 8 | `results/source_data/Fig17_stratified_estimates.csv`；工作簿 `Figure 3`、`Supplementary Figure 8`。 |
| 补充图 6、7 | `results/episodes.csv.gz`、`per_scene_summary.csv`、`seed_level.csv`、`source_data/Fig16_curves_and_intervals.csv`，以及对应工作表。 |
| 表 1 的安全滤波结果、图 3f/g、补充表 2 | 工作簿 `Figure 3`、`Supplementary Figure 10`、`Supplementary Figure 11`。 |
| 图 4、5、6 及其余定量补充图 | 工作簿中按相应图号命名的工作表。 |
| 补充图 9：验证及最终检查点比较 | `results/analysis_v3.json`、`results/all_test_episodes.csv.gz`、`models/benchmark_three_seed/*/validation_history.json`，以及 `Supplementary Figure 9` 工作表。 |

CSV 文件名前缀 `Fig15`、`Fig16`、`Fig17` 是保留的旧分析图号，不是当前论文图号。`Fig15_estimates_and_intervals.csv` 中的 8.18 pp 对应“允许自碰撞的恢复率”下 MM-PPO 相对点云配置的增益；主要终点“无碰撞恢复率”的增益为 8.51 pp，见 `benchmark_retrained.json` 的 `collision_free` 和表 1。比较数值时应同时核对终点定义。


## 数据

手术室测试平台的实测点云（1,990 个场景及目标坐标）、数值记录和按图源数据已在 [Zenodo 数据记录](https://doi.org/10.5281/zenodo.23601833)中公开。点云与本仓库分开提供：解压 `Point Clouds.zip` 后，将 `NC_POINTDATA` 设为 `Point Clouds/pointclouds` 的绝对路径。重新运行仿真或训练还需要 [`simulation/README.md`](simulation/README.md) 中说明的兼容原生后端及运行依赖。对本仓库内数值记录的重新分析无需下载这些点云。

三种子逐回合仿真记录和分析结果位于 `simulation/benchmark_three_seed/results/`；各图数值数据已包含在仓库根目录的 [Source Data.xlsx](Source%20Data.xlsx) 中，该文件为正式投稿使用的 19 工作表版本。本文报告的输入模态分析使用种子 29、47、83，每种输入配置包含 3 个策略，共 9 个策略；检查点由验证集选择。仓库提供当前分析规范、模型训练清单及数值记录。患者数据、手术录像、账号凭据、设备序列号和厂商 SDK 二进制文件均不随本仓库分发。

## 第三方软件

原生后端已附 KDL/Eigen 源码及 Windows 运行库许可。LIBSVM、海康封装代码的归属与许可范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。其余依赖按 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) 单独安装。

## 引用

论文发表后补充。

## 许可证

项目自有代码和训练好的策略采用 MIT 许可证；第三方和厂商组件保留各自许可证，详见 `THIRD_PARTY_NOTICES.md`。Zenodo 数据集及按图源数据采用 CC BY 4.0。

## 文件校验

`.gitattributes` 禁用 Git 自动换行符转换，保证克隆后的文件字节与 SHA-256 清单一致。运行 `python tools/regenerate_package_manifest.py --check` 验证；有意增删或修改文件后，运行 `python tools/regenerate_package_manifest.py` 重新生成清单。清单不包含自身，已忽略的本地运行输出也不会纳入。
