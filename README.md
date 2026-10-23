# AutoSurgView

[English](README.md) | [简体中文](README_CN.md)

AutoSurgView is an occlusion-aware robotic camera for open surgery. A surgeon designates the region to be observed, and the robot keeps that region in view when members of the surgical team block the camera, so that the operative field can be recorded and streamed to remote experts.

The robot evaluates occlusion of the designated target in three dimensions: a viewing pyramid from the task camera to the target splits the depth-camera point cloud into points inside and outside the pyramid, and an occlusion score and an avoidance direction are computed from the inside points. A multimodal reinforcement-learning policy (MM-PPO) fuses both point sets with a compact task state and outputs a planar displacement of the camera. This repository contains the robot software, the simulation and evaluation code, and the trained policies. It accompanies a manuscript in preparation; citation information will be added on publication.


> **Research software and safety notice:** this repository controls physical robotic hardware. It is not a medical device and must not be used clinically. Validate limits, emergency-stop behaviour, calibration and collision handling in simulation and on an isolated test rig before enabling actuators.

## Get the repository

This repository is private. Sign in with a GitHub account that has access, then clone it:

```sh
git clone https://github.com/ngkhila67-gif/AutoSurgView.git
cd AutoSurgView
```

Alternatively, open [the repository](https://github.com/ngkhila67-gif/AutoSurgView), select **Code > Download ZIP**, extract it, and work from the extracted repository directory. See `PACKAGE CONTENTS.txt` for the included models and records.

The current benchmark and reproduction use training seeds **29, 47 and 83**, with three input configurations and nine policies. The analysis specification and required inputs are described in `simulation/benchmark_three_seed/results/analysis_protocol.json`.

## Repository layout

| Folder | Content |
|---|---|
| `robot-controller/` | Windows C++ controller: depth sensing, viewing-pyramid occlusion model, MM-PPO inference, arm/PTU control. Derived from the controller of 23 January 2026, retaining MM-PPO recovery and manual takeover. |
| `operator-console/` | Windows WPF operator interface. |
| `simulation/` | Python/C++ simulation and evaluation: environment, kinematics backend, MM-PPO network, three-seed analysis, control-barrier-function (CBF) safety filter, and the occlusion-score/occlusion-rate calibration. See [`simulation/README.md`](simulation/README.md). |
| `models/` | Trained policies: the 9 reported benchmark policies (three input configurations × three seeds) and the policy deployed on the robot. See [`models/README.md`](models/README.md). |
| `docs/` | Dependencies and components that are not redistributed. |

## What the code implements

| Component | Where |
|---|---|
| Viewing pyramid, inside/outside partition, occlusion score `V` and guidance vector | `robot-controller/occlusion_avoidance.cpp`; `simulation/ncsim/geometry.py` |
| MM-PPO network (two point-wise encoders with attention pooling, state encoder, actor/critic) | `simulation/ncsim/policies.py` (`AblationAttentionPointNet`) |
| Reward, termination and the 40-decision episode | `simulation/ncsim/historical_backend.py`, `simulation/ncsim/learning_env.py` |
| Three-seed analysis (validation-based checkpoint selection, 398 scenes × 40 starts) | `simulation/benchmark_three_seed/` |
| CBF safety filter with human and equipment envelopes | `simulation/ncsim/cbf_*.py`, `simulation/ncsim/human_*.py` |
| Relation between occlusion score and image occlusion rate | `simulation/occlusion_calibration.py` |
| Deployed recovery trigger (continuous monitoring; recovery starts at a score of 20 and stops below 20) | `robot-controller/occlusion_avoidance.cpp` |

## Robot controller

1. Install Visual Studio 2022 with **Desktop development with C++**, CMake, and the dependencies in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
2. Put non-redistributable headers and libraries under `robot-controller/vendor/` as documented.
3. Copy `robot-controller/config/parameters.example.txt` to `parameters.txt` and validate every setting.
4. Build from the repository root:

```powershell
cmake -S robot-controller -B build/robot-controller
cmake --build build/robot-controller --config Release
```

The controller loads `robot-controller/models/ppo_AnzhenRobot_20.onnx` (inputs: inside points 256 × 3, outside points 512 × 3, task state 6). Set `AUTOSURGVIEW_CONFIG`, `AUTOSURGVIEW_MODEL` and `AUTOSURGVIEW_LOG_DIR` to override the default paths; occlusion-avoidance events are logged under `<log dir>/RunLogs`.

## Operator console

Install Visual Studio 2022 with **.NET desktop development** and the .NET Framework 4.8 developer pack:

```powershell
dotnet restore operator-console/AutoSurgView.OperatorConsole.csproj
dotnet build operator-console/AutoSurgView.OperatorConsole.csproj -c Release
```

Camera functions also require licensed Hikvision native SDK binaries in `operator-console/native/`.

Some controller and operator-console functions retain paths specific to the original test rig, including image/video outputs and optional resources. Review and configure these paths for the deployment environment before running the hardware software; see [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

## Simulation and evaluation

See [`simulation/README.md`](simulation/README.md) for the environment, the data layout, building the native backend and reproducing the three-seed analysis.

## Data

The measured point clouds of the operating-room test bed (1,990 scenes with target coordinates) are not included in this repository. The files at the [previous Zenodo record](https://doi.org/10.5281/zenodo.23456180) are currently restricted. Training requires an authorized local copy of the point-cloud data; set `NC_POINTDATA` to that copy as described in [`simulation/README.md`](simulation/README.md). The bundled numerical records can be reanalysed without downloading those point clouds.

The three-seed episode records and analysis outputs are included under `simulation/benchmark_three_seed/results/`. Figure-level numerical data are included at the repository root in [Source Data.xlsx](Source%20Data.xlsx), the 19-worksheet journal submission file. The reported input-modality analysis uses seeds 29, 47 and 83 for each of three input configurations (nine policies in total). Checkpoints are selected on the validation set. The current analysis specification, model training manifests and numerical records are included in the repository. Patient data, surgical recordings, credentials, device serial numbers and proprietary SDK binaries are not distributed with this repository.

## Third-party software

Orocos KDL (LGPL-2.1), Eigen (MPL-2.0) and the other dependencies listed in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) are installed separately and retain their licences.

## Citation

Citation information will be added on publication.

## License

MIT for the code and trained policies in this repository. Third-party and vendor components retain their own licences.
