# AutoSurgView

[English](README.md) | [简体中文](README_CN.md)

AutoSurgView is an occlusion-aware robotic camera for open surgery. A surgeon designates the region to be observed, and the robot keeps that region in view when members of the surgical team block the camera, so that the operative field can be recorded and streamed to remote experts.

The robot evaluates occlusion of the designated target in three dimensions: a viewing pyramid from the task camera to the target splits the depth-camera point cloud into points inside and outside the pyramid, and an occlusion score and an avoidance direction are computed from the inside points. A multimodal reinforcement-learning policy (MM-PPO) fuses both point sets with a compact task state and outputs a planar displacement of the camera. This repository contains the robot software, the simulation and evaluation code, and the trained policies. It accompanies a manuscript under review; citation information will be added on publication.


> **Research software and safety notice:** this repository controls physical robotic hardware. It is not a medical device and must not be used clinically. Validate limits, emergency-stop behaviour, calibration and collision handling in simulation and on an isolated test rig before enabling actuators.

## Get the repository

This repository is public. Clone it without a GitHub sign-in:

```sh
git clone https://github.com/ngkhila67-gif/AutoSurgView.git
cd AutoSurgView
```

Alternatively, open [the repository](https://github.com/ngkhila67-gif/AutoSurgView), select **Code > Download ZIP**, extract it, and work from the extracted repository directory. See `PACKAGE CONTENTS.txt` for the included models and records.

The current benchmark and reproduction use training seeds **29, 47 and 83**, with three input configurations and nine policies. The analysis specification and required inputs are described in `simulation/benchmark_three_seed/results/analysis_protocol.json`.

## Repository layout

| Folder | Content |
|---|---|
| `robot-controller/` | Windows C++ controller: depth sensing, viewing-pyramid occlusion model, MM-PPO inference, arm/PTU control. Adapted from the controller of 23 January 2026 for MM-PPO recovery and manual takeover; packaging and portability changes are documented. |
| `operator-console/` | Windows WPF operator interface. |
| `simulation/` | Python/C++ simulation and evaluation: environment, kinematics backend, MM-PPO network, three-seed analysis, control-barrier-function (CBF) safety filter, and the occlusion-score/occlusion-rate calibration. See [`simulation/README.md`](simulation/README.md). |
| `models/` | Trained policies: the 9 reported benchmark policies (three input configurations × three seeds) and the policy deployed on the robot. See [`models/README.md`](models/README.md). |
| `docs/` | Dependencies and components that are not redistributed. |

## What the code implements

| Component | Where |
|---|---|
| Viewing pyramid, inside/outside partition, occlusion score `V` and guidance vector | `robot-controller/occlusion_avoidance.cpp`; `simulation/ncsim/geometry.py` |
| MM-PPO network (two point-wise encoders with attention pooling, state encoder, actor/critic) | `simulation/ncsim/policies.py` (`AblationAttentionPointNet`) |
| Reward, termination and the 40-decision episode | `simulation/ncsim/historical_backend.py`, `simulation/benchmark_three_seed/modality3_study.py` |
| Three-seed analysis (validation-based checkpoint selection, 398 scenes × 40 starts) | `simulation/benchmark_three_seed/` |
| CBF safety filter with human and equipment envelopes | `simulation/ncsim/cbf_*.py`, `simulation/ncsim/human_*.py` |
| Relation between occlusion score and image occlusion rate | `simulation/occlusion_calibration.py` |
| Deployed recovery trigger (continuous monitoring; recovery starts at a score of 20 and stops below 20) | `robot-controller/occlusion_avoidance.cpp` |

## Robot controller

1. Install Visual Studio 2022 with **Desktop development with C++**, CMake, and the dependencies in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
2. Obtain the 13 project-owned planning headers listed in `docs/DEPENDENCIES.md` from the corresponding author, Yueri Cai (caiyueri@buaa.edu.cn), and place them under `robot-controller/vendor/include/`. They are required to compile the hardware controller. Install the separate vendor headers and libraries as documented.
3. Copy `robot-controller/config/parameters.example.txt` to `parameters.txt` and validate every setting.
4. Build from the repository root:

```powershell
cmake -S robot-controller -B build/robot-controller
cmake --build build/robot-controller --config Release
```

The controller loads `robot-controller/models/ppo_AnzhenRobot_20.onnx` (inputs: inside points 256 × 3, outside points 512 × 3, task state 6). Set `AUTOSURGVIEW_CONFIG` and `AUTOSURGVIEW_MODEL` to override the configuration and model paths.

## Operator console

Install Visual Studio 2022 with **.NET desktop development** and the .NET Framework 4.8 developer pack:

```powershell
dotnet restore operator-console/AutoSurgView.OperatorConsole.csproj
dotnet build operator-console/AutoSurgView.OperatorConsole.csproj -c Release
```

Camera functions also require licensed Hikvision native SDK binaries in `operator-console/native/`.

Some controller and operator-console functions retain paths specific to the original test rig, including image/video outputs and optional resources. Review and configure these paths for the deployment environment before running the hardware software; see [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

## Simulation and evaluation

Windows and Linux native backends, their manifests and geometry source are included. Run `python simulation/native/check_native.py` to verify the local native setup. With `NC_POINTDATA` set, `python simulation/benchmark_three_seed/replay_reported_policies.py` reruns the nine supplied policies without training; add `--scene-limit 1` for a small installation check. See [`simulation/README.md`](simulation/README.md) for the data layout, rebuilding and training the manuscript configuration.

## Manuscript results and repository files

`Source Data.xlsx` is the figure-by-figure submission workbook. The paths below are relative to the repository root; `results/` in the explanations means `simulation/benchmark_three_seed/results/`.

| Manuscript result | File and field or worksheet |
|---|---|
| Table 1, input-modality results; Supplementary Table 1 | `results/benchmark_retrained.json`: `collision_free`, `summary`, `pooled_rates`; `results/seed_level.csv` provides the nine individual policies. Image-region endpoints are in `results/analysis_v3.json`. |
| Fig. 3a, collision-free recovery | `Source Data.xlsx` → `Figure 3`, panel a; `results/seed_level.csv` → `macro_strict_success`; `results/benchmark_retrained.json` → `collision_free`. |
| Fig. 3b, terminal score | `Source Data.xlsx` → `Figure 3`, panel b; `results/seed_level.csv` → `mean_final`. |
| Fig. 3c, final image-region recovery | `Source Data.xlsx` → `Figure 3`, panel c; `results/analysis_v3.json` → `secondary_O_le_20`; original final-state measurements are in `results/final_state_O.csv.gz`. |
| Fig. 3d; Supplementary Fig. 8 | `results/source_data/Fig17_stratified_estimates.csv`; workbook sheets `Figure 3` and `Supplementary Figure 8`. |
| Supplementary Fig. 6; Supplementary Fig. 7 | `results/episodes.csv.gz`, `results/per_scene_summary.csv`, `results/seed_level.csv`, and `results/source_data/Fig16_curves_and_intervals.csv`; corresponding workbook sheets. |
| Table 1 safety-filter rows; Fig. 3f,g; Supplementary Table 2 | `Source Data.xlsx` → `Figure 3`, `Supplementary Figure 10` and `Supplementary Figure 11`. |
| Fig. 4; Fig. 5; Fig. 6 and the other quantitative supplementary figures | Their explicitly numbered worksheets in `Source Data.xlsx`. |
| Supplementary Fig. 9, validation and final-checkpoint comparisons | `results/analysis_v3.json`, `results/all_test_episodes.csv.gz`, `models/benchmark_three_seed/*/validation_history.json`, and workbook sheet `Supplementary Figure 9`. |

The CSV prefixes `Fig15`, `Fig16` and `Fig17` are retained **legacy analysis figure numbers**, not the current manuscript figure numbers. `Fig15_estimates_and_intervals.csv` includes recovery allowing self-collisions: its 8.18 pp MM-PPO–point-cloud gain belongs to that endpoint. The primary collision-free gain is 8.51 pp in `benchmark_retrained.json` → `collision_free` and Table 1. Use the endpoint and worksheet mappings above when comparing values.


## Data

The measured point clouds of the operating-room test bed (1,990 scenes with target coordinates), numerical records and figure-level source data are publicly available in the [Zenodo dataset](https://doi.org/10.5281/zenodo.23601833). The point clouds are distributed separately from this repository: extract `Point Clouds.zip` and set `NC_POINTDATA` to the absolute path of `Point Clouds/pointclouds`. Fresh simulation or training also requires the compatible native backends and runtime dependencies described in [`simulation/README.md`](simulation/README.md). The bundled numerical records can be reanalysed without downloading those point clouds.

The three-seed episode records and analysis outputs are included under `simulation/benchmark_three_seed/results/`. Figure-level numerical data are included at the repository root in [Source Data.xlsx](Source%20Data.xlsx), the 19-worksheet journal submission file. The reported input-modality analysis uses seeds 29, 47 and 83 for each of three input configurations (nine policies in total). Checkpoints are selected on the validation set. The current analysis specification, model training manifests and numerical records are included in the repository. Patient data, surgical recordings, credentials, device serial numbers and proprietary SDK binaries are not distributed with this repository.

## Third-party software

The native backends include their KDL/Eigen sources and Windows runtime notice. LIBSVM and Hikvision wrapper attribution, licence exclusions and rebuild-source locations are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Other dependencies are installed separately as described in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

## Citation

Citation information will be added on publication.

## License

MIT for project-owned code and trained policies. Third-party and vendor components retain their own licences; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The Zenodo dataset and figure source data are CC BY 4.0.

## Package integrity

`.gitattributes` disables automatic line-ending conversion so cloned file bytes match the SHA-256 manifest. Verify with `python tools/regenerate_package_manifest.py --check` (also supported for a downloaded ZIP). After intentional additions or edits in a Git checkout, regenerate with `python tools/regenerate_package_manifest.py`. The manifest excludes itself; local ignored outputs are not included.
