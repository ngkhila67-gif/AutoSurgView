# Dependencies and components that are not redistributed

## Robot controller (Windows x64, C++17)

Open-source dependencies inferred directly from includes are OpenCV, Eigen3, Boost.Thread, PCL, Intel RealSense SDK 2, ONNX Runtime, LibTorch, AprilTag 3, Slamtec RPLIDAR SDK, Orocos KDL, `wjwwood/serial`, JsonCpp, OpenMP, and OpenGL/GLUT.

Components that are not included and must be obtained separately:

- `ECANVci.h` and its CAN adapter library/DLL.
- `yanHuaDriver.h` and its digital-I/O library/DLL.
- Hikvision SDK runtime for the operator console.

Project-owned planning headers that are not yet released:

- `rrt.h`, `rrt_star.h`, `rrt_connect.h`, `rrt_connect_star.h`, `Q_rrt_star.h`.
- `D_rrt.h`, `D_rrt_star.h`, `D_rrt_connect.h`, `D_rrt_connect_star.h`.
- `MP_rrt.h`, `MP_rrt_star.h`, `MP_rrt_connect.h`, `MP_rrt_connect_star.h`.

Expected private layout:

```text
robot-controller/vendor/
  include/
  lib/
  bin/
```

Optional paths also reference a heart-detection ONNX model, LibTorch and SVM compensation models for the visual target, robot STL meshes, UI icons/audio, and an external Record Tool. Keep those features disabled until the artefacts and redistribution rights are established.

## Operator console (.NET Framework 4.8)

NuGet dependencies are declared in the project: MaterialDesignThemes, OpenCvSharp4, MathNet.Numerics, SharpGL, and SharpGL.WPF. Put licensed Hikvision x64 runtime files in `operator-console/native/`; Git ignores them.

## Simulation and evaluation (`simulation/`)

- Python 3.13 with numpy 2.4.4, scipy 1.17.1, pandas 3.0.3, matplotlib 3.10.9, torch 2.14.1, stable-baselines3 2.9.0, gymnasium 1.3.0 and onnxruntime 1.26.0 (versions used for the manuscript; see `simulation/environment_versions.json`).
- Native backend: MSYS2 UCRT64 GCC toolchain (`MSYS2_UCRT64`), Orocos KDL and Eigen (`NC_SSBOT_LIB`), and the robot's original kinematics and self-collision sources (`robot-controller/`, overridable with `NC_ROBOT_SOURCE`).
- Measured point clouds of the operating-room test bed are provided separately in the public [Zenodo dataset](https://doi.org/10.5281/zenodo.23601833). Extract `Point Clouds.zip` and set `NC_POINTDATA` to the absolute path of `Point Clouds/pointclouds`. Bundled numerical records remain sufficient for numerical reanalysis.
- The three-seed training entry point additionally requires `benchmark_three_seed/native/original_backend.dll` and `build_manifest.json`, plus the compatible 0.5405 m backend under `NC_HISTORY`: `historical_backend_05405.dll`, `historical_backend_manifest.json`, the geometry source referenced by that manifest, and its runtime libraries. These are native execution prerequisites; downloading the point-cloud archive does not supply them. See `simulation/README.md`.
- Robot part meshes used for three-dimensional playback are not redistributed; the benchmark itself uses the original self-collision routine and does not require them.

## Paths for a new deployment

The hardware code retains paths from the original test rig. In particular, `operator-console/Page3.xaml.cs`, `operator-console/Page9.xaml.cs`, and `robot-controller/rs_camera_work.cpp` contain fixed image/video or calibration-output paths. Optional icons, audio, compensation models and recording tools also use rig-specific paths. Review these locations and configure them for the actual deployment environment before running the relevant functions. This repository upload preserves the original hardware code; it does not adapt those functions to a different computer.
