# Native execution prerequisites

The package includes the archived Windows x64 `.dll` and Linux x86-64 `.so` backends at both loader defaults:

* `simulation/native/` and `simulation/benchmark_three_seed/native/`: `original_backend` plus `build_manifest.json`.
* `simulation/outputs/historical_reproduction_20261002/`: `historical_backend_05405`, its manifest and `geometry_source.py`.

The historical chain has lengths 0.7, 0.5 and 0.5405 m. Its four Python camera-geometry functions were extracted with unchanged abstract syntax trees from the original geometry module. KDL/collision source files and the supplied binaries retain their recorded hashes. Native prerequisite paths in the manifests are relative to the manifest file, so the repository can be moved or cloned to a different directory. `NC_HISTORY` is an optional override.

Windows uses the included `runtime/libwinpthread-1.dll` and the operating system's Universal C Runtime. Linux binaries were built on Ubuntu 24.04.1 x86-64 with GCC 13.3.0; use the native loader error to identify a missing system runtime on another distribution, or rebuild locally. Linux binaries have been recovered from the original build archive and verified against its manifests; this packaging update is tested on Windows, not a new Linux execution run.

From the repository root, check loading and basic geometry without connecting robot hardware:

```sh
python simulation/native/check_native.py
```

To rebuild, install a C++ compiler and OpenCV development headers/libraries. KDL 1.5.1, Eigen 3.3.7 and collision sources are included in `source/`; their licences and original notices are retained.

On Windows, use the MSYS2 UCRT64 toolchain, with `MSYS2_UCRT64` pointing to its UCRT64 directory (default `C:/msys64/ucrt64`). On Linux, `g++` and `pkg-config --cflags opencv4` must be available. The builder performs no installation or downloads:

```sh
python simulation/native/build_native.py
python simulation/native/check_native.py
```

The builder verifies the source manifest, builds both libraries, installs them at all three default directories and updates each platform's binary hash. Existing manifests retain the other platform's binary record. Rebuilding creates local file changes; refresh the package manifest only if intentionally preparing a new package. A successful native smoke check is not a complete policy-training run or a claim of bit-identical Windows/Linux trajectories.

The Windows packaging checks rebuilt both libraries from the included sources and matched FK, IK and collision outputs. A 360-episode check (the first scene, all 40 starts and all nine supplied policies) matched the saved success classifications and decision counts. The largest final-score difference was 0.0006833 score units; the packaged and original Windows backends produced identical results for that case. A separate 16-step PPO smoke run completed two optimizer updates. Model weights and supplied numerical records were not changed.
