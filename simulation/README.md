# Simulation and evaluation

The current input ablation analysis uses seeds **29, 47 and 83** for each of three input configurations (nine reported policies). Checkpoint selection uses the validation set. The analysis scope is recorded in `benchmark_three_seed/results/analysis_protocol.json`.

The reproduction entry point, `modality3_study.py`, uses 29, 47 and 83. Numerical reanalysis uses the bundled episode records, seed-level values, final-state occlusion records and endpoint summary. The current analysis specification lists these inputs and their checksums.

## Reproduce the reported numerical analysis

After cloning or downloading the public repository, run the following from its root directory in a Python environment:

```sh
python -m pip install -r simulation/requirements.txt
cd simulation/benchmark_three_seed
python reanalyse_reported_seeds.py --out outputs/reanalysis
```

This reads the bundled per-episode records and recorded final-state occlusion rates, recomputes per-seed values, scene averages, paired differences, seed-level t intervals (2 degrees of freedom), and acquisition-folder bootstrap intervals (47 folders, 2,000 resamples, random seed 20261002). The selected policies' original weights and training manifests are in `models/benchmark_three_seed/` at the repository root. Figure-level numerical data are included in the repository-root [Source Data.xlsx](../Source%20Data.xlsx) (19 worksheets); no original clinical images or recordings are included.

## Replay the supplied policies

Set `NC_POINTDATA` to the extracted `Point Clouds/pointclouds` directory, then run `python simulation/benchmark_three_seed/replay_reported_policies.py` from the repository root. It uses the nine supplied validation-selected models and all 398 × 40 fixed starts without training. `--scene-limit 1` checks the first scene; `--method mmppo --seed 29` selects one policy. Outputs go to a new directory given by `--out` and do not overwrite the supplied results.

## Train a new reproduction

Download `Point Clouds.zip` from the public [Zenodo dataset](https://doi.org/10.5281/zenodo.23601833), extract it, and set `NC_POINTDATA` to the absolute path of `Point Clouds/pointclouds`. The numerical reanalysis above uses bundled records and does not require those point clouds.

The package supplies both Windows x64 and Linux x86-64 native libraries, relative-path manifests and the historical geometry source. `benchmark_three_seed/native/` contains `original_backend` and `build_manifest.json`; `outputs/historical_reproduction_20261002/` contains the 0.5405 m backend, its manifest and `geometry_source.py`. `NC_HISTORY` is optional. Check the native prerequisites from the repository root before training:

```sh
python simulation/native/check_native.py
```

The original archived binaries are bundled without numerical changes. Linux binaries were built on Ubuntu 24.04.1 x86-64 with GCC 13.3.0. Rebuild on an incompatible platform with `python simulation/native/build_native.py`; it installs both backends in the default locations and updates their platform hashes. This requires a compiler and OpenCV development headers/libraries, but the original KDL 1.5.1, Eigen 3.3.7 and collision sources are included. Windows uses the bundled winpthreads runtime. See [`native/README.md`](native/README.md) for build and platform details. Then, from `simulation`, run:

```sh
cd benchmark_three_seed
python modality3_study.py prepare
python modality3_study.py run --parallel 3
python modality3_study.py aggregate
```

The default seeds are 29, 47 and 83 across all three input configurations. `prepare` creates a fresh reproduction protocol, verifies the split and fixed starts, and records source hashes before execution. Training settings are PPO learning rate 3e-4, rollout 2,048, mini-batch 256, 10 epochs and 1,273,856 environment steps per run. Checkpoints are selected on 162 fixed validation starts. The supplied model weights and per-model training manifests are preserved without modification. New training creates its own protocol and job manifests; the numerical reanalysis does not require model training or a training protocol file. `setup_run_cloud.sh` launches the same nine jobs on a configured Linux machine.

## Other components

`ncsim/` contains the environment, KDL bindings, geometry, network and CBF safety-filter evaluation. `native/` contains the C++ backend and build scripts. `occlusion_calibration.py` relates geometric score to image occlusion rate. Environment versions are recorded in `environment_versions.json`; external hardware SDKs are obtained separately as described in `../docs/DEPENDENCIES.md`.

## Result-file interpretation

See the [manuscript mapping in the main README](../README.md#manuscript-results-and-repository-files). `Fig15/16/17` CSV prefixes are legacy analysis labels. The 8.18 pp value in `Fig15_estimates_and_intervals.csv` is the recovery gain allowing self-collisions; the primary collision-free gain is 8.51 pp. The supported manuscript reproduction entry point is `benchmark_three_seed/modality3_study.py`.
