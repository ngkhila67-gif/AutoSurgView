# Simulation and evaluation

The current input ablation analysis uses seeds **29, 47 and 83** for each of three input configurations (nine reported policies). Checkpoint selection uses the validation set. The analysis scope is recorded in `benchmark_three_seed/results/analysis_protocol.json`.

The reproduction entry point, `modality3_study.py`, uses 29, 47 and 83. Numerical reanalysis uses the bundled episode records, seed-level values, final-state occlusion records and endpoint summary. The current analysis specification lists these inputs and their checksums.

## Reproduce the reported numerical analysis

After cloning or downloading the private repository, run the following from its root directory in a Python environment:

```sh
python -m pip install -r simulation/requirements.txt
cd simulation/benchmark_three_seed
python reanalyse_reported_seeds.py --out outputs/reanalysis
```

This reads the bundled per-episode records and recorded final-state occlusion rates, recomputes per-seed values, scene averages, paired differences, seed-level t intervals (2 degrees of freedom), and acquisition-folder bootstrap intervals (47 folders, 2,000 resamples, random seed 20261002). The selected policies' original weights and training manifests are in `models/benchmark_three_seed/` at the repository root. Figure-level numerical data are included in the repository-root [Source Data.xlsx](../Source%20Data.xlsx) (19 worksheets); no original clinical images or recordings are included.

## Train a new reproduction

From the `simulation` directory, install the Python dependencies, build the native backend with `python native/build_native.py`, and set `NC_POINTDATA` to an authorized local copy of the point-cloud data and `NC_HISTORY` to the compiled native-backend folder. This repository does not include the point clouds. Files at the [previous Zenodo record](https://doi.org/10.5281/zenodo.23456180) are currently restricted. The numerical reanalysis above uses bundled records and does not require those point clouds.

```sh
cd benchmark_three_seed
python modality3_study.py prepare
python modality3_study.py run --parallel 3
python modality3_study.py aggregate
```

The default seeds are 29, 47 and 83 across all three input configurations. `prepare` creates a fresh reproduction protocol, verifies the split and fixed starts, and records source hashes before execution. Training settings are PPO learning rate 3e-4, rollout 2,048, mini-batch 256, 10 epochs and 1,273,856 environment steps per run. Checkpoints are selected on 162 fixed validation starts. The supplied model weights and per-model training manifests are preserved without modification. New training creates its own protocol and job manifests; the numerical reanalysis does not require model training or a training protocol file. `setup_run_cloud.sh` launches the same nine jobs on a configured Linux machine.

## Other components

`ncsim/` contains the environment, KDL bindings, geometry, network and CBF safety-filter evaluation. `native/` contains the C++ backend and build scripts. `occlusion_calibration.py` relates geometric score to image occlusion rate. Environment versions are recorded in `environment_versions.json`; vendor dependencies are installed separately as described in `../docs/DEPENDENCIES.md`.
