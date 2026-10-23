# Trained policies

SHA-256 checksums of all policy files are listed in `SHA256SUMS.json`.

## `benchmark_three_seed/`

Nine reported policies of the simulation benchmark: three input configurations with seeds 29, 47 and 83 under the same recorded training protocol (Stable-Baselines3 PPO, 1,273,856 environment steps per run).

| Folder prefix | Input configuration |
|---|---|
| `mmppo_seed_*` | MM-PPO: inside points (256 × 3), outside points (512 × 3) and the six-dimensional task state |
| `pc_seed_*` | Point clouds only |
| `state_seed_*` | Task state only (normalised camera position, truncated occlusion score, guidance vector, consecutive unreachable actions) |

Each folder contains `best_model.zip` (checkpoint selected on the validation set; Stable-Baselines3 format), `final_normalization.pkl` (observation normalisation statistics), `best_validation.json`, `validation_history.json` and `training_manifest.json`. Load a policy with the code in `../simulation/benchmark_three_seed/` (custom feature extractor `AblationAttentionPointNet` in `ncsim/policies.py`).

## `deployed/`

`ppo_AnzhenRobot_20` is the MM-PPO policy trained earlier with the same network and training settings and used in the physical and clinical experiments. `ppo_AnzhenRobot_20.onnx` is the file loaded by the robot controller (also at `robot-controller/models/`); `ppo_AnzhenRobot_20.zip` and `ppo_AnzhenRobot_20_vec_normalize.pkl` are the corresponding Stable-Baselines3 checkpoint and normalisation statistics.
