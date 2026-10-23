"""Frozen, paired evaluation for newly trained multimodal policies.

This module never selects cases by a policy's performance. Historical checkpoints
and published reconstruction results are not modified. A single training seed is
a pilot, not evidence of state-of-the-art performance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import ROOT, load_scene, natural_key, sha256
from .geometry import INITIAL_Q, self_collision
from .historical_backend import HistoricalEnv, configure, inverse

VERSION = "new_learning_paired_evaluation_v1"
HISTORY = ROOT / "outputs/historical_reproduction_20261002"
THRESHOLD = 20.0
MAX_STEPS = 40


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def select_group_scenes(rows, scenes_per_group=1):
    """Equally spaced files within each capture group, selected without outcomes."""
    if not isinstance(scenes_per_group, int) or scenes_per_group < 1:
        raise ValueError("scenes_per_group must be a positive integer")
    groups = defaultdict(list)
    for row in rows:
        groups[row["capture_group"]].append(row)
    selected = []
    for group in sorted(groups, key=natural_key):
        ordered = sorted(groups[group], key=lambda row: natural_key(row["scene_id"]))
        # One representative is the middle file; multiple representatives span
        # the group, regardless of score, model, or success.
        if scenes_per_group == 1:
            indices = [len(ordered) // 2]
        else:
            indices = np.linspace(0, len(ordered) - 1,
                                  min(len(ordered), scenes_per_group)).round().astype(int)
        selected.extend(ordered[int(index)] for index in indices)
    return selected


def _case_seed(seed, scene_id, start_index):
    # Stable if unrelated scenes are added to or removed from the evaluation.
    return int.from_bytes(hashlib.sha256(
        f"{seed}|{scene_id}|{start_index}".encode("utf-8")).digest()[:4], "big")


def _case_path(output):
    output = Path(output)
    return output if output.suffix.lower() == ".json" else output / "cases.json"


def freeze_cases(rows, starts_per_scene, seed, output):
    """Freeze independently sampled reachable, collision-free initial poses.

    All starts are retained, including initially clear views. The primary recovery
    population is separately flagged with initial_score >= 20. Rows retain all
    source and split metadata supplied by the caller.
    """
    rows = list(rows)
    if not rows or len({row["scene_id"] for row in rows}) != len(rows):
        raise ValueError("Supply a nonempty list of unique scene rows")
    if not isinstance(starts_per_scene, int) or starts_per_scene < 1:
        raise ValueError("starts_per_scene must be positive")
    path = _case_path(output)
    specification = dict(version=VERSION, rows=rows, starts_per_scene=starts_per_scene,
                         seed=int(seed), threshold=THRESHOLD, max_steps=MAX_STEPS,
                         initial_pose="reachable and native self-collision-free",
                         sampling="stable per-scene/start Python Random seed",
                         historical_backend_sha256=sha256(HISTORY / "historical_backend_05405.dll"),
                         collision_backend_sha256=sha256(ROOT / "native/original_backend.dll"))
    spec_path = path.with_name(path.stem + "_protocol.json")
    if path.exists():
        if not spec_path.exists():
            raise ValueError("Cases exist without their frozen protocol")
        stored = json.loads(spec_path.read_text(encoding="utf-8"))
        if stored.get("specification") != specification or stored.get("cases_sha256") != sha256(path):
            raise ValueError("Frozen case protocol, source rows, or cases changed")
        cases = json.loads(path.read_text(encoding="utf-8"))
        _validate_cases(cases)
        return cases
    if spec_path.exists():
        raise ValueError("Incomplete frozen case output; choose a fresh directory")
    configure(HISTORY)
    cases = []
    for scene_index, row in enumerate(rows):
        scene = load_scene(row)
        env = HistoricalEnv([scene], max_steps=MAX_STEPS)
        for start_index in range(starts_per_scene):
            case_seed = _case_seed(seed, row["scene_id"], start_index)
            rng = random.Random(case_seed)
            for attempt in range(1, 10001):
                xy = np.array([rng.uniform(-0.2, 0.5), rng.uniform(0.91, 1.71)])
                q, reachable = inverse(xy, INITIAL_Q)
                if reachable and not self_collision(q):
                    break
            else:
                raise RuntimeError(f"No reachable collision-free start: {row['scene_id']}")
            case_id = f"{row['scene_id']}::start_{start_index + 1:02d}"
            start = dict(seed=case_seed, xy=xy.tolist(), q=q.tolist(), attempts=attempt,
                         initial_self_collision=False, start_index=start_index, case_id=case_id)
            env.reset_start(start)
            cases.append(dict(case_id=case_id, scene_id=row["scene_id"],
                              capture_group=row["capture_group"], scene_index=scene_index,
                              start_index=start_index, row=dict(row), start=start,
                              initial_score=float(env.initial_score),
                              eligible=bool(env.initial_score >= THRESHOLD)))
    _validate_cases(cases)
    _save(path, cases)
    _save(spec_path, dict(specification=specification, cases_sha256=sha256(path),
                         case_count=len(cases), eligible=sum(case["eligible"] for case in cases)))
    return cases


def _validate_cases(cases):
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation cases must be nonempty and unique")
    for case in cases:
        if case["row"]["scene_id"] != case["scene_id"]:
            raise ValueError("Case scene does not match source row")
        if case["eligible"] != (case["initial_score"] >= THRESHOLD):
            raise ValueError("Case eligibility does not match the frozen threshold")
        if case["start"].get("initial_self_collision", False):
            raise ValueError("New learning evaluation excludes initially colliding poses")


def summarize_episodes(episodes, expected_cases=None):
    """Keep recovery success, safety, and all-eligible path metrics distinct."""
    episodes = list(episodes)
    eligible = [row for row in episodes if row["eligible"]]
    successful = [row for row in eligible if row["strict_success"]]

    def mean(rows, key):
        return float(np.mean([row[key] for row in rows])) if rows else None

    def percent(rows, key):
        result = mean(rows, key)
        return None if result is None else result * 100

    groups = defaultdict(list)
    for row in episodes:
        groups[row["scene_id"]].append(row)
    per_scene = []
    for scene_id, scene_rows in groups.items():
        selected = [row for row in scene_rows if row["eligible"]]
        per_scene.append(dict(scene_id=scene_id, candidates=len(scene_rows), eligible=len(selected),
                              success_percent=percent(selected, "success"),
                              strict_success_percent=percent(selected, "strict_success")))
    nonempty = [row for row in per_scene if row["eligible"]]
    total_steps = sum(row["decision_steps"] for row in eligible)
    collision_count = sum(bool(row["self_collision"]) for row in eligible)
    return dict(
        evidence_status="pilot_not_sota", complete=expected_cases is None or len(episodes) == expected_cases,
        expected_cases=expected_cases, candidates=len(episodes), eligible=len(eligible),
        initially_clear=len(episodes) - len(eligible), scene_count=len(per_scene),
        capture_group_count=len({row["capture_group"] for row in episodes}),
        scenes_without_eligible_starts=len(per_scene) - len(nonempty),
        recovery_successes=sum(row["success"] for row in eligible),
        collision_free_recovery_successes=len(successful),
        pooled_success_percent=percent(eligible, "success"),
        pooled_strict_success_percent=percent(eligible, "strict_success"),
        scene_macro_success_percent=mean(nonempty, "success_percent"),
        scene_macro_strict_success_percent=mean(nonempty, "strict_success_percent"),
        self_collision_cases=collision_count, self_collision_percent=percent(eligible, "self_collision"),
        sampled_path_self_collision_cases=sum(bool(row.get("sampled_path_self_collision", row["self_collision"])) for row in eligible),
        any_self_collision_cases=sum(bool(row.get("any_self_collision", row["self_collision"])) for row in eligible),
        collision_definition="self_collision is the native endpoint check; sampled_path_self_collision is finite joint interpolation, not a continuous guarantee",
        mean_path_length_m_all_eligible=mean(eligible, "path_length_m"),
        mean_path_length_m_collision_free_successes=mean(successful, "path_length_m"),
        mean_final_score_all_eligible=mean(eligible, "final_score"),
        mean_final_score_collision_free_successes=mean(successful, "final_score"),
        mean_decision_steps_all_eligible=mean(eligible, "decision_steps"),
        mean_inference_ms_per_decision=(sum(row["inference_seconds"] for row in eligible) / total_steps * 1000
                                        if total_steps else None),
        mean_environment_ms_per_decision=(sum(row["environment_seconds"] for row in eligible) / total_steps * 1000
                                          if total_steps else None),
        total_filtered_actions=sum(row["filtered_actions"] for row in eligible),
        total_ik_failures=sum(row["ik_failures"] for row in eligible),
        per_scene=per_scene,
        limitations=["Single checkpoint evaluation is a pilot, not a SOTA claim or a training-seed confidence interval.",
                     "Static recorded point clouds and kinematic actions; inference timing is not physical recovery time.",
                     "Endpoint self-collision uses the original native robot check; finite joint-path samples are additional checks, not continuous or clinical human-contact safety guarantees.",
                     "Path means over successful trials are conditional and can favor a method with fewer successes; all eligible means are also reported."])


def _write_csv(path, episodes):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        if episodes:
            writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
    temporary.replace(path)


def _load_model(path, algorithm, device):
    from stable_baselines3 import PPO, SAC, TD3
    # Evaluation never samples or updates a replay buffer. SB3 otherwise creates
    # the entire saved training buffer capacity when loading off-policy models.
    options = {} if algorithm == "PPO" else {"buffer_size": 1}
    return {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algorithm].load(str(path), device=device, **options)


def _new_env(scene, safety_filter, seed, state_mode="original6"):
    from .learning_env import HistoricalLearningEnv
    return HistoricalLearningEnv([scene], reward_mode="shaped", seed=seed,
                                 safety_filter=safety_filter, max_steps=MAX_STEPS,
                                 state_mode=state_mode)


def _checkpoint_state_mode(model):
    """Match the evaluation observation to the saved model, never guess/pad it."""
    try:
        shape = tuple(model.observation_space["robot_state"].shape)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Checkpoint must contain a robot_state observation space") from exc
    modes = {(6,): "original6", (12,): "joint12"}
    if shape not in modes:
        raise ValueError(f"Unsupported checkpoint robot_state shape: {shape}; expected (6,) or (12,)")
    return modes[shape], list(shape)


def _frame(env, info, action=None):
    core = env.core
    return dict(step=int(info.get("steps", core.steps)), score=float(info.get("final_score", core.score)),
                xy=np.asarray(core.xy).tolist(), q=np.asarray(core.q).tolist(),
                action=None if action is None else np.asarray(action).tolist(),
                path_length_m=float(info.get("path_length_m", core.path_length)),
                reason=info.get("reason", core.reason), self_collision=bool(info.get("self_collision", core.collided)),
                sampled_path_self_collision=bool(info.get("swept_self_collision", False)))


def _source_checksums():
    names = ("learning_evaluation.py", "learning_env.py", "historical_backend.py",
             "paper_evaluation.py", "environment.py", "geometry.py", "native_backend.py", "policies.py",
             "learning_features.py")
    return {name: sha256(ROOT / "ncsim" / name) for name in names}


def evaluate_checkpoint(model_path, algorithm, cases, out_dir, safety_filter=False,
                        progress_callback=None, trace_limit=3, device="cpu", resume=True):
    """Evaluate one SB3 checkpoint with resumable, model-independent paired cases.

    ``progress_callback`` receives a dictionary after each completed case. Model
    predictions are deterministic; inference and native environment work are
    separately timed. A shield must be compared with the same shield for all
    algorithms and is explicitly encoded in the output identity.
    """
    if isinstance(cases, (str, Path)):
        cases = json.loads(Path(cases).read_text(encoding="utf-8"))
    cases = list(cases)
    _validate_cases(cases)
    algorithm = algorithm.upper().replace("MM-", "")
    if algorithm not in {"PPO", "SAC", "TD3"}:
        raise ValueError("algorithm must be PPO, SAC, or TD3")
    if trace_limit < 0:
        raise ValueError("trace_limit must be nonnegative")
    model_path, out_dir = Path(model_path).resolve(), Path(out_dir).resolve()
    # Load before freezing identity: the saved observation space determines the
    # required physical-state representation, including for resumed evaluations.
    model = _load_model(model_path, algorithm, device)
    state_mode, state_shape = _checkpoint_state_mode(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    specification = dict(version=VERSION, algorithm=algorithm, checkpoint=str(model_path),
                         checkpoint_sha256=sha256(model_path), cases_sha256=_digest(cases),
                         candidates=len(cases), threshold=THRESHOLD, max_steps=MAX_STEPS,
                         state_mode=state_mode, robot_state_shape=state_shape,
                         safety_filter=bool(safety_filter), deterministic=True, device=device,
                         evidence_status="pilot_not_sota", source_sha256=_source_checksums(),
                         historical_backend_sha256=sha256(HISTORY / "historical_backend_05405.dll"),
                         collision_backend_sha256=sha256(ROOT / "native/original_backend.dll"))
    protocol_path = out_dir / "evaluation_protocol.json"
    if protocol_path.exists():
        if not resume:
            raise FileExistsError("Evaluation already exists and resume=False")
        stored = json.loads(protocol_path.read_text(encoding="utf-8"))
        if stored != specification:
            raise ValueError("Cannot resume: checkpoint, case set, environment, or protocol changed")
    elif any(out_dir.iterdir()):
        raise ValueError("Output directory is not empty and has no evaluation protocol")
    else:
        _save(protocol_path, specification)
    protocol_digest = _digest(specification)
    journal = out_dir / "episodes.jsonl"
    completed = {}
    if journal.exists():
        with journal.open(encoding="utf-8") as handle:
            for line in handle:
                saved = json.loads(line)
                row = saved["episode"]
                if (saved["protocol_sha256"] != protocol_digest or row["case_id"] in completed
                        or saved.get("episode_sha256") != _digest(row)):
                    raise ValueError("Evaluation journal protocol mismatch or duplicate case")
                completed[row["case_id"]] = row
        if not set(completed).issubset({case["case_id"] for case in cases}):
            raise ValueError("Evaluation journal contains an unknown case")
        for case in cases:
            if case["case_id"] in completed:
                row = completed[case["case_id"]]
                if (row["scene_id"] != case["scene_id"] or row["start_seed"] != case["start"]["seed"]
                        or row["eligible"] != case["eligible"]
                        or row["initial_score"] != case["initial_score"]
                        or not 0 <= row["decision_steps"] <= MAX_STEPS):
                    raise ValueError("Evaluation journal no longer matches the frozen starts")
    configure(HISTORY)
    scene, env, last_scene = None, None, None
    try:
        with journal.open("a", encoding="utf-8") as handle:
            for index, case in enumerate(cases):
                if case["case_id"] in completed:
                    continue
                if case["scene_id"] != last_scene:
                    if env is not None:
                        env.close()
                    scene = load_scene(case["row"])
                    env = _new_env(scene, safety_filter, case["start"]["seed"], state_mode)
                    last_scene = case["scene_id"]
                episode_start = time.perf_counter()
                observation, info = env.reset(options={"scene_index": 0, "start": case["start"]})
                if not np.isclose(info["initial_score"], case["initial_score"], rtol=0, atol=1e-8):
                    raise ValueError(f"Initial score differs from frozen case: {case['case_id']}")
                trace = [_frame(env, info)] if index < trace_limit else None
                inference_seconds, environment_seconds, total_reward, decision_steps = 0.0, 0.0, 0.0, 0
                # Initially clear observations are counted, but not acted on or
                # called recoveries. This preserves the original population rule.
                terminated = not case["eligible"]
                truncated = False
                while not (terminated or truncated):
                    before = time.perf_counter()
                    action, _ = model.predict(observation, deterministic=True)
                    inference_seconds += time.perf_counter() - before
                    before = time.perf_counter()
                    observation, reward, terminated, truncated, info = env.step(action)
                    environment_seconds += time.perf_counter() - before
                    total_reward += float(reward)
                    decision_steps += 1
                    if trace is not None:
                        trace.append(_frame(env, info, action))
                    if decision_steps > MAX_STEPS:
                        raise RuntimeError("Environment exceeded the frozen evaluation horizon")
                collided = bool(info["self_collision"])
                sampled_collision = bool(info.get("swept_self_collision", collided))
                any_collision = collided or sampled_collision
                final_score = float(info["final_score"])
                success = bool(case["eligible"] and final_score < THRESHOLD)
                row = dict(case_id=case["case_id"], scene_id=case["scene_id"],
                           capture_group=case["capture_group"], start_index=case["start_index"],
                           start_seed=case["start"]["seed"], algorithm=algorithm, safety_filter=bool(safety_filter),
                           initial_x_m=float(case["start"]["xy"][0]), initial_y_m=float(case["start"]["xy"][1]),
                           eligible=bool(case["eligible"]), initially_clear=not case["eligible"],
                           initial_score=float(case["initial_score"]), final_score=final_score,
                           success=success, strict_success=success and not any_collision,
                           self_collision=collided, sampled_path_self_collision=sampled_collision,
                           any_self_collision=any_collision, path_length_m=float(info["path_length_m"]),
                           decision_steps=decision_steps, termination_reason=str(info["reason"]),
                           filtered_actions=int(info.get("filtered_actions", 0)),
                           ik_failures=int(info.get("ik_failures", 0)), total_reward=total_reward,
                           inference_seconds=inference_seconds, environment_seconds=environment_seconds,
                           episode_wall_seconds=time.perf_counter() - episode_start)
                if trace is not None:
                    _save(out_dir / "traces" / f"case_{index:05d}.json",
                          dict(case_id=case["case_id"], protocol_sha256=protocol_digest, frames=trace))
                handle.write(_canonical(dict(protocol_sha256=protocol_digest,
                                             episode_sha256=_digest(row), episode=row)) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed[case["case_id"]] = row
                progress = dict(status="running", completed_cases=len(completed), total_cases=len(cases),
                                algorithm=algorithm, safety_filter=bool(safety_filter), last_case_id=case["case_id"])
                _save(out_dir / "progress.json", progress)
                if progress_callback is not None:
                    progress_callback(progress)
    finally:
        if env is not None:
            env.close()
        episodes = [completed[case["case_id"]] for case in cases if case["case_id"] in completed]
        _write_csv(out_dir / "episodes.csv", episodes)
        summary = summarize_episodes(episodes, expected_cases=len(cases))
        summary.update(algorithm=algorithm, safety_filter=bool(safety_filter),
                       state_mode=state_mode, robot_state_shape=state_shape,
                       checkpoint_sha256=specification["checkpoint_sha256"],
                       cases_sha256=specification["cases_sha256"])
        _save(out_dir / "summary.json", summary)
    _save(out_dir / "progress.json", dict(status="complete", completed_cases=len(completed),
                                         total_cases=len(cases), algorithm=algorithm,
                                         safety_filter=bool(safety_filter)))
    return summary


def paired_comparison(left, right):
    """Paired differences, including common-success paths; no invented seed CI."""
    left_by_id = {row["case_id"]: row for row in left}
    right_by_id = {row["case_id"]: row for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right) or left_by_id.keys() != right_by_id.keys():
        raise ValueError("Paired comparisons require identical, unique case sets")
    pairs = []
    for case_id, lhs in left_by_id.items():
        rhs = right_by_id[case_id]
        if lhs["eligible"] != rhs["eligible"] or not np.isclose(lhs["initial_score"], rhs["initial_score"], rtol=0, atol=1e-8):
            raise ValueError("Paired comparison starts do not match")
        for key in ("start_seed", "initial_x_m", "initial_y_m"):
            if lhs.get(key) != rhs.get(key):
                raise ValueError("Paired comparison initial poses or sampling seeds do not match")
        if lhs.get("safety_filter") != rhs.get("safety_filter"):
            raise ValueError("Compare algorithms at the same safety-filter setting")
        if lhs["eligible"]:
            pairs.append((lhs, rhs))
    common = [(lhs, rhs) for lhs, rhs in pairs if lhs["strict_success"] and rhs["strict_success"]]
    difference = lambda rows, key: float(np.mean([float(lhs[key]) - float(rhs[key]) for lhs, rhs in rows])) if rows else None
    success_difference = difference(pairs, "strict_success")
    return dict(evidence_status="pilot_not_sota", direction="left minus right", eligible_pairs=len(pairs),
                strict_success_difference_percentage_points=None if success_difference is None else success_difference * 100,
                path_difference_m_all_eligible=difference(pairs, "path_length_m"),
                final_score_difference_all_eligible=difference(pairs, "final_score"),
                self_collision_case_difference=sum(int(lhs["self_collision"]) - int(rhs["self_collision"]) for lhs, rhs in pairs),
                common_success_pairs=len(common), path_difference_m_common_success=difference(common, "path_length_m"),
                left_only_successes=sum(lhs["strict_success"] and not rhs["strict_success"] for lhs, rhs in pairs),
                right_only_successes=sum(rhs["strict_success"] and not lhs["strict_success"] for lhs, rhs in pairs),
                training_seed_confidence_interval=None)
