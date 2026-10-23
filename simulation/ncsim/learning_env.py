"""Gymnasium adapter for new learning runs on the archived 0.5405 m geometry.

Old evaluation code and checkpoints are unchanged. Actions here are normalized
2-vectors: 0.08 * clip(a, -1, 1) / max(1, ||clip(a, -1, 1)||).
Both reward modes use the new normalized actions and the same sampled-sweep
collision termination. "legacy" preserves only the historical reward formula.
The shaped reward is practical reward shaping, not a claim of policy-invariant
potential-based shaping.
The optional joint12 observation retains unwrapped joint angles, the previous
executed displacement, and the remaining horizon. The core stays original6.
The optional filter only checks native self-collision along sampled joint-linear
segments. This is not human/obstacle avoidance or a continuous safety guarantee.
"""
from functools import lru_cache
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data import ROOT
from .geometry import HEIGHT, INITIAL_Q
from . import historical_backend as historical
from .native_backend import self_collision

HISTORY = ROOT / "outputs" / "historical_reproduction_20261002"
ACTION_STEP_M = 0.08
THRESHOLD = 20.0
REWARD_CONFIG = dict(potential_scale=100.0, potential_weight=2.0,
                     potential_mapping="score / (score + potential_scale)",
                     policy_invariant_shaping=False,
                     path_weight=0.1, step_cost=0.01, success_bonus=5.0,
                     collision_penalty=10.0, ik_failure_penalty=0.5,
                     timeout_penalty=1.0)
RESET_CONFIG = dict(max_attempts=1024, max_clear_candidates=32,
                    workspace_low=[-0.2, 0.91], workspace_high=[0.5, 1.71],
                    reject_initial_self_collision=True, prefer_occluded=True)
SAFETY_CONFIG = dict(max_joint_increment_degrees=2.0,
                     scales=[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
                     collision_model="native NC self-collision only",
                     path_model="linear interpolation in joint coordinates")


@lru_cache(maxsize=1)
def configure_historical():
    """Configure the immutable historical backend once per worker process."""
    historical.configure(HISTORY)


def checked_joint_segment(q0, q1):
    """Return (any native collision, endpoint collision, sample count).

    Includes both endpoints and spaces samples by at most 2 degrees in every
    joint. No angle wrapping: checks the actual joint-coordinate interpolation.
    """
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if q0.shape != (3,) or q1.shape != (3,) or not np.isfinite([q0, q1]).all():
        raise ValueError("Joint segment requires two finite 3-vectors")
    endpoint = bool(self_collision(q1))
    if endpoint:
        return True, True, 1
    increments = max(1, int(np.ceil(np.max(np.abs(q1-q0)) /
                                  np.deg2rad(SAFETY_CONFIG['max_joint_increment_degrees']))))
    count = 1
    for i in range(increments):
        count += 1
        if self_collision(q0 + (q1-q0) * (i/increments)):
            return True, False, count
    return False, False, count


class HistoricalLearningEnv(gym.Env):
    """Shared training/evaluation interface; ``core`` exposes HistoricalEnv.

    Fixed starts preserve their xy/q/sampling seed exactly. Training resets reject
    initial collisions and seek an occluded pose, falling back to a clear safe
    pose after 32 clear candidates or 1024 attempted IK solves. The selection is
    disclosed in every reset info. No test/validation split is selected here.
    """
    metadata = {"render_modes": []}

    def __init__(self, scenes, reward_mode="shaped", seed=20261001,
                 safety_filter=False, max_steps=40, state_mode="original6"):
        super().__init__()
        if reward_mode not in ("shaped", "legacy"):
            raise ValueError("reward_mode must be shaped or legacy")
        if state_mode not in ("original6", "joint12"):
            raise ValueError("state_mode must be original6 or joint12")
        if not scenes or int(max_steps) != max_steps or max_steps < 1:
            raise ValueError("Require scenes and a positive integer max_steps")
        configure_historical()
        self.scenes = list(scenes)
        self.reward_mode = reward_mode
        self.state_mode = state_mode
        self.safety_filter = bool(safety_filter)
        self.max_steps = int(max_steps)
        self.default_seed = int(seed)
        self.core = historical.HistoricalEnv([self.scenes[0]], threshold=THRESHOLD,
                                             max_steps=self.max_steps,
                                             action_step_m=ACTION_STEP_M)
        self.raw_env = self.core
        observation_spaces = dict(self.core.observation_space.spaces)
        if state_mode == "joint12":
            observation_spaces["robot_state"] = spaces.Box(-np.inf, np.inf, (12,), np.float32)
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = self.core.action_space
        self._has_reset = False
        self._done = True

    @property
    def configuration(self):
        state_order = ["(camera_x_m + 1) / 2", "(camera_y_m - 0.8) / 0.95",
                       "min(score / 500, 1)" if self.state_mode == "original6" else "score / (score + 100)",
                       "occlusion_guide_x", "occlusion_guide_y", "consecutive_unreachable / 5"]
        if self.state_mode == "joint12":
            state_order += ["q1_unwrapped_rad / pi", "q2_unwrapped_rad / pi", "q3_unwrapped_rad / pi",
                            "previous_executed_delta_x_m / 0.08", "previous_executed_delta_y_m / 0.08",
                            "remaining_decision_steps / max_steps"]
        return dict(geometry="archived_0.5405_m", history=str(HISTORY),
                    state_mode=self.state_mode, robot_state_dimension=len(state_order),
                    robot_state_order=state_order,
                    state_note="joint12: q is neither clipped nor wrapped; previous displacement is actual execution; reset history is zero and remaining horizon is one; core observations remain original6",
                    reward_mode=self.reward_mode, reward=dict(REWARD_CONFIG),
                    legacy_scope="Historical reward formula only; new shared actions and sampled-sweep termination",
                    collision_termination="native_endpoint_or_sampled_joint_segment",
                    normalized_action=True, action_step_m=ACTION_STEP_M,
                    threshold=THRESHOLD, max_steps=self.max_steps,
                    safety_filter=self.safety_filter, safety=dict(SAFETY_CONFIG),
                    reset=dict(RESET_CONFIG), default_seed=self.default_seed)

    @staticmethod
    def displacement(action):
        a = np.asarray(action, dtype=np.float32).copy()
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Action must be a finite 2-vector")
        a = np.clip(a, -1.0, 1.0)
        return a * np.float32(ACTION_STEP_M / max(1.0, float(np.linalg.norm(a))))

    def _random_start(self, scene):
        points = scene.points[(scene.points[:, 2] >= scene.target[2]+0.2) &
                              (scene.points[:, 2] <= HEIGHT-0.05)]
        fallback = None
        rejected_collision = rejected_ik = clear = 0
        for attempt in range(1, RESET_CONFIG['max_attempts']+1):
            xy = self.np_random.uniform(RESET_CONFIG['workspace_low'], RESET_CONFIG['workspace_high'])
            q, ok = historical.inverse(xy, INITIAL_Q)
            if not ok:
                rejected_ik += 1
                continue
            if self_collision(q):
                rejected_collision += 1
                continue
            score = historical.classify(points, xy, scene.target, q)[1]
            candidate = dict(xy=xy.tolist(), q=q.tolist(),
                             seed=int(self.np_random.integers(0, 2**31-1)))
            if score >= THRESHOLD:
                return candidate, dict(policy="random_safe_prefer_occluded", attempts=attempt,
                    rejected_initial_collision=rejected_collision, rejected_initial_ik=rejected_ik,
                    clear_candidates=clear, used_clear_fallback=False)
            clear += 1
            if fallback is None:
                fallback = candidate
            if clear >= RESET_CONFIG['max_clear_candidates']:
                break
        if fallback is None:
            raise RuntimeError("No native-reachable collision-free training start found")
        return fallback, dict(policy="random_safe_prefer_occluded", attempts=attempt,
            rejected_initial_collision=rejected_collision, rejected_initial_ik=rejected_ik,
            clear_candidates=clear, used_clear_fallback=True)

    def reset(self, *, seed=None, options=None):
        if seed is None and not self._has_reset:
            seed = self.default_seed
        super().reset(seed=seed)
        self._has_reset = True
        options = options or {}
        index = int(options['scene_index']) if 'scene_index' in options else int(self.np_random.integers(len(self.scenes)))
        if not 0 <= index < len(self.scenes):
            raise ValueError("scene_index out of range")
        scene = self.scenes[index]
        if scene.target[2] >= HEIGHT-0.25:
            raise ValueError("Target height incompatible with archived camera height")
        self.core.scenes = [scene]
        if 'start' in options:
            supplied = options['start']
            xy, q = np.asarray(supplied['xy'], dtype=float), np.asarray(supplied['q'], dtype=float)
            if xy.shape != (2,) or q.shape != (3,) or not np.isfinite(xy).all() or not np.isfinite(q).all():
                raise ValueError("Fixed start requires finite xy(2), q(3), and a sampling seed")
            start = dict(xy=xy.tolist(), q=q.tolist(), seed=int(supplied['seed']))
            self.start_selection = dict(policy="fixed_start_unmodified", attempts=0,
                rejected_initial_collision=0, rejected_initial_ik=0, clear_candidates=0,
                used_clear_fallback=False)
        else:
            start, self.start_selection = self._random_start(scene)
        self.start = start
        self._last_obs = self.core.reset_start(start)
        self.initial_collision = bool(self_collision(self.core.q))
        self.core.collided = self.initial_collision
        self.filtered_actions = self.rejected_collision = self.rejected_endpoint_collision = 0
        self.ik_failures = self.ik_solver_failures = 0
        self.swept_collision = self.initial_collision
        self.collision_checks = 0
        self._last = dict(candidate_self_collision=False, candidate_swept_self_collision=False,
                          executed_self_collision=self.initial_collision,
                          executed_swept_self_collision=self.initial_collision,
                          action_scale=1.0, executed_displacement_m=[0.0, 0.0],
                          requested_displacement_m=[0.0, 0.0], safety_rejected=False,
                          ik_ok=True, ik_attempts=0, reward_components={})
        self._initial_terminal = self.initial_collision or self.core.initial_score < THRESHOLD
        if self.initial_collision:
            self.core.finished, self.core.reason = True, "initial_self_collision"
        self._done = False
        return self._copy_observation(), self._info()

    def _copy_observation(self):
        obs = {key: value.copy() for key, value in self._last_obs.items()}
        if self.state_mode == "joint12":
            state = obs["robot_state"]
            score = float(self.core.score)
            state[2] = score / (score + 100.0)
            previous_delta = np.asarray(self._last["executed_displacement_m"], dtype=float) / ACTION_STEP_M
            remaining = (self.max_steps - self.core.steps) / self.max_steps
            obs["robot_state"] = np.concatenate((state, self.core.q / np.pi, previous_delta, [remaining])).astype(np.float32)
        return obs

    def _info(self):
        c = self.core
        endpoint_success = bool(c.score < THRESHOLD and not c.collided)
        success = bool(endpoint_success and not self.swept_collision)
        return dict(scene_id=c.scene.scene_id, capture_group=c.scene.capture_group,
            initial_score=float(c.initial_score), final_score=float(c.score), score=float(c.score),
            path_length_m=float(c.path_length), self_collision=bool(c.collided),
            swept_self_collision=bool(self.swept_collision), initial_self_collision=self.initial_collision,
            success=success, is_success=success, endpoint_success=endpoint_success,
            collision_free_success=success, reason=c.reason, termination_reason=c.reason,
            steps=c.steps, initially_clear=bool(c.initial_score < THRESHOLD),
            filtered_actions=self.filtered_actions, rejected_collision=self.rejected_collision,
            rejected_endpoint_collision=self.rejected_endpoint_collision,
            ik_failures=self.ik_failures, ik_solver_failures=self.ik_solver_failures,
            unreachable_count=c.unreachable_count, collision_checks=self.collision_checks,
            start_selection=dict(self.start_selection), start={k:list(v) if isinstance(v,list) else v for k,v in self.start.items()},
            reward_mode=self.reward_mode, state_mode=self.state_mode, safety_filter=self.safety_filter, **self._last)

    def _candidate(self, delta):
        q, ok = historical.inverse(self.core.xy + delta, self.core.q)
        if not ok:
            self.ik_solver_failures += 1
            return q, False, False, False
        swept, endpoint, checks = checked_joint_segment(self.core.q, q)
        self.collision_checks += checks
        return q, True, swept, endpoint

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset before stepping a finished episode")
        delta = self.displacement(action)
        c = self.core
        if self._initial_terminal:
            self._initial_terminal = False
            self._done = True
            c.finished = True
            return self._copy_observation(), 0.0, True, False, self._info()
        old = float(c.score)
        old_xy = c.xy.copy()
        q, ok, swept, endpoint = self._candidate(delta)
        first_ok, first_swept, first_endpoint = ok, swept, endpoint
        attempts = 1
        scale = 1.0
        rejected = False
        if self.safety_filter and (not ok or swept):
            self.filtered_actions += 1
            if swept:
                self.rejected_collision += 1
            if endpoint:
                self.rejected_endpoint_collision += 1
            rejected = True
            for scale in SAFETY_CONFIG['scales'][1:]:
                q, ok, swept, endpoint = self._candidate(delta * scale)
                attempts += 1
                if ok and not swept:
                    rejected = False
                    break
            if rejected:
                scale, q, ok, swept, endpoint = 0.0, c.q.copy(), True, False, bool(self_collision(c.q))
        c.steps += 1
        if not first_ok:
            self.ik_failures += 1
        if ok and not rejected:
            c.q = q
            c.xy = old_xy + delta * scale
            c.path_length += float(np.linalg.norm(c.xy-old_xy))
            c.unreachable_count = 0
            c._measure()
        elif rejected or not ok:
            c.unreachable_count += 1
        c.collided = bool(self_collision(c.q))
        executed_swept = bool(swept and ok and not rejected)
        self.swept_collision = bool(self.swept_collision or executed_swept or c.collided)
        moved = float(np.linalg.norm(c.xy-old_xy))
        terminated = truncated = False
        collision_failure = c.collided or executed_swept
        if collision_failure:
            c.reason, terminated = ("self_collision" if c.collided else "swept_self_collision"), True
        elif ok and not rejected and c.score < THRESHOLD:
            c.reason, terminated = "criterion_met", True
        elif c.unreachable_count > 5:
            c.reason, terminated = "repeated_unreachable", True
        elif c.steps >= self.max_steps:
            c.reason, truncated = "step_limit", True
        else:
            c.reason = "running"
        if self.reward_mode == 'legacy':
            reward = (-float(c.score)/max(old, 1e-9) + (0.5 if c.score < old else 0.0)) if ok and not rejected else -float(c.unreachable_count)
            bonus = -5.0 if collision_failure or c.reason in ('repeated_unreachable','step_limit') else (5.0 if c.reason == 'criterion_met' else 0.0)
            parts = dict(legacy_step=reward, terminal=bonus)
        else:
            cfg = REWARD_CONFIG
            parts = dict(potential=cfg['potential_weight'] * (old/(old+cfg['potential_scale'])-float(c.score)/(float(c.score)+cfg['potential_scale'])),
                path=-cfg['path_weight'] * moved/ACTION_STEP_M, step=-cfg['step_cost'],
                ik_failure=-cfg['ik_failure_penalty'] if not first_ok else 0.0,
                collision=-cfg['collision_penalty'] if collision_failure else 0.0,
                success=cfg['success_bonus'] if c.reason == 'criterion_met' else 0.0,
                timeout=-cfg['timeout_penalty'] if c.reason in ('step_limit','repeated_unreachable') else 0.0)
        reward = float(sum(parts.values()))
        c.finished = self._done = bool(terminated or truncated)
        self._last = dict(candidate_self_collision=bool(first_endpoint),
            candidate_swept_self_collision=bool(first_swept), executed_self_collision=bool(c.collided),
            executed_swept_self_collision=executed_swept, action_scale=float(scale),
            executed_displacement_m=(c.xy-old_xy).tolist(), requested_displacement_m=delta.tolist(),
            safety_rejected=rejected, ik_ok=bool(ok and not rejected), ik_attempts=attempts,
            reward_components=parts)
        self._last_obs = c._observation()
        return self._copy_observation(), reward, bool(terminated), bool(truncated), self._info()

    def close(self):
        self.core.close()
