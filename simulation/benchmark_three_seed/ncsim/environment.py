import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .geometry import INITIAL_Q, HEIGHT, forward, inverse, classify_and_score, sample_points, self_collision
from .native_backend import ENGINE
from .robot_model import assembly_transforms


class SurgicalCameraEnv(gym.Env):
    """Static captured-point-cloud replay with archived native KDL and collision.

    One step is a camera decision, NOT a physical second. No robot I/O is used.
    """
    metadata = {"render_modes": []}

    def __init__(self, scenes, threshold=20.0, max_steps=40, action_step_m=.08, action_mode="clip_norm"):
        super().__init__()
        if not scenes:
            raise ValueError("At least one real point-cloud scene is required")
        if threshold <= 0 or max_steps < 1 or action_step_m <= 0:
            raise ValueError("Invalid environment configuration")
        if action_mode not in ("clip_norm","fixed_length"):
            raise ValueError("Unknown action mode")
        self.scenes = scenes
        self.threshold,self.max_steps,self.action_step,self.action_mode = threshold,max_steps,action_step_m,action_mode
        self.observation_space = spaces.Dict({
            "robot_state": spaces.Box(-np.inf,np.inf,(6,),np.float32),
            "point_cloud_in": spaces.Box(-np.inf,np.inf,(256,3),np.float32),
            "point_cloud_out": spaces.Box(-np.inf,np.inf,(512,3),np.float32)})
        self.action_space = spaces.Box(-1.0,1.0,(2,),np.float32)
        self.finished = True

    def _measure(self):
        self.inside,self.score,self.guide,self.geometry = classify_and_score(self.points,self.xy,self.scene.target,self.q)

    def _observation(self):
        T = self.geometry[0]
        # Independent per-step RNG prevents a planning baseline's lookahead from
        # changing the point samples subsequently seen by any learned policy.
        rng = np.random.default_rng(np.random.SeedSequence([self.observation_seed,self.steps]))
        return {
            "robot_state": np.array([(self.xy[0]+1)/2,(self.xy[1]-.8)/.95,min(self.score/500,1),
                                      self.guide[0],self.guide[1],self.unreachable_count/5],dtype=np.float32),
            "point_cloud_in": sample_points(self.points[self.inside],self.scene.target,T,256,rng),
            "point_cloud_out": sample_points(self.points[~self.inside],self.scene.target,T,512,rng)}

    def _info(self):
        return dict(scene_id=self.scene.scene_id,score=self.score,success=bool(self.score<self.threshold and not self.collided),
                    termination_reason=self.reason,steps=self.steps,path_length_m=self.path_length,
                    initial_score=self.initial_score,initially_clear=bool(self.initial_score<self.threshold),
                    unreachable_count=self.unreachable_count,self_collision=self.collided,
                    engine=ENGINE,time_unit="decision_step")

    def reset(self, *,seed=None,options=None):
        super().reset(seed=seed)
        options = options or {}
        self.scene = self.scenes[int(options.get("scene_index",self.np_random.integers(len(self.scenes))))]
        target_z = self.scene.target[2]
        if target_z >= HEIGHT-.25:
            raise ValueError("Target height incompatible with the archived camera height")
        self.points = self.scene.points[(self.scene.points[:,2]>=target_z+.2)&(self.scene.points[:,2]<=HEIGHT-.05)].copy()
        self.q = INITIAL_Q.copy()
        if "initial_xy" in options:
            candidate = np.asarray(options["initial_xy"],dtype=float)
            self.q,ok = inverse(candidate,self.q)
            if not ok or self_collision(self.q):
                raise ValueError("Requested start is not reachable and self-collision-free")
        else:
            for _ in range(1000):
                candidate = self.np_random.uniform([-.2,.91],[.5,1.71])
                q,ok = inverse(candidate,INITIAL_Q)
                if ok and not self_collision(q):
                    self.q = q
                    break
            else:
                raise RuntimeError("No feasible initial pose after 1000 attempts")
        self.xy = forward(self.q)[-1]
        self.observation_seed = int(self.np_random.integers(0,2**31-1))
        self.steps,self.unreachable_count,self.path_length = 0,0,0.0
        self.collided,self.finished,self.reason = False,False,"running"
        self._measure()
        self.initial_score = self.score
        return self._observation(),self._info()

    def displacement(self,action):
        action = np.asarray(action,dtype=float).copy()
        if action.shape!=(2,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 2-vector")
        action = np.clip(action,-1,1)
        norm = np.linalg.norm(action)
        if norm > 0 and (norm>self.action_step or self.action_mode=="fixed_length"):
            action *= self.action_step/norm
        return action

    def candidate(self,action):
        delta = self.displacement(action)
        q,ok = inverse(self.xy+delta,self.q)
        return delta,q,ok

    def step(self,action):
        if self.finished:
            raise RuntimeError("Reset before stepping a finished episode")
        if self.steps==0 and self.initial_score<self.threshold:
            self.finished,self.reason = True,"initially_clear"
            return self._observation(),0.0,True,False,self._info()
        old = self.score
        delta,q,ok = self.candidate(action)
        self.steps += 1
        if ok:
            self.unreachable_count = 0
            previous_xy = self.xy.copy()
            self.q,self.xy = q,forward(q)[-1]
            self.path_length += float(np.linalg.norm(self.xy-previous_xy))
            self._measure()
            reward = -self.score/max(old,1e-9)+(0.5 if self.score<old else 0)
            self.collided = self_collision(self.q)
        else:
            self.unreachable_count += 1
            reward = -float(self.unreachable_count)
        terminated = False
        truncated = False
        if self.collided:
            self.reason,terminated,reward = "self_collision",True,reward-5
        elif self.score<self.threshold:
            self.reason,terminated,reward = "criterion_met",True,reward+5
        elif self.unreachable_count>5:
            self.reason,terminated,reward = "repeated_unreachable",True,reward-5
        elif self.steps>=self.max_steps:
            self.reason,truncated,reward = "step_limit",True,reward-5
        self.finished = terminated or truncated
        return self._observation(),float(reward),terminated,truncated,self._info()

    def snapshot(self):
        T,center,axis,corners = self.geometry
        return dict(step=self.steps,score=self.score,xy=self.xy.tolist(),q=self.q.tolist(),
                    camera=center.tolist(),corners=corners.tolist(),target=self.scene.target.tolist(),
                    robot_transforms=assembly_transforms(self.q,self.xy,self.scene.target),
                    arm=np.column_stack((forward(self.q),np.full(4,HEIGHT))).tolist(),
                    inside_indices=np.flatnonzero(self.inside).tolist(),reason=self.reason,
                    path_length_m=self.path_length)
