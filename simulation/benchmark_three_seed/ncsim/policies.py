import io
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from .data import sha256
from .geometry import self_collision, classify_and_score


class AblationAttentionPointNet(BaseFeaturesExtractor):
    """Architecture transcribed from the original training feature extractor."""
    def __init__(self,observation_space,embed_dim=128,robot_embed_dim=64,ablation_mode="fusion"):
        dimensions = {"fusion":2*embed_dim+robot_embed_dim,"only_pc":2*embed_dim,"only_state":robot_embed_dim}
        super().__init__(observation_space,dimensions[ablation_mode])
        self.ablation_mode = ablation_mode
        if ablation_mode in ("fusion","only_pc"):
            def mlp():
                return nn.Sequential(nn.Linear(3,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),
                                     nn.Linear(128,embed_dim),nn.ReLU())
            def attn():
                return nn.Sequential(nn.Linear(embed_dim,32),nn.Tanh(),nn.Linear(32,1))
            self.mlp_in,self.mlp_out,self.attn_in,self.attn_out = mlp(),mlp(),attn(),attn()
        if ablation_mode in ("fusion","only_state"):
            self.state_encoder = nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,robot_embed_dim),nn.ReLU())

    def forward(self,observations):
        features = []
        if self.ablation_mode in ("fusion","only_pc"):
            for name in ("in","out"):
                f = getattr(self,"mlp_"+name)(observations["point_cloud_"+name].float())
                a = torch.softmax(getattr(self,"attn_"+name)(f),dim=1)
                features.append((f*a).sum(dim=1))
        if self.ablation_mode in ("fusion","only_state"):
            features.append(self.state_encoder(observations["robot_state"].float()))
        return torch.cat(features,dim=1)


class ArchivedPolicy:
    def __init__(self,path,env):
        self.path = Path(path)
        with zipfile.ZipFile(path) as z:
            metadata = json.loads(z.read("data"))
            mode = metadata["policy_kwargs"]["features_extractor_kwargs"]["ablation_mode"]
            # Deserialize only tensor weights, not archived executable cloudpickle.
            weights = torch.load(io.BytesIO(z.read("policy.pth")),map_location="cpu",weights_only=True)
        self.model = MultiInputActorCriticPolicy(env.observation_space,env.action_space,lambda _:3e-4,
                        features_extractor_class=AblationAttentionPointNet,
                        features_extractor_kwargs=dict(ablation_mode=mode),
                        net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=nn.Tanh,ortho_init=False)
        self.model.load_state_dict(weights,strict=True)
        self.model.eval()
        self.metadata = dict(path=str(self.path),sha256=sha256(path),mode=mode,
                             archived_training_steps=metadata["num_timesteps"])

    def raw_action(self,obs):
        tensors = {k:torch.from_numpy(v[None]) for k,v in obs.items()}
        with torch.no_grad():
            return self.model(tensors,deterministic=True)[0][0].numpy()

    def predict(self,obs,env):
        return np.clip(self.raw_action(obs),-1,1)


class NoAvoidance:
    metadata = {"description":"Zero camera displacement on the same static point cloud"}
    def predict(self,obs,env):
        return np.zeros(2,dtype=np.float32)


class GreedyViewSearch:
    metadata = {"description":"One-step 16-direction geometric search for platform diagnostics"}
    def predict(self,obs,env):
        best_score,best = env.score,np.zeros(2,dtype=np.float32)
        for angle in np.linspace(0,2*np.pi,16,endpoint=False):
            action = env.action_step*np.array([np.cos(angle),np.sin(angle)])
            delta,q,ok = env.candidate(action)
            if not ok or self_collision(q):
                continue
            score = classify_and_score(env.points,env.xy+delta,env.scene.target,q)[1]
            if score < best_score:
                best_score,best = score,action
        return best


LABELS = {"fusion":"MM-PPO 融合输入", "pointcloud":"仅点云", "robotstate":"仅机器人状态",
          "greedy":"一步几何搜索", "no_avoidance":"未规避"}


def policy_set(config,env):
    root = Path(config["archive_root"])/"model"/"RL_Point"
    return {"fusion":ArchivedPolicy(root/"ppo_AnzhenRobot_20.zip",env),
            "pointcloud":ArchivedPolicy(root/"ppo_AnzhenRobot_21.zip",env),
            "robotstate":ArchivedPolicy(root/"ppo_AnzhenRobot_22.zip",env),
            "greedy":GreedyViewSearch(),"no_avoidance":NoAvoidance()}
