"""Shared PointNet architecture for newly trained PPO, SAC, and TD3 policies.

The archived extractor is unchanged. Only the state encoder's input layer adapts
to the explicitly supported original6 and joint12 observation specifications.
The two point-cloud branches and fused output dimensions remain identical.
"""
from torch import nn
from .policies import AblationAttentionPointNet


class LearningPointNet(AblationAttentionPointNet):
    def __init__(self, observation_space, embed_dim=128, robot_embed_dim=64,
                 ablation_mode="fusion"):
        try:
            shape = observation_space.spaces["robot_state"].shape
        except (AttributeError, KeyError) as exc:
            raise ValueError("LearningPointNet requires a Dict with robot_state") from exc
        if shape not in ((6,), (12,)):
            raise ValueError("LearningPointNet supports only robot_state shapes (6,) and (12,)")
        if ablation_mode not in ("fusion", "only_pc", "only_state"):
            raise ValueError("Unsupported learning feature ablation mode")
        super().__init__(observation_space, embed_dim=embed_dim,
                         robot_embed_dim=robot_embed_dim, ablation_mode=ablation_mode)
        self.state_dim = shape[0]
        if self.state_dim != 6 and ablation_mode in ("fusion", "only_state"):
            self.state_encoder[0] = nn.Linear(self.state_dim, 64)

    def forward(self, observations):
        if observations["robot_state"].shape[-1:] != (self.state_dim,):
            raise ValueError("robot_state input dimension differs from the declared observation space")
        return super().forward(observations)
