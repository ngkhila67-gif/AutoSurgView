from datetime import datetime
import json
import random
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from .data import ROOT,configuration,scene_index,load_scene,sha256
from .evaluate import make_env
from .policies import AblationAttentionPointNet
from .native_backend import provenance


def train(steps=2048,mode="fusion",seed=20261001,smoke=False):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config=configuration()
    rows=[r for r in scene_index()["scenes"] if r["new_training_split"]=="train" and r["duplicate_of"] is None]
    if smoke:
        rows=rows[:4]
    scenes=[load_scene(r) for r in rows]
    out=ROOT/"outputs"/(("training_smoke_" if smoke else "training_")+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    env=DummyVecEnv([lambda:Monitor(make_env(scenes,config),filename=str(out/"monitor.csv"))])
    env=VecNormalize(env,norm_obs=False,norm_reward=True,gamma=.99)
    rollout=256 if smoke else 2048
    model=PPO("MultiInputPolicy",env,learning_rate=3e-4,n_steps=rollout,batch_size=64 if smoke else 256,
              n_epochs=2 if smoke else 10,gamma=.99,gae_lambda=.95,clip_range=.2,
              ent_coef=0,vf_coef=.5,max_grad_norm=.5,seed=seed,device="cpu",verbose=1,
              policy_kwargs=dict(features_extractor_class=AblationAttentionPointNet,
                                 features_extractor_kwargs=dict(ablation_mode=mode)))
    model.learn(total_timesteps=steps)
    model.save(out/"new_policy.zip")
    env.save(out/"reward_normalization.pkl")
    # Verify save/reload inference on a separately designated validation scene.
    validation_row=next(r for r in scene_index()["scenes"] if r["new_training_split"]=="validation")
    validation_env=make_env([load_scene(validation_row)],config)
    obs,_=validation_env.reset(seed=seed+1)
    reloaded=PPO.load(out/"new_policy.zip",device="cpu")
    action,_=reloaded.predict(obs,deterministic=True)
    if not np.isfinite(action).all():
        raise AssertionError("Saved model failed reload prediction")
    record=dict(purpose="training_pipeline_smoke_test" if smoke else "new_training_run_not_yet_evaluated",
                requested_steps=steps,actual_steps=model.num_timesteps,seed=seed,mode=mode,
                num_training_scenes=len(rows),training_scene_ids=[r["scene_id"] for r in rows],
                config=config,native_backend=provenance(),validation_inference_scene=validation_row["scene_id"],
                reload_prediction=action.tolist(),model_sha256=sha256(out/"new_policy.zip"),
                note="No claim of convergence. Test partition was not used. Single validation inference checks loading only.")
    (out/"training_manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    env.close()
    return out
