import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import MiniMetroRLEnv

def make_env(seed: int = 42):
    def _init():
        env = MiniMetroRLEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

if __name__ == "__main__":
    logdir = "runs/ppo_minimetro"
    os.makedirs(logdir, exist_ok=True)

    venv = DummyVecEnv([make_env(42)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.02,
        vf_coef=0.5,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=None,
    )

    model.learn(total_timesteps=300_000)

    model.save(os.path.join(logdir, "model"))
    venv.save(os.path.join(logdir, "vecnormalize.pkl"))
    print("Saved to:", logdir)