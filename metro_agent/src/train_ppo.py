import math
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan, VecMonitor
from torch.nn import Tanh

from src.rl_env import MiniMetroRLEnv


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "model" / "ppo_minimetro"
TB_DIR = RUN_DIR / "tb"
CKPT_DIR = RUN_DIR / "checkpoints"
MODEL_PATH = RUN_DIR / "model.zip"
NORM_PATH = RUN_DIR / "vecnormalize.pkl"


def make_env(seed: int = 42):
    def _init():
        env = MiniMetroRLEnv()
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(seed: int = 42, load_norm: bool = True):
    base = DummyVecEnv([make_env(seed)])
    base = VecCheckNan(base, raise_exception=True)
    base = VecMonitor(base)

    if load_norm and NORM_PATH.exists():
        venv = VecNormalize.load(str(NORM_PATH), base)
        print(f"Loaded VecNormalize stats from {NORM_PATH}")
    else:
        probe_env = MiniMetroRLEnv()
        fps = 1000.0 / probe_env.decision_interval_ms   # 250ms -> 4 decisions/sec
        half_life_seconds = 20.0
        gamma = math.exp(math.log(0.5) / (fps * half_life_seconds))

        venv = VecNormalize(
            base,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
        )
        print(f"Created new VecNormalize with gamma={gamma:.6f}")

    return venv


def build_new_model(venv):
    policy_kwargs = dict(
        activation_fn=Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=venv.gamma if hasattr(venv, "gamma") else 0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=str(TB_DIR),
        device="auto",
    )
    return model


def save_all(model, venv, tag: str = "latest"):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if tag == "latest":
        model.save(str(MODEL_PATH.with_suffix("")))
        venv.save(str(NORM_PATH))
        print(f"Saved latest model to {MODEL_PATH}")
        print(f"Saved latest vec stats to {NORM_PATH}")
    else:
        model.save(str(CKPT_DIR / f"model_{tag}"))
        venv.save(str(CKPT_DIR / f"vecnormalize_{tag}.pkl"))
        print(f"Saved checkpoint: model_{tag}.zip / vecnormalize_{tag}.pkl")


if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    seed = 42
    train_interval = 100_000
    checkpoint_freq = 50_000
    hard_save_every = 100_000

    # --- env ---
    venv = build_vec_env(seed=seed, load_norm=True)

    # --- model ---
    if MODEL_PATH.exists():
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(str(MODEL_PATH), env=venv, device="auto")
    else:
        print("No existing model found. Creating a new PPO model.")
        model = build_new_model(venv)

    # --- callback ---
    callback = CheckpointCallback(
        save_freq=max(1, checkpoint_freq // venv.num_envs),
        save_path=str(CKPT_DIR),
        name_prefix="ppo_metro",
    )

    try:
        next_hard_save = model.num_timesteps + hard_save_every

        while True: #model.num_timesteps < 300000:
            print(f"\nTraining step starting at {model.num_timesteps} timesteps")
            model.learn(
                total_timesteps= train_interval,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=True,
            )

            save_all(model, venv, tag="latest")

            if model.num_timesteps >= next_hard_save:
                save_all(model, venv, tag=str(model.num_timesteps))
                next_hard_save += hard_save_every

    except KeyboardInterrupt:
        print("\nTraining error")

    finally:
        print("Saving latest")
        save_all(model, venv, tag="latest")
        venv.close()