# space to pause
# d to show/hide panel
# esc to escape

import os
import sys
import numpy as np
import pygame

print("before torch")
from stable_baselines3 import PPO
print("after torch")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.config import Config
from src.rl_env import MiniMetroRLEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "python3.11agent", "ppo_minimetro")
MODEL_PATH = os.path.join(RUN_DIR, "model.zip")
NORM_PATH = os.path.join(RUN_DIR, "vecnormalize.pkl")



def build_eval_env(seed: int = 42):
    raw_env = MiniMetroRLEnv()
    raw_env.reset(seed=seed)

    venv = DummyVecEnv([lambda: raw_env])

    if not os.path.exists(NORM_PATH):
        raise FileNotFoundError(
            f"Missing VecNormalize stats: {NORM_PATH}\n"
            "Train PPO first so vecnormalize.pkl exists."
        )

    venv = VecNormalize.load(NORM_PATH, venv)
    venv.training = False
    venv.norm_reward = False

    try:
        obs = venv.reset(seed=seed)
    except TypeError:
        obs = venv.reset()

    return raw_env, venv, obs


def ensure_action_shape(action):
    a = np.asarray(action, dtype=np.int64)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 0:
        return np.array([[int(a), 0, 0]], dtype=np.int64)
    return a


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing PPO model: {MODEL_PATH}\n"
            "Train PPO first so model.zip exists."
        )
    pygame.init()

    pygame.display.set_caption("Mini Metro PPO Viewer")
    screen = pygame.display.set_mode((Config.screen_width, Config.screen_height))

    clock = pygame.time.Clock()

    raw_env, venv, obs = build_eval_env(seed=42)
    print("build eval env")

    model = PPO.load(MODEL_PATH, env=venv)
    print("model loaded")

    raw_env.engine.set_clock(clock)
    raw_env.engine.showing_debug = True

    paused = False
    done = False
    step_idx = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_d:
                    raw_env.engine.showing_debug = not raw_env.engine.showing_debug

        if not paused:
            action, _ = model.predict(obs, deterministic=True)
            action = ensure_action_shape(action)

            obs, reward, done_arr, infos = venv.step(action)

            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            r = float(reward[0]) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
            done = bool(done_arr[0]) if isinstance(done_arr, (list, tuple, np.ndarray)) else bool(done_arr)

            print(
                f"t={step_idx:4d} "
                f"action={np.asarray(action).tolist()} "
                f"reward={r:7.3f} "
                f"maxQ={info0.get('max_queue')} "
                f"totalW={info0.get('total_waiting')} "
                f"paths={info0.get('num_paths')} "
                f"invalid={info0.get('invalid_action', False)}"
            )

            step_idx += 1

        screen.fill((0, 0, 0))
        raw_env.engine.render(screen)
        pygame.display.flip()

        clock.tick(10)

    venv.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    print("hello")
    main()