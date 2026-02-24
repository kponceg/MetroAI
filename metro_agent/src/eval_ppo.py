import os
import numpy as np
import matplotlib.pyplot as plt

from src.rl_env import MiniMetroRLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    # keep with training/eval
    env = MiniMetroRLEnv()
    return Monitor(env)

def make_vec_env(seed, norm_path: str | None):
    base = DummyVecEnv([lambda: make_env()])
    if norm_path and os.path.exists(norm_path):
        venv = VecNormalize.load(norm_path, base)
        venv.training = False
        venv.norm_reward = False  # raw reward for comparability
    else:
        venv = base

    # reset with seed
    try:
        _ = venv.reset(seed=seed)
    except TypeError:
        _ = venv.reset()
    return venv

def ensure_action_shape(action):
    """Ensure action is shaped for DummyVecEnv with n_envs=1: (1, 3)."""
    a = np.asarray(action, dtype=np.int64)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 0:
        return np.array([[int(a), 0, 0]], dtype=np.int64)
    return a

def extract_op(action):
    a = np.asarray(action, dtype=np.int64)
    if a.ndim == 2:
        return int(a[0, 0])
    return int(a[0])

def run_episode(venv, policy_fn, max_steps=20000):
    obs = venv.reset()
    cum_ret = 0.0

    totalw, maxq, numpaths = [], [], []
    invalid, rew, cum = [], [], []
    op_hist = []
    rew_hist = []
    last_info = {}

    count_by_op = np.zeros(5, dtype=int)
    invalid_by_op = np.zeros(5, dtype=int)

    for t in range(max_steps):
        action = policy_fn(obs, venv)
        action = ensure_action_shape(action)

        op = extract_op(action)
        op_hist.append(op)

        obs, reward, done, infos = venv.step(action)

        r = float(reward[0]) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        last_info = info0

        cum_ret += r
        rew_hist.append(reward)

        totalw.append(float(info0.get("total_waiting", 0.0)))
        maxq.append(float(info0.get("max_queue", 0.0)))
        numpaths.append(float(info0.get("num_paths", 0.0)))

        inv_flag = int(bool(info0.get("invalid_action", False)))
        invalid.append(float(inv_flag))

        count_by_op[op] += 1
        invalid_by_op[op] += inv_flag

        rew.append(r)
        cum.append(cum_ret)

        if bool(done[0]) if isinstance(done, (list, tuple, np.ndarray)) else bool(done):
            break

    # print action summary
    print("count_by_op:", count_by_op)
    print("invalid_rate_by_op:", invalid_by_op / np.maximum(1, count_by_op))

    return {
        "total_waiting": totalw,
        "max_queue": maxq,
        "num_paths": numpaths,
        "invalid": invalid,
        "reward": rew,
        "cum_return": cum,
        "op_hist": op_hist,
        "last_info": last_info,
        "episode_return": cum_ret,
        "steps": len(rew),
        "rew_hist": rew_hist,
    }

def plot_two(series_a, series_b, title, ylabel, label_a="Random", label_b="PPO"):
    plt.figure()
    plt.plot(series_a, label=label_a)
    plt.plot(series_b, label=label_b)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def main():
    seed = 42
    logdir = "runs/ppo_minimetro"
    model_path = os.path.join(logdir, "model.zip")
    norm_path = os.path.join(logdir, "vecnormalize.pkl")

    if not os.path.exists(model_path) or not os.path.exists(norm_path):
        raise FileNotFoundError(
            "Missing PPO artifacts. Need:\n"
            f"  {model_path}\n"
            f"  {norm_path}\n"
            "train ppo first."
        )

    # --- build envs (same norm stats, same seed) ---
    env_random = make_vec_env(seed=seed, norm_path=norm_path)
    env_ppo = make_vec_env(seed=seed, norm_path=norm_path)

    # --- load model ---
    model = PPO.load(model_path, env=env_ppo)

    # --- policies ---
    def random_policy(obs, venv):
        return venv.action_space.sample()  # MultiDiscrete sample

    def ppo_policy(obs, venv):
        action, _ = model.predict(obs, deterministic=True)
        return action

    # --- run ---
    random_roll = run_episode(env_random, random_policy)
    ppo_roll = run_episode(env_ppo, ppo_policy)

    print("\n=== RANDOM ===")
    print("Episode return:", random_roll["episode_return"])
    print("Steps:", random_roll["steps"])
    print("Last info:", random_roll["last_info"])

    print("\n=== PPO ===")
    print("Episode return:", ppo_roll["episode_return"])
    print("Steps:", ppo_roll["steps"])
    print("Last info:", ppo_roll["last_info"])

    # --- plots: same figure per metric, two lines ---
    plot_two(random_roll["total_waiting"], ppo_roll["total_waiting"],
             "Total waiting over time", "total_waiting")

    plot_two(random_roll["max_queue"], ppo_roll["max_queue"],
             "Max queue over time", "max_queue")

    plot_two(random_roll["num_paths"], ppo_roll["num_paths"],
             "Number of paths over time", "num_paths")

    plot_two(random_roll["cum_return"], ppo_roll["cum_return"],
             "Cumulative return over time", "cumulative_return")

    plot_two(random_roll["rew_hist"], ppo_roll["rew_hist"],
             "Reward history over time", "rew_hist")

    # invalid action rate (running average)
    def running_avg(x):
        x = np.asarray(x, dtype=np.float64)
        return np.cumsum(x) / (np.arange(len(x)) + 1)

    plot_two(running_avg(random_roll["invalid"]), running_avg(ppo_roll["invalid"]),
             "Invalid action rate (running avg)", "invalid_rate")

    # option frequency comparison
    plt.figure()
    cnt_r = np.bincount(np.asarray(random_roll["op_hist"], dtype=int), minlength=4)
    cnt_p = np.bincount(np.asarray(ppo_roll["op_hist"], dtype=int), minlength=4)

    x = np.arange(4)
    w = 0.35
    plt.bar(x - w/2, cnt_r, width=w, label="Random")
    plt.bar(x + w/2, cnt_p, width=w, label="PPO")
    plt.title("Action op frequency")
    plt.xlabel("op (0=None, 1=Create, 2=Expand, 3=Remove)")
    plt.ylabel("count")
    plt.xticks(x, [0, 1, 2, 3])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
