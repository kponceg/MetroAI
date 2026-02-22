from src.rl_env import MiniMetroRLEnv
import matplotlib.pyplot as plt


env = MiniMetroRLEnv(
    dt_ms=16,
    congest_threshold=3,   # start LOW so you actually see congestion
    fail_threshold=10,# start LOW so episodes can en
    max_stations=20,
    max_episode_steps=2000
)

obs, info = env.reset(seed=42)

congested_steps = 0
maxq_hist = []
totalw_hist = []
reward_hist = []

max_steps = 2000


print("Before:", info)
# check the system actually connect
obs, r, terminated, truncated, info = env.step(1)
print("After action=1:", info)


pm = env.engine.path_manager
c = env.engine._components

for t in range(max_steps):
    # for now: random select an action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    maxq_hist.append(info["max_queue"])
    totalw_hist.append(info["total_waiting"])
    congested_steps += int(info["congested"])
    reward_hist.append(reward)

    # print every 50 steps so you can see it evolve
    if t % 50 == 0:
        print(f"t={t:4d} maxQ={info['max_queue']} totalW={info['total_waiting']}"
              f" congested={info['congested']} invalid={info.get('invalid_action', False)} r = {reward:.3f}")

    if terminated or truncated:
        break

survival_time = t + 1 # 2000?
congestion_rate = congested_steps / survival_time

print("Survival time:", survival_time)
print("Congestion rate:", congestion_rate)
print("Final max queue:", maxq_hist[-1])
print("Final total waiting:", totalw_hist[-1])

plt.figure()
plt.plot(maxq_hist)
plt.title("Max station queue over time")
plt.xlabel("Timestep")
plt.ylabel("Max queue")
plt.show()

plt.figure()
plt.plot(totalw_hist)
plt.title("Total waiting passengers over time")
plt.xlabel("Timestep")
plt.ylabel("Total waiting")
plt.show()


plt.figure()
plt.plot(reward_hist)
plt.title("Reward over time (random)")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.show()