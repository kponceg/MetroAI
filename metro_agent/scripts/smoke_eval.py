from src.rl_env import MiniMetroRLEnv
import matplotlib.pyplot as plt


env = MiniMetroRLEnv(
    dt_ms=16,
    congest_threshold=3,   # start LOW so you actually see congestion
    fail_threshold=10      # start LOW so episodes can en
)

obs, info = env.reset(seed=42)

congested_steps = 0
maxq_hist = []
totalw_hist = []

max_steps = 2000

for t in range(max_steps):
    # for now: do nothing (we're just testing metrics)
    obs, reward, done, _, info = env.step(action_id=0)

    maxq_hist.append(info["max_queue"])
    totalw_hist.append(info["total_waiting"])
    congested_steps += int(info["congested"])

    # print every 50 steps so you can see it evolve
    if t % 50 == 0:
        print(f"t={t:4d} maxQ={info['max_queue']} totalW={info['total_waiting']} congested={info['congested']}")

    if done:
        break

survival_time = t + 1
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