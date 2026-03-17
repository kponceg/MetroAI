---
layout: default
title: Final Report
---



## Project Summary

We implemented a reinforcement learning agent to play Mini Metro in a custom simulation environment. The environment exposes a Gym-like API that allows the agent to observe the transit system state and choose high-level network editing actions from the custom game engine. These actions include creating, expanding, and replacing transit lines in order to manage congestion under growing passenger demand.

Our goal was to train a reinforcement learning agent that learns how to maintain a functional metro network over time while minimizing passenger waiting and avoiding station overflow failures. Designing metro networks is difficult because congestion evolves dynamically, and small structural changes in the network can have delayed system-wide effects. This makes the problem well suited for reinforcement learning, where an agent can learn strategies through interaction with the environment.

We train a **Proximal Policy Optimization (PPO)** agent to learn a control policy that maximizes survival time while minimizing passenger waiting and overcrowding. During development we also analyze how action design, reward structure, and invalid actions affect system stability and learning behavior.

For evaluation we compare the trained PPO agent against a random baseline policy that selects actions uniformly. By comparing these two policies we can determine whether the learned policy improves system performance beyond naive decision making.

## Approach

## Evaluation

## Resources Used

We used the following resources during development:

Libraries
- Stable-Baselines3 (PPO implementation)
- Gymnasium environment API
- NumPy and Matplotlib
- PyGame for visualization

References
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. Proximal Policy Optimization Algorithms.
-- https://arxiv.org/abs/1707.06347

- Stable-Baselines3 Documentation
-- https://stable-baselines3.readthedocs.io

- Gymnasium Documentation
-- https://gymnasium.farama.org

- Mini Metro Simulation Engine
-- https://github.com/autosquash/python_mini_metro_extended

We also used ChatGPT for assistance in understanding reinforcement learning algorithms, debugging environment integration issues, and helping explain code behavior in the final report.
