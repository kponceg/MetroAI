---
layout: default
title: Status
---

## Project Summary

We implemented a reinforcement learning agent to play Mini Metro in a custom simulation environment. The environment exposes a Gym-like API to help the agent to oberve the transit system state and choose high level network edit actions from the custom game engine. The actions include creating, expanding, and replacing transit lines to manage congestion under growing passenger demand. We train a Proximal Policy Optimization (PPO) agent to learn a control policy that maximizes survival time while minimizing passenger waiting and overcrowding. As well as analyzing how action design and reward structure affect system performance and stability.

## Approach

Our main algorithm to train an agent to play Mini Metro is Proximal Policy Optimization (PPO). This is an on policy algorithm that learns a neural network policy to approximate an optimal control strategy for metro network editions.

### Markov Decision Process
We model the game as an Markov Decision Process. At each timestep $t$, the agent receives an observation ($O_t$), selects an action ($A_t$), and the simulator advance by a fixed time step (detla t)

### State/Observation
We use a compact feature vector as the observation:

$O_t$ ​= $[q_1​, …, q_N, d_1​, …, d_N​, P]$
- $q_i$: the stataion "i" current queue/occupation
- $d_i$: the station "i" dgree (how many paths include the this station)
- $P$: list of [number of paths, number of stations, time]

### Action Space
We use discrete high level operatiors, implemeted as parameteried acitons such as (action id, $station_i$ , $station_j$). Parameters are choosen by the agent.
- action 0: Do nothing
- action 1: create and connect $station_i$ to $station_j$
- action 2: expand the network from $station_i$ to $station_j$
- action 3: remove a low utility path, then create a new connection between most congested $station_i$ and $station_j$
- action 4: connect a least connected but high demand $station_i$ to $station_j$

### Reward 
We use reward to enovurages the policy to reduce crowding on stations and increase survial time by prevent overflow.

### PPO
We train the agent using Proximal Policy Optimization (PPO) 
$q_i$

PPO uses a clipped policy update. Define the probability ratio:
```math
p_t(\theta)=\frac{{\pi_\theta}(a_s | s_t)}{{\pi_\theta}{\text{old}}(a_s | s_t)}
```

**Code Methods**
- Environment: `rl_env.py` (`MiniMetroRLEnv`)
- Observation/state features: `rl_env.py` → `_get_obs()`
- Reward + termination: `rl_env.py` → `step()`
- Action operators: `rl_env.py` → `_apply_action()`

## Evaluation


## Remaining Goals and Challenges

Our current prototype successfully trains a PPO agent and compares it to a random baseline, but evaluation remains limited. We plan to conduct multi-seed evaluations, compare our agent against stronger heuristic baselines, and systematically analyze the impact of reward design and action structure. We also aim to refine the action space to better capture realistic network planning decisions.

A key challenge is how delayed our credit assignment is. Structural network edits may have long-term effects that are difficult for PPO to learn. Additionally, the MultiDiscrete action space can produce invalid or low impact actions, slowing convergence. To address this, we plan to explore improved reward shaping and possibly action masking.


## Resources Used

We used a Stable-Baselines3 (PPO), Gymnasium, and an open-source Mini Metro simulation repository as the foundation of our environment. We also used ChatGPT as a source to explain the source code for us to understand, and helped us debug environment integration issues.

Stable-Baslines3:
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

PPO Algorithms:
- https://arxiv.org/abs/1707.06347

Gymnasium:
- https://gymnasium.farama.org

Python custom mini metro game engine:
- https://github.com/autosquash/python_mini_metro_extended 
