---
layout: default
title: Status
---

## Project Summary
We implement a reinforcement learning agent to play Mini Metro in a custom simulation environment. The environment exposes a Gym-like API to help the agent to oberve the transit system state and choose high level network edit actions from the custom game engine. The agent selects high-level network edit actions that includes creating, expanding, and replacing transit lines to manage congestion under growing passenger demand.
We train a Proximal Policy Optimization (PPO) agent to learn a control policy that maximizes survival time while minimizing passenger waiting and overcrowding. As well as analyzing how action design and reward structure affect system performance and stability.

## Approach
Our main algorithm to train an agent to play Mini Metro is Proximal Policy Optimization (PPO). This is an on policy algorithm that learns a neural network policy to approximate an optimal control strategy for metro network editions.

### Markov Decision Process
We model the game as an Markov Decision Process. At each timestep "t", the agent receives an observation (O_t), selects an action (A_t), and the simulator advance by a fixed time step (detlat)

### State/Observation
We use a compact feature vector as the observation:

O_t ​= [q_1​, …, q_N, d_1​, …, d_N​, P]
- q_i: the stataion "i" current queue/occupation
- d_i: the station "i" dgree (how many paths include the this station)
- P: list of [number of paths, number of stations, time]

### Action Space
We use discrete high level operatiors, implemeted as parameteried acitons such as (action id, station i , station j). Parameters are choosen by the agent.
- action 0: Do nothing
- action 1: create and connect station_i to station_j
- action 2: expand the network from station_i to station_j
- action 3: remove a low utility path, then create a new connection between ,ost congested station_i and station_j
- action 4: connect a least connected but high demand station_i to station_j

### Reward 
We use reward to enovurages the policy to reduce crowding on stations and increase survial time by prevent overflow.

### PPO

**Code Methods**
- Environment: `rl_env.py` (`MiniMetroRLEnv`)
- Observation/state features: `rl_env.py` → `_get_obs()`
- Reward + termination: `rl_env.py` → `step()`
- Action operators: `rl_env.py` → `_apply_action()`

## Evaluation

## Remaining Goals and Challenges

## Resources Used
Python custom mini metro game engine: https://github.com/autosquash/python_mini_metro_extended 
