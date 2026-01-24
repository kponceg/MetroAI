---
layout: default
title:  Proposal
---

## Summary
The purpose of this project is to build an AI agent that designs and manages diverse metro networks with the objective of making the most efficient and feasible connections. We are using a Mini Metro style environment to minimize congestion under growing passenger demand. The agent receives as input a representation of the current system state, including the existing network map, passenger distributions and queues at stations, train capacities, and system constraints. Based on this information, the agent outputs network-level actions such as adding or removing connections, extending or rerouting lines, or reallocating limited resources. Our main objective for this project is to learn how to manage an AI agent and different heuristics in a video game that reflects real-life problems, and provides feedback on circumstances that aren't completely unknown to the real world.

## Project Goals

**Minimum Goal:**
- An agent that replaces the connections to minimize the congestion of routes.
- Evaluation of changes on the agent depending on the heuristic.

**Realistic Goal:**
- Build ontop the environment, adding resource mechanics.
- Train a robust RL angent that outperform stroing heuristics.
- Optimatize the queue/waiting time and survial time.

**Moonshot Goal:**
- Achieve strong generalization to unseen layouts and demand/resource schedules by using a graph-based or hierarchical policy.
- Implement a resource reward system. Where after surviving a fixed interval the agent will be reward a locomotive and choose among a set of resource (e.g., new line, carriage, interchange or tunnel), and learn a policy that optimizes both network expansion and resource allocation under this decision space.

## Algorithms
We plan on using model-free, on-policy deep reinforcement learning such as PPO with neural function approximator.
We also considering using GNN algorithm.

## Evaluation

__Quantitative Evaluation:__ We will evaluate policies on a suite of scenarios, fixed maps and random seeds, by using the main metric, survival time, which counts the timesteps until failure. A second metric is the congestion rate, which is the fraction of timesteps any station queue exceeds a threshold. The last two metrics that are relative to the first two are the average passenger waiting time per game, and the total passengers delivered per game. For baselines, we will have random actions that select uniformly, a greedy heuristic that prioritizes to lessen the most congested station, and a simple heuristic that will focus on allocating stations. We will train agents under a fixed budget of training sessions, maximum timesteps per game, and number of random seeds. We expect for RL to improve waiting time and congestion relative to heuristics, and our goal is to measure how design choices affect performance.

__Qualitative Analysis:__ To make sure our project is working, we will visually demonstrate the building of the stations and present the common types of failure and successful modes. This will provide the station queue trajectories for each method to verify that improvements correspond to behaviors rather than outputs. 

## Tools

We will be using an open source Mini Metro Python simulation repository. 
This will work as a base environment framework, and allow us to set up a metro network simulation while we focus on reinforcement learning. We will also be using ChatGPT as an assistive tool to help us analyze and identify features in the open source code for a better understanding, and for us to be able to modify it for our project. 
