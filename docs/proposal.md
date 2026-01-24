---
layout: default
title:  Proposal
---

# Summary
The purpose of this project is to build an AI agent that designs and manages diverse metro networks with the objective of making the most efficient and feasible connections. We are using a Mini Metro style environment to minimize congestion under growing passenger demand. The agent receives as input a representation of the current system state, including the existing network map, passenger distributions and queues at stations, train capacities, and system constraints. Based on this information, the agent outputs network-level actions such as adding or removing connections, extending or rerouting lines, or reallocating limited resources. Our main objective for this project is to learn how to manage an AI agent and different heuristics in a video game that reflects real-life problems, and provides feedback on circumstances that aren't completely unknown to the real world.

# Project Goals
Minimum Goal:
• An agent that replaces the connections to minimize the congestion of routes.
• Evaluation of changes on the agent depending on the heuristic.

# Algorithms

# Evaluation

__Quantitative Evaluation:__ We will evaluate policies on a suite of scenarios, fixed maps and random seeds, using the main metric, survival time, which counts the timesteps until failure. A second metric is the congestion rate, which is the fraction of timesteps any station queue exceeds a threshold. The last two metrics that impact the first two are the average passenger waiting time per game, and the total passengers delivered per game. For baselines, we will have random actions, a greedy heuristic that prioritizes to lessen the most congested station, and a simple heuristic that will focus on allocating stations. 

__Qualitative Analysis:__ To make sure our project is working, we will visually demonstrate the building of the stations and present the common types of failure and successful modes. This will provide the station queue trajectories for each method to verify that improvements correspond to behaviors rather than outputs.

# Tools

We will be using an open source Mini Metro Python simulation repositary. 
This will work as a base environment framework, and allow us to set up a metro network simulation while we focus on reinforcement learning. We will also be using ChatGPT as an assistive tool to help us analyze and identify features in the open source code for a better understanding, and for us to be able to modify it for our project. 
