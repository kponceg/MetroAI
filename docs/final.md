---
layout: default
title: Final Report
---
[demo video](metro_agent_play.mp4)


## Project Summary

We implemented a reinforcement learning agent to play Mini Metro in a custom simulation environment. The environment exposes a Gym-like API that allows the agent to observe the transit system state and choose high-level network editing actions from the custom game engine. These actions include creating, expanding, and replacing transit lines in order to manage congestion under growing passenger demand.

Our goal was to train a reinforcement learning agent that learns how to maintain a functional metro network over time while minimizing passenger waiting and avoiding station overflow failures. Designing metro networks is difficult because congestion evolves dynamically, and small structural changes in the network can have delayed system-wide effects. This makes the problem well suited for reinforcement learning, where an agent can learn strategies through interaction with the environment.

We train a **Proximal Policy Optimization (PPO)** agent to learn a control policy that maximizes survival time while minimizing passenger waiting and overcrowding. During development we also analyze how action design, reward structure, and invalid actions affect system stability and learning behavior.

For evaluation we compare the trained PPO agent against a random baseline policy that selects actions uniformly. By comparing these two policies we can determine whether the learned policy improves system performance beyond naive decision making.

## Approach
### Setup
We formualte this as a reinforcement learnig problem in a custom Mini Metro game engine, using Gymnasium environment. At each deision step, the agent observes the current metro network, choose a high level editing action, and then the simulator advances forward before the next decision is made. In our implementation, the internal simulatior runs with 16ms, while the agent only acts every 250ms, which aimed to reduces the frequency of decisions and make training stable. In addition, the game terminates when episodes reaches to 4000 steps, and when a station remians at capacity long enough to trigger overflow failture. 

To make sure the problem is more learnable, we do no expose the raw game UI controls to the agent. Instead, we simplify the action space into five high-level operations, implemented as `MultiDiscrete([total number actions, total number of stations, total number of stations])`. 

**Actions:**
* Do Nothing
* Creating a new path
* Expanding a path
* Removing and replacing a path
* Guided expansion action that connects a high demand station to an existing path

### Observation Space
This observation is a compact feature vector. It contains three types of features: station-level, path-level, and global features. 

For each station, the agent observes information such as the queue ratio, normalized occupancy, whether the station is already on a path, as well as the distribution of passenger destinations at that station. 

For each path, the observation includes information such as whether the path exists, normalized origin and destination indices, normalized path length, normalized path load, and queue and degree information for both endpoints. 

Global features summarize the network status through the number of active paths, remaining path budget, passenger generation progress, maximum queue ratio, average queue ratio, and the proportion of unserved stations. 
   
### Improvements
We first started by verifying that the environment produced sensible signals under random interaction. Our smoke_test evaluation rolled out random actions and logged metrics such as survival time, congestion, maximum queue, total waiting passenager, and cumulative reward return. This gave us a baseline for how badly the system behaves without learning and also helped us confirm that the environment was working as expected.

After establishing that baseline, the first major improvement was simplifying the action sapce into high level network eddition operatiors instead of directly exposing low-level controls. 
The second improvement was using a compact observation vector that explicitly describes the system, where includes congestion, converage, and path structure.
Next we improve our reward system. Rather than only rewarding the score, we added several additional signals to ensure the agent receive denser feedback during training. These include reward on survial longer, increasing score, reducing waiting passengers, and lowering the maximum queue; We also penalities the agent for invalid actions and terminal failture.

Over time, it became clear that one of the main challenges was not just survive longer, but learning to build useful routes before congestion becomes irreversible. We also identified that some parts of the observation may still be more complex than necessary, and one future improvment is to reduce less useful structureal features.

### Baseline approach
The baseline approaches was to evaluate the environment under random actions. In this setup, the agent samples uniformly from the action sapce but does not use any learned strategy. This baseline is useful becasue it establishes a lower bound; if PPO can not outperform random editing, then it is not actually learning meaningful control. 

The main advantage of this baseline is that it requires to training and provides a simple reference point for comparison. However, it does not adapt to congestion, oftern wastes the resources, and frequently produces invalid edits.

### Reward Design
The reward function is one of the most critical components of our method. At each step, the agent receives a small survival reward; it receives additional rewards when the game score increases; and it also receives rewards when the system reduces the total number of waiting passengers or decreases the maximum queue length. 

At the same time, the agent is penalized for situations such as excessively high total waiting numbers, severe congestion, maintaining too many paths, reaching warning-level congestion, excessively high action costs, invalid actions, or terminal failures. We have also incorporated subtle incentives for specific actions: creating new paths yields a reward, extending paths yields a higher reward, while removing paths results in a penalty. 

This design not only encourages the agent to prolong its survival but also motivates it to actively alleviate congestion and avoid wasting actions on invalid or harmful edits.

However, we have observed that simply adding rewards for various scenarios may increase the overall complexity of the system. Therefore, in our latest training, we removed some rewards and retained only those related to score, survival time, total waiting passenger and termination. Unfortunately, we didn’t have enough time to verify whether this approach, reduce the complxity of the reward system, could lead to better learning.

### PPO
We train the agent using Proximal Policy Optimization (PPO) (Schulman, J)

We used PPO from Stable-Baselines3 with an MLP poliocy. PPO is well-suited for this task because it typically performs more consistently than value-based methods in environments with shaping rewards and structured observation. Since the task requires sequential decision-making with delayed consequences, PPO’s pruning strategy updates and advantage-based learning provide a more reliable training process than purely greedy action-value learners. 

At each PPO iteration:
Collect n_steps transitions \( (O_t, A_t, r_t, O_{t+1}) \) using the current policy compute advantage estimates $\hat{A}_t$ optimize the PPO losses for n_epochs epochs over minibatches of size batch_size

PPO uses a clipped policy update. Define the probability ratio:

$$
\rho_t(\theta)=\frac{\pi_\theta(A_t \mid O_t)}{\pi_{\theta_{\text{old}}}(A_t \mid O_t)}
$$

$$
\mathcal{L}^{\text{CLIP}}(\theta)=\mathbb{E}_{t}\left[
\min\Big(
\rho_t(\theta)\hat{A}_t,\;
\text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\Big)
\right]
$$

- $( O_t )$ : observation in step $( t )$
- $( A_t )$: action in step \( t \)
- $( r_t )$: reward in step \( t \)
- $( \theta \)$: current policy
- $( \pi_\theta(A_t \mid O_t) \)$: probability that the current policy assigns to taking action $\( A_t \)$ given observation $\( O_t \)$
- $( \rho_t(\theta) \)$: ratio between new and old action probability
- $( \mathcal{L}^{\text{CLIP}} \)$: PPO's clipped policy objective
- $( \text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon) \)$: clamps the ratio into $\( [1-\epsilon,1+\epsilon] \)$ to prevent overly large policy updates
- $( \hat{A}_t )$: advantage estimate at time $( t )$

Our PPO model uses MlpPolicy with separate policy and value networks, each with two hidden layers of size 256 and Tanh activations. The main hyperparameters are: 
- `learning_rate = 3e-4`, 
- `n_steps = 2048`, 
- `batch_size = 256`, 
- `n_epochs = 10`, 
- `gae_lambda = 0.95`, 
- `clip_range = 0.2`, 
- `ent_coef = 0.03`,
- `vf_coef = 0.5`
- `gamma` = math.exp(math.log(0.5) / (fps * half_life_seconds))

One notebale design choice in our training setup is how we set the gamma factor. Rather than manually selecting a default value, we derived the **gamma** value based on a 20-second reward half-life and the environment’s decision rate. Since the agent performs an action every 250 milliseconds—equivalent to four decisions per second—the gamma value was calculated based on this timescale.

The training process will continue until we end it manually, but we do save checkpoints every 50,000 steps and also hard save of full model and normalization statistics every 100,000 steps. This setup lets us resume long-running training while keeping evaluation consistent with the training distribution.

The reward function is one of the most critical components of our PPO architecture. In each iteration step, the agent receives a base survival reward; it receives an additional reward when the score improves; and it receives yet another reward when the total waiting time or maximum queue length decreases. At the same time, the agent is penalized for situations such as excessive total waiting time, severe congestion, maintaining too many paths, performing high-cost operations, or selecting invalid operations. Terminal failures result in significant negative rewards. This design aims to balance short-term congestion control with long-term network quality, while preventing agents from adopting behaviors that appear simple but are eventually not usefull.

Within our framework, the main advantage of PPO is that it can achieve stable policy improvement under complex reward functions and constrained action spaces. It can learn from normalized observation data and adjust its behavior over the course of many episodes. Its main limitations are that it is highly sensitive to the quality of observations and reward design. If the state representation omits critical information, or if the reward places excessive emphasis on shaping terms, the policy may converge to incorrect behaviors.

## Evaluation

To see how well our reinforcement learning approach actually performed, we compared our trained PPO agent to a simple random action baseline in our custom MiniMetro environment. We kept everything consistent between the two, same setup, the same timestep limit, and the same observations, so the comparison would be fair. The PPO agent was trained using Stable-Baselines3, with a reward function that considers factors like queue sizes, how many passengers are waiting, survival bonuses, and penalties for invalid actions.

For our evaluation, we focused on a few key metrics:
- Total waiting passengers over time
- Maximum queue length
- Cumulative return over time
- Reward history
- Invalid action rate

These metrics give us a good overall picture. It shows us how the system is performing and also how well the agent is interacting with the environment.

<img src="myplot1.png" width="300" height="300">

**Total Waiting Passengers**

The initial plot presents the aggregate waiting time, serving as an indicator of the agent's efficacy in managing passenger flow throughout the observed period. The data reveal that both agents exhibit an upward trend in total waiting time, an expected pattern given the environment's escalating difficulty. However, the evaluation indicates that the PPO agent generally has a longer total waiting time than the random model.

This suggests that, despite learning a policy, the PPO agent is not effectively minimizing congestion. A conclusion we reached is that the reward function tends to develop a bias toward actions that do not directly affect the flow of waiting passengers. This shows a common challenge in reinforcement learning, which is a misalignment between reward design and the expected system behavior.

<img src="myplot2.png" width="300" height="300">

**Maximum Queue Length**

Maximum queue length indicates how severe congestion gets at its worst point in the system, which is important because high values usually mean the system is close to failing. Looking at the results, both the PPO agent and the random agent behave pretty similarly. In both cases, the maximum queue keeps increasing over time and eventually reaches a similar critical level. The PPO agent doesn’t really do a better job than random at keeping the peak congestion levels under control.

This suggests that the agent isn’t effectively focusing on the stations that need the most attention or preventing bottlenecks from forming. Ideally, a stronger policy would keep those peaks lower or at least delay when they happen, but we don’t really see that here.

<img src="myplot3.png" width="300" height="300">

**Invalid Action Rate**

Invalid actions are when the agent tries to do something that isn’t allowed, like making a path change that doesn’t make sense. When this happens a lot, it usually means the agent doesn’t really understand how the environment works yet. This is actually where PPO improves the most compared to random. The random agent keeps making invalid moves most of the time, while the PPO agent brings that down a lot to about 30% by the end.

So even though PPO isn’t necessarily better at reducing congestion, it clearly learns what actions are valid and avoids doing obviously bad or impossible things. That’s a good sign that the agent is starting to understand the structure of the environment, even if it hasn’t fully figured out how to optimize performance yet.

<img src="myplot4.png" width="300" height="300">

**Cumulative Return over time**

Cumulative reward indicates how well the agent is performing overall under the reward system we designed. From the graph, the PPO agent consistently outperforms the random agent, indicating it is actually learning and optimizing for the rewards we provided. At the same time, both curves continue to decline over time, indicating that the environment becomes harder and penalties start to outweigh any positive rewards.

The important part here is that the PPO agent is learning, but it’s learning exactly what we told it to optimize, not necessarily what we actually care about. In this case, that means it follows the reward function well, but that reward function doesn’t fully reflect the real goal of reducing congestion. That gap between the reward and the true objective is one of the biggest limitations of our approach.

<img src="myplot5.png" width="300" height="300">

**Reward History**

The reward history shows how much reward the agent gets at each step, which helps us understand how stable its behavior is and how it’s learning over time. Looking at the graph, both agents have pretty noisy rewards, with random spikes at some points. The PPO agent looks a bit more consistent at the beginning. Still, over time, both agents end up with sharp drops in reward near the end of the episode, which usually means the system is failing, for example, overcrowded stations.

Overall, this tells us a few things. The environment itself is pretty unpredictable and hard to control; the agent never fully learns how to avoid those failure states, and by the end, most of the rewards are actually penalties rather than positive gains.

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
