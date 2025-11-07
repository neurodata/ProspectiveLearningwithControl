import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from src.utils import reward_simulate, simulate_data_raw
from src.utils import path_opt, compute_normalized_future_rewards
from src.rl import FQILearner

from collections import deque
import random
from tqdm import tqdm

base_reward = 10.0
map_name = "short"
decay_rate = 0.6
reward_period = 10 
session_duration = 500000
rewards_in_period = []

total_steps = session_duration
for period_start in range(0, total_steps, reward_period):
    period_end = min(period_start + reward_period, total_steps)
    steps_in_period = np.arange(period_start, period_end)
    rewards_in_period.extend(base_reward * (decay_rate ** (steps_in_period - period_start)))

pattern = [1,1,1,1,1,1,1,2,3,4,5,5,5,5,5,5,5,4,3,2]
optimal_states = np.array(pattern*(session_duration // len(pattern))).reshape(-1,1)

class ForagingEnv:
    def __init__(self, rewards_in_period):
        self.rewards_in_period = rewards_in_period
        self.reset()

    def reset(self):
        self.current_step = 0
        self.state = 1  # Start at state 1
        return self.state
    
    def get_next_state(self, action):
        next_state = self.state + action
        if self.state == 0 and next_state < 0:
            next_state = 0
        if self.state == 6 and next_state > 6:
            next_state = 6
        return next_state

    def step(self, action):
        # Action: -1 (left), 0 (stay), +1 (right)
        reward = reward_simulate(self.state, self.current_step, self.rewards_in_period)
        next_state = self.get_next_state(action)
        self.state = next_state
        self.current_step += 1
        return next_state, reward

    def get_state(self):
        return self.state

    def get_time(self):
        return self.current_step
    
def get_next_state(current_state, action):
    next_state = current_state + action
    if current_state == 0 and next_state < 0:
        next_state = 0
    if current_state == 6 and next_state > 6:
        next_state = 6
    return next_state

def evaluate(learner, current_state, t, eval_period=100, gamma=0.5):
    pred_states = [current_state]
    for step in range(t, t+eval_period):
        action = learner.greedy_policy(current_state, step)
        current_state = get_next_state(current_state, action)
        pred_states.append(current_state)

    # evaluate
    irewards_test = [reward_simulate(pred_states[i], t+i, rewards_in_period) for i in np.arange(1, len(pred_states))]
    preward = compute_normalized_future_rewards(irewards_test, eval_period, gamma)
    optimal_states, ireward_opt = path_opt(pred_states[0], t, eval_period)
    preward_opt_test = compute_normalized_future_rewards(ireward_opt[1:], eval_period, gamma)
    pregret = (np.sum(preward_opt_test).item() - np.sum(preward).item()) / eval_period
    return pregret

def run_replicate(include_time=False):
    env = ForagingEnv(rewards_in_period)
    learner = FQILearner(include_time=include_time)

    experiences = []
    epsilon = 0.5
    batch_size = 200
    eval_period = 100
    update_period = 100
    terminal_time = 5000
    n_trees = 100

    t_list = []
    pregret_list = []
    for t in tqdm(range(0, terminal_time)):
        # get current state
        state = env.get_state()

        # evaluate
        if (t + 1) % eval_period == 0:
            pregret = evaluate(learner, state, t)
            pregret_list.append(pregret)
            t_list.append(t+1)

        # select an action using episilon-greedy policy
        if np.random.rand() < max(0.01, epsilon * (0.999 ** (t))):
            action = np.random.choice([-1, 0, 1])  # Explore
        else:
            action = learner.greedy_policy(state, t)  # Exploit

        # execute the action in the environment
        next_state, reward = env.step(action)

        # store the interaction in the experience buffer
        experiences.append((state, action, reward, t))

        if len(experiences) >= batch_size and (t + 1) % update_period == 0:
            # Update on all the experiences
            batch = experiences

            # update the Q-function using the batch
            learner.fit(list(batch), num_iterations=20, n_trees=n_trees)

    pregret_list = np.array(pregret_list)
    t_list = np.array(t_list)
    return pregret_list, t_list

num_reps = 10

results_notime = Parallel(n_jobs=5)(delayed(run_replicate)(include_time=False) for _ in range(num_reps))
results_time = Parallel(n_jobs=5)(delayed(run_replicate)(include_time=True) for _ in range(num_reps))

pregret_list_notime = []
for pregret_list, t_list in results_notime:
    pregret_list_notime.append(pregret_list)

pregret_list_time = []
for pregret_list, t_list in results_time:
    pregret_list_time.append(pregret_list)

np.savez("results/RL/online_pregrets_fqi.npz", pregret_list_notime=pregret_list_notime, pregret_list_time=pregret_list_time, t_list=t_list)