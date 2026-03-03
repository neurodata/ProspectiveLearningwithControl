import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from src.utils import reward_simulate, simulate_data_raw
from src.utils import path_opt, compute_normalized_future_rewards
from src.dreamer import DreamerV3Learner

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
        self.action_space = [-1, 0, 1]  # left, stay, right
        self.reset()

    def reset(self):
        self.current_step = 0
        self.state = 1  # Start at state 1
        return self.state

    def get_next_state(self, action):
        action = self.action_space[action]  # map action index to actual action
        next_state = self.state + action
        if self.state == 0 and next_state < 0:
            next_state = 0
        if self.state == 6 and next_state > 6:
            next_state = 6
        return next_state

    def step(self, action):
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
    action = [-1, 0, 1][action]  # map action index to actual action
    next_state = current_state + action
    if current_state == 0 and next_state < 0:
        next_state = 0
    if current_state == 6 and next_state > 6:
        next_state = 6
    return next_state

def evaluate(learner, current_state, t, eval_period=100, gamma=0.5, epsilon=0.1):
    learner.reset_eval_state()
    pred_states = [current_state]
    for step in range(t, t+eval_period):
        if np.random.rand() < epsilon:
            action = np.random.choice([0,1,2])
        else:
            action = learner.select_action(current_state, step, eval_mode=True)
        current_state = get_next_state(current_state, action)
        pred_states.append(current_state)

    # evaluate
    irewards_test = [reward_simulate(pred_states[i], t+i, rewards_in_period) for i in np.arange(1, len(pred_states))]
    preward = compute_normalized_future_rewards(irewards_test, eval_period, gamma, normalization=True)
    optimal_states, ireward_opt = path_opt(pred_states[0], t, eval_period)
    preward_opt_test = compute_normalized_future_rewards(ireward_opt[1:], eval_period, gamma, normalization=True)
    pregret = (np.sum(preward_opt_test).item() - np.sum(preward).item()) / eval_period
    return pregret

def run_replicate(include_time=True):
    env = ForagingEnv(rewards_in_period)
    learner = DreamerV3Learner(include_time=include_time)

    batch_size = 16
    eval_period = 100
    terminal_time = 100000

    t_list = []
    pregret_list = []

    progress_bar = tqdm(range(0, terminal_time))
    for t in progress_bar:
        # get current state
        state = env.get_state()

        # evaluate
        if (t + 1) % eval_period == 0:
            pregret = evaluate(learner, state, t)
            pregret_list.append(pregret)
            t_list.append(t+1)

        # Encode observation for buffer
        obs = learner.encode_obs(state, t).detach().numpy().flatten()

        # Select action
        action = learner.select_action(state, t)

        # Step in environment
        next_state, reward = env.step(action)

        # Store transition
        learner.replay_buffer.push(obs, action, reward)

        # Update DreamerV3
        learner.update(batch_size)

    pregret_list = np.array(pregret_list)
    t_list = np.array(t_list)
    return pregret_list, t_list

num_reps = 10

results_time = Parallel(n_jobs=10)(delayed(run_replicate)(include_time=True) for _ in range(num_reps))
results_notime = Parallel(n_jobs=10)(delayed(run_replicate)(include_time=False) for _ in range(num_reps))

pregret_list_time = []
for pregret_list, t_list in results_time:
    pregret_list_time.append(pregret_list)

pregret_list_notime = []
for pregret_list, t_list in results_notime:
    pregret_list_notime.append(pregret_list)

np.savez("../results/RL/stochastic_online_pregrets_dreamer.npz",
         pregret_list_time=pregret_list_time,
         pregret_list_notime=pregret_list_notime,
         t_list=t_list)
