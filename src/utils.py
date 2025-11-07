import sys
import numpy as np
import os

total_steps = 5000000
base_reward = 10.0
decay_rate = 0.6
reward_period = 10 

rewards_in_period = []
for period_start in range(0, total_steps, reward_period):
    period_end = min(period_start + reward_period, total_steps)
    steps_in_period = np.arange(period_start, period_end)
    rewards_in_period.extend(base_reward * (decay_rate ** (steps_in_period - period_start)))



def time_embedding_np(t, tdim=50):
    freqs = (2 * np.pi) / np.arange(2, tdim + 1, 2)  # shape: (tdim//2,)
    angles = np.outer(t, freqs)
    sin_emb = np.sin(angles)
    cos_emb = np.cos(angles)
    emb = np.concatenate([sin_emb, cos_emb], axis=-1)  # shape: (len(t), tdim)
    if emb.shape[0] == 1:
        return emb[0]  # return (tdim,) for scalar input
    return emb  # (len(t), tdim)

def reward_simulate(state, t,rewards_in_period):
    phase = t % 20
    if state in [0, 2, 3, 4,6]:
        val = 0.0
    elif state == 1 and phase < 10:
        val = rewards_in_period[t].item()
    elif state == 5 and phase >= 10:
        val = rewards_in_period[t].item()
    else:
        val = 0.0
    return float(val)  

def simulate_data_raw(rewards_in_period, session_duration = 10000,tdim=50, n_sessions=10,seed  = 515):
    """
    Simulate training and testing data with rewards from a given environment.


    Returns:
    - actions, state, ireward, times: np.ndarrays
    """

    np.random.seed(seed)
    actions, states, irewards,times = [], [], [], []
    # state_initial = int(np.random.choice(np.arange(7)))
    state_initial = 1
    for episode in range(n_sessions):
        t = 0
        episode_data = []
        state = state_initial
        for _ in range(session_duration):
            ireward = reward_simulate(state, t, rewards_in_period)
            state_vec = np.array([state])  # shape (1, state_dim)
            time_emb = time_embedding_np(t+1, tdim=tdim)
            features = np.concatenate([state_vec, time_emb])
            next_state = np.random.choice([state-1, state, state+1])
            if state == 0 and next_state <0:
                next_state = 0
            if state ==6 and next_state>6:
                next_state = 6
            
            action = next_state - state
            episode_data.append((action,features,ireward))
            state = next_state
            
            t += 1
        for i, (action, features, ireward) in enumerate(episode_data):
            actions.append(action)
            states.append(features)
            irewards.append(ireward)
            times.append(i)

    return (
        np.array(actions),
        np.array(states),
        np.array(irewards),
        np.array(times),
    )

def make_weighted_target(rewards, gamma,T,normalization):
    """
    Given a 1D array `rewards` of length T (indexed 0..T-1),
    returns an array Y of length T such that for each t:
      Y[t] = sum_{k=1}^{T-t-1} alpha_k * rewards[t+k],
    where alpha_k ∝ gamma^(k-1) and sum alpha_k = 1.
    """
    Y = np.zeros(T)

    # Precompute all gamma-powers up to T-1
    gammas = gamma ** np.arange(0,T)  # [γ^0, γ^1, γ^2, …] 
    
    for t in range(T):
        L = T - (t+1)              # number of future steps
        if L <= 0:
            Y[t] = 0.0            # no future reward
            continue

        # raw weights ∝ [γ^0, γ^1, ..., γ^{L-1}]
        # print(L,t+1+L)
        raw = gammas[:L]          # length L
        # 
        if normalization:
            weights = raw / raw.sum() # normalize to sum=1
        else:
            weights =raw

        future_rewards = rewards[t+1 : t+1+L]
        # print(weights.shape, future_rewards.shape)
        Y[t] = np.dot(weights, future_rewards)

    return Y

def best_next_state(X_state, t, model, pe_type):
    if X_state == 0:
        candidates = [X_state+1,X_state]
        actions = [2,0]
    elif X_state == 6:
        candidates = [X_state-1,X_state]
        actions = [1,0]
    else:
        candidates = [X_state-1,X_state,X_state+1]
        actions = [1,0,2]
    time_emb = time_embedding_np(t+2, tdim=50) # time embedding begins at 1, so the time_emb for next state is t+2
    preds = []
    for candidate in candidates:
        if pe_type == 'onehot':
            cand_encode = position_encoder(np.array([candidate]), type="onehot")
        if pe_type == 'pe':
            cand_encode = position_encoder(np.array([candidate]), type="pe")
        features = np.hstack([cand_encode, time_emb.reshape(1,-1)])
        preds.append(np.round(model.predict(features),10))
    if len(candidates)==3:
        if preds[0] == preds[1] == preds[2]:
            time_emb2 = time_embedding_np(t+3, tdim=50) # Lookahead one step
            # best_idx = np.random.choice([0,1,2])
            preds = []
            for cand1 in candidates:
                # generate 2nd‐level candidates at t+2
                if cand1 == 0:
                    cands2 = [cand1+1, cand1]
                elif cand1 == 6:
                    cands2 = [cand1-1, cand1]
                else:
                    cands2 = [cand1-1, cand1, cand1+1]
                preds2 = []
                for cand2 in cands2:
                    if pe_type == 'onehot':
                        cand2_encode = position_encoder(np.array([cand2]), type="onehot")
                    if pe_type == 'pe':
                        cand2_encode = position_encoder(np.array([cand2]), type="pe")
                    feat2 = np.hstack([cand2_encode, time_emb2.reshape(1,-1)])
                    preds2.append(model.predict(feat2)[0])
                # best future reward for this branch
                best_scores = np.round(np.max(preds2),10)
                preds.append(best_scores)
            if len(candidates)==3:
                if preds[0] == preds[1] == preds[2]:
                    # print('same')
                    best_idx = 1
                elif preds[1] == preds[0] and preds[1] > preds[2]:
                    # print('same')
                    best_idx = 1
                elif preds[1] == preds[2] and preds[1] > preds[0]:
                    # print('same')
                    best_idx = 1
                else:
                    best_idx = int(np.argmax(preds))
            else:
                best_idx = int(np.argmax(preds))
        elif preds[1] == preds[0] and preds[1] > preds[2]:
            best_idx = 1
        elif preds[1] == preds[2] and preds[1] > preds[0]:
            best_idx = 1
        else:
            best_idx = int(np.argmax(preds))
    elif len(candidates)==2 & (preds[0] == preds[1]):
        # best_idx = np.random.choice([0,1])
        time_emb2 = time_embedding_np(t+3, tdim=50)
        preds = []
        for cand1 in candidates:
            # generate 2nd‐level candidates at t+2
            if cand1 == 0:
                cands2 = [cand1+1, cand1]
            elif cand1 == 6:
                cands2 = [cand1-1, cand1]
            else:
                cands2 = [cand1-1, cand1, cand1+1]
            preds2 = []
            for cand2 in cands2:
                if pe_type == 'onehot':
                    cand2_encode = position_encoder(np.array([cand2]), type="onehot")
                if pe_type == 'pe':
                    cand2_encode = position_encoder(np.array([cand2]), type="pe")
                feat2 = np.hstack([cand2_encode, time_emb2.reshape(1,-1)])
                feat2 = np.hstack([cand2_encode.reshape(1,-1), time_emb2.reshape(1,-1)])
                preds2.append(model.predict(feat2)[0])
            # best future reward for this branch
            best_scores= np.round(np.max(preds2),10)
            preds.append(best_scores)
        best_idx = int(np.argmax(preds))
    else:
        best_idx = int(np.argmax(preds))
    x_best   = candidates[best_idx]
    action = actions[best_idx]
    # print(t,preds)
    return x_best, action

def position_encoder(states, type="onehot"):
    """
    Encode state positions into either one-hot or position-encoding.
    
    Parameters
    ----------
    states : np.ndarray
        Input array of shape (n_samples, 1) where each entry is a state index [0..6].
    type : str, optional
        Encoding type. Options:
        - 'onehot': classic one-hot encoding
        - 'pe'    : position-enhanced encoding (neighbors weighted)
    
    Returns
    -------
    np.ndarray
        Encoded state representation of shape (n_samples, 7).
    """
    n_samples = states.shape[0]
    state_posencode = np.zeros((n_samples, 7))

    for i in range(n_samples):
        current_state = int(states[i])

        if type == "onehot":
            # one-hot encoding
            state_posencode[i, current_state] = 1

        elif type == "pe":
            # position encoding
            if current_state == 0:
                state_posencode[i, current_state] = 1
                state_posencode[i, current_state + 1] = 0.66
            elif current_state == 6:
                state_posencode[i, current_state - 1] = 0.66
                state_posencode[i, current_state] = 1
            else:
                state_posencode[i, current_state - 1] = 0.33
                state_posencode[i, current_state] = 1
                state_posencode[i, current_state + 1] = 0.33
        else:
            raise ValueError("Encoding type must be either 'onehot' or 'pe'.")

    return state_posencode

def _successors(s: int):
    if s <= 0:  return (0, 1)
    if s >= 6:  return (5, 6)
    return (s-1, s, s+1)

def _step_toward(s: int, target: int) -> int:
    # move one step toward target (respect edges)
    if s < target:
        s_next = s + 1
    elif s > target:
        s_next = s - 1
    else:
        s_next = s
    return s_next 

def _target_for_time(T: int) -> int:
    r = T % 20
    if 0 <= r <= 5:     # move/stay at 1 until r==6
        return 1
    if 6 <= r <= 15:    # move/stay at 5 through r==15
        return 5
    return 1

def path_opt(state: int, t: int, len_future: int):
    """
    Return the next `len_future` states following the periodic rule.
    state ∈ {0..6}. At edges, agent can only stay or move inward.
    """
    s = state
    path = [state]
    ireward = [reward_simulate(s,t,rewards_in_period)]
    if len_future == 0:
        return path,ireward
    else:
        for k in range(1,len_future+1):
            T = t + k
            target = _target_for_time(T-1)
            s = _step_toward(s, target)
            path.append(s)
            ireward.append(reward_simulate(s,T,rewards_in_period))
        return path,ireward

def preward_opt(ireward,gamma,normalization):
    gammas = gamma ** np.arange(0,len(ireward))  # [γ^0, γ^1, γ^2, …]
    if normalization: 
        weights = gammas / gammas.sum() # normalize to sum=1
    else:
        weights = gammas
    return np.dot(weights, ireward)
    # return np.sum(ireward)


def compute_normalized_future_rewards(rewards,T,gamma,normalization):
    Y = np.zeros(T, dtype=float)
    gammas  = gamma ** np.arange(T)  # [γ^0, γ^1, …, γ^{T-1}]

    for t in range(T):
        # number of future steps
        L = T - (t + 1)
        if L <= 0:
            continue

        # normalized weights for the next L rewards
        raw     = gammas[:L]            # [γ^0, γ^1, …, γ^{L-1}]
        if normalization:
            weights = raw / raw.sum()
        else:
            weights = raw

        future_rewards = rewards[t+1 : t+1+L]
        # dot-product returns the weighted average
        Y[t] = np.dot(weights, future_rewards)

    return Y




