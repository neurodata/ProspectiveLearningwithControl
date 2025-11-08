import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .utils import time_embedding_np

class TabularQLearner:
    def __init__(self, gamma=0.9, num_states=7, num_actions=3, include_time=True, tsize=20):
        if include_time:
            self.Q_table = np.zeros((num_states, tsize, num_actions))
        else:
            self.Q_table = np.zeros((num_states, 1, num_actions))
        self.include_time = include_time
        self.tsize = tsize
        self.gamma = gamma

    def reset(self):
        self.Q_table = np.zeros_like(self.Q_table)

    def fit(self, experiences, num_iterations=100):
        for _ in range(num_iterations):
            for step in range(len(experiences)-1):
                
                s, a, r, t = experiences[step]
                s_, _, _, t_ = experiences[step+1]

                a += 1 # shift action from [-1,0,1] to [0,1,2] for indexing (better to define the action space instead?)

                if self.include_time:
                    t = t % self.tsize
                    t_ = t_ % self.tsize
                else:
                    t, t_ = 0, 0

                self.Q_table[s, t, a] = r + self.gamma * np.max(self.Q_table[s_, t_, :])

        return self.Q_table

    def greedy_policy(self, state, time):   
        if self.include_time:
            time = time % self.tsize
        else: 
            time = 0
        return np.argmax(self.Q_table[state, time, :])-1 # shift back action from [0,1,2] to [-1,0,1]
    

class FQILearner:
    def __init__(self, gamma=0.9, num_states=7, num_actions=3, include_time=True):
        self.Q = None
        self.include_time = include_time
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions

    def reset(self):
        self.Q = None

    def encode(self, s, a, t):
        # s_enc = np.zeros(self.num_states)
        # s_enc[int(s)] = 1
        # a_enc = np.zeros(self.num_actions)
        # a_enc[int(a)] = 1
        s_enc = [s]
        a_enc = [a]
        if self.include_time:
            # t_enc = [(t % i) for i in range(5, 30+1, 5)]
            # t_enc = [(t % 20)]
            t_enc = time_embedding_np(t)
            return np.concatenate([s_enc, a_enc, t_enc])
        else:
            return np.concatenate([s_enc, a_enc])

    def fit(self, experiences, num_iterations=100, n_trees=1000):
        ## -- Simple version --
        # for _ in range(num_iterations):
        #     inputs, targets = [], []
        #     for step in range(len(experiences)-1):
                
        #         s, a, r, t = experiences[step]
        #         s_, _, _, t_ = experiences[step+1]
        #         a += 1 # shift action from [-1,0,1] to [0,1,2] for indexing

        #         if self.Q is None:
        #             q_next = 0
        #         else:
        #             q_next = max(self.Q.predict([self.encode(s_, a_, t_)]) for a_ in np.arange(self.num_actions))
        #         y = r + self.gamma * q_next

        #         inputs.append(self.encode(s, t, a))
        #         targets.append(y)

        #     # Fit the regressor
        #     self.Q = RandomForestRegressor(n_estimators=100)
        #     self.Q.fit(inputs, targets)

        ## -- Vectorized (efficient) version --
        experiences = np.array(experiences)
        s = experiences[:-1, 0].reshape(-1,1)      # state
        a = experiences[:-1, 1].reshape(-1,1)      # action
        a += 1  # shift action from [-1,0,1] to [0,1,2] for indexing
        r = experiences[:-1, 2]                    # reward
        t = experiences[:-1, 3].reshape(-1,1)      # time
        s_next = experiences[1:, 0].reshape(-1,1)  # next state
        t_next = experiences[1:, 3].reshape(-1,1)  # next state

        x_sat = np.hstack([s, a, t])
        X = np.array([self.encode(x[0], x[1], x[2]) for x in x_sat])

        s_next_all = np.repeat(s_next, self.num_actions, axis=0)
        t_next_all = np.repeat(t_next, self.num_actions, axis=0)
        a_next_all = np.tile(np.arange(self.num_actions), len(s_next)).reshape(-1, 1)
        x_next_all = np.hstack([s_next_all, a_next_all, t_next_all])
        X_next = np.array([self.encode(x[0], x[1], x[2]) for x in x_next_all])

        for _ in range(num_iterations):
            if self.Q is None:
                y = r
            else:
                q_next_all = self.Q.predict(X_next).reshape(len(s_next), self.num_actions)
                q_next_max = q_next_all.max(axis=1)
                y = r + self.gamma * q_next_max

            # Fit the regressor
            self.Q = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
            self.Q.fit(X, y)

    def greedy_policy(self, state, time):   
        if self.Q is None:
            return np.random.choice([-1,0,1])
        else:
            return np.argmax([self.Q.predict([self.encode(state, a_, time)]) for a_ in np.arange(self.num_actions)])-1


