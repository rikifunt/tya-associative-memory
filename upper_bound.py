from dataclasses import dataclass
from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from semantic_memory_game import SemanticMemoryGame


class SemanticMemoryGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, game: SemanticMemoryGame):
        self.game = game
        self.observation_space = spaces.Discrete(1 + 4**len(self.game.oracle))
        self.action_space = spaces.Discrete(len(self.game.oracle))
        self.game_state = None

    def observe(self):
        if self.game_state is None:
            return 0
        # map from array of values in 0-3 to single integer
        return 1 + (4**np.arange(len(self.game.oracle)) * self.game_state).sum()

    def reward(self, s, s1):
        if s1 is None:
            print("WIN")
            return 10.0
        if self.game.have_new_match(s, s1):
            # print("MATCH")
            return 1.0
        return -1.0

    def reset(self, seed=None, options=None):
        self.game_state = self.game.reset()
        return self.observe(), {}

    def step(self, action):
        last_state = self.game_state
        self.game_state = self.game.step(self.game_state, action)
        term = self.game_state is None
        trunc = False
        return self.observe(), self.reward(last_state, self.game_state), term, trunc, {}

from gymnasium.envs.registration import register

register(
    id="AssociativeMemoryGameEnv-v0",
    entry_point="upper_bound:AssociativeMemoryGameEnv",
    kwargs={
        "game": SemanticMemoryGame(),
    },
    max_episode_steps=100,
)


class TwoAgentHandoverWrapper(gym.Wrapper):
    """Wrapper introducing a 2-agent handover action to the environment.
    
    An additional action is added to the environment, which allows the current
    agent to hand over control to a second agent. The actual handover is
    delegated to the caller of env.step.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete)
        self.action_space = spaces.Discrete(env.action_space.n + 1)
        self.observation_space = env.observation_space
        self.last_step = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_step = obs, 0, False, False, info
        return obs, info

    def step(self, action):
        if action == 0:
            return self.last_step
        self.last_step = self.env.step(action-1)
        return self.last_step


def tabular_q_learning(env: gym.Env, steps=1000, alpha=0.1, gamma=0.99, epsilon=0.1, max_episode_steps=100):
    assert isinstance(env.observation_space, spaces.Discrete)
    assert isinstance(env.action_space, spaces.Discrete)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    term = True
    trunc = False
    episode_steps = 0
    for _ in range(steps):
        if term or trunc or episode_steps >= max_episode_steps:
            state = env.reset()
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state])
        next_state, reward, term, trunc, _ = env.step(action)
        q[state, action] += alpha * (reward + term * gamma * np.max(q[next_state]) - q[state, action])
        state = next_state
        episode_steps += 1
    return q



def main():
    pass


if __name__ == "__main__":
    main()

