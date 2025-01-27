from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Iterable, Iterator, NamedTuple, TypeVar

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pygame
import math
import gymnasium as gym
from gymnasium import spaces



def hamper_competence(card, category, competence):
    if card in category:
        return card if np.random.rand() < competence else np.random.choice(category)
    return card


@dataclass
class Player:
    """A player with a given policy and CTF on a single category."""

    policy: Callable[[object], object]
    category: np.ndarray
    ctf: float

    def __call__(self, s):
        return self.policy(s)


class MixedHumanAIWrapper(gym.Wrapper):
    # Inserts:
    # - trust factor given as observation to the robot (discretized)
    # - a second (human) player with a given policy and CTF for a single category (can be dynamically changed)
    # - a handover action for both players to switch roles
    # NOTE: when the human is playing, we still ask actions from the AI, but they are ignored

    # Why not collapse human steps?
    # 1. episodes may end at reset time
    # 2. what to do with the human reward? accumulating messes up discounting

    def __init__(self, env: gym.Env, human: Player, tf_levels: int, ai_hampered_competence: float):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Discrete)
        assert isinstance(env.action_space, spaces.Discrete)
        self.observation_space = spaces.Discrete(env.observation_space.n*tf_levels)
        self.action_space = spaces.Discrete(env.action_space.n + 1)
        self.env_n_states = env.observation_space.n
        self.human = human
        self.tf_levels = tf_levels
        self.ai_hampered_competence = ai_hampered_competence
        # episode state variables
        self.active_player = None
        self.last_step = None

    def wrap_state(self, s):
        ctf_level = self.human.ctf
        s_wrapped = ctf_level*self.env_n_states + s
        return s_wrapped

    def reset(self, **kwargs):
        s, info = self.env.reset(**kwargs)
        self.last_step = s, 0, False, False, info
        # TODO set seed?
        self.active_player = np.random.choice(['human', 'ai'])
        return s, info

    def step(self, action):
        if action == 0:
            self.active_player = 'ai' if self.active_player == 'human' else 'human'
            return self.last_step

        if self.active_player == 'human':
            action = self.human.policy(self.last_step[0])
        else:
            assert self.active_player == 'ai'
            action = hamper_competence(action-1, self.human.category, self.ai_hampered_competence)

        self.last_step = self.env.step(action-1)
        return self.last_step

