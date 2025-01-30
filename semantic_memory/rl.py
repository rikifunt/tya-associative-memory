from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Iterable, Iterator, NamedTuple, TypeVar

import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces



# General MDP utilities

class TimeStep(NamedTuple):
    state: object
    action: object
    next_state: object
    reward: float
    term: bool
    trunc: bool
    info: dict
    next_info: dict

    @property
    def done(self):
        return self.term or self.trunc

def steps(env: gym.Env, policy: Callable[[object], object]):
    term = True
    trunc = False
    while True:
        if term or trunc:
            state, info = env.reset()
        action = policy(state)
        next_state, reward, term, trunc, next_info = env.step(action)
        yield TimeStep(state, action, next_state, reward, term, trunc, info, next_info)
        state = next_state
        info = next_info

def episode(env: gym.Env, policy: Callable[[object], object]):
    # Can't use take while here because we need to yield the last step
    for ts in steps(env, policy):
        yield ts
        if ts.done:
            break

class Statistic:
    def __init__(self, op, initial):
        self.op = op
        self.value = initial

    def update(self, ts: TimeStep) -> 'Statistic':
        self.value = self.op(self.value, ts)
        return self

class Stats:
    @staticmethod
    def return_():
        return Statistic(lambda x, ts: x + ts.reward, 0)
    
    @staticmethod
    def length():
        return Statistic(lambda x, ts: x + 1, 0)

    @staticmethod
    def terminated():
        return Statistic(lambda x, ts: x or ts.term, False)

def eval_policy(env: gym.Env, policy: Callable[[object], object], statistics: Iterable[Statistic] = ()) -> tuple:
    return map(
        lambda stat: stat.value,
        reduce(
            lambda stats, ts: tuple(stat.update(ts) for stat in stats),
            episode(env, policy),
            statistics,
        )
    )

def evals(env: gym.Env, policy: Callable[[object], object], make_stat_fns, n=10) -> tuple[np.ndarray, ...]:
    stats = list(zip(*[eval_policy(env, policy, (make_stat() for make_stat in make_stat_fns)) for _ in range(n)]))
    return tuple(np.array(stat) for stat in stats)

T = TypeVar('T')

@dataclass(frozen=True)
class Take(Iterable[T]):
    n: int
    iterable: Iterable[T]

    def __len__(self):
        return self.n

    def __iter__(self) -> Iterator[T]:
        for i, x in enumerate(self.iterable):
            if i >= self.n:
                break
            yield x


# Tabular Q-Learning

@dataclass
class TabularQLearning:
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_schedule: Callable[[int], float] = lambda i: 0.1

    @staticmethod
    def decay_schedule(initial: float= 0.1, decay: float = 0.999):
        return lambda i: initial * decay**i
    
    @staticmethod
    def exponential_epsilon_decay(epsilon_start=1, epsilon_end=0.01, epsilon_decay=5000):
        return lambda i: epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * i / epsilon_decay)

    @staticmethod
    def linear_schedule(initial: float = 0.1, final: float = 0.01, n: int = 10000):
        return lambda i: max(final, initial - i * (initial - final) / n)

    @dataclass
    class State:
        q: np.ndarray
        steps: int
        algo: 'TabularQLearning'

        @property
        def epsilon(self):
            return self.algo.epsilon_schedule(self.steps)

        def best_action(self, s):
            return np.random.choice(np.flatnonzero(self.q[s] == self.q[s].max()))
            # return np.argmax(self.q[s])

        def epsilon_greedy_action(self, s):
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.q.shape[1])
            return self.best_action(s)

    def update(self, q: np.ndarray, ts: TimeStep):
        V_next = (1-ts.term)*self.gamma * np.max(q[ts.next_state])
        q[ts.state, ts.action] += self.alpha * (ts.reward + V_next - q[ts.state, ts.action])

    def run(self, env: gym.Env, q: np.ndarray | None = None) -> Iterator[State]:
        assert isinstance(env.observation_space, spaces.Discrete)
        assert isinstance(env.action_space, spaces.Discrete)
        if q is None:
            q = np.zeros((env.observation_space.n, env.action_space.n))
        algo_state = TabularQLearning.State(q, 0, self)
        for i, ts in enumerate(steps(env, algo_state.epsilon_greedy_action)):
            self.update(algo_state.q, ts)
            algo_state.steps = i
            yield TabularQLearning.State(algo_state.q.copy(), i, self)
