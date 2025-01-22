from dataclasses import dataclass, field
from functools import reduce
from itertools import takewhile
from typing import Callable, Iterable, Iterator, NamedTuple, TypeVar

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from semantic_memory_game import MemoryGame, MemoryGameEnv


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

def episode(env: gym.Env, policy: Callable[[object], object]):
    return takewhile(lambda ts: not ts.done, steps(env, policy))

def eval_policy(env: gym.Env, policy: Callable[[object], object]):
    return reduce(lambda acc, ts: (acc[0] + ts.reward, acc[0]+1), episode(env, policy), (0, 0))

def mean_eval(env: gym.Env, policy: Callable[[object], object], n=10):
    evals = np.array([eval_policy(env, policy) for _ in range(n)])
    return evals[:,0].mean(), evals[:,1].mean()

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
            return np.argmax(self.q[s])

        def epsilon_greedy_action(self, s):
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.q.shape[1])
            return self.best_action(s)

    def update(self, q: np.ndarray, ts: TimeStep):
        q[ts.state, ts.action] += self.alpha * (ts.reward + ts.term * self.gamma * np.max(q[ts.next_state]) - q[ts.state, ts.action])

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


def main():
    def make_env():
        return MemoryGameEnv(MemoryGame(oracle=np.array([1, 0, 3, 2])))
    env = make_env()
    eval_env = make_env()

    n_steps = 500_000
    epsilon_schedule = TabularQLearning.linear_schedule(initial=0.5, n=n_steps)
    algo = TabularQLearning(epsilon_schedule=epsilon_schedule)

    eval_returns = []
    eval_lens = []
    epsilons = []
    for i, agent in enumerate(tqdm(Take(n_steps, algo.run(env)))):
        # print(f'i: {i}')
        if i % 1000 == 0:
            # evals.append(eval_policy(eval_env, agent.best_action))
            ret, len = mean_eval(eval_env, agent.best_action)
            eval_returns.append(ret)
            eval_lens.append(len)
            epsilons.append(agent.epsilon)
    print(f'learned Q: {agent.q.shape}')

    plt.plot(eval_returns, label='eval returns')
    plt.show()
    # plt.legend()

    plt.plot(eval_lens, label='eval episode lengths')
    plt.show()

    plt.plot(epsilons, label='epsilon')
    plt.show()



if __name__ == "__main__":
    main()

