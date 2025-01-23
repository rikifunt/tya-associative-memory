from dataclasses import dataclass, field
from functools import reduce
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
    # Can't use take while here because we need to yield the last step
    for ts in steps(env, policy):
        yield ts
        if ts.done:
            break

def eval_policy(env: gym.Env, policy: Callable[[object], object]):
    return reduce(lambda acc, ts: (acc[0] + ts.reward, acc[1]+1, ts.term), episode(env, policy), (0, 0, None))

def evals(env: gym.Env, policy: Callable[[object], object], n=10):
    evals = np.array([eval_policy(env, policy) for _ in range(n)])
    return evals[:,0], evals[:,1], evals[:,2]

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
        return MemoryGameEnv(MemoryGame(oracle=np.array([1, 0, 3, 2, 5, 4])))
    env = make_env()
    eval_env = make_env()

    n_steps = 100_000
    epsilon_schedule = TabularQLearning.linear_schedule(initial=0.5, n=n_steps)
    algo = TabularQLearning(
        alpha=5e-3,
        epsilon_schedule=epsilon_schedule
    )

    random_policy_rets, random_policy_lens, random_policy_wins = evals(eval_env, lambda _: np.random.choice(4), n=1000)
    random_policy_ret = random_policy_rets.mean()
    random_policy_len = random_policy_lens.mean()
    random_policy_wr = random_policy_wins.mean()

    eval_returns = []
    eval_lens = []
    eval_wrs = []
    epsilons = []
    best_wr_agent = None
    best_wr = -1
    for i, agent in enumerate(tqdm(Take(n_steps, algo.run(env)))):
        # print(f'i: {i}')
        if i % 100 == 0:
            # evals.append(eval_policy(eval_env, agent.best_action))
            rets, ep_lens, terms = evals(eval_env, agent.best_action, n=20)
            eval_returns.append(rets.mean())
            eval_lens.append(ep_lens.mean())
            eval_wrs.append(terms.mean())
            # for term in terms:
            #     if term:
            #         print(f'[{i}] found eval WIN: ')
            epsilons.append(agent.epsilon)
            # Early stopping
            if terms.mean() > best_wr:
                best_wr_agent = agent
                best_wr = terms.mean()
    # print(f'learned Q: {agent.q.shape}')
    best_agent = best_wr_agent

    rets, ep_lens, terms = evals(eval_env, best_agent.best_action, n=1000)
    print(f'final eval returns: {rets.mean()} +- {rets.std():.2f} (random: {random_policy_ret} +- {random_policy_rets.std():.2f})')
    print(f'final eval episode lengths: {ep_lens.mean()} +- {ep_lens.std():.2f} (random: {random_policy_len} +- {random_policy_lens.std():.2f})')
    print(f'final eval win rates: {terms.mean()} +- {terms.std():.2f} (random: {random_policy_wr} +- {random_policy_wins.std():.2f})')

    # plt.plot(eval_returns, label='eval returns')
    # plt.plot([random_policy_ret] * len(eval_returns), label='random policy returns')
    # plt.legend()
    # plt.show()

    plt.plot(eval_lens, label='eval episode lengths')
    plt.plot([random_policy_len] * len(eval_lens), label='random policy episode lengths')
    plt.legend()
    plt.show()

    plt.plot(eval_wrs, label='eval win rates')
    plt.plot([random_policy_wr] * len(eval_wrs), label='random policy win rates')
    plt.legend()
    plt.show()

    # plt.plot(epsilons, label='epsilon')
    # plt.show()



if __name__ == "__main__":
    main()

