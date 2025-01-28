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

from semantic_memory_game import MemoryGame, MemoryGameEnv
import rl


def main():
    def make_env():
        return MemoryGameEnv(MemoryGame(oracle=np.array([1, 0, 3, 2, 5, 4])))
    env = make_env()
    eval_env = make_env()

    n_steps = 10000
    epsilon_schedule = rl.TabularQLearning.exponential_epsilon_decay(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=n_steps)
    algo = rl.TabularQLearning(
        alpha=0.99,
        gamma=0.99,
        epsilon_schedule=epsilon_schedule
    )

    random_policy_rets, random_policy_lens, random_policy_wins = rl.evals(eval_env, lambda _: np.random.choice(4), n=1000)
    random_policy_ret = random_policy_rets.mean()
    random_policy_len = random_policy_lens.mean()
    random_policy_wr = random_policy_wins.mean()

    eval_returns = []
    eval_lens = []
    eval_wrs = []
    epsilons = []
    best_wr_agent = None
    best_wr = -1
    for i, agent in enumerate(tqdm(rl.Take(n_steps, algo.run(env)))):
        # print(f'i: {i}')
        if i % 100 == 0:
            # evals.append(eval_policy(eval_env, agent.best_action))
            rets, ep_lens, terms = rl.evals(eval_env, agent.best_action, n=20)
            eval_returns.append(rets.mean())
            eval_lens.append(ep_lens.mean())
            eval_wrs.append(terms.mean())
            # for term in terms:
            #     if term:
            #         print(f'[{i}] found eval WIN: ')
            epsilons.append(agent.epsilon)
            # Early stopping
            # if terms.mean() > best_wr:
            #     best_wr_agent = agent
            #     best_wr = terms.mean()
    # print(f'learned Q: {agent.q.shape}')
    best_agent = best_wr_agent

    rets, ep_lens, terms = rl.evals(eval_env, agent.best_action, n=1000)
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

    plt.plot(epsilons, label='epsilon')
    plt.show()



if __name__ == "__main__":
    main()

