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


import rl
from semantic_memory_game import MemoryGame, MemoryGameEnv


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


class SemanticMemoryGameEnv(MemoryGameEnv):
    """
    Like MemoryGameEnv, but there is also:
    - a trust factor given as observation to the robot (discretized)
    - a second (human) player with a given policy and CTF for a single category (can be dynamically changed)
    - a handover action for both players to switch roles (a=N)
    
    When the human is playing, all their steps are condensed into a single step
    where the rewards are accumulated and given as a single value. If
    the human starts the episode and solves the game, the episode is skipped
    and a new one is started instead, until a non-empty episode is started. 

    This means that using a gamma < 1 will not work as if the agent was
    playing alone: we suggest setting gamma=1 and using a small negative
    reward for each step where no other reward is given, to still optimize
    for the shortest path to the goal.
    """

    @property
    def handover_action(self):
        return self.action_space.n-1

    def __init__(self, game: MemoryGame, human: Player, ai_hampered_competence: float = 0, tf_levels: int = 3, max_steps=None):
        super().__init__(game, max_steps)
        self.memory_game_states = self.observation_space.n
        self.observation_space = spaces.Discrete(self.memory_game_states * tf_levels)
        self.action_space = spaces.Discrete(1 + self.action_space.n)
        self.human = human
        self.tf_levels = tf_levels
        self.ai_hampered_competence = ai_hampered_competence

    def observe(self):
        s = super().observe()
        tf_level = round((self.tf_levels-1)*self.human.ctf)
        s_wrapped = tf_level*self.memory_game_states + s
        return s_wrapped

    def human_steps(self):
        # return value if human handover is immediate
        s, R, term, trunc, info = self.observe(), 0, False, False, {}
        n_steps = 0
        while True:
            action = self.human(self.game_state)
            if action == self.handover_action:
                self.steps += 1
                trunc = self.timed_out
                break
            s, r, term, trunc, info = super().step(action)
            R += r
            if term or trunc:
                break
            n_steps += 1
        return s, R, term, trunc, info

    def reset(self, seed=None, options=None):
        # TODO set seed for self.env reset?
        n_reset = 0
        while True:
            # Reset and pick active player
            s, info = super().reset(seed, options)
            n_reset += 1
            human_starts = np.random.choice([True, False])
            # If AI starts, we are good
            if not human_starts:
                break
            # If human starts, we need to play their turn
            s, _, term, trunc, info = self.human_steps()
            # If the human didn't solve the game, we are good, otherwise we
            # start over
            if not (term or trunc):
                break
        # We can safely skip the reward that the human accumulated at reset
        # time, since it will provide no learning signal to an RL agent
        return s, info

    def step(self, action):
        if action == self.handover_action:
            # AI hands over to human
            self.steps += 1
            s, *step = self.human_steps()
        else:
            action = hamper_competence(action, self.human.category, self.ai_hampered_competence)
            s, *step = super().step(action)
        return s, *step



def make_test_human(oracle, category, ctf):
    N = len(oracle)

    def policy(game_state: MemoryGame.State):
        # if there is no card face up, pick a random one
        if game_state.face_up == 0:
            return np.random.choice(N)
        
        # if a card face up is part of our category, choose its matched card
        # (the env takes care of picking a random card if the matched card is
        # still unseen)
        if game_state.face_up-1 in category:
            paired = oracle[game_state.face_up-1]
            return paired if np.random.rand() < ctf else np.random.choice(N)

        # if a card face up is not part of our category, handover
        return N

    return Player(policy, category, ctf)



def main():
    game = MemoryGame(np.array([1, 0, 3, 2, 5, 4, 7, 6]))
    # ai_category = np.array([1, 0])
    human = make_test_human(game.oracle, np.array([2, 3, 4, 5]), 1.0)

    def make_env():
        return SemanticMemoryGameEnv(game, human)
    env = make_env()
    eval_env = make_env()

    stats = (
        rl.Stats.return_,
        rl.Stats.length,
        rl.Stats.terminated,
    )

    n_steps = 100_000
    epsilon_schedule = rl.TabularQLearning.exponential_epsilon_decay(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=n_steps)
    algo = rl.TabularQLearning(
        alpha=0.2,
        gamma=0.99,
        epsilon_schedule=epsilon_schedule
    )

    random_policy_rets, random_policy_lens, random_policy_wins = rl.evals(eval_env, lambda _: np.random.choice(env.action_space.n), stats, n=1000)
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
            rets, ep_lens, terms = rl.evals(eval_env, agent.best_action, stats, n=20)
            eval_returns.append(rets.mean())
            eval_lens.append(ep_lens.mean())
            eval_wrs.append(terms.mean())
            # for term in terms:
            #     if term:
            #         print(f'[{i}] found eval WIN: ')
            epsilons.append(agent.epsilon)
            # Early stopping
            if terms.mean() >= best_wr:
                best_wr_agent = agent
                best_wr = terms.mean()
    # print(f'learned Q: {agent.q.shape}')
    # best_agent = agent
    best_agent = best_wr_agent
    print(f'Best seen win rate: {best_wr}')

    rets, ep_lens, terms = rl.evals(eval_env, best_agent.best_action, stats, n=1000)
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


