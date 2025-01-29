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


class HamperedAISemanticMemoryGameEnv(MemoryGameEnv):

    def __init__(self, game, ai_hampered_competence, category, max_steps=None):
        super().__init__(game, max_steps)
        self.ai_hampered_competence = ai_hampered_competence
        self.category = category

    def step(self, action):
        action = hamper_competence(action, self.category, self.ai_hampered_competence)
        return super().step(action)




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
        info['human_steps'] = n_steps
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

    @staticmethod
    def team_len():
        return rl.Statistic(lambda n, ts: n + 1 + ts.info.get('human_steps', 0), 0)



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


def train_omniscent_ai_only():
    def make_env():
        return MemoryGameEnv(MemoryGame(oracle=np.array([1, 0, 3, 2, 5, 4, 7, 6])))
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

    rets, ep_lens, terms = rl.evals(eval_env, best_agent.best_action, stats, n=1000)

    return {
        'train_eval_returns': eval_returns,
        'train_eval_lens': eval_lens,
        'train_eval_wrs': eval_wrs,
        'eval_rets': rets,
        'eval_lens': ep_lens,
        'eval_terms': terms
    }


# TODO hamper AI policy in the same way of train_mixed_team
def train_partial_ai_only():
    def make_env():
        return HamperedAISemanticMemoryGameEnv(
            MemoryGame(oracle=np.array([1, 0, 3, 2, 5, 4, 7, 6])),
            ai_hampered_competence=0,
            category=np.array([2, 3, 4, 5]),
        )
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

    rets, ep_lens, terms = rl.evals(eval_env, best_agent.best_action, stats, n=1000)

    return {
        'train_eval_returns': eval_returns,
        'train_eval_lens': eval_lens,
        'train_eval_wrs': eval_wrs,
        'eval_rets': rets,
        'eval_lens': ep_lens,
        'eval_terms': terms
    }


def train_mixed_team():
    game = MemoryGame(np.array([1, 0, 3, 2, 5, 4, 7, 6]))
    # ai_category = np.array([1, 0])
    human = make_test_human(game.oracle, np.array([2, 3, 4, 5]), 1.0)

    def make_env():
        return SemanticMemoryGameEnv(game, human)
    env = make_env()
    eval_env = make_env()

    stats = (
        rl.Stats.return_,
        # rl.Stats.length,
        SemanticMemoryGameEnv.team_len,
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
    best_len_agent = None
    best_len = np.inf
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
            if ep_lens.mean() <= best_len:
                best_len_agent = agent
                best_len = ep_lens.mean()
    # print(f'learned Q: {agent.q.shape}')
    # best_agent = agent
    # best_agent = best_len_agent
    best_agent = best_wr_agent
    print(f'Best seen win rate: {best_wr}')
    print(f'Best seen episode length: {best_len}')

    rets, ep_lens, terms = rl.evals(eval_env, best_agent.best_action, stats, n=1000)
    print(f'final eval returns: {rets.mean()} +- {rets.std():.2f} (random: {random_policy_ret} +- {random_policy_rets.std():.2f})')
    print(f'final eval episode lengths: {ep_lens.mean()} +- {ep_lens.std():.2f} (random: {random_policy_len} +- {random_policy_lens.std():.2f})')
    print(f'final eval win rates: {terms.mean()} +- {terms.std():.2f} (random: {random_policy_wr} +- {random_policy_wins.std():.2f})')

    results = {
        'train_eval_returns': eval_returns,
        'train_eval_lens': eval_lens,
        'train_eval_wrs': eval_wrs,
        'eval_rets': rets,
        'eval_lens': ep_lens,
        'eval_terms': terms
    }
    return results


def run_and_save_trials(n_trials, train_fn):
    trial_results = [train_fn() for _ in range(n_trials)]
    # stack results into single arrays
    results = {
        key: np.array([trial[key] for trial in trial_results])
        for key in trial_results[0]
    }
    np.savez(f'../results/{train_fn.__name__}_results.npz', **results)
    return results


def eval_only_human():
    pass


def eval_only_random():
    def make_env():
        return MemoryGameEnv(MemoryGame(oracle=np.array([1, 0, 3, 2, 5, 4, 7, 6])))
    eval_env = make_env()

    stats = (
        rl.Stats.return_,
        rl.Stats.length,
        rl.Stats.terminated,
    )

    random_policy_rets, random_policy_lens, random_policy_wins = \
        rl.evals(eval_env, lambda _: np.random.choice(eval_env.action_space.n), stats, n=1000)

    return {
        'returns': random_policy_rets,
        'lens': random_policy_lens,
        'wrs': random_policy_wins
    }



def smooth(x, window_len=100):
    return np.convolve(x, np.ones(window_len)/window_len, mode='same')


def plot_mean_std(x, y, yerr, label, smoothing_window=None, color=None, ax=plt):
    if ax is None:
        ax = plt.gca()

    if smoothing_window is not None:
        y = smooth(y, window_len=smoothing_window)
        # yerr = smooth(yerr, window_len=smoothing_window)

    p = ax.plot(x, y, label=label, color=color)
    ax.fill_between(x, y-yerr, y+yerr, color=p[0].get_color(), alpha=0.15)


def plot_results():
    train_fns = [
        'train_omniscent_ai_only',
        'train_partial_ai_only',
        'train_mixed_team',
    ]
    train_results = {
        fn: np.load(f'../results/{fn}_results.npz')
        for fn in train_fns
    }

    fixed_evals = [
        'random'
    ]
    fixed_results = {
        k: globals()[f'eval_only_{k}']()
        for k in fixed_evals
    }

    train_xs = np.arange(100_000, step=100)
    for k in ['returns', 'lens']:
        plt.title(k)
        for fn, results in train_results.items():
            plot_mean_std(
                train_xs,
                results[f'train_eval_{k}'].mean(axis=0),
                results[f'train_eval_{k}'].std(axis=0),
                label=fn,
                smoothing_window=10,
            )
        for fixed_policy, results in fixed_results.items():
            plot_mean_std(
                train_xs,
                np.full_like(train_xs, results[k].mean()),
                np.full_like(train_xs, results[k].std()),
                label=fixed_policy
            )
        plt.legend()
        plt.show()


def main():
    # TODO plots
    # return train
    # puzzle solution time
    # visual repr of optimal policy execution:
    # - show which categories are chosen by each agent
    # - handover/steps percentage per agent
    pass


if __name__ == "__main__":
    # one CLI arg telling which train function, or 'plot'
    import sys
    if len(sys.argv) != 2:
        print('Usage: python mixed_team.py <cmd>')
        sys.exit(1)

    if sys.argv[1] == 'plot':
        plot_results()
        sys.exit(0)

    train_fn = globals().get('train_' + sys.argv[1])
    run_and_save_trials(5, train_fn)


