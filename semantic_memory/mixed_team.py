from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Iterable, Iterator, NamedTuple, TypeVar

from tqdm import tqdm
from matplotlib import pyplot as plt
# uncomment for headless
# import matplotlib
# matplotlib.use("Agg")
import numpy as np
import pygame
import math
import gymnasium as gym
from gymnasium import spaces


import rl
from semantic_memory_game import MemoryGame, MemoryGameEnv
from humans import high as high_human, policy_bridge


def hamper_competence(card, category, competence, n_cards):
    if card in category:
        return card if np.random.rand() < competence else np.random.choice(n_cards)
    return card


class HamperedAISemanticMemoryGameEnv(MemoryGameEnv):

    def __init__(self, game, ai_hampered_competence, category, max_steps=None):
        super().__init__(game, max_steps)
        self.ai_hampered_competence = ai_hampered_competence
        self.category = category

    def step(self, action):
        action = hamper_competence(action, self.category, self.ai_hampered_competence, self.game.n_cards)
        return super().step(action)


class HumanOnlySemanticMemoryGameEnv(MemoryGameEnv):

    def __init__(self, game, human, max_steps=None):
        super().__init__(game, max_steps)
        self.human = human

    def observe(self):
        return self.game_state

    def step(self, action):
        # override human handover
        if action == self.action_space.n:
            # this is apparently worse than random :|
            # if np.random.rand() < 0.5:
            #     # random seen face down card
            #     seen_face_down = \
            #         (self.game_state.seen == MemoryGame.CardState.SEEN) \
            #         & (self.game_state.face_up-1 == np.arange(self.game.n_cards))
            #     if seen_face_down.any():
            #         action = np.random.choice(np.where(seen_face_down)[0])
            #     else:
            #         # random unseen card
            #         action = self.game.n_cards
            # else:
            #     # random unseen card
            #     action = self.game.n_cards
            action = np.random.choice(self.game.n_cards+1)
            # print(f'Overriding handover to {action}')
        # print(f'action: {action}')
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
        n_matches = 0
        while True:
            action = self.human(self.game_state)
            if action == self.handover_action:
                n_steps += 1
                self.total_human_steps += 1
                self.steps += 1
                trunc = self.timed_out
                break
            s, r, term, trunc, info = super().step(action)
            n_steps += 1
            self.total_human_steps += 1
            if r == 1.0:
                n_matches += 1
            R += r
            if term or trunc:
                break
        info['human_steps'] = n_steps
        info['human_matches'] = n_matches
        return s, R, term, trunc, info

    def reset(self, seed=None, options=None):
        # TODO set seed for self.env reset?
        n_reset = 0
        while True:
            # Reset and pick active player
            s, info = super().reset(seed, options)
            self.total_human_steps = 0
            n_reset += 1
            human_starts = np.random.choice([True, False])
            # If AI starts, we are good
            if not human_starts:
                break
            # If human starts, we need to play their turn
            # TODO provide reward as info to count for team return
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
            action = hamper_competence(action, self.human.category, self.ai_hampered_competence, self.game.n_cards)
            s, *step = super().step(action)
        step[3]['total_human_steps'] = self.total_human_steps
        # if step[2] and step[3].get('total_human_steps', 0) > 0:
        #     print(f'found trunc: {step[2]}')
        #     print(f'game state: {self.game_state}')
        #     print(f'self.n_steps: {self.steps}')
        #     print(f'self.total_human_steps: {self.total_human_steps}, info[total_human_steps]: {step[3]["total_human_steps"]}')
        #     ans = input('continue? Y/n: ')
        #     if ans == 'n':
        #         import sys; sys.exit(1)
        # if step[2]:
        #     import sys; sys.exit(1)
        return s, *step

    @staticmethod
    def ai_matches():
        return rl.Statistic(lambda n, ts: n + int(ts.reward == 1.0), 0)

    @staticmethod
    def human_matches():
        return rl.Statistic(lambda n, ts: n + ts.info.get('human_matches', 0), 0)

    @staticmethod
    def team_matches():
        return rl.Statistic(lambda n, ts: n + ts.info.get('human_matches', 0) + int(ts.reward == 1.0), 0)

    @staticmethod
    def team_len():
        return rl.Statistic(lambda n, ts: n + 1 + ts.info.get('human_steps', 0), 0)

    @staticmethod
    def human_moves():
        return rl.Statistic(lambda n, ts: n + ts.info.get('human_steps', 0), 0)

    @staticmethod
    def total_human_moves():
        return rl.Statistic(lambda n, ts: ts.next_info.get('total_human_steps', -1), -2)

    @staticmethod
    def invalid_actions():
        return rl.Statistic(lambda n, ts: n + int(ts.state == ts.next_state), 0)



def make_test_human(oracle, category, ctf):
    N = len(oracle)

    def policy(game_state: MemoryGame.State):
        seen = game_state.seen == MemoryGame.CardState.SEEN
        if game_state.face_up == 0:
            # if there is no card face up and we have seen a card from our
            # category, choose the first seen card from our category
            in_category = np.zeros(N, dtype=bool)
            for c in category:
                in_category |= (np.arange(N) == c)
            seen_in_category = seen & in_category
            if seen_in_category.any():
                # print(f'[H] random seen from category')
                return np.random.choice(np.nonzero(seen_in_category)[0])
            # if there is no card face up and we have not seen a card from our
            # category, randomly handover or choose a random unseen card
            # print(f'[H] handover or random unseen')
            return np.random.choice([N, N+1])
        
        # if a card face up is part of our category, try to choose its matched
        # card
        if game_state.face_up-1 in category:
            paired = oracle[game_state.face_up-1]
            if seen[paired]:
                # print(f'[H] match attempt of {game_state.face_up-1} with {paired}')
                return paired if np.random.rand() < ctf else np.random.choice(N)
            # paired card was not seen, choose a random unseen card
            return N

        # if a card face up is not part of our category, handover
        # print(f'[H] handover')
        return N+1

    return Player(policy, category, ctf)


def make_llm_human(category, ctf_level):
    assert ctf_level == 1 # medium (0.5 ctf)
    card_contents = [None]*8
    human_category = ["France", "Paris", "Italy", "Rome"]
    ai_category = ["7*8", "2*(5-3)", "56", "4"]
    for i, card in enumerate(category):
        card_contents[card] = human_category[i]
    for i in range(8):
        if card_contents[i] is None:
            card_contents[i] = ai_category.pop()
    
    policy = policy_bridge.PolicyBridge(card_contents, high_human.humanDecision)
    human = Player(policy, category, 0.5)
    return human


# Experiments

def get_make_env(env_id):
    game = MemoryGame(np.array([1, 0, 3, 2, 5, 4, 7, 6]))
    human_category = np.array([2, 3, 4, 5])

    if env_id == 'mixed_team':
        def make_env():
            return SemanticMemoryGameEnv(game, make_llm_human(human_category, 1))
        return make_env

    if env_id == 'mixed_team_test':
        def make_env():
            return SemanticMemoryGameEnv(game, make_test_human(game.oracle, human_category, 1.0))
        return make_env

    if env_id == 'partial_ai_only':
        def make_env():
            return HamperedAISemanticMemoryGameEnv(game, 0, human_category)
        return make_env
    
    if env_id == 'omniscent_ai_only':
        def make_env():
            return MemoryGameEnv(game)
        return make_env

    if env_id == 'test_human_only':
        def make_env():
            return HumanOnlySemanticMemoryGameEnv(game, make_test_human(game.oracle, human_category, 1.0))
        return make_env

    if env_id == 'human_only':
        def make_env():
            return HumanOnlySemanticMemoryGameEnv(game, make_llm_human(human_category, 1))
        return make_env

    raise ValueError(f'Unknown env_id: {env_id}')


def train(env_id):
    make_env = get_make_env(env_id)
    env = make_env()
    eval_env = make_env()

    stats = (
        rl.Stats.return_,
        SemanticMemoryGameEnv.team_len, # also counts human steps
        rl.Stats.terminated,
        SemanticMemoryGameEnv.human_moves,
    )

    n_steps = 200_000
    epsilon_schedule = rl.TabularQLearning.exponential_epsilon_decay(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=n_steps)
    algo = rl.TabularQLearning(
        alpha=0.2,
        gamma=0.99,
        epsilon_schedule=epsilon_schedule
    )

    random_policy_rets, random_policy_lens, random_policy_wins, random_policy_hsteps = \
        rl.evals(eval_env, lambda _: np.random.choice(env.action_space.n), stats, n=1000)
    random_policy_ret = random_policy_rets.mean()
    random_policy_len = random_policy_lens.mean()
    random_policy_wr = random_policy_wins.mean()
    random_policy_hsteps = random_policy_hsteps.mean()

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
        if i % (n_steps // 100) == 0:
            # evals.append(eval_policy(eval_env, agent.best_action))
            rets, ep_lens, terms, _ = rl.evals(eval_env, agent.best_action, stats, n=20)
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

    rets, ep_lens, terms, hsteps = rl.evals(eval_env, best_agent.best_action, stats, n=1000)
    print(f'final eval returns: {rets.mean()} +- {rets.std():.2f} (random: {random_policy_ret} +- {random_policy_rets.std():.2f})')
    print(f'final eval episode lengths: {ep_lens.mean()} +- {ep_lens.std():.2f} (random: {random_policy_len} +- {random_policy_lens.std():.2f})')
    print(f'final eval win rates: {terms.mean()} +- {terms.std():.2f} (random: {random_policy_wr} +- {random_policy_wins.std():.2f})')
    print(f'final eval human steps: {hsteps.mean()} +- {hsteps.std():.2f} (random: {random_policy_hsteps} +- {random_policy_hsteps.std():.2f})')

    results = {
        'train_eval_returns': eval_returns,
        'train_eval_lens': eval_lens,
        'train_eval_wrs': eval_wrs,
        'eval_rets': rets,
        'eval_lens': ep_lens,
        'eval_terms': terms
    }
    return results


def run_and_save_trials(n_trials, env_id):
    trial_results = [train(env_id) for _ in range(n_trials)]
    # stack results into single arrays
    results = {
        key: np.array([trial[key] for trial in trial_results])
        for key in trial_results[0]
    }
    np.savez(f'../results/{env_id}_results.npz', **results)
    return results


def eval_fixed_pi(pi_id):
    if pi_id == 'random':
        make_env = get_make_env('omniscent_ai_only')
        env = make_env()
        policy = lambda _: np.random.choice(env.action_space.n)
    elif pi_id == 'human':
        make_env = get_make_env('human_only')
        env = make_env()
        policy = env.human.policy
    else:
        raise ValueError(f'Unknown pi_id: {pi_id}')

    print(f'Evaluating {pi_id} fixed policy')

    stats = (
        rl.Stats.return_,
        SemanticMemoryGameEnv.team_len,
        rl.Stats.terminated,
        SemanticMemoryGameEnv.team_matches,
        SemanticMemoryGameEnv.invalid_actions,
    )

    results = rl.evals(env, policy, stats, n=1000)
    rets, lens, wins, n_matches, n_invalid = results

    if not (~wins | (n_matches == (env.game.n_cards // 2) - 1)).all():
        print(f'WARNING: policy did not find all matches in all episodes')
        matches_on_win = n_matches[wins]
        matches_on_loss = n_matches[~wins]
        print(f'         mean matches on win: {matches_on_win.mean()} +- {matches_on_win.std()}')
        print(f'         mean matches on loss: {matches_on_loss.mean()} +- {matches_on_loss.std()}')

    for stat, result in zip(stats, results):
        print(f'  {stat.__name__}: {result.mean()} +- {result.std()}')

    return {
        'returns': rets,
        'lens': lens,
        'wrs': wins
    }

def get_fixed_evals():
    fixed_evals = [
        'random',
        'human',
    ]
    fixed_results = {
        k: eval_fixed_pi(k)
        for k in fixed_evals
    }
    fixed_names = {
        'random': 'Random policy',
        'human': 'Human policy',
    }
    return fixed_results, fixed_names


def smooth(x, window_len):
    # return np.convolve(x, np.ones(window_len)/window_len, mode='full')
    # running mean of window_len (or less) values
    smoothed = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window_len + 1)
        smoothed[i] = np.mean(x[start:i+1])
    assert smoothed.shape == x.shape
    return smoothed


def plot_mean_std(x, y, yerr, label, smoothing_window=None, color=None, ax=plt):
    if ax is None:
        ax = plt.gca()

    if smoothing_window is not None:
        y = smooth(y, window_len=smoothing_window)
        yerr = smooth(yerr, window_len=smoothing_window)

    p = ax.plot(x, y, label=label, color=color)
    ax.fill_between(x, y-yerr, y+yerr, color=p[0].get_color(), alpha=0.15)


def print_evals():
    train_env_ids = [
        # 'omniscent_ai_only',
        # 'partial_ai_only',
        'mixed_team',
    ]
    train_results = {
        fn: np.load(f'../results/{fn}_results.npz')
        for fn in train_env_ids
    }
    train_fn_names = {
        'omniscent_ai_only': 'AI only (omniscent)',
        'partial_ai_only': 'AI only (partial competence)',
        'mixed_team': 'AI (partial competence) + human',
    }

    fixed_results, fixed_names = get_fixed_evals()

    for fn, results in train_results.items():
        print(f'{train_fn_names[fn]}:')
        for k, v in results.items():
            if k.startswith('eval'):
                print(f'  {k}: {v.mean()} +- {v.std()}')
    for k, v in fixed_results.items():
        print(f'{fixed_names[k]}:')
        for k, v in v.items():
            print(f'  {k}: {v.mean()} +- {v.std()}')


def plot_results():
    train_env_ids = [
        # 'omniscent_ai_only',
        # 'partial_ai_only',
        'mixed_team',
    ]
    train_results = {
        fn: np.load(f'../results/{fn}_results.npz')
        for fn in train_env_ids
    }
    train_fn_names = {
        'omniscent_ai_only': 'AI only (omniscent)',
        'partial_ai_only': 'AI only (partial competence)',
        'mixed_team': 'AI (partial competence) + human',
    }

    fixed_results, fixed_names = get_fixed_evals()

    infos = {
        'returns': {
            'title': 'Cumulative reward',
            'ylabel': 'Episode Return',
        },
        'lens': {
            'title': '(Team) steps to solution',
            'ylabel': 'Episode length',
        },
    }

    n_steps = 200_000
    train_xs = np.arange(n_steps, step=n_steps // 100)
    for k in ['returns', 'lens']:
        plt.title(infos[k]['title'])
        plt.ylabel(infos[k]['ylabel'])
        plt.xlabel('Training steps')
        for fn, results in train_results.items():
            plot_mean_std(
                train_xs,
                results[f'train_eval_{k}'].mean(axis=0),
                results[f'train_eval_{k}'].std(axis=0),
                label=train_fn_names[fn],
                smoothing_window=10,
            )
        for fixed_policy, results in fixed_results.items():
            plot_mean_std(
                train_xs,
                np.full_like(train_xs, results[k].mean()),
                np.full_like(train_xs, results[k].std()),
                label=fixed_names[fixed_policy],
            )
        plt.legend()
        plt.savefig(f'../images/{k}.png')
        plt.show()


def eval_only_random_with_human():
    make_env = get_make_env('mixed_team')
    eval_env = make_env()

    stats = (
        rl.Stats.return_,
        SemanticMemoryGameEnv.team_len,
        rl.Stats.terminated,
        SemanticMemoryGameEnv.team_matches,
        SemanticMemoryGameEnv.ai_matches,
        SemanticMemoryGameEnv.human_matches,
        SemanticMemoryGameEnv.human_moves,
        SemanticMemoryGameEnv.total_human_moves,
    )

    random_policy_rets, random_policy_lens, random_policy_wins, n_matches, n_hmatches, n_aimatches, hsteps, t_hsteps = \
        rl.evals(eval_env, lambda _: np.random.choice(eval_env.action_space.n), stats, n=1000)

    # TODO these metrics don't actually work (probably cause r == 1 is not always counting matches)
    # if not (~random_policy_wins | (n_matches == (eval_env.game.n_cards // 2) - 1)).all():
    #     print(f'WARNING: random policy did not find all matches in all episodes')
    #     matches_on_win = n_matches[random_policy_wins]
    #     matches_on_loss = n_matches[~random_policy_wins]
    #     print(f'         mean matches on win: {matches_on_win.mean()} +- {matches_on_win.std()}')
    #     print(f'         mean matches on loss: {matches_on_loss.mean()} +- {matches_on_loss.std()}')
    #     print(f'         mean human matches on win: {n_hmatches[random_policy_wins].mean()} +- {n_hmatches[random_policy_wins].std()}')
    #     print(f'         mean human matches on loss: {n_hmatches[~random_policy_wins].mean()} +- {n_hmatches[~random_policy_wins].std()}')
    #     print(f'         mean ai matches on win: {n_aimatches[random_policy_wins].mean()} +- {n_aimatches[random_policy_wins].std()}')
    #     print(f'         mean ai matches on loss: {n_aimatches[~random_policy_wins].mean()} +- {n_aimatches[~random_policy_wins].std()}')

    if not (random_policy_wins | (random_policy_lens == 256)).all():
        print(f'WARNING: random policy win and timeout inconsistency')
        loss_lens = random_policy_lens[~random_policy_wins]
        print(f'         mean episode length on loss: {loss_lens.mean()} +- {loss_lens.std()}')
        print(f'         mean human steps on loss: {hsteps[~random_policy_wins].mean()} +- {hsteps[~random_policy_wins].std()}')
        print(f'         mean total human steps on loss: {t_hsteps[~random_policy_wins].mean()} +- {t_hsteps[~random_policy_wins].std()}')

    print(f'random policy returns: {random_policy_rets.mean()} +- {random_policy_rets.std()}')
    print(f'random policy episode lengths: {random_policy_lens.mean()} +- {random_policy_lens.std()}')
    print(f'random policy win rate: {random_policy_wins.mean()} +- {random_policy_wins.std()}')
    print(f'random policy human steps: {hsteps.mean()} +- {hsteps.std()}')


def main():
    # TODO plots
    # return train
    # puzzle solution time
    # visual repr of optimal policy execution:
    # - show which categories are chosen by each agent
    # - handover/steps percentage per agent
    # - omniscent AI + human
    # - multiple humans
    pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) not in {2,3}:
        print('Wrong number of arguments')
        sys.exit(1)

    if sys.argv[1] == 'plot':
        assert len(sys.argv) == 2
        plot_results()
        sys.exit(0)

    if sys.argv[1] == 'print':
        assert len(sys.argv) == 2
        print_evals()
        sys.exit(0)

    if sys.argv[1] == 'eval':
        assert len(sys.argv) == 3
        eval_fixed_pi(sys.argv[2])
        sys.exit(0)

    if sys.argv[1] == 'train':
        assert len(sys.argv) == 3
        run_and_save_trials(5, sys.argv[2])
        sys.exit(0)

    if sys.argv[1] == 'test':
        assert len(sys.argv) == 2
        eval_only_random_with_human()
        sys.exit(0)

