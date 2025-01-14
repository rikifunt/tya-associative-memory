from dataclasses import dataclass
from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from associative_memory_game import AssociativeMemoryGame


class AssociativeMemoryGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, game: AssociativeMemoryGame, reward_scheme: str = "sparse"):
        self.game = game
        self.reward_scheme = reward_scheme
        self.id_to_card_and_category = tuple(self.game.cards_and_categories())
        self.card_and_category_to_id = {pair: i for i, pair in enumerate(self.id_to_card_and_category)}

        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                0, 1+len(self.id_to_card_and_category),
                shape=(len(self.id_to_card_and_category),),
                dtype=int,
            ),
            "turn": spaces.Discrete(2),
            "cardA": spaces.Discrete(len(self.id_to_card_and_category)+1),
            "cardB": spaces.Discrete(len(self.id_to_card_and_category)+1),
        })
        self.action_space = spaces.Discrete(len(self.id_to_card_and_category))

        self.game_state = None
        self.board_omniscent = None
        # Any card that has ever been faceup
        self.faceup_historical = None

    # each element in board is:
    #  0: card was never seen
    #  1...N: card was seen
    def observe(self):
        # State space has at most cardinality:
        #   (N+1)! * 2 * N * (N-1)
        # which is actually very large for tabular RL?
        assert self.game_state is not None
        board = 1 + self.board_omniscent
        board[~self.faceup_historical] = 0
        return {
            "board": board,
            "turn": self.game_state.turn,
            "cardA": 1+self.game_state.cardA if self.game_state.cardA is not None else 0,
            "cardB": 1+self.game_state.cardB if self.game_state.cardB is not None else 0,
        }

    def reward(self):
        if self.reward_scheme == "sparse":
            if self.game_state is None:
                print("WIN")
                return 1.0
            else:
                return 0.0
        elif self.reward_scheme == "dense":
            if self.game_state is None:
                print("WIN")
                return 10.0
            if self.game.has_new_match(self.game_state):
                # print("MATCH")
                return 1.0
            else:
                return -1.0
        else:
            raise ValueError(f"Unknown reward scheme: {self.reward_scheme}")

    def reset(self, seed=None, options=None):
        self.game_state = self.game.reset(seed)
        self.board_omniscent = np.array([
            self.card_and_category_to_id[category, card]
            for card, category in zip(self.game_state.cards, self.game_state.categories)
        ])
        self.faceup_historical = np.zeros(len(self.id_to_card_and_category), dtype=bool)
        return self.observe(), {}

    def step(self, action):
        self.game_state = self.game.step(self.game_state, action)
        self.faceup_historical |= self.game_state.faceup
        term = self.game_state is None
        trunc = False
        return self.observe(), self.reward(), term, trunc, {}

from gymnasium.envs.registration import register

register(
    id="AssociativeMemoryGameEnv-v0",
    entry_point="upper_bound:AssociativeMemoryGameEnv",
    kwargs={
        "game": AssociativeMemoryGame.make(
            letters=(
                ("A", "a"), ("B", "b"),
            ),
            numbers=(
                ("1", "one"), ("2", "two"),
            ),
        ),
        "reward_scheme": "dense"
    },
    max_episode_steps=100,
)



def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # Parallel environments
    vec_env = make_vec_env("AssociativeMemoryGameEnv-v0", n_envs=4)

    model = PPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=500000)
    model.save("ppo")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo")

    obs = vec_env.reset()
    return_ = 0.0
    steps = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        return_ += rewards[0]
        steps += 1
        done = dones[0]
        # vec_env.render("human")
    print(f"Return: {return_}")
    print(f"Steps: {steps}")

if __name__ == "__main__":
    main()



# class RobotPreprocessor:
#     """Takes care of converting string-based card observations to numerical.
    
#     For each category, it has a different preprocessor that returns an integer
#     from a string, e.g.:
#     - an ID for an association found in a DB;
#     - the solution of a math problem in numerical representation;
#     - ...
#     """

#     preprocessors: dict[str, Callable[[str], int]]

#     def __call__(self, category: str, card: str):
#         return self.preprocessors[category](card)


# @dataclass(frozen=True)
# class DBPreprocessor:
#     """Query a DB of known associations for the association ID.
#     """

#     # Each entry must have exactly two strings, and each string must be unique.
#     db: tuple[frozenset[str]]
#     default: int = -1

#     def __call__(self, card: str):
#         for i, pair in enumerate(self.db):
#             if card in pair:
#                 return i
#         return self.default
