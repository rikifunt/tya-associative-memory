from dataclasses import dataclass, field
from enum import IntEnum
from functools import reduce
from logging import warning
from typing import Iterable, Iterator, NamedTuple

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces



@dataclass
class MemoryGame:
    class CardState(IntEnum):
        UNSEEN = 0 # card was never seen
        FACE_DOWN = 1 # card was seen and is face down
        FACE_UP = 2 # card was seen and is face up
        SOLVED = 3 # card is solved (face up, matched with the correct card)

    oracle: np.ndarray[int] # for each card, the card it matches with

    @property
    def n_cards(self) -> int:
        return len(self.oracle)

    def __post_init__(self):
        if len(self.oracle.shape) != 1:
            raise ValueError("Oracle must be a 1D array")
        if len(np.unique(self.oracle)) != len(self.oracle):
            raise ValueError("Each card must have a unique match")

    def reset(self, seed=None) -> np.ndarray[int]:
        return np.full(self.n_cards, MemoryGame.CardState.UNSEEN)

    # TODO try having 1 special action for turn random card, and make invalid
    # turning up unseen cards (no-op)
    def step(self, state: np.ndarray[int], action: int) -> np.ndarray[int] | None:
        if (state == MemoryGame.CardState.SOLVED).all():
            # Game is over (all cards are solved)
            # print(f'[Game] All cards are solved')
            return state.copy()

        if state[action] == MemoryGame.CardState.FACE_UP or state[action] == MemoryGame.CardState.SOLVED:
            # Do nothing (invalid action)
            # print(f'[Game] Card {action} is already face up or solved')
            return state.copy()

        # Action is valid: turn card face up (if unseen card chosen, randomize
        # the choice)

        if state[action] == MemoryGame.CardState.UNSEEN:
            # Choose a random unseen card
            action = np.random.choice(np.where(state == MemoryGame.CardState.UNSEEN)[0])
            # print(f'[Game] choosing random unseen card: {action}')

        return self.turn_face_up(state, action)

    def turn_face_up(self, state: np.ndarray[int], card: int) -> np.ndarray[int]:
        next_state = state.copy()
        # print(f'[Game] Turning card {card} face up')
        # Turn card face up
        next_state[card] = MemoryGame.CardState.FACE_UP
        # If there are two face up cards, check if they match
        face_up = np.where(next_state == MemoryGame.CardState.FACE_UP)[0]
        assert len(face_up) <= 2
        if len(face_up) == 2:
            if self.oracle[face_up[0]] == face_up[1]:
                # print(f'[Game] Matched cards {face_up[0]} and {face_up[1]} :)')
                next_state[face_up] = MemoryGame.CardState.SOLVED
            else:
                # print(f'[Game] Cards {face_up[0]} and {face_up[1]} do not match :(')
                next_state[face_up] = MemoryGame.CardState.FACE_DOWN
        return next_state

    def have_new_match(self, state: np.ndarray[int], next_state: np.ndarray[int]) -> bool:
        # number of solved cards increased
        return np.sum(next_state == MemoryGame.CardState.SOLVED) > np.sum(state == MemoryGame.CardState.SOLVED)


@dataclass
class InteractiveMatch:
    game: MemoryGame
    width: int
    height: int
    cards: tuple[str, ...]
    categories: tuple[str, ...]
    rows: int = field(init=False)
    cols: int = field(init=False)
    
    def __post_init__(self):
        self.rows, self.cols = self.grid_dimensions(self.game.n_cards)

    @staticmethod
    def grid_dimensions(num_cards):
        factors = [i for i in range(1, num_cards + 1) if num_cards % i == 0]
        cols = min(factors, key=lambda x: abs(x - np.sqrt(num_cards)))
        rows = int(num_cards // cols)
        return rows, cols

    @property
    def padding(self):
        return min(self.width, self.height) // 25

    @property
    def card_width(self):
        return (self.width - (self.cols + 1) * self.padding) // self.cols
    
    @property
    def card_height(self):
        return (self.height - (self.rows + 2) * self.padding) // self.rows

    # Draw using pygame (needs pygame.init() to be called before)
    def render(self, state: np.ndarray) -> pygame.Surface:
        # Constants
        FONT_SIZE = 24
        BG_COLOR = (30, 30, 30)
        COLORS = (
            (200, 200, 200), # unseen
            (255, 0, 0), # seen, face down
            (255, 255, 0), # face up
            (0, 255, 0), # solved
        )
        TEXT_COLOR = (0, 0, 0)
        STATE_LINE_COLOR = (255, 255, 255)

        # Create display surface
        canvas = pygame.Surface((self.width, self.height))

        # Font setup
        font = pygame.font.Font(None, FONT_SIZE)

        # Fill background
        canvas.fill(BG_COLOR)

        # Draw cards
        for card, card_state in enumerate(state):
            row, col = divmod(card, self.cols)
            x = col * (self.card_width + self.padding) + self.padding
            y = row * (self.card_height + self.padding) + self.padding

            color = COLORS[card_state]
            pygame.draw.rect(canvas, color, (x, y, self.card_width, self.card_height))

            if card_state != MemoryGame.CardState.UNSEEN:
                text = self.cards[card]
                text = f'[{self.cards[card]}] {text}'
                text_surface = font.render(text, True, TEXT_COLOR)
                text_rect = text_surface.get_rect(center=(x + self.card_width // 2, y + self.card_height // 2))
                canvas.blit(text_surface, text_rect)

        # Print text at the bottom
        bottom_text = f"..."
        bottom_text_surface = font.render(bottom_text, True, STATE_LINE_COLOR)
        bottom_text_rect = bottom_text_surface.get_rect(center=(self.width // 2, self.height - self.padding))
        canvas.blit(bottom_text_surface, bottom_text_rect)

        return canvas

    def cursor_to_card(self, x, y):
        col = x // (self.card_width + self.padding)
        row = y // (self.card_height + self.padding)
        card = col + row * self.cols
        return card

    def play(self):
        pygame.init()
        state = self.game.reset()
        screen = pygame.display.set_mode((self.width, self.height))
        screen.blit(self.render(state), (0, 0))
        pygame.display.flip()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                action = None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    card = self.cursor_to_card(*event.pos)
                    # print('[RENDER] Chosen card:', card)
                    if card < self.game.n_cards:
                        action = card
                if action is not None:
                    state = self.game.step(state, action)
                    if state is None:
                        # print('[RENDER] Game over (Win)')
                        running = False
                    else:
                        screen.blit(self.render(state), (0, 0))
                        pygame.display.flip()
        pygame.quit()

def test_interactive_match():
    m = InteractiveMatch(
        MemoryGame(np.array([1, 0, 3, 2])),
        width=800, height=600,
        cards=('Paris', 'France', '2*(5-3)', '4'),
        categories=('Capitals', 'Capitals', 'Math', 'Math')
    )
    m.play()



class MemoryGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, game: MemoryGame, max_steps=None):
        self.game = game
        self.max_steps = max_steps if max_steps is not None else 2**self.game.n_cards
        self.observation_space = spaces.Discrete(4**self.game.n_cards)
        self.action_space = spaces.Discrete(self.game.n_cards)
        self.final_state = np.full(self.game.n_cards, MemoryGame.CardState.SOLVED)
        self.game_state = np.full(self.game.n_cards, MemoryGame.CardState.UNSEEN)
        self.steps = None

    def observe(self):
        # map from array of values in 0-3 to single integer
        return (4**np.arange(self.game.n_cards) * self.game_state).sum()

    def reward(self, s, s1):
        if (s1 == self.final_state).all():
            return 100.0
        if self.game.have_new_match(s, s1):
            # print("MATCH")
            return 1.0
        return 0.0

    def reset(self, seed=None, options=None):
        # print(f'[env] reset')
        self.game_state = self.game.reset()
        self.steps = 0
        return self.observe(), {}

    def step(self, action):
        last_state = self.game_state
        # print(f'[env] last_state is None: {last_state is None}')
        self.game_state = self.game.step(self.game_state, action)
        self.steps += 1
        term = (self.game_state==self.final_state).all()
        trunc = self.steps >= self.max_steps
        # print(f'[env] state is None: {self.game_state is None}, term: {term}')
        return self.observe(), self.reward(last_state, self.game_state), term, trunc, {}

# from gymnasium.envs.registration import register

# register(
#     id="AssociativeMemoryGameEnv-v0",
#     entry_point="upper_bound:AssociativeMemoryGameEnv",
#     kwargs={
#         "game": SemanticMemoryGame(),
#     },
#     max_episode_steps=100,
# )

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



if __name__ == '__main__':
    test_interactive_match()
