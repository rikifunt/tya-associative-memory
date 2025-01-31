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
        UNSEEN = 0 # card was never seen (position is unknown)
        SEEN = 1 # card was seen (position is known)
        SOLVED = 2 # card is solved (face up and matched correctly)

    class State(NamedTuple):
        seen: np.ndarray[int] # one CardState per card
        face_up: int # 0 for no card face up, 1..N for card i-1 face up temporarily

        def copy(self):
            return MemoryGame.State(seen=self.seen.copy(), face_up=self.face_up)

        def __eq__(self, other: 'MemoryGame.State') -> bool:
            return (self.seen == other.seen).all() and self.face_up == other.face_up

        @property
        def n_solved(self) -> int:
            return np.sum(self.seen == MemoryGame.CardState.SOLVED)

        @property
        def solved(self) -> bool:
            return (self.seen == MemoryGame.CardState.SOLVED).all()


    oracle: np.ndarray[int] # for each card, the card it matches with

    @property
    def n_cards(self) -> int:
        return len(self.oracle)

    def __post_init__(self):
        if len(self.oracle.shape) != 1:
            raise ValueError("Oracle must be a 1D array")
        if len(np.unique(self.oracle)) != len(self.oracle):
            raise ValueError("Each card must have a unique match")

    def reset(self, seed=None) -> State:
        return MemoryGame.State(np.full(self.n_cards, MemoryGame.CardState.UNSEEN), 0)

    def step(self, state: State, action: int) -> State:
        seen, face_up = state
        if state.solved:
            # Game is over (all cards are solved)
            # print(f'[Game] All cards are solved')
            return state.copy()

        if action == (face_up-1) or action < self.n_cards and seen[action] in {MemoryGame.CardState.SOLVED, MemoryGame.CardState.UNSEEN}:
            # Do nothing (invalid action)
            # print(f'[Game] Card {action} is unseen or solved')
            return state.copy()

        # Action is valid: turn card face up

        if action == self.n_cards:
            # Choose a random unseen card
            unseen_cards = np.where(seen == MemoryGame.CardState.UNSEEN)[0]
            if len(unseen_cards) == 0:
                # No unseen cards left, invalid action
                # print(f'[Game] No unseen cards left')
                return state.copy()
            action = np.random.choice(unseen_cards)
            # print(f'[Game] choosing random unseen card: {action}')

        return self.turn_face_up(state, action)

    def turn_face_up(self, state: State, card: int) -> State:
        seen, face_up = state
        assert seen[card] != MemoryGame.CardState.SOLVED

        next_seen, next_face_up = state.copy()
        # Mark the card to be turned face up as seen
        # print(f'[Game] Turning card {card} face up')
        next_seen[card] = MemoryGame.CardState.SEEN

        # If there is no face up card, turn this card face up
        if face_up == 0:
            # print(f'[Game] No other card face up, turning card {card} face up')
            next_face_up = card + 1
            return MemoryGame.State(next_seen, next_face_up)

        # If there are two face up cards, check if they match
        assert seen[face_up-1] == MemoryGame.CardState.SEEN
        if self.oracle[face_up-1] == card:
            # print(f'[Game] Cards {face_up-1} and {card} match')
            next_seen[face_up-1] = next_seen[card] = MemoryGame.CardState.SOLVED
        next_face_up = 0
        return MemoryGame.State(next_seen, next_face_up)

    def have_new_match(self, state: State, next_state: State) -> bool:
        # number of solved cards increased
        return next_state.n_solved > state.n_solved


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
    def render(self, state: MemoryGame.State) -> pygame.Surface:
        # Constants
        FONT_SIZE = 24
        BG_COLOR = (30, 30, 30)
        COLORS = (
            (200, 200, 200), # unseen
            (255, 0, 0), # seen, face down
            (0, 255, 0), # solved
            (255, 255, 0), # face up temporarily
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
        for card, card_state in enumerate(state.seen):
            if card == state.face_up - 1:
                card_state = MemoryGame.CardState.SOLVED + 1
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
                    print('[RENDER] Chosen card:', card)
                    if card < self.game.n_cards:
                        action = card
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = self.game.n_cards
                if action is not None:
                    state = self.game.step(state, action)
                    if state.solved:
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
        N = self.game.n_cards
        self.max_steps = max_steps if max_steps is not None else 2**N
        self.observation_space = spaces.Discrete((N+1) * 3**N)
        self.action_space = spaces.Discrete(1 + N)
        self.game_state = None
        self.steps = None

    def observe(self):
        N = self.game.n_cards
        return self.game_state.face_up*(3**N) + (3**np.arange(N) * self.game_state.seen).sum()

    def reward(self, s, s1):
        if s1.solved:
            # print("SOLVED")
            return 100.0
        if self.game.have_new_match(s, s1):
            # print("MATCH")
            return 1.0
        # print("NO MATCH")
        return -1.0

    @property
    def timed_out(self):
        return self.steps >= self.max_steps

    def reset(self, seed=None, options=None):
        # print(f'[env] reset')
        self.game_state = self.game.reset()
        self.steps = 0
        self.n_matches = 0
        # print(f'[GAME] END RESET')
        return self.observe(), {}

    def step(self, action):
        last_state = self.game_state
        # print(f'[env] last_state is None: {last_state is None}')
        self.game_state = self.game.step(self.game_state, action)
        self.steps += 1
        term = self.game_state.solved
        # print(f'[GAME] steps: {self.steps}, max_steps: {self.max_steps}, term: {term}, trunc: {self.timed_out}')
        # print(f'[env] state is None: {self.game_state is None}, term: {term}')
        if self.game.have_new_match(last_state, self.game_state):
            self.n_matches += 1
        assert not term or self.n_matches == self.game.n_cards // 2
        return self.observe(), self.reward(last_state, self.game_state), term, self.timed_out, {}

# from gymnasium.envs.registration import register

# register(
#     id="AssociativeMemoryGameEnv-v0",
#     entry_point="upper_bound:AssociativeMemoryGameEnv",
#     kwargs={
#         "game": SemanticMemoryGame(),
#     },
#     max_episode_steps=100,
# )


if __name__ == '__main__':
    test_interactive_match()
