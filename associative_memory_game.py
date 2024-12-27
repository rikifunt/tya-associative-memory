from dataclasses import dataclass
from functools import reduce
from logging import warning
from typing import Iterable, Iterator, NamedTuple

import numpy as np
import pygame


@dataclass
class AssociativeMemoryGame:
    class State(NamedTuple):
        # All cards on the board with their contents
        cards: np.ndarray[np.str_]
        # Boolean mask of which cards are face up
        faceup: np.ndarray[bool]
        # Active player
        turn: bool
        # Card A chosen by active player (if any)
        cardA: int | None
        # Suggestion in the form (card, category) for card B by non-active player
        suggestionB: tuple[int, int] | None
    
    # Correct associations between cards. NOTE: each string must only appear
    # *once* for consistency. Each entry must contain exactly two strings.
    oracle: tuple[frozenset[str], ...]

    @staticmethod
    def as_oracle(*pairs: tuple[str, str]) -> tuple[frozenset[str], ...]:
        return tuple(map(frozenset, pairs))

    def cards(self) -> Iterator[str]:
        cards = reduce(lambda A, B: tuple(A) + tuple(B), self.oracle, ())
        return cards

    def __post_init__(self):
        if not all(len(pair) == 2 for pair in self.oracle):
            raise ValueError("Each entry in oracle must contain exactly two strings")
        cards = self.cards()
        if not len(cards) == len(set(cards)):
            raise ValueError("Each string must appear exactly once in the oracle")

    def reset(self, seed=42) -> State:
        rng = np.random.default_rng(seed)
        cards = np.array(self.cards())
        rng.shuffle(cards)
        return self.State(
            cards=cards,
            faceup=np.zeros(len(cards), dtype=bool),
            turn=0,
            cardA=None,
            suggestionB=None
        )

    def step(self, state: State, action: tuple[int, int]) -> State | None:
        card, category = action
        if card not in range(len(state.cards)):
            raise ValueError(f"Invalid card index {card}")

        if state.cardA is None:
            # Active player chooses card A
            assert state.suggestionB is None, "Suggestion B chosen before card A"
            assert state.faceup.sum() % 2 == 0, f"Odd number of face-up cards"
            if state.faceup[card]:
                # Card is already face up, do nothing
                warning(f'Card A already face up: {card}')
                return state
            faceup = state.faceup.copy()
            faceup[card] = True
            # print(f'Card A chosen: {card}')
            return state._replace(cardA=card, faceup=faceup)
        
        if state.suggestionB is None:
            # Non-active player suggests card B
            # print(f'Suggestion B chosen: {card}, {category}')
            return state._replace(suggestionB=(card, category))

        # Both cards are chosen
        if state.faceup[card]:
            # Card B is already face up, do nothing
            warning(f'Card B already face up: {card}')
            return state

        if {state.cards[state.cardA], state.cards[card]} in self.oracle:
            # Match found, keep cards face up
            faceup = state.faceup.copy()
            faceup[card] = True
            # print(f'Match found: {state.cardA}, {card}')
            if faceup.sum() == len(state.cards):
                # All cards are face up, game over
                return None
            return state._replace(faceup=faceup, cardA=None, suggestionB=None)
        
        # No match, cover cards and switch turn
        faceup = state.faceup.copy()
        faceup[state.cardA] = False
        # print(f'No match: {state.cardA}, {card}')
        return state._replace(faceup=faceup, cardA=None, suggestionB=None, turn=not state.turn)

    @staticmethod
    def grid_dimensions(num_cards):
        factors = [i for i in range(1, num_cards + 1) if num_cards % i == 0]
        cols = min(factors, key=lambda x: abs(x - np.sqrt(num_cards)))
        rows = int(num_cards // cols)
        return rows, cols

    @staticmethod
    def card_render_dims(width, height, rows, cols):
        PADDING = min(width, height) // 25
        CARD_WIDTH = (width - (cols + 1) * PADDING) // cols
        CARD_HEIGHT = (height - (rows + 2) * PADDING) // rows  # Reserve space for text at the bottom
        return CARD_WIDTH, CARD_HEIGHT, PADDING

    # TODO fix bug of cardA remain faceup sometimes
    # Draw using pygame (needs pygame.init() to be called before)
    def render(self, state: State, width: int, height: int, rows: int, cols: int) -> pygame.Surface:
        # Constants
        FONT_SIZE = 24
        BG_COLOR = (30, 30, 30)
        CARD_COLOR = (200, 200, 200)
        FACEUP_COLOR = (255, 255, 255)
        TEXT_COLOR = (0, 0, 0)
        STATE_LINE_COLOR = (255, 255, 255)

        # Create display surface
        canvas = pygame.Surface((width, height))

        # Calculate card dimensions and padding
        CARD_WIDTH, CARD_HEIGHT, PADDING = self.card_render_dims(width, height, rows, cols)

        # Font setup
        font = pygame.font.Font(None, FONT_SIZE)

        # Fill background
        canvas.fill(BG_COLOR)

        # Draw cards
        for i, card in enumerate(state.cards):
            row, col = divmod(i, cols)
            x = col * (CARD_WIDTH + PADDING) + PADDING
            y = row * (CARD_HEIGHT + PADDING) + PADDING

            if state.faceup[i]:
                color = FACEUP_COLOR
                text = str(card)
                print(f'[U] Card {i}: {card}, faceup: {state.faceup[i]}')
            else:
                color = CARD_COLOR
                text = ""
                print(f'[D] Card {i}: {card}, faceup: {state.faceup[i]}')

            pygame.draw.rect(canvas, color, (x, y, CARD_WIDTH, CARD_HEIGHT))
            text_surface = font.render(text, True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT // 2))
            canvas.blit(text_surface, text_rect)

        # Print state at the bottom
        state_text = f"Turn: {'Player 1' if state.turn == 0 else 'Player 2'}, Card A: {state.cardA}, Suggestion B: {state.suggestionB}"
        state_surface = font.render(state_text, True, STATE_LINE_COLOR)
        state_rect = state_surface.get_rect(center=(width // 2, height - PADDING))
        canvas.blit(state_surface, state_rect)

        # Convert pygame surface to numpy array
        # view = pygame.surfarray.array3d(canvas)
        # pygame.quit()
        # return np.transpose(view, (1, 0, 2))
        return canvas

    def play(self):
        pygame.init()
        state = self.reset()
        W, H = 800, 600
        screen = pygame.display.set_mode((W, H))
        running = True
        rows, cols = self.grid_dimensions(len(state.cards))
        CARD_WIDTH, CARD_HEIGHT, PADDING = self.card_render_dims(W, H, rows, cols)
        screen.blit(self.render(state, W, H, rows, cols), (0, 0))
        pygame.display.flip()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    col = x // (CARD_WIDTH + PADDING)
                    row = y // (CARD_HEIGHT + PADDING)
                    card = col + row * cols
                    print('Card:', card)
                    state = self.step(state, (card, 0))
                    if state is None:
                        running = False
                    else:
                        screen.blit(self.render(state, W, H, rows, cols), (0, 0))
                        pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    game = AssociativeMemoryGame(AssociativeMemoryGame.as_oracle(
        ("A", "a"), ("B", "b"),
        ("C", "c"), ("D", "d"),
    ))
    game.play()

