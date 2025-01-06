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
        # Categories for each card
        categories: np.ndarray[np.str_]
        # Boolean mask of which cards are face up
        faceup: np.ndarray[bool]
        # Active player
        turn: bool
        # Card A chosen by active player (if any)
        cardA: int | None
        # Card B chosen by active player (if any)
        cardB: int | None
        # Suggestion in the form (card, category) for card A by non-active player
        suggestionA: tuple[int, int] | None
        # Suggestion in the form (card, category) for card B by non-active player
        suggestionB: tuple[int, int] | None

        # Using a singleton since None is a valid value for some fields
        class _Missing: pass

        def deep_update(
            self,
            cards=_Missing,
            categories=_Missing,
            faceup=_Missing,
            turn=_Missing,
            cardA=_Missing,
            cardB=_Missing,
            suggestionA=_Missing,
            suggestionB=_Missing
        ):
            missing = AssociativeMemoryGame.State._Missing
            return AssociativeMemoryGame.State(
                cards=self.cards.copy() if cards is missing else cards,
                categories=self.categories.copy() if categories is missing else categories,
                faceup=self.faceup.copy() if faceup is missing else faceup,
                turn=self.turn if turn is missing else turn,
                cardA=self.cardA if cardA is missing else cardA,
                cardB=self.cardB if cardB is missing else cardB,
                suggestionA=self.suggestionA if suggestionA is missing else suggestionA,
                suggestionB=self.suggestionB if suggestionB is missing else suggestionB,
            )

    # Correct associations between cards. NOTE: each string must only appear
    # *once* in each category for consistency. Each item has the category as
    # key, and the entry in the form {textA, textB} as value.
    oracle: dict[str, tuple[frozenset[str], ...]]

    # @staticmethod
    # def as_oracle(*pairs: tuple[str, str]) -> tuple[frozenset[str], ...]:
    #     return tuple(map(frozenset, pairs))

    @staticmethod
    def make(**items: Iterable[tuple[str, str]]) -> "AssociativeMemoryGame":
        return AssociativeMemoryGame({k: tuple(frozenset(vi) for vi in v) for k, v in items.items()})

    def cards_and_categories(self) -> Iterator[str]:
        for category, associations in self.oracle.items():
            for cardA, cardB in associations:
                yield category, cardA
                yield category, cardB

    def __post_init__(self):
        for category, associations in self.oracle.items():
            if not all(len(pair) == 2 for pair in associations):
                raise ValueError(f"Each entry in a category {category} must contain exactly two strings")
            # TODO check each string appears only once for category
            # if not len(associations) == len(set(associations)):
            #     raise ValueError(f"Each pair of strings must appear exactly once in category {category}")

    def reset(self, seed=42) -> State:
        rng = np.random.default_rng(seed)
        cards_and_categories = list(self.cards_and_categories())
        rng.shuffle(cards_and_categories)
        cards = np.array([card for _, card in cards_and_categories], dtype=np.str_)
        categories = np.array([category for category, _ in cards_and_categories], dtype=np.str_)
        return self.State(
            cards=cards,
            categories=categories,
            faceup=np.zeros(len(cards), dtype=bool),
            turn=0,
            cardA=None,
            cardB=None,
            suggestionA=None,
            suggestionB=None
        )

    def step(self, state: State, action: tuple[int, int]) -> State | None:
        card, category = action
        if card not in range(len(state.cards)):
            raise ValueError(f"Invalid card index {card}")

        if state.suggestionA is None:
            # Non-active player suggests card A
            # print(f'Suggestion A chosen: {card}, {category}')
            return state.deep_update(suggestionA=(card, category))

        if state.cardA is None:
            # Active player chooses card A
            assert state.cardB is None, "Card B chosen before card A"
            assert state.suggestionB is None, "Suggestion B chosen before card A"
            assert state.faceup.sum() % 2 == 0, f"Odd number of face-up cards"
            if state.faceup[card]:
                # Card is already face up, do nothing
                warning(f'Card A already face up: {card}')
                return state.deep_update()
            faceup = state.faceup.copy()
            faceup[card] = True
            # print(f'Card A chosen: {card}')
            return state.deep_update(cardA=card, faceup=faceup)
        
        if state.suggestionB is None:
            # Non-active player suggests card B
            # print(f'Suggestion B chosen: {card}, {category}')
            return state.deep_update(suggestionB=(card, category))

        # Both cards are chosen (card B could still be face down)

        # Turn card B face up
        if state.cardB is None:
            if state.faceup[card]:
                # Card B is already face up, do nothing
                warning(f'Card B already face up: {card}')
                return state.deep_update()
            # Active player chooses card B
            faceup = state.faceup.copy()
            faceup[card] = True
            # print(f'Card B chosen: {card}')
            return state.deep_update(cardB=card, faceup=faceup)

        # Both cards are face up, check for match
        if state.categories[state.cardA] == state.categories[state.cardB]:
            if {state.cards[state.cardA], state.cards[state.cardB]} in self.oracle[state.categories[state.cardA]]:
                # Match found, keep cards face up and switch turn
                faceup = state.faceup.copy()
                # print(f'Match found: {state.cardA}, {card}')
                if faceup.sum() == len(state.cards):
                    # All cards are face up, game over
                    return None
                return state.deep_update(faceup=faceup, cardA=None, cardB=None, suggestionA=None, suggestionB=None, turn=not state.turn)
        
        # No match, cover cards and switch turn
        faceup = state.faceup.copy()
        faceup[state.cardA] = False
        faceup[state.cardB] = False
        # print(f'No match: {state.cardA}, {state.cardB}')
        return state.deep_update(faceup=faceup, cardA=None, cardB=None, suggestionA=None, suggestionB=None, turn=not state.turn)

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
                text = f'[{state.categories[i]}] {text}'
                # print(f'[U] Card {i}: {card}, faceup: {state.faceup[i]}')
            else:
                color = CARD_COLOR
                text = ""
                # print(f'[D] Card {i}: {card}, faceup: {state.faceup[i]}')

            pygame.draw.rect(canvas, color, (x, y, CARD_WIDTH, CARD_HEIGHT))
            text_surface = font.render(text, True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT // 2))
            canvas.blit(text_surface, text_rect)

        # Print state at the bottom
        state_text = f"Turn: {'Player 1' if state.turn == 0 else 'Player 2'}, Suggestion A: {state.suggestionA}, Card A: {state.cardA}, Suggestion B: {state.suggestionB}"
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
                        print('Game over (Win)')
                        running = False
                    else:
                        screen.blit(self.render(state, W, H, rows, cols), (0, 0))
                        pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    game = AssociativeMemoryGame.make(
        letters=(
            ("A", "a"), ("B", "b"),
        ),
        numbers=(
            ("1", "one"), ("2", "two"),
        ),
    )
    game.play()

