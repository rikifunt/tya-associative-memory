from dataclasses import dataclass
from enum import Enum
from functools import reduce
from logging import warning
from typing import Iterable, Iterator, NamedTuple

import numpy as np
import pygame


@dataclass
class SemanticMemoryGame:
    class CardState(Enum):
        UNSEEN = 0 # card was never seen
        FACE_DOWN = 1 # card was seen and is face down
        FACE_UP = 2 # card was seen and is face up
        SOLVED = 3 # card is solved (face up, matched with the correct card)

    oracle: np.ndarray[int] # for each card, the card it matches with

    def __post_init__(self):
        if len(self.oracle.shape) != 1:
            raise ValueError("Oracle must be a 1D array")
        if len(np.unique(self.oracle)) != len(self.oracle):
            raise ValueError("Each card must have a unique match")

    def reset(self, seed=None) -> np.ndarray[int]:
        return np.full(len(self.oracle), SemanticMemoryGame.CardState.UNSEEN)

    def step(self, state: np.ndarray[int], action: int) -> np.ndarray[int] | None:
        next_state = state.copy()

        if state[action] == SemanticMemoryGame.CardState.UNSEEN:
            # Choose a random unseen card
            card = np.random.choice(np.where(state == SemanticMemoryGame.CardState.UNSEEN)[0])
            next_state[card] = SemanticMemoryGame.CardState.FACE_UP
            return next_state

        if state[action] == SemanticMemoryGame.CardState.FACE_DOWN:
            # Turn card face up
            next_state[action] = SemanticMemoryGame.CardState.FACE_UP
            # If there are two face up cards, check if they match
            face_up = np.where(next_state == SemanticMemoryGame.CardState.FACE_UP)[0]
            assert len(face_up) <= 2
            if len(face_up) == 2:
                if self.oracle[face_up[0]] == face_up[1]:
                    next_state[face_up] = SemanticMemoryGame.CardState.SOLVED
                else:
                    next_state[face_up] = SemanticMemoryGame.CardState.FACE_DOWN
            return next_state

        if state[action] == SemanticMemoryGame.CardState.FACE_UP or state[action] == SemanticMemoryGame.CardState.SOLVED:
            # Do nothing (invalid action)
            return next_state

    def have_new_match(self, state: np.ndarray[int], next_state: np.ndarray[int]) -> bool:
        # number of solved cards increased
        return np.sum(next_state == SemanticMemoryGame.CardState.SOLVED) > np.sum(state == SemanticMemoryGame.CardState.SOLVED)

