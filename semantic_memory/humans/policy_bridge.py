

from dataclasses import dataclass
from typing import Callable


from semantic_memory_game import MemoryGame


@dataclass
class PolicyBridge:
    card_contents: list[str]
    raw_policy: Callable[[object], object]

    def __call__(self, game_state: MemoryGame.State):

        state = [
            "unknown" if card_state == MemoryGame.CardState.UNSEEN else self.card_contents[card]
            for card, card_state in enumerate(game_state.seen)
        ]
        face_up = "None" if game_state.face_up == 0 else self.card_contents[game_state.face_up-1]
        state.append(face_up)

        assert all(isinstance(card, str) for card in state), f'Expected all cards to be strings, got {state}'
        action = self.raw_policy(state)
        # TODO filter invalid actions?
        return action


