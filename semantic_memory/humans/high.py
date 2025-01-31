"""
ChatGPT comment:

Here is the function implementing the human-like decision-making process based on the given constraints:

This function incorporates stochastic decision-making based on knowledge levels and trust, ensuring it avoids invalid actions while simulating human-like behavior. Let me know if you want any tweaks!
"""

import random

def humanDecision(state):
    known_cards = {k: v for k, v in state.items() if v not in ["unknown", "solved", "None"] and k != 8}
    uncovered_card = state[8]
    
    capitals_nations = {"Paris": "France", "France": "Paris", "Rome": "Italy", "Italy": "Rome"}
    math_operations = {"2*(5-3)": "4", "4": "2*(5-3)", "7*8": "56", "56": "7*8"}
    
    # If no known cards and no uncovered card, either uncover a random one or handover
    if not known_cards and uncovered_card is None:
        return random.choices([8, 9], weights=[0.3, 0.7])[0]  # Prefer handing over due to high trust
    
    # If an uncovered card exists
    if uncovered_card:
        if uncovered_card in capitals_nations:  # Good at capitals-nations
            associated_card = capitals_nations[uncovered_card]
            if associated_card in known_cards.values():
                return next(k for k, v in known_cards.items() if v == associated_card)  # Match it
            else:
                return random.choices([8, 9], weights=[0.7, 0.3])[0]  # Try uncovering a card first
        elif uncovered_card in math_operations:  # Bad at math
            return random.choices([8, 9], weights=[0.2, 0.8])[0]  # Mostly handover due to low knowledge
    
    # If there are known cards (but no uncovered card)
    for card, index in known_cards.items():
        if card in capitals_nations:  # Good at capitals-nations
            return index  # Try uncovering the known good association first
    
    # If we have known cards but only bad at them
    return random.choices([8, 9], weights=[0.1, 0.9])[0]  # Prefer to handover due to low trust in ability
