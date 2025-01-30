import random

def human_decision(state):
    # Extract relevant states from the dictionary
    uncovered_card = state[8]
    cards = state[:8]

    # Determine level of knowledge and trust
    knowledge_capitals = "medium"
    knowledge_math = "bad"
    trust_level = "high"
    
    # Decide based on the current uncovered card or whether to pass to the partner
    if uncovered_card == "None":
        # If no card is uncovered, check the current state of cards
        if all(val == "unknown" for val in cards):
            # If all cards are unknown, make a random guess (more likely to handover)
            if trust_level == "high" and random.random() < 0.8:
                return 8  # Handover to partner
            else:
                return random.choice([i for i, val in enumerate(cards) if val == "unknown"])
        else:
            # If some cards are known, try to make an informed choice
            possible_choices = [i for i, val in enumerate(cards) if val == "unknown"]
            if len(possible_choices) == 1:
                return possible_choices[0]
            else:
                # Use knowledge to choose: "medium" for capitals/nations, "bad" for math
                if knowledge_capitals == "medium":
                    # Choose cards related to capitals/nations
                    pairs = [("France", "Paris"), ("Italy", "Rome")]
                    for i in possible_choices:
                        if cards[i] in [p[0] for p in pairs] or cards[i] in [p[1] for p in pairs]:
                            return i
                # If no good match, make a random guess
                return random.choice(possible_choices)
    else:
        # If a card is uncovered, make a decision based on current knowledge
        if uncovered_card in ["Paris", "France", "Rome", "Italy"]:
            # High trust in partner, may trust them to solve the match
            if trust_level == "high" and random.random() < 0.8:
                return 8  # Handover to partner
            else:
                # Try to solve if it’s a capital-nation pair
                if uncovered_card == "Paris":
                    return cards.index("France") if "France" in cards else 8
                elif uncovered_card == "France":
                    return cards.index("Paris") if "Paris" in cards else 8
                elif uncovered_card == "Rome":
                    return cards.index("Italy") if "Italy" in cards else 8
                elif uncovered_card == "Italy":
                    return cards.index("Rome") if "Rome" in cards else 8
        elif "7*8" in uncovered_card or "2*(5-3)" in uncovered_card:
            # Bad knowledge on math, trust partner more in this case
            if trust_level == "high" and random.random() < 0.9:
                return 8  # Handover to partner
            else:
                # Randomly guess for math problems
                return random.choice([i for i, val in enumerate(cards) if val == "unknown"])
        else:
            # Default random guess
            return random.choice([i for i, val in enumerate(cards) if val == "unknown"])
