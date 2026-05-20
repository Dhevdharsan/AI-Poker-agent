# strong_rule_player.py — a tighter, smarter baseline than RandomPlayer
from pypokerengine.players import BasePokerPlayer
class StrongRulePlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # Parse hand strength
        r1 = self._rank(hole_card[0])
        r2 = self._rank(hole_card[1])
        street = round_state.get("street", "preflop")

        # Preflop: only play strong hands
        if street == "preflop":
            if r1 + r2 >= 22 or r1 == r2:  # high cards or pair
                return "raise" if "raise" in [a["action"] for a in valid_actions] else "call"
            elif r1 + r2 >= 16:
                return "call"
            else:
                return "fold"

        # Post-flop: call most of the time, raise occasionally
        return "call"

    def _rank(self, card):
        rank_map = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
        return rank_map.get(card[1].upper(), 2)

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, rc, hc, s):
        pass

    def receive_street_start_message(self, s, rs):
        pass

    def receive_game_update_message(self, a, rs):
        pass

    def receive_round_result_message(self, w, hi, rs):
        pass


def setup_ai():
    return StrongRulePlayer()