from pypokerengine.players import BasePokerPlayer

class AlwaysRaisePlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        actions = [a["action"] for a in valid_actions]
        
        # Always raise if the rules allow it
        if "raise" in actions:
            return "raise"
        # Fallback to call if the raise cap is reached
        if "call" in actions:
            return "call"
        return "fold"

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass