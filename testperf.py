import sys
sys.path.insert(0, './pypokerengine/api/')
import game as poker_game
setup_config = poker_game.setup_config
start_poker  = poker_game.start_poker
import time
from argparse import ArgumentParser
from strong_rule_player import StrongRulePlayer as StrongRulePlayer2

""" =========== Import your agents here =========== """
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from pypokerengine.players import BasePokerPlayer
from always_raise import AlwaysRaisePlayer
try:
    from mcts_player import MCTSPlayer
except ImportError:
    MCTSPlayer = None
""" =============================================== """


class StrongRulePlayer(BasePokerPlayer):
    """
    Tight-aggressive rule-based agent.
    Folds weak hands, raises strong ones.
    Used as a benchmark opponent harder than RandomPlayer.
    """
    def declare_action(self, valid_actions, hole_card, round_state):
        actions  = [a["action"] for a in valid_actions]
        street   = round_state.get("street", "preflop")
        r1       = self._rank(hole_card[0])
        r2       = self._rank(hole_card[1])
        rank_sum = r1 + r2
        is_pair  = (r1 == r2)
        is_suited = (hole_card[0][0] == hole_card[1][0])

        if street == "preflop":
            if rank_sum >= 22 or is_pair:
                return "raise" if "raise" in actions else "call"
            elif rank_sum >= 18 or (rank_sum >= 16 and is_suited):
                return "call"
            else:
                return "fold"
        elif street == "flop":
            if rank_sum >= 20:
                return "raise" if "raise" in actions else "call"
            elif rank_sum >= 14:
                return "call"
            else:
                return "fold"
        elif street == "turn":
            if rank_sum >= 22:
                return "raise" if "raise" in actions else "call"
            elif rank_sum >= 16:
                return "call"
            else:
                return "fold"
        else:  # river
            if rank_sum >= 20:
                return "raise" if "raise" in actions else "call"
            elif rank_sum >= 14:
                return "call"
            else:
                return "fold"

    def _rank(self, card):
        rank_map = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,
                    "8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}
        return rank_map.get(card[1].upper(), 2)

    def receive_game_start_message(self, game_info):    pass
    def receive_round_start_message(self, rc, hc, s):   pass
    def receive_street_start_message(self, s, rs):      pass
    def receive_game_update_message(self, a, rs):       pass
    def receive_round_result_message(self, w, hi, rs):  pass


"""
How to run:

  GauntletAgent vs RandomPlayer:
    python3 testperf.py -n1 "GauntletAgent" -a1 RaisedPlayer -n2 "Random" -a2 RandomPlayer

  GauntletAgent vs StrongRulePlayer:
    python3 testperf.py -n1 "GauntletAgent" -a1 RaisedPlayer -n2 "StrongRule" -a2 StrongRulePlayer

  GauntletAgent vs itself (~50/50 sanity check):
    python3 testperf.py -n1 "Agent1" -a1 RaisedPlayer -n2 "Agent2" -a2 RaisedPlayer
"""

AGENT_MAP = {
    "RaisedPlayer":      RaisedPlayer,
    "RandomPlayer":      RandomPlayer,
    "StrongRulePlayer":  StrongRulePlayer,
    "StrongRulePlayer2": StrongRulePlayer2,
    "AlwaysRaisePlayer": AlwaysRaisePlayer,
}
if MCTSPlayer is not None:
    AGENT_MAP["MCTSPlayer"] = MCTSPlayer


def testperf(agent_name1, agent1_class, agent_name2, agent2_class):
    num_game          = 50
    max_round         = 100
    initial_stack     = 10000
    smallblind_amount = 20

    agent1_pot = 0
    agent2_pot = 0

    for game_num in range(1, num_game + 1):
        print("Game number: ", game_num)

        config = setup_config(
            max_round=max_round,
            initial_stack=initial_stack,
            small_blind_amount=smallblind_amount,
        )
        config.register_player(name=agent_name1, algorithm=agent1_class())
        config.register_player(name=agent_name2, algorithm=agent2_class())

        game_result = start_poker(config, verbose=0)
        agent1_pot += game_result['players'][0]['stack']
        agent2_pot += game_result['players'][1]['stack']

    print("\n After playing {} games of {} rounds, the results are:".format(
        num_game, max_round))
    print("\n " + agent_name1 + "'s final pot: ", agent1_pot)
    print("\n " + agent_name2 + "'s final pot: ", agent2_pot)

    total  = agent1_pot + agent2_pot
    margin = agent1_pot - agent2_pot
    pct    = abs(margin) / total * 100 if total > 0 else 0
    print("\n Chip difference: {:+,}  ({:.2f}% margin)".format(margin, pct))

    if agent1_pot > agent2_pot:
        print("\n Congratulations! " + agent_name1 + " has won.")
    elif agent2_pot > agent1_pot:
        print("\n Congratulations! " + agent_name2 + " has won.")
    else:
        print("\n It's a draw!")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-n1', '--agent_name1', default="RaisedPlayer", type=str)
    parser.add_argument('-a1', '--agent1',      default="RaisedPlayer", type=str)
    parser.add_argument('-n2', '--agent_name2', default="Random",       type=str)
    parser.add_argument('-a2', '--agent2',      default="RandomPlayer", type=str)
    args = parser.parse_args()

    if args.agent1 not in AGENT_MAP:
        raise ValueError("Unknown agent '{}'. Choose from: {}".format(
            args.agent1, list(AGENT_MAP.keys())))
    if args.agent2 not in AGENT_MAP:
        raise ValueError("Unknown agent '{}'. Choose from: {}".format(
            args.agent2, list(AGENT_MAP.keys())))

    return (args.agent_name1, AGENT_MAP[args.agent1],
            args.agent_name2, AGENT_MAP[args.agent2])


if __name__ == '__main__':
    name1, agent1, name2, agent2 = parse_arguments()
    start = time.time()
    testperf(name1, agent1, name2, agent2)
    end = time.time()
    print("\n Time taken to play: %.4f seconds" % (end - start))