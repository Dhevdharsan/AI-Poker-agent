"""
train.py  —  Offline Training Script for RaisedPlayer
COMPSCI 683 Artificial Intelligence — Spring 2026, UMass Amherst

================================================================
TRAINING STRATEGY  (Strategy D: Reinforcement Learning on Weights)
================================================================

Overview:
  We train the 8 base evaluation weights of RaisedPlayer's linear
  evaluation function using a 3-phase reinforcement learning loop.
  The 6 conditional features (bluffs, traps, weak calls) are left 
  untouched to preserve their hand-tuned strategic intent.

Update rule (Policy Gradient Delta):
  w_j = w_j + (alpha * reward * phi_j_avg * sign(w_j))

  where:
    w_j       = weight for feature j
    alpha     = learning rate (decays from 0.05 to 0.01 over training)
    reward    = (agent_stack - opp_stack) / initial_stack
    phi_j_avg = average activation of feature j across the batch
    sign(w_j) = preserves the strategic intent. A positive reward 
                makes a bonus more positive, and a penalty more negative.

3-Phase Training Schedule:
  Phase 1 — vs RandomPlayer (iterations 0 to PHASE1_END-1)
  Phase 2 — vs StrongRulePlayer (iterations PHASE1_END to PHASE2_END-1)
  Phase 3 — Self-play (iterations PHASE2_END to NUM_ITERATIONS-1)

How to run:
  python train.py

Output:
  hybrid_weights.json
================================================================
"""

import sys
import os
import json
import copy
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer


# ============================================================
# STRONG RULE-BASED OPPONENT  (Phase 2)
# ============================================================

class StrongRulePlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        actions   = [a["action"] for a in valid_actions]
        street    = round_state.get("street", "preflop")
        pot       = self._get_pot(round_state)
        r1        = self._rank(hole_card[0])
        r2        = self._rank(hole_card[1])
        rank_sum  = r1 + r2
        is_pair   = (r1 == r2)
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
                return "fold" if pot < 100 else "call"
        elif street == "turn":
            if rank_sum >= 22:
                return "raise" if "raise" in actions else "call"
            elif rank_sum >= 16:
                return "call"
            else:
                return "fold"
        else:
            if rank_sum >= 20:
                return "raise" if "raise" in actions else "call"
            elif rank_sum >= 14:
                return "call"
            else:
                return "fold"

    def _rank(self, card):
        rank_map = {
            "2": 2,  "3": 3,  "4": 4,  "5": 5,  "6": 6,
            "7": 7,  "8": 8,  "9": 9,  "T": 10,
            "J": 11, "Q": 12, "K": 13, "A": 14,
        }
        return rank_map.get(card[1].upper(), 2)

    def _get_pot(self, round_state):
        pot = round_state.get("pot", {})
        if isinstance(pot, dict):
            main = pot.get("main", 0)
            return main.get("amount", 0) if isinstance(main, dict) else int(main)
        return int(pot) if isinstance(pot, (int, float)) else 0

    def receive_game_start_message(self, game_info):   pass
    def receive_round_start_message(self, rc, hc, s):  pass
    def receive_street_start_message(self, s, rs):     pass
    def receive_game_update_message(self, a, rs):      pass
    def receive_round_result_message(self, w, hi, rs): pass


# ============================================================
# HYPERPARAMETERS
# ============================================================

GAMES_PER_BATCH = 20
NUM_ITERATIONS  = 180
MAX_ROUND       = 200
INITIAL_STACK   = 10000
SMALL_BLIND     = 10

SAVE_PATH = "hybrid_weights.json"

LR_START = 0.05
LR_END   = 0.01

PHASE1_END = 60
PHASE2_END = 120


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_learning_rate(iteration):
    progress = iteration / max(1, NUM_ITERATIONS - 1)
    return LR_START + progress * (LR_END - LR_START)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

def extract_features(hole_card, round_state, agent):
    """
    Extract ONLY the base features. We exclude conditional features 
    (like strong_raise_bonus) so they retain their hand-tuned default 
    values and don't get overwritten by generalized math.
    """
    # Reduce monte carlo sims slightly during training to speed it up
    hand_strength = agent._estimate_hand_strength_mc(hole_card, round_state, nb_simulation = 10) 
    pot_amount      = agent._extract_pot_amount(round_state)
    pot_bucket      = agent._bucketize(pot_amount, [20, 60, 120, 250]) / 4.0
    pressure        = agent._extract_betting_pressure(round_state)
    pressure_bucket = agent._bucketize(pressure, [10, 30, 60, 120]) / 4.0
    opp_stats       = agent._infer_opponent_stats_from_history(round_state)
    opp_aggression  = opp_stats["raise_rate"]
    street_name     = round_state.get("street", "preflop")
    street_index    = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(street_name, 0)
    street_progress = street_index / 3.0
    card_feats      = agent._hole_card_features(hole_card)

    return {
        "hand_strength":      hand_strength,
        "pot_bucket":         pot_bucket,
        "pressure_bucket":    pressure_bucket,
        "opp_aggression":     opp_aggression,
        "street_progress":    street_progress,
        "pair_bonus":         1.0 if card_feats["pair"]      else 0.0,
        "suited_bonus":       1.0 if card_feats["suited"]    else 0.0,
        "connected_bonus":    1.0 if card_feats["connected"] else 0.0,
    }


# ============================================================
# FEATURE-TRACKING WRAPPER
# ============================================================

class TrackingPlayer(RaisedPlayer):
    """
    Subclass of RaisedPlayer that logs feature observations.
    """
    def __init__(self):
        super().__init__()
        self.feature_log = []

    def declare_action(self, valid_actions, hole_card, round_state):
        action = super().declare_action(valid_actions, hole_card, round_state)
        try:
            feats = extract_features(hole_card, round_state, self)
            feats["action"] = action
            self.feature_log.append(feats)
        except Exception:
            pass
        return action

    def reset_log(self):
        self.feature_log = []

    def get_average_features(self):
        if not self.feature_log:
            return {}
        totals = {}
        for record in self.feature_log:
            for k, v in record.items():
                if k != "action":
                    totals[k] = totals.get(k, 0.0) + v
        n = max(1, len(self.feature_log))
        return {k: v / n for k, v in totals.items()}


# ============================================================
# REWARD FUNCTION
# ============================================================

def compute_reward(game_result, agent_name):
    players     = game_result.get("players", [])
    agent_stack = None
    opp_stack   = None

    for p in players:
        if p.get("name") == agent_name:
            agent_stack = p.get("stack", INITIAL_STACK)
        else:
            opp_stack = p.get("stack", INITIAL_STACK)

    if agent_stack is None or opp_stack is None:
        return 0.0

    return (agent_stack - opp_stack) / INITIAL_STACK


# ============================================================
# WEIGHT UPDATE  (Fixed Math)
# ============================================================

def update_weights(weights, reward, avg_features, alpha):
    new_weights = {}
    for key in weights:
        # If we didn't track the feature (e.g. conditional bonuses),
        # carry the exact weight forward unchanged.
        if key not in avg_features:
            new_weights[key] = weights[key]
            continue
            
        phi_j = avg_features.get(key, 0.0)
        
        # We multiply by the sign of the original weight.
        # This ensures that if the agent wins, a penalty becomes 
        # MORE negative, and a bonus becomes MORE positive.
        sign = 1.0 if weights[key] >= 0 else -1.0
        
        delta = alpha * reward * phi_j * sign
        new_weights[key] = weights[key] + delta
        
        # Clip to prevent the math from exploding over many iterations
        new_weights[key] = max(-10.0, min(10.0, new_weights[key]))
        
    return new_weights


# ============================================================
# SINGLE BATCH
# ============================================================

def run_batch(agent, opponent, agent_name, opp_name):
    total_reward = 0.0
    games_played = 0

    for _ in range(GAMES_PER_BATCH):
        config = setup_config(
            max_round=MAX_ROUND,
            initial_stack=INITIAL_STACK,
            small_blind_amount=SMALL_BLIND,
        )
        config.register_player(name=agent_name, algorithm=agent)
        config.register_player(name=opp_name,   algorithm=opponent)

        try:
            result        = start_poker(config, verbose=0)
            total_reward += compute_reward(result, agent_name)
            games_played += 1
        except Exception:
            pass

    return total_reward / max(1, games_played)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train():
    total_games = NUM_ITERATIONS * GAMES_PER_BATCH

    print("=" * 62)
    print("RaisedPlayer Training  —  3-Phase + Decaying LR")
    print(f"  Iterations       : {NUM_ITERATIONS}  ({total_games} total games)")
    print(f"  Games / batch    : {GAMES_PER_BATCH}")
    print(f"  Learning rate    : {LR_START} → {LR_END} (linear decay)")
    print(f"  Phase 1 (Random)     : iter   0 – {PHASE1_END - 1}")
    print(f"  Phase 2 (StrongRule) : iter  {PHASE1_END} – {PHASE2_END - 1}")
    print(f"  Phase 3 (Self-play)  : iter {PHASE2_END} – {NUM_ITERATIONS - 1}")
    print(f"  Output           : {SAVE_PATH}")
    print("=" * 62)

    base_agent      = RaisedPlayer()
    current_weights = base_agent._load_hybrid_weights(SAVE_PATH)
    best_weights    = copy.deepcopy(current_weights)
    best_reward     = -float("inf")
    reward_history  = []

    start_time = time.time()

    for iteration in range(NUM_ITERATIONS):

        if iteration < PHASE1_END:
            phase     = 1
            opp_label = "Random"
        elif iteration < PHASE2_END:
            phase     = 2
            opp_label = "StrongRule"
        else:
            phase     = 3
            opp_label = "SelfPlay"

        alpha = get_learning_rate(iteration)

        agent = TrackingPlayer()
        agent._hybrid_weights = copy.deepcopy(current_weights)
        agent.reset_log()

        if phase == 1:
            opponent = RandomPlayer()
        elif phase == 2:
            opponent = StrongRulePlayer()
        else:
            opponent = TrackingPlayer()
            opponent._hybrid_weights = copy.deepcopy(current_weights)

        avg_reward = run_batch(
            agent=agent,
            opponent=opponent,
            agent_name="Agent",
            opp_name=opp_label,
        )
        reward_history.append(avg_reward)

        avg_features    = agent.get_average_features()
        current_weights = update_weights(
            weights=current_weights,
            reward=avg_reward,
            avg_features=avg_features,
            alpha=alpha,
        )

        if avg_reward > best_reward:
            best_reward  = avg_reward
            best_weights = copy.deepcopy(current_weights)
            with open(SAVE_PATH, "w") as f:
                json.dump(best_weights, f, indent=2)

        if (iteration + 1) % 10 == 0:
            elapsed    = time.time() - start_time
            recent_avg = sum(reward_history[-10:]) / 10
            print(
                f"  Iter {iteration+1:4d}/{NUM_ITERATIONS} | "
                f"Phase {phase} ({opp_label:10s}) | "
                f"Reward: {avg_reward:+.3f} | "
                f"10-avg: {recent_avg:+.3f} | "
                f"LR: {alpha:.4f} | "
                f"{elapsed:.0f}s"
            )

    total_time = time.time() - start_time
    print("=" * 62)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best batch reward  : {best_reward:+.3f}")
    print(f"Weights saved to   : {SAVE_PATH}")
    print("=" * 62)
    print("\nFinal learned weights vs defaults:")

    dummy    = RaisedPlayer()
    defaults = dummy._load_hybrid_weights("__no_such_file__")
    for k, v in best_weights.items():
        d      = defaults.get(k, 0.0)
        change = v - d
        arrow  = "▲" if change > 0.01 else ("▼" if change < -0.01 else "─")
        print(f"  {k:25s}: {v:+7.4f}  (default {d:+.4f},  {arrow} {change:+.4f})")

    return best_weights


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()