"""
raise_player.py  —  RaisedPlayer AI Agent
COMPSCI 683 Artificial Intelligence — Spring 2026, UMass Amherst

================================================================
STRATEGY OVERVIEW
================================================================

Core engine: Strategy B + D combined

  - Strategy B: Minimax adversarial search with alpha-beta pruning
    * Searches the abstract game tree to depth 3
    * At our nodes (max nodes): pick the action with highest value
    * At opponent nodes: use a profile-weighted expected value model
      (not pure min, because the opponent's hand is hidden and they
       do not always play optimally — we model their likely response
       distribution based on observed play style)
    * Alpha-beta pruning cuts branches that cannot affect the result,
      keeping runtime safely within the 0.5-second action time limit

  - Strategy D: Learned linear evaluation function at cutoff nodes
    * phi(s) = sum_j  w_j * phi_j(s)
    * Features: hand strength, pot size, betting pressure,
      opponent aggression, street progress, hole card quality
    * Weights w_j are loaded from hybrid_weights.json, produced by
      the offline training script train.py
    * Falls back to hand-tuned defaults if no trained file exists

Bluff layer (checked before minimax search each turn):

  - Semi-bluff (~25% of eligible situations):
    Raise with a flush or straight draw on flop/turn against a
    non-aggressive opponent. NOT a pure bluff — we have real equity
    (~35% chance to complete by river) so raising is justified.

  - Reverse bluff / Slow play (~15% of eligible situations):
    Call with a very strong hand (> 0.85) on preflop/flop against
    an aggressive opponent when the pot is small. Goal: disguise
    strength, keep the aggressive opponent betting into us, and
    extract maximum value from the hand.

Card format note:
  The engine's Card.__str__() returns suit+rank:
    e.g.  "CA" = Club Ace,  "H9" = Heart Nine,  "DT" = Diamond Ten
  _parse_card() is written to match this format exactly.

Hand strength note:
  _estimate_hand_strength() uses Monte Carlo simulation (50 runs)
  via estimate_hole_card_win_rate from pypokerengine.utils.card_utils.
  This accounts for actual community cards on the board, making
  post-flop decisions significantly more accurate than a preflop
  heuristic. Falls back to a fast heuristic if the MC call fails.
================================================================
"""

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
import os
import json


class RaisedPlayer(BasePokerPlayer):
    def _get_canonical_hand(self, hole_card):
        """Converts two hole cards into their canonical string (e.g., 'AKs', '72o')."""
        if len(hole_card) != 2: return "XX"
        r1, s1 = self._parse_card(hole_card[0])
        r2, s2 = self._parse_card(hole_card[1])
        
        if r1 < r2:
            r1, r2 = r2, r1
            s1, s2 = s2, s1
            
        rank_str = {14:"A", 13:"K", 12:"Q", 11:"J", 10:"T", 9:"9", 8:"8", 7:"7", 6:"6", 5:"5", 4:"4", 3:"3", 2:"2"}
        str1, str2 = rank_str.get(r1, "X"), rank_str.get(r2, "X")
        
        if r1 == r2: return f"{str1}{str2}"
        elif s1 == s2: return f"{str1}{str2}s"
        else: return f"{str1}{str2}o"

    # ============================================================
    # ENTRY POINT — called by the engine on every turn
    # ============================================================

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        Main decision pipeline. Returns one of: "fold", "call", "raise".

        Stages:
          1. Extract legal actions from engine-provided valid_actions list
          2. Lazy-load trained weights (once per game instance)
          3. Compute shared context: opponent profile, hand strength, etc.
          4. Semi-bluff check  — may return "raise" early
          5. Reverse bluff check — may return "call" early
          6. Minimax search (depth 3, alpha-beta pruned)
             with learned linear evaluation at cutoff nodes
        """

        # Stage 1: Get the list of legal action strings
        legal_actions = [a["action"] for a in valid_actions]

        # Stage 2: Lazy-load trained weights once per game instance
        if not hasattr(self, "_hybrid_weights"):
            self._hybrid_weights = self._load_hybrid_weights("hybrid_weights.json")

        # Stage 3: Compute shared context (used by bluff checks AND minimax)
        opp_stats     = self._get_opponent_stats(round_state)
        opp_profile   = self._classify_opponent_profile(opp_stats)
        street        = round_state.get("street", "preflop")
        pot           = self._extract_pot_amount(round_state)
        hand_strength = self._estimate_hand_strength_mc(hole_card, round_state)

        # Stage 4: Semi-bluff check
        # Condition: flush or straight draw + flop/turn + not vs aggressive opp
        # Trigger:   ~25% of eligible situations (deterministic, not random)
        # Rationale: drawing hands have ~35% equity to improve by river,
        #            making a raise profitable even if called occasionally.
        # SKIP against calling_station — they never fold so bluffs have no value
        if (
            "raise" in legal_actions
            and street in ("flop", "turn")
            and opp_profile not in ("aggressive", "calling_station")
            and self._has_draw_potential(hole_card, round_state)
        ):
            if self._deterministic_trigger(round_state, salt=7) < 0.25:
                return "raise"  # Semi-bluff: raise with real draw equity

        # Stage 5: Reverse bluff (slow play) check
        # Condition: monster hand + early street + aggressive opp + small pot
        # Trigger:   ~15% of eligible situations
        # Rationale: calling disguises strength; aggressive opponents will
        #            keep betting, letting us extract more chips than an
        #            immediate raise would.
        if (
            "call" in legal_actions
            and hand_strength > 0.85
            and street in ("preflop", "flop")
            and opp_profile == "aggressive"
            and pot < 100
        ):
            if self._deterministic_trigger(round_state, salt=13) < 0.15:
                return "call"  # Reverse bluff: trap the aggressive opponent

        # Stage 5.2: Pure bluff on river
        # Condition: weak hand + river + passive or balanced opponent + small pot
        # Trigger:   ~20% of eligible situations
        # Rationale: on the river there are no more cards to come, so a pure bluff
        #            only works if the opponent folds. Against passive/balanced
        #            opponents who fold ~25-30% of the time, a small raise on the
        #            river with a weak hand is profitable in expectation:
        #              EV = 0.27 * pot - 0.73 * raise_cost
        #            With pot=$60 and raise=$10: EV = 0.27*60 - 0.73*10 = +$8.90
        #            We only bluff in small pots to limit downside if called.
        # NOT against calling_station (never folds) or aggressive (may re-raise)
        call_amount = next((a.get("amount", 0) for a in valid_actions if a.get("action") == "call"), 0)
        if (
            "raise" in legal_actions
            and street == "river"
            and hand_strength < 0.35
            and opp_profile in ("passive", "balanced")
            and pot < 80
            and call_amount == 0
        ):
            if self._deterministic_trigger(round_state, salt=31) < 0.20:
                return "raise"  # Pure bluff: bet weak hand hoping opponent folds

        # ==========================================================
        # Stage 5.5: Heuristic Pruning (Trash Hand Filter)
        # ==========================================================
        # If our hand is bad and we have no draws, we shouldn't ask Minimax. We should just fold.
        # However, if the opponent checked (call amount is 0), we check for free instead of folding.
        
        
        if hand_strength < 0.40 and not self._has_draw_potential(hole_card, round_state):
            if call_amount > 0:
                return "fold"
            elif "call" in legal_actions:
                return "call"  # Checking for free
        # ==========================================================

        # Stage 6: Minimax adversarial search (standard path)
        root_state = self._build_abstract_state(
            hole_card=hole_card,
            round_state=round_state,
            legal_actions=legal_actions,
            opp_stats=opp_stats,
            opp_profile=opp_profile,
            hand_strength=hand_strength,
        )

        # Score each legal action using alpha-beta minimax
        action_scores = {}
        alpha = -float("inf")
        beta  =  float("inf")

        for action in legal_actions:
            next_state = self._transition(root_state, action, actor="self")
            score = self._minimax(
                state=next_state,
                depth=2,           # depth 3 total including root action
                maximizing=False,  # opponent responds next
                alpha=alpha,
                beta=beta,
            )
            action_scores[action] = score
            alpha = max(alpha, score)  # update alpha at root level

        # Deterministic tie-break: raise > call > fold
        return self._argmax_with_tiebreak(action_scores, legal_actions)

    # ============================================================
    # BLUFF HELPERS
    # ============================================================

    def _has_draw_potential(self, hole_card, round_state):
        """
        Returns True if our hole cards + community cards give us a
        flush draw (4 cards of the same suit) OR a straight draw
        (4 consecutive ranks in any 5-card window).

        This is used to gate semi-bluff raises — we only semi-bluff
        when we have real equity behind the raise (draw potential).
        Only meaningful on flop/turn; the caller already filters by street.
        """
        community = round_state.get("community_card", [])
        if not community:
            return False

        all_cards = hole_card + community

        # Flush draw: 4+ cards of the same suit among hole + community
        suits = [self._parse_card(c)[1] for c in all_cards]
        if any(suits.count(s) >= 4 for s in set(suits)):
            return True

        # Straight draw: 4 consecutive ranks in a 5-rank window
        ranks = sorted(set(self._parse_card(c)[0] for c in all_cards))
        for lo in range(2, 12):
            if sum(1 for r in ranks if lo <= r <= lo + 4) >= 4:
                return True

        return False

    def _deterministic_trigger(self, round_state, salt):
        """
        Produces a stable float in [0, 1) derived from game state + salt.

        Why deterministic instead of random.random()?
          - Same game state + same salt → same bluff decision (reproducible)
          - Different salts distinguish semi-bluff vs reverse-bluff triggers
          - Still varies across rounds/streets so bluffs are not predictable
          - Avoids dependency on random seed

        Parameters:
          round_state : current round state dict from the engine
          salt        : integer to differentiate the two bluff triggers
        """
        round_count = round_state.get("round_count", 1)
        pot         = int(self._extract_pot_amount(round_state))
        street_idx  = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(
            round_state.get("street", "preflop"), 0
        )
        raw = (round_count * salt * 17 + pot * 3 + street_idx * 11) % 100
        return raw / 100.0

    # ============================================================
    # MINIMAX WITH ALPHA-BETA PRUNING  (Strategy B)
    # ============================================================

    def _minimax(self, state, depth, maximizing, alpha, beta):
        """
        Alpha-beta minimax search over the abstract state space.

        Max nodes (maximizing=True — our turn):
          Pick the child with the highest value.
          Beta-cutoff: prune if value >= beta (opponent won't allow this branch).

        Min-like nodes (maximizing=False — opponent's turn):
          Use a profile-weighted expected value over opponent responses.
          We do NOT use pure minimax (pick the worst child for us) because:
            (a) the opponent's hole cards are hidden from us
            (b) real opponents don't always play optimally
          Instead we model their likely action distribution based on
          their observed play profile (aggressive / passive / balanced).

        Base cases:
          depth == 0    : evaluate with the linear heuristic _evaluate()
          terminalish   : a fold occurred; return terminal_value directly
          no legal acts : evaluate current state
        """
        if depth == 0 or state.get("terminalish", False):
            return self._evaluate(state)

        legal = state.get("legal_actions", [])
        if not legal:
            return self._evaluate(state)

        if maximizing:
            # Our turn: standard max node with alpha-beta pruning
            value = -float("inf")
            for action in legal:
                child     = self._transition(state, action, actor="self")
                child_val = self._minimax(child, depth - 1, False, alpha, beta)
                value     = max(value, child_val)
                alpha     = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cutoff — prune remaining branches
            return value

        else:
            # Opponent's turn: profile-weighted expected value
            child_values = []
            for action in legal:
                child     = self._transition(state, action, actor="opponent")
                child_val = self._minimax(child, depth - 1, True, alpha, beta)
                child_values.append((action, child_val))

            return self._opponent_response_value(child_values, state)

    def _opponent_response_value(self, child_values, state):
        """
        Compute the expected value at an opponent node using a
        probability distribution over their likely actions.

        Distribution conditioned on opponent's observed play profile:
          aggressive : raise=55%, call=30%, fold=15%
          passive    : raise=15%, call=55%, fold=30%
          balanced   : raise=33%, call=44%, fold=23%

        Returns: weighted average of child values under this distribution.
        """
        profile = state.get("opp_profile", "balanced")
        dist = {
            "aggressive":      {"raise": 0.55, "call": 0.30, "fold": 0.15},
            "passive":         {"raise": 0.15, "call": 0.55, "fold": 0.30},
            "balanced":        {"raise": 0.33, "call": 0.44, "fold": 0.23},
            "calling_station": {"raise": 0.05, "call": 0.90, "fold": 0.05},
        }.get(profile, {"raise": 0.33, "call": 0.44, "fold": 0.23})

        total, norm = 0.0, 0.0
        for action, val in child_values:
            w     = dist.get(action, 0.0)
            total += w * val
            norm  += w

        if norm == 0:
            return sum(v for _, v in child_values) / max(1, len(child_values))
        return total / norm

    def _transition(self, state, action, actor):
        """
        Abstract state transition model.

        Updates abstract features based on the action taken by 'actor'
        ("self" or "opponent"). Does NOT call the real game engine —
        this is a fast approximation that runs in microseconds per node,
        keeping total search time within the 0.5-second time limit.

        Transition rules:
          self raise   : pot grows (+1 bucket), pressure on us drops (-1)
          self call    : pressure on us drops slightly (-1)
          self fold    : terminal, terminal_value = -3.0  (we lose pot)
          opp raise    : pot grows (+1), pressure on us increases (+1)
          opp call     : no significant state change
          opp fold     : terminal, terminal_value = +2.0  (we win pot)

        street_index advances by 1 after every action (approximation).
        """
        ns = dict(state)  # shallow copy — all values are scalars/immutables
        ns["legal_actions"] = ["fold", "call", "raise"]

        if actor == "self":
            ns["last_self_action"] = action
            if action == "raise":
                ns["pot_bucket"]      = min(4, state["pot_bucket"] + 1)
                ns["pressure_bucket"] = max(0, state["pressure_bucket"] - 1)
            elif action == "call":
                ns["pressure_bucket"] = max(0, state["pressure_bucket"] - 1)
            elif action == "fold":
                ns["terminalish"]    = True
                ns["terminal_value"] = 0.0

        else:  # opponent
            ns["last_opp_action"] = action
            if action == "raise":
                ns["pot_bucket"]      = min(4, state["pot_bucket"] + 1)
                ns["pressure_bucket"] = min(4, state["pressure_bucket"] + 1)
            elif action == "fold":
                ns["terminalish"]    = True
                ns["terminal_value"] = 2.0
            # opponent call: no major feature change

        ns["street_index"] = min(3, state["street_index"] + 1)
        return ns

    # ============================================================
    # LINEAR EVALUATION FUNCTION  (Strategy D)
    # ============================================================

    def _evaluate(self, state):
        """
        Linear evaluation function at search cutoff and terminal nodes.

        phi(s) = sum_j  w_j * phi_j(s)

        Base features phi_j(s):
          hand_strength   : estimated win probability [0,1]
          pot_bucket      : normalized pot size [0,1]
          pressure_bucket : normalized betting pressure [0,1]  (negative weight)
          opp_aggression  : opponent raise rate [0,1]          (negative weight)
          street_progress : how far into the hand we are [0,1]
          pair            : pocket pair flag {0,1}
          suited          : suited hole cards {0,1}
          connected       : connected hole cards {0,1}

        Conditional bonus/penalty terms:
          strong_raise_bonus : raised with a strong hand (>= 0.70)
          semi_bluff_bonus   : raised with a draw hand (< 0.45 but has_draw)
          weak_raise_penalty : raised with a weak hand (< 0.45, no draw)
          slow_play_bonus    : called with a monster hand (> 0.85)
          weak_call_penalty  : called with a very weak hand (< 0.35)
          fold_bias          : small penalty for folding (avoids over-folding)

        All weights loaded from hybrid_weights.json (trained by train.py).
        """
        if state.get("terminalish", False):
            return state.get("terminal_value", 0.0)

        w = self._hybrid_weights

        # Normalize features to [0, 1] range
        hand_strength   = state["hand_strength"] - 0.5
        pot_bucket      = state["pot_bucket"]      / 4.0
        pressure_bucket = state["pressure_bucket"] / 4.0
        opp_aggression  = state["opp_aggression"]
        street_progress = state["street_index"]    / 3.0
        pair            = 1.0 if state["pair"]      else 0.0
        suited          = 1.0 if state["suited"]    else 0.0
        connected       = 1.0 if state["connected"] else 0.0
        has_draw        = state.get("has_draw", False)

        # Base linear score
        score  = w["hand_strength"]   * hand_strength
        #score += w["pot_bucket"]      * pot_bucket
        score += (w["pot_bucket"] * pot_bucket) * max(0, hand_strength)
        score += w["pressure_bucket"] * pressure_bucket
        score += w["opp_aggression"]  * opp_aggression
        score += w["street_progress"] * street_progress
        score += w["pair_bonus"]      * pair
        score += w["suited_bonus"]    * suited
        score += w["connected_bonus"] * connected

        # Conditional bonus/penalty based on most recent self action
        last_self = state.get("last_self_action")

        if last_self == "raise":
            if hand_strength >= 0.70:
                score += w["strong_raise_bonus"]    # value bet — good
            elif has_draw:
                score += w["semi_bluff_bonus"]      # semi-bluff — acceptable
            else:
                score += w["weak_raise_penalty"]    # weak raise — bad

        elif last_self == "call":
            if hand_strength > 0.85:
                score += w["slow_play_bonus"]       # slow play trap — good
            elif hand_strength < 0.35:
                score += w["weak_call_penalty"]     # calling too light — bad

        elif last_self == "fold":
            score += w["fold_bias"]                 # small penalty: avoid over-folding

        return score

    # ============================================================
    # ABSTRACT STATE CONSTRUCTION
    # ============================================================

    def _build_abstract_state(
        self, hole_card, round_state, legal_actions,
        opp_stats, opp_profile, hand_strength
    ):
        """
        Build the root abstract state dict for the minimax search.

        The abstract state contains only the discretized/normalized features
        needed by _evaluate() and _transition() — not the full game state.
        This abstraction is what makes the search tractable: instead of
        branching over thousands of possible card combinations, we work
        with a small set of bucketed features.

        Pre-computed arguments (opp_stats, opp_profile, hand_strength) are
        passed in to avoid redundant computation — they were already
        calculated earlier in declare_action().
        """
        card_feats = self._hole_card_features(hole_card)

        pot_amount      = self._extract_pot_amount(round_state)
        pot_bucket      = self._bucketize(pot_amount, [20, 60, 120, 250])

        pressure        = self._extract_betting_pressure(round_state)
        pressure_bucket = self._bucketize(pressure, [10, 30, 60, 120])

        street_name  = round_state.get("street", "preflop")
        street_index = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(
            street_name, 0
        )

        has_draw = self._has_draw_potential(hole_card, round_state)

        return {
            # Hand quality features
            "hand_strength":    hand_strength,
            "pair":             card_feats["pair"],
            "suited":           card_feats["suited"],
            "connected":        card_feats["connected"],
            "has_draw":         has_draw,
            # Game situation features
            "pot_bucket":       pot_bucket,
            "pressure_bucket":  pressure_bucket,
            "street_index":     street_index,
            # Opponent model
            "opp_profile":      opp_profile,
            "opp_aggression":   opp_stats["raise_rate"],
            # Search bookkeeping
            "legal_actions":    list(legal_actions),
            "terminalish":      False,
            "terminal_value":   0.0,
            "last_self_action": None,
            "last_opp_action":  None,
        }

    # ============================================================
    # OPPONENT MODELING
    # ============================================================

    def _infer_opponent_stats_from_history(self, round_state):
        """
        Compute opponent action frequencies from action_histories in round_state.

        We skip our own uuid (only model the opponent).
        We skip SMALLBLIND / BIGBLIND / ANTE (forced bets — not strategic choices).

        Returns dict:
          raise_rate, call_rate, fold_rate : frequencies in [0, 1]
          total : raw count of observed voluntary opponent actions
        """
        histories = round_state.get("action_histories", {})
        my_uuid   = getattr(self, "uuid", None)
        counts    = {"raise": 0, "call": 0, "fold": 0, "total": 0}

        for street, acts in histories.items():
            for act in acts:
                actor_uuid = act.get("uuid") or act.get("player_uuid")
                move       = act.get("action", "").upper()

                if actor_uuid == my_uuid:
                    continue  # skip our own actions

                if move == "RAISE":
                    counts["raise"] += 1;  counts["total"] += 1
                elif move in ("CALL", "CHECK"):
                    counts["call"]  += 1;  counts["total"] += 1
                elif move == "FOLD":
                    counts["fold"]  += 1;  counts["total"] += 1
                # SMALLBLIND, BIGBLIND, ANTE are forced — skip

        total = max(1, counts["total"])
        counts["raise_rate"] = counts["raise"] / total
        counts["call_rate"]  = counts["call"]  / total
        counts["fold_rate"]  = counts["fold"]  / total
        return counts

    def _classify_opponent_profile(self, opp_stats):
        """
        Classify the opponent as 'aggressive', 'passive', or 'balanced'.

        We need at least 4 voluntary actions before trusting the estimate;
        before that we default to 'balanced' (the safest assumption).

        Classification rules:
          aggressive : raise_rate >= 0.45
          passive    : fold_rate  >= 0.35 AND raise_rate < 0.20
                       OR call_rate >= 0.50 AND raise_rate < 0.25
          balanced   : everything else
        """
        if opp_stats["total"] < 4:
            return "balanced"
        if opp_stats["raise_rate"] >= 0.45:
            return "aggressive"
        # Calling station: almost never folds, almost never raises
        # Bluffing against them is pointless — they always call
        # Only value bet strong hands against calling stations
        if opp_stats["call_rate"] >= 0.60 and opp_stats["fold_rate"] < 0.15:
            return "calling_station"
        if opp_stats["fold_rate"]  >= 0.35 and opp_stats["raise_rate"] < 0.20:
            return "passive"
        if opp_stats["call_rate"]  >= 0.50 and opp_stats["raise_rate"] < 0.25:
            return "passive"
        return "balanced"

    def _get_opponent_stats(self, round_state):
        """
        NEW function — returns opponent action frequencies using persistent
        counts accumulated across ALL rounds played so far, supplemented
        by actions visible in the current round.

        Why persistent?
          _infer_opponent_stats_from_history only sees the current round.
          Between rounds action_histories resets, so all previous hands
          are lost. After 200 hands we have a much clearer picture of
          the opponent's style — this function preserves that information.

        Counters are initialised in receive_game_start_message and
        flushed at round end via _commit_round_opp_stats.
        """
        if not hasattr(self, "_opp_raise_count"):
            self._opp_raise_count = 0
            self._opp_call_count  = 0
            self._opp_fold_count  = 0
            self._opp_total       = 0
            self._opp_last_seen   = 0

        current = self._infer_opponent_stats_from_history(round_state)
        new_actions = current["total"] - self._opp_last_seen

        if new_actions > 0:
            self._opp_raise_count += round(new_actions * current["raise_rate"])
            self._opp_call_count  += round(new_actions * current["call_rate"])
            self._opp_fold_count  += round(new_actions * current["fold_rate"])
            self._opp_total       += new_actions
            self._opp_last_seen    = current["total"]

        total = max(1, self._opp_total)
        return {
            "raise":      self._opp_raise_count,
            "call":       self._opp_call_count,
            "fold":       self._opp_fold_count,
            "total":      self._opp_total,
            "raise_rate": self._opp_raise_count / total,
            "call_rate":  self._opp_call_count  / total,
            "fold_rate":  self._opp_fold_count  / total,
        }

    def _commit_round_opp_stats(self, round_state):
        """
        NEW function — called at end of each round to flush the final
        opponent actions into persistent counters before action_histories
        resets, then resets the within-round seen counter to zero.
        """
        if not hasattr(self, "_opp_raise_count"):
            return

        final = self._infer_opponent_stats_from_history(round_state)
        new_actions = final["total"] - self._opp_last_seen

        if new_actions > 0:
            self._opp_raise_count += round(new_actions * final["raise_rate"])
            self._opp_call_count  += round(new_actions * final["call_rate"])
            self._opp_fold_count  += round(new_actions * final["fold_rate"])
            self._opp_total       += new_actions

        self._opp_last_seen = 0

    # ============================================================
    # WEIGHT MANAGEMENT  (Strategy D)
    # ============================================================

    def _load_hybrid_weights(self, path):
        """
        Load offline-trained evaluation weights from a JSON file.
        Falls back to hand-tuned defaults if the file is missing.

        Weights are produced by train.py and should NOT be trained
        live during gameplay (violates project rules and time limit).

        Weight keys and their strategic meaning:
          hand_strength      : importance of raw hand strength
          pot_bucket         : reward for having a large pot to win
          pressure_bucket    : penalty for heavy betting pressure against us
          opp_aggression     : penalty when facing an aggressive opponent
          street_progress    : reward for later streets (more information)
          pair_bonus         : pocket pairs are stronger starting hands
          suited_bonus       : suited cards have flush draw potential
          connected_bonus    : connected cards have straight draw potential
          strong_raise_bonus : reward for value-betting a strong hand
          semi_bluff_bonus   : reward for raising with a draw (semi-bluff)
          weak_raise_penalty : penalty for raising weak with no draw
          slow_play_bonus    : reward for slow-playing a monster hand
          weak_call_penalty  : penalty for calling with very weak holdings
          fold_bias          : small penalty to discourage over-folding
        """
        defaults = {
            "hand_strength":      4.8,
            "pot_bucket":         0.9,
            "pressure_bucket":   -1.0,
            "opp_aggression":    -1.1,
            "street_progress":    0.5,
            "pair_bonus":         1.5,
            "suited_bonus":       0.6,
            "connected_bonus":    0.4,
            "strong_raise_bonus": 1.1,
            "semi_bluff_bonus":   0.8,
            "weak_raise_penalty":-2.0,
            "slow_play_bonus":    0.6,
            "weak_call_penalty": -1.4,
            "fold_bias":         -0.15,
        }

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    loaded = json.load(f)
                for k, v in loaded.items():
                    if k in defaults:
                        defaults[k] = float(v)
            except Exception:
                pass  # silently keep defaults on any read/parse error

        return defaults

    # ============================================================
    # CARD PARSING AND FEATURE EXTRACTION
    # ============================================================

    def _parse_card(self, card_str):
        """
        Parse a card string in engine format: suit + rank.

        Engine format (Card.__str__):
          Position 0 = suit : C (Club), D (Diamond), H (Heart), S (Spade)
          Position 1 = rank : 2-9, T (ten), J, Q, K, A

        Examples:
          "CA" = Ace of Clubs     rank=14, suit='C'
          "H9" = Nine of Hearts   rank=9,  suit='H'
          "DT" = Ten of Diamonds  rank=10, suit='D'
          "SK" = King of Spades   rank=13, suit='S'

        Returns: (rank_int, suit_char)
        """
        if not isinstance(card_str, str) or len(card_str) != 2:
            return 2, "X"

        suit = card_str[0].upper()
        rank = card_str[1].upper()

        rank_map = {
            "2":  2, "3":  3, "4":  4, "5":  5,
            "6":  6, "7":  7, "8":  8, "9":  9,
            "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
        }
        return rank_map.get(rank, 2), suit

    def _hole_card_features(self, hole_card):
        """
        Extract three boolean quality features from the two hole cards.

        pair      : both cards have the same rank (pocket pair)
                    e.g. KK, AA, 77 — strong starting hands
        suited    : both cards share a suit (flush draw potential)
                    adds roughly +8% equity vs unsuited equivalent
        connected : ranks differ by exactly 1 (straight draw potential)
                    e.g. 89, JQ, KA — adds roughly +5% equity
        """
        if len(hole_card) != 2:
            return {"pair": False, "suited": False, "connected": False}

        r1, s1 = self._parse_card(hole_card[0])
        r2, s2 = self._parse_card(hole_card[1])

        return {
            "pair":      r1 == r2,
            "suited":    s1 == s2,
            "connected": abs(r1 - r2) == 1,
        }

    def _estimate_hand_strength(self, hole_card, round_state):
        """
        Original fast deterministic hand strength estimate in [0, 1].

        UNTOUCHED — kept exactly as provided by the course staff.
        Used as fallback by _estimate_hand_strength_mc if MC fails,
        and used directly by train.py for fast feature extraction.

        Scoring logic:
          base           = (rank1 + rank2) / 28   max = 1.0 for A+A
          pocket pair    : +0.35 + rank/20         high pairs worth more
          suited         : +0.08                   flush draw equity
          gap == 1       : +0.08                   connected (straight draw)
          gap == 2       : +0.03                   one-gap draw
          broadway cards : +0.04 per T/J/Q/K/A     high card combos
          weak hand      : -0.08 if low+unconnected e.g. 2-7o

        Street adjustment: small bonuses for later streets reflecting
        increased certainty as community cards are revealed.
        """
        if len(hole_card) != 2:
            return 0.3

        r1, s1 = self._parse_card(hole_card[0])
        r2, s2 = self._parse_card(hole_card[1])

        hi  = max(r1, r2)
        lo  = min(r1, r2)
        gap = abs(r1 - r2)

        score = (hi + lo) / 28.0

        if r1 == r2:
            score += 0.35 + (hi / 20.0)    # pocket pair bonus

        if s1 == s2:
            score += 0.08                   # suited bonus

        if gap == 1:
            score += 0.08                   # connected
        elif gap == 2:
            score += 0.03                   # one-gap

        broadway = sum(1 for r in (r1, r2) if r >= 10)
        score += 0.04 * broadway

        if hi <= 9 and gap >= 3 and r1 != r2:
            score -= 0.08                   # weak unconnected low cards

        street = round_state.get("street", "preflop")
        if   street == "flop":  score += 0.03
        elif street == "turn":  score += 0.05
        elif street == "river": score += 0.07

        return max(0.0, min(1.0, score))

    def _lookup_preflop_equity(self, hole_card):
        """
        Look up preflop win rate from precomputed table (memoization_table.json).
        Returns None if table not loaded or hand not found.
        """
        # Load table once per game instance
        if not hasattr(self, "_preflop_table"):
            try:
                import json, os
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "memoization_table.json"
                )
                with open(path, "r") as f:
                    self._preflop_table = json.load(f)
            except Exception:
                self._preflop_table = {}

        if not self._preflop_table:
            return None

        # Try both orderings of the two cards
        key1 = str(tuple(sorted(hole_card, key=lambda x: (x[1], x[0]))))
        key2 = str((hole_card[0], hole_card[1]))
        key3 = str((hole_card[1], hole_card[0]))

        for key in (key1, key2, key3):
            entry = self._preflop_table.get(key)
            if entry and entry.get("total_visits", 0) > 0:
                return entry["wins"] / entry["total_visits"]

        return None

    # def _estimate_hand_strength_mc(self, hole_card, round_state, nb_simulation=50):
    #     street = round_state.get("street", "preflop")
        
    #     if street == "preflop":
    #         # Load the full precomputed lookup table if available
    #         win_rate = self._lookup_preflop_equity(hole_card)
    #         if win_rate is not None:
    #             return win_rate
    #         # Fall back to hand-tuned table for premium hands
    #         canonical = self._get_canonical_hand(hole_card)
    #         PREFLOP_EQUITY = {
    #             "AA": 0.85, "KK": 0.82, "QQ": 0.80, "JJ": 0.77,
    #             "TT": 0.75, "99": 0.72, "88": 0.69,
    #             "AKs": 0.67, "AQs": 0.66, "AJs": 0.65, "ATs": 0.64,
    #             "KQs": 0.63, "KJs": 0.62,
    #             "AKo": 0.65, "AQo": 0.64, "AJo": 0.63,
    #             "ATo": 0.62, "KQo": 0.60
    #         }
    #         if canonical in PREFLOP_EQUITY:
    #             return PREFLOP_EQUITY[canonical]
    #         return self._estimate_hand_strength(hole_card, round_state)

    #     # Post-flop Monte Carlo (unchanged)
    #     community_str = round_state.get("community_card", [])
    #     try:
    #         hole_card_objs      = gen_cards(hole_card)
    #         community_card_objs = gen_cards(community_str) if community_str else []
    #         win_rate = estimate_hole_card_win_rate(
    #             nb_simulation=nb_simulation, nb_player=2,
    #             hole_card=hole_card_objs, community_card=community_card_objs
    #         )
    #         return float(win_rate)
    #     except Exception:
    #         return self._estimate_hand_strength(hole_card, round_state)
    # ============================================================
    # POT AND PRESSURE EXTRACTION
    # ============================================================

    def _estimate_hand_strength_mc(self, hole_card, round_state):
        street = round_state.get("street", "preflop")
        
        if street == "preflop":
            # Load the full precomputed lookup table if available
            win_rate = self._lookup_preflop_equity(hole_card)
            if win_rate is not None:
                return win_rate
            # Fall back to hand-tuned table for premium hands
            canonical = self._get_canonical_hand(hole_card)
            PREFLOP_EQUITY = {
                "AA": 0.85, "KK": 0.82, "QQ": 0.80, "JJ": 0.77,
                "TT": 0.75, "99": 0.72, "88": 0.69,
                "AKs": 0.67, "AQs": 0.66, "AJs": 0.65, "ATs": 0.64,
                "KQs": 0.63, "KJs": 0.62,
                "AKo": 0.65, "AQo": 0.64, "AJo": 0.63,
                "ATo": 0.62, "KQo": 0.60
            }
            if canonical in PREFLOP_EQUITY:
                return PREFLOP_EQUITY[canonical]
            return self._estimate_hand_strength(hole_card, round_state)

        # ============================================================
        # Post-Flop: Time-Bounded Anytime Monte Carlo
        # ============================================================
        community_str = round_state.get("community_card", [])
        try:
            hole_card_objs      = gen_cards(hole_card)
            community_card_objs = gen_cards(community_str) if community_str else []
            
            import time
            start_time = time.time()
            time_limit = 0.10  # 150 milliseconds maximum budget
            
            total_wins = 0.0
            total_sims = 0

            # Run in fast batches of 10 until time runs out
            while time.time() - start_time < time_limit:
                batch_win_rate = estimate_hole_card_win_rate(
                    nb_simulation=10, 
                    nb_player=2,
                    hole_card=hole_card_objs, 
                    community_card=community_card_objs
                )
                total_wins += (batch_win_rate * 10)
                total_sims += 10

            # Failsafe if the loop couldn't even run once
            if total_sims == 0:
                return self._estimate_hand_strength(hole_card, round_state)

            return float(total_wins / total_sims)

        except Exception:
            return self._estimate_hand_strength(hole_card, round_state)

    def _extract_pot_amount(self, round_state):
        """
        Extract the total pot size (main pot + all side pots).
        Handles all dict/int/float formats the engine may return.
        """
        pot = round_state.get("pot", {})

        if isinstance(pot, dict):
            main = pot.get("main", 0)
            side = pot.get("side", [])
            main_amount = main.get("amount", 0) if isinstance(main, dict) else int(main)
            side_amount = sum(p.get("amount", 0) for p in side if isinstance(p, dict))
            return main_amount + side_amount

        if isinstance(pot, (int, float)):
            return pot

        return 0

    def _extract_betting_pressure(self, round_state):
        """
        Estimate the betting pressure we are currently facing.

        Approximated as pot_amount × street_multiplier, where later streets
        have higher multipliers (bets are larger relative to stack depth).
        """
        pot    = self._extract_pot_amount(round_state)
        street = round_state.get("street", "preflop")
        mult   = {"preflop": 0.20, "flop": 0.25, "turn": 0.30, "river": 0.35}.get(
            street, 0.25
        )
        return pot * mult

    # ============================================================
    # UTILITY HELPERS
    # ============================================================

    def _bucketize(self, x, cutoffs):
        """
        Discretize a continuous value x into a bucket index [0, len(cutoffs)].

        Example: _bucketize(75, [20, 60, 120, 250]) → 2
          (75 > 20 → bucket 1, 75 > 60 → bucket 2, 75 <= 120 → stop)

        This is the key abstraction that compresses continuous pot/pressure
        values into a small number of discrete states, making the search tree
        tractable.
        """
        bucket = 0
        for c in cutoffs:
            if x > c:
                bucket += 1
            else:
                break
        return bucket

    def _argmax_with_tiebreak(self, action_scores, legal_actions):
        """
        Return the highest-scoring legal action.
        On ties, apply fixed priority: raise > call > fold.

        Fully deterministic — no randomness in final action selection.
        """
        best_score   = -float("inf")
        best_actions = []

        for action in legal_actions:
            s = action_scores.get(action, -float("inf"))
            if s > best_score:
                best_score   = s
                best_actions = [action]
            elif s == best_score:
                best_actions.append(action)

        for pref in ("raise", "call", "fold"):
            if pref in best_actions:
                return pref

        return best_actions[0]

    # ============================================================
    # ENGINE CALLBACKS  (required by BasePokerPlayer interface)
    # ============================================================

    def receive_game_start_message(self, game_info):
        """
        Called once at the start of each game.
        Resets weights and initialises persistent opponent counters.
        """
        if hasattr(self, "_hybrid_weights"):
            del self._hybrid_weights

        self._opp_raise_count = 0
        self._opp_call_count  = 0
        self._opp_fold_count  = 0
        self._opp_total       = 0
        self._opp_last_seen   = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """
        Called at the end of each round.
        Flushes final opponent actions into persistent counters
        before action_histories resets for the next round.
        """
        self._commit_round_opp_stats(round_state)


# ============================================================
# REQUIRED FACTORY FUNCTION
# ============================================================

def setup_ai():
    """
    Returns a fresh RaisedPlayer instance.
    Called by testperf.py and the tournament framework.
    """
    return RaisedPlayer()