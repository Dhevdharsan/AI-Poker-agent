from pypokerengine.players import BasePokerPlayer
import time
import math
import random
import os
import json

# ============================================================
# MCTS NODE
# ============================================================
class MCTSNode:
    def __init__(self, state, action=None, parent=None):
        self.state        = state
        self.action       = action
        self.parent       = parent
        self.children     = []
        self.visits       = 0
        self.total_reward = 0.0
        self.untried      = list(state.get("legal_actions", []))

    @property
    def is_fully_expanded(self):
        return len(self.untried) == 0

    @property
    def is_terminal(self):
        return self.state.get("terminalish", False) or \
               len(self.state.get("legal_actions", [])) == 0

    def ucb1(self, c=1.4):
        if self.visits == 0:
            return float("inf")
        return (self.total_reward / self.visits) + \
               c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c=1.4):
        return max(self.children, key=lambda n: n.ucb1(c))

    def most_visited_child(self):
        return max(self.children, key=lambda n: n.visits)

# ============================================================
# MAIN PLAYER CLASS
# ============================================================
# MODIFICATION 1: Renamed to V2Player
class V2Player(BasePokerPlayer):

    MCTS_BUDGET     = 0.38   
    MINIMAX_BUDGET  = 0.40   
    C_PARAM         = 1.4    
    PHASE2_THRESHOLD = 6      
    PHASE3_THRESHOLD = 15    

    def declare_action(self, valid_actions, hole_card, round_state):
        legal_actions = [a["action"] for a in valid_actions]

        # MODIFICATION 2: Load the isolated weights
        if not hasattr(self, "_weights"):
            self._weights = self._load_weights("v2_weights.json")

        self._update_opponent_stats(round_state)
        opp_total    = getattr(self, "_opp_total", 0)
        opp_type     = self._classify_opponent()
        opp_profile  = self._classify_profile_for_minimax()

        hand_strength = self._estimate_hand_strength(hole_card, round_state)
        street        = round_state.get("street", "preflop")
        pot           = self._extract_pot_amount(round_state)

        # Bluff layer
        if ("raise" in legal_actions and street in ("flop", "turn") and 
            opp_profile != "aggressive" and self._has_draw_potential(hole_card, round_state)):
            if self._det_trigger(round_state, salt=7) < 0.25:
                return "raise"   

        if ("call" in legal_actions and hand_strength > 0.85 and 
            street in ("preflop", "flop") and opp_profile == "aggressive" and pot < 100):
            if self._det_trigger(round_state, salt=13) < 0.15:
                return "call"    

        # Exploit layer
        if opp_total >= self.PHASE3_THRESHOLD:
            exploit = self._exploit_action(opp_type, hand_strength, legal_actions, round_state)
            if exploit is not None:
                return exploit

        root_state = self._build_abstract_state(
            hole_card, round_state, legal_actions, opp_profile, hand_strength
        )

        if opp_total < self.PHASE2_THRESHOLD:
            action = self._mcts_search(root_state, informed=False)
        elif opp_total < self.PHASE3_THRESHOLD:
            action = self._minimax_search(root_state)
        else:
            action = self._mcts_search(root_state, informed=True)

        return action if action in legal_actions else legal_actions[0]

    def _mcts_search(self, root_state, informed=True):
        root     = MCTSNode(root_state)
        deadline = time.time() + self.MCTS_BUDGET

        while time.time() < deadline:
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            reward = self._simulate(node.state, informed=informed)
            self._backprop(node, reward)

        if not root.children:
            return root_state.get("legal_actions", ["call"])[0]

        return root.most_visited_child().action

    def _select(self, node):
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child(self.C_PARAM)
        return node

    def _expand(self, node):
        action    = node.untried.pop(random.randrange(len(node.untried)))
        actor     = "self" if node.state.get("our_turn", True) else "opponent"
        new_state = self._transition(node.state, action, actor)
        child     = MCTSNode(new_state, action=action, parent=node)
        node.children.append(child)
        return child

    def _simulate(self, state, informed=True, max_depth=8):
        s     = dict(state)
        depth = 0
        while not s.get("terminalish", False) and depth < max_depth:
            actions = s.get("legal_actions", [])
            if not actions: break
            our_turn = s.get("our_turn", True)
            action   = self._rollout_policy(s, our_turn, informed)
            actor    = "self" if our_turn else "opponent"
            s        = self._transition(s, action, actor)
            depth   += 1
        return self._terminal_reward(s)

    def _rollout_policy(self, state, our_turn, informed):
        actions = state.get("legal_actions", ["call"])
        hs      = state.get("hand_strength", 0.5)

        if our_turn:
            if hs >= 0.70 and "raise" in actions: return "raise"
            if hs >= 0.40 and "call" in actions: return "call"
            return "fold" if "fold" in actions else actions[0]
        else:
            if not informed: return random.choice(actions)
            af   = getattr(self, "_opp_af",   1.0)
            vpip = getattr(self, "_opp_vpip", 0.5)
            r    = random.random()
            if af > 2.0:
                if r < 0.45 and "raise" in actions: return "raise"
                if r < 0.80 and "call"  in actions: return "call"
                return "fold" if "fold" in actions else actions[0]
            elif vpip < 0.25:
                if r < 0.45: return "fold" if "fold" in actions else actions[0]
                if r < 0.85 and "call" in actions: return "call"
                return "raise" if "raise" in actions else actions[0]
            else:
                if r < 0.30 and "raise" in actions: return "raise"
                if r < 0.75 and "call"  in actions: return "call"
                return "fold" if "fold" in actions else actions[0]

    def _backprop(self, node, reward):
        while node is not None:
            node.visits       += 1
            node.total_reward += reward
            node = node.parent

    def _terminal_reward(self, state):
        if state.get("terminalish", False):
            tv = state.get("terminal_value", 0.0)
            return (tv + 3.0) / 5.0  
        return state.get("hand_strength", 0.5)

    def _minimax_search(self, root_state):
        legal      = root_state.get("legal_actions", [])
        alpha      = -float("inf")
        beta       =  float("inf")
        best_score = -float("inf")
        best_acts  = []

        for action in legal:
            next_state = self._transition(root_state, action, actor="self")
            score = self._minimax(next_state, depth=2, maximizing=False, alpha=alpha, beta=beta)
            if score > best_score:
                best_score = score
                best_acts  = [action]
            elif score == best_score:
                best_acts.append(action)
            alpha = max(alpha, score)

        for pref in ("raise", "call", "fold"):
            if pref in best_acts: return pref
        return best_acts[0] if best_acts else legal[0]

    def _minimax(self, state, depth, maximizing, alpha, beta):
        if depth == 0 or state.get("terminalish", False):
            return self._evaluate(state)
        legal = state.get("legal_actions", [])
        if not legal: return self._evaluate(state)

        if maximizing:
            value = -float("inf")
            for action in legal:
                child = self._transition(state, action, actor="self")
                value = max(value, self._minimax(child, depth-1, False, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta: break
            return value
        else:
            child_values = []
            for action in legal:
                child = self._transition(state, action, actor="opponent")
                cv    = self._minimax(child, depth-1, True, alpha, beta)
                child_values.append((action, cv))
            return self._opponent_response_ev(child_values, state)

    def _opponent_response_ev(self, child_values, state):
        profile = state.get("opp_profile", "balanced")
        dist = {
            "aggressive": {"raise": 0.55, "call": 0.30, "fold": 0.15},
            "passive":    {"raise": 0.15, "call": 0.55, "fold": 0.30},
            "balanced":   {"raise": 0.33, "call": 0.44, "fold": 0.23},
        }.get(profile, {"raise": 0.33, "call": 0.44, "fold": 0.23})

        total, norm = 0.0, 0.0
        for action, val in child_values:
            w      = dist.get(action, 0.0)
            total += w * val
            norm  += w
        if norm == 0: return sum(v for _, v in child_values) / max(1, len(child_values))
        return total / norm

    def _evaluate(self, state):
        if state.get("terminalish", False): return state.get("terminal_value", 0.0)
        w = self._weights
        hs       = state["hand_strength"]
        pot      = state["pot_bucket"]      / 4.0
        pressure = state["pressure_bucket"] / 4.0
        opp_agg  = state["opp_aggression"]
        street   = state["street_index"]    / 3.0
        pair     = 1.0 if state["pair"]     else 0.0
        suited   = 1.0 if state["suited"]   else 0.0
        conn     = 1.0 if state["connected"] else 0.0
        has_draw = state.get("has_draw", False)

        score  = w["hand_strength"]   * hs
        score += w["pot_bucket"]      * pot
        score += w["pressure_bucket"] * pressure
        score += w["opp_aggression"]  * opp_agg
        score += w["street_progress"] * street
        score += w["pair_bonus"]      * pair
        score += w["suited_bonus"]    * suited
        score += w["connected_bonus"] * conn

        last = state.get("last_self_action")
        if last == "raise":
            if hs >= 0.70:      score += w["strong_raise_bonus"]
            elif has_draw:      score += w["semi_bluff_bonus"]
            else:               score += w["weak_raise_penalty"]
        elif last == "call":
            if hs > 0.85:       score += w["slow_play_bonus"]
            elif hs < 0.35:     score += w["weak_call_penalty"]
        elif last == "fold":
            score += w["fold_bias"]
        return score

    def _exploit_action(self, opp_type, hand_strength, legal_actions, round_state):
        if opp_type == "calling_station":
            if hand_strength >= 0.60 and "raise" in legal_actions: return "raise"
            if hand_strength >= 0.35 and "call" in legal_actions: return "call"
            return "fold" if "fold" in legal_actions else legal_actions[0]
        if opp_type == "maniac":
            if hand_strength > 0.82 and "call" in legal_actions: return "call"  
            if hand_strength > 0.92 and "raise" in legal_actions: return "raise"
            if hand_strength < 0.38: return "fold" if "fold" in legal_actions else legal_actions[0]
            return None         
        if opp_type == "nit":
            last = self._last_opponent_action(round_state)
            if last == "raise" and hand_strength < 0.72: return "fold" if "fold" in legal_actions else legal_actions[0]
            if hand_strength >= 0.42 and "raise" in legal_actions: return "raise" 
            return None
        if opp_type == "tag":
            last = self._last_opponent_action(round_state)
            if last == "raise" and hand_strength < 0.68: return "fold" if "fold" in legal_actions else legal_actions[0]
            return None
        return None   

    def _build_abstract_state(self, hole_card, round_state, legal_actions, opp_profile, hand_strength):
        card_feats      = self._hole_card_features(hole_card)
        pot_amount      = self._extract_pot_amount(round_state)
        pot_bucket      = self._bucketize(pot_amount, [20, 60, 120, 250])
        pressure        = self._extract_betting_pressure(round_state)
        pressure_bucket = self._bucketize(pressure, [10, 30, 60, 120])
        street_name     = round_state.get("street", "preflop")
        street_index    = {"preflop":0,"flop":1,"turn":2,"river":3}.get(street_name, 0)
        has_draw        = self._has_draw_potential(hole_card, round_state)
        af              = getattr(self, "_opp_af", 1.0)

        return {
            "hand_strength":    hand_strength,
            "pair":             card_feats["pair"],
            "suited":           card_feats["suited"],
            "connected":        card_feats["connected"],
            "has_draw":         has_draw,
            "pot_bucket":       pot_bucket,
            "pressure_bucket":  pressure_bucket,
            "street_index":     street_index,
            "opp_profile":      opp_profile,
            "opp_aggression":   min(1.0, af / 4.0),
            "legal_actions":    list(legal_actions),
            "terminalish":      False,
            "terminal_value":   0.0,
            "our_turn":         True,
            "last_self_action": None,
            "last_opp_action":  None,
        }

    def _transition(self, state, action, actor):
        ns = dict(state)
        ns["legal_actions"] = ["fold", "call", "raise"]
        if actor == "self":
            ns["our_turn"]         = False
            ns["last_self_action"] = action
            if action == "raise":
                ns["pot_bucket"]      = min(4, state["pot_bucket"] + 1)
                ns["pressure_bucket"] = max(0, state["pressure_bucket"] - 1)
            elif action == "call":
                ns["pressure_bucket"] = max(0, state["pressure_bucket"] - 1)
            elif action == "fold":
                ns["terminalish"]    = True
                ns["terminal_value"] = -3.0
        else:
            ns["our_turn"]        = True
            ns["last_opp_action"] = action
            if action == "raise":
                ns["pot_bucket"]      = min(4, state["pot_bucket"] + 1)
                ns["pressure_bucket"] = min(4, state["pressure_bucket"] + 1)
            elif action == "fold":
                ns["terminalish"]    = True
                ns["terminal_value"] = 2.0
        ns["street_index"] = min(3, state["street_index"] + 1)
        return ns

    def _update_opponent_stats(self, round_state):
        if not hasattr(self, "_opp_raise_count"):
            self._opp_raise_count = 0
            self._opp_call_count  = 0
            self._opp_fold_count  = 0
            self._opp_total       = 0
            self._opp_vpip_hands  = 0
            self._opp_pfr_hands   = 0
            self._opp_hands_seen  = 0
            self._seen_actions    = set()

        my_uuid   = getattr(self, "uuid", None)
        histories = round_state.get("action_histories", {})

        for street, acts in histories.items():
            for i, act in enumerate(acts):
                uid  = act.get("uuid") or act.get("player_uuid")
                move = act.get("action", "").upper()
                key  = (street, i, uid, move)

                if key in self._seen_actions or uid == my_uuid: continue
                if move in ("SMALLBLIND", "BIGBLIND", "ANTE"): continue

                self._seen_actions.add(key)
                self._opp_total += 1

                if move == "RAISE":
                    self._opp_raise_count += 1
                    if street == "preflop":
                        self._opp_pfr_hands  += 1
                        self._opp_vpip_hands += 1
                elif move in ("CALL", "CHECK"):
                    self._opp_call_count += 1
                    if street == "preflop":
                        self._opp_vpip_hands += 1
                elif move == "FOLD":
                    self._opp_fold_count += 1

        calls_folds = max(1, self._opp_call_count + self._opp_fold_count)
        hands       = max(1, self._opp_hands_seen + 1)

        self._opp_af        = self._opp_raise_count / calls_folds
        self._opp_vpip      = self._opp_vpip_hands  / hands
        self._opp_pfr       = self._opp_pfr_hands   / hands
        self._opp_fold_rate = self._opp_fold_count  / max(1, self._opp_total)

    def _classify_opponent(self):
        if not hasattr(self, "_opp_af") or getattr(self, "_opp_total", 0) < 6: return "balanced"
        vpip = self._opp_vpip
        pfr  = self._opp_pfr
        af   = self._opp_af
        if vpip > 0.55 and af < 1.0:  return "calling_station"
        if pfr  > 0.45 and af > 2.5:  return "maniac"
        if vpip < 0.20:                return "nit"
        if vpip < 0.35 and af > 1.8:  return "tag"
        return "balanced"

    def _classify_profile_for_minimax(self):
        if not hasattr(self, "_opp_af") or getattr(self, "_opp_total", 0) < 4: return "balanced"
        if self._opp_af >= 1.8:                                        return "aggressive"
        if self._opp_fold_rate >= 0.35 and self._opp_af < 1.0:         return "passive"
        if getattr(self, "_opp_vpip", 0.5) >= 0.50 and self._opp_af < 1.0: return "passive"
        return "balanced"

    def _last_opponent_action(self, round_state):
        my_uuid   = getattr(self, "uuid", None)
        histories = round_state.get("action_histories", {})
        for street in ("river", "turn", "flop", "preflop"):
            for act in reversed(histories.get(street, [])):
                uid  = act.get("uuid") or act.get("player_uuid")
                move = act.get("action", "").upper()
                if uid != my_uuid and move not in ("SMALLBLIND","BIGBLIND","ANTE"):
                    return move.lower()
        return None

    def _has_draw_potential(self, hole_card, round_state):
        community = round_state.get("community_card", [])
        if not community: return False
        all_cards = hole_card + community
        suits = [self._parse_card(c)[1] for c in all_cards]
        if any(suits.count(s) >= 4 for s in set(suits)): return True
        ranks = sorted(set(self._parse_card(c)[0] for c in all_cards))
        for lo in range(2, 12):
            if sum(1 for r in ranks if lo <= r <= lo + 4) >= 4: return True
        return False

    def _det_trigger(self, round_state, salt):
        round_count = round_state.get("round_count", 1)
        pot         = int(self._extract_pot_amount(round_state))
        street_idx  = {"preflop":0,"flop":1,"turn":2,"river":3}.get(round_state.get("street","preflop"), 0)
        raw = (round_count * salt * 17 + pot * 3 + street_idx * 11) % 100
        return raw / 100.0

    def _parse_card(self, card_str):
        if not isinstance(card_str, str) or len(card_str) != 2: return 2, "X"
        suit = card_str[0].upper()
        rank = card_str[1].upper()
        rank_map = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}
        return rank_map.get(rank, 2), suit

    def _hole_card_features(self, hole_card):
        if len(hole_card) != 2: return {"pair": False, "suited": False, "connected": False}
        r1, s1 = self._parse_card(hole_card[0])
        r2, s2 = self._parse_card(hole_card[1])
        return {"pair": r1 == r2, "suited": s1 == s2, "connected": abs(r1 - r2) == 1}

    def _estimate_hand_strength(self, hole_card, round_state):
        if len(hole_card) != 2: return 0.3
        r1, s1 = self._parse_card(hole_card[0])
        r2, s2 = self._parse_card(hole_card[1])
        hi, lo  = max(r1, r2), min(r1, r2)
        gap     = abs(r1 - r2)
        score   = (hi + lo) / 28.0

        if r1 == r2:    score += 0.35 + (hi / 20.0)
        if s1 == s2:    score += 0.08
        if gap == 1:    score += 0.08
        elif gap == 2:  score += 0.03

        broadway = sum(1 for r in (r1, r2) if r >= 10)
        score   += 0.04 * broadway
        if hi <= 9 and gap >= 3 and r1 != r2: score -= 0.08

        street = round_state.get("street", "preflop")
        if   street == "flop":  score += 0.03
        elif street == "turn":  score += 0.05
        elif street == "river": score += 0.07

        return max(0.0, min(1.0, score))

    def _extract_pot_amount(self, round_state):
        pot = round_state.get("pot", {})
        if isinstance(pot, dict):
            main = pot.get("main", 0)
            side = pot.get("side", [])
            main_amount = main.get("amount", 0) if isinstance(main, dict) else int(main)
            side_amount = sum(p.get("amount", 0) for p in side if isinstance(p, dict))
            return main_amount + side_amount
        if isinstance(pot, (int, float)): return pot
        return 0

    def _extract_betting_pressure(self, round_state):
        pot    = self._extract_pot_amount(round_state)
        street = round_state.get("street", "preflop")
        mult   = {"preflop":0.20,"flop":0.25,"turn":0.30,"river":0.35}.get(street, 0.25)
        return pot * mult

    def _bucketize(self, x, cutoffs):
        bucket = 0
        for c in cutoffs:
            if x > c: bucket += 1
            else: break
        return bucket

    def _load_weights(self, path):
        defaults = {
            "hand_strength":      4.8, "pot_bucket":          0.9, "pressure_bucket":   -1.0,
            "opp_aggression":    -1.1, "street_progress":     0.5, "pair_bonus":          1.5,
            "suited_bonus":       0.6, "connected_bonus":     0.4, "strong_raise_bonus":  1.1,
            "semi_bluff_bonus":   0.8, "weak_raise_penalty": -2.0, "slow_play_bonus":     0.6,
            "weak_call_penalty": -1.4, "fold_bias":          -0.15,
        }
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    loaded = json.load(f)
                for k, v in loaded.items():
                    if k in defaults: defaults[k] = float(v)
            except Exception: pass
        return defaults

    def receive_game_start_message(self, game_info):
        for attr in ("_opp_raise_count","_opp_call_count","_opp_fold_count",
                     "_opp_total","_opp_vpip_hands","_opp_pfr_hands",
                     "_opp_hands_seen","_seen_actions",
                     "_opp_af","_opp_vpip","_opp_pfr","_opp_fold_rate",
                     "_weights"):
            if hasattr(self, attr): delattr(self, attr)

    def receive_round_start_message(self, round_count, hole_card, seats):
        if hasattr(self, "_opp_hands_seen"): self._opp_hands_seen += 1

    def receive_street_start_message(self, street, round_state):   pass
    def receive_game_update_message(self, action, round_state):    pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass