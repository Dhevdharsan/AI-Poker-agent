# RaisedPlayer — AI Poker Agent

**COMPSCI 683: Artificial Intelligence | UMass Amherst | Spring 2026**

An AI poker agent for heads-up Limit Texas Hold'em, built from scratch over the course of a semester. The agent combines depth-limited adversarial search, persistent opponent modeling, a bluff layer, and a linear evaluation function whose weights were trained through two rounds of machine learning — first reinforcement learning, then a genetic algorithm when RL hit a wall.

This is the full development story: the ideas that worked, the ones that didn't, and what we'd do differently knowing what we know now.

---

## The Journey

### Phase 0 — Where We Started

The course gave us a `BasePokerPlayer` interface and a `RandomPlayer` baseline. Our first working version just computed a hand strength heuristic from the two hole cards (rank sum, pocket pair bonus, suited bonus) and used a fixed threshold: raise if strong, call if medium, fold if weak.

It beat the random agent. It lost to everything else.

The problem was obvious in hindsight: poker is a game of *hidden information and sequential decisions*. A static threshold ignores the opponent entirely, ignores what street you're on, and ignores pot size. You need to look ahead.

### Phase 1 — Adversarial Search (Alpha-Beta Minimax)

The first real architecture introduced **depth-limited Minimax with Alpha-Beta pruning** (`raise_player.py`). Instead of making a decision in isolation, the agent now searches a tree of possible action sequences 3 levels deep and picks the move with the best expected outcome.

The key design choices:

**State abstraction.** The real game state has continuous pot sizes, arbitrary stack depths, and 52-card combinatorics. We compressed it into a small set of discrete features:
- Pot size → 5 buckets (`[20, 60, 120, 250]` chip thresholds)
- Betting pressure → 5 buckets
- Street → index 0–3
- Opponent profile → one of 5 categories

This made the search tree tiny enough to run in well under the 0.5-second time limit.

**Profile-weighted opponent nodes.** Pure Minimax assumes the opponent plays optimally (always picks the worst move for you). Real opponents don't. Instead of a Min node, we model the opponent as a probability distribution over actions — aggressive opponents raise more, passive ones call more — and take the expected value. This is closer to how actual poker solvers model opponent ranges.

**Linear evaluation function at leaf nodes.** When the search hits depth 0, it needs a score. We defined:

```
φ(s) = Σ wⱼ · φⱼ(s)
```

Features: hand strength, pot bucket, pressure bucket, opponent aggression, street progress, pocket pair, suited, connected. The weights `wⱼ` started hand-tuned. Training them was the next challenge.

### Phase 2 — Reinforcement Learning (and its failure)

We wrote `train.py`: a 3-phase policy gradient RL loop.

**Phase 1 (iterations 0–59):** Agent vs RandomPlayer  
**Phase 2 (iterations 60–119):** Agent vs StrongRulePlayer (tight-aggressive rule-based bot)  
**Phase 3 (iterations 120–179):** Self-play

The update rule:

```
wⱼ ← wⱼ + α · reward · φⱼ_avg · sign(wⱼ)
```

where `reward = (agent_stack - opp_stack) / initial_stack` and `α` decayed from 0.05 → 0.01.

**What happened:** The agent learned quickly against RandomPlayer, but stalled in Phase 2. Worse, it drifted into a local optimum we called the **"additive trap"**: the pot bucket weight grew very large because winning big pots correlated with high rewards. The agent learned to *chase* large pots — calling with mediocre hands when the pot was already big, falling for the sunk-cost fallacy. It would rather lose 300 chips in a big pot than fold and preserve 50.

Local gradient descent can't escape this. The gradient always points "more pot chasing is correlated with winning" because the training data is biased toward hands where the agent happened to win large pots.

### Phase 3 — Genetic Algorithm

To escape the local optimum, we replaced gradient descent with a **population-based genetic algorithm** (`genetic_train.py`).

**Setup:**
- Population of 8 agents, each with randomly perturbed weight vectors
- Each generation: play 5 games vs StrongRulePlayer, measure fitness (chips won per game)
- Selection: keep the top 3
- Crossover: breed 5 children — each weight independently inherited from one of the top 3 parents
- Mutation: randomly nudge weights by ±0.3–1.0 with 40% probability

**Why GA over RL here:**
- RL follows one gradient direction and gets stuck. GA explores 8 directions simultaneously and recombines the best parts.
- Fitness is measured over complete games, not individual action sequences — this gives a much cleaner signal of what actually wins.
- No learning rate to tune; the population naturally preserves diversity.

**Result:** The GA found a dramatically different weight landscape. It cut the pot bucket weight by half, increased the hand strength weight, and made pressure bucket and opponent aggression strongly negative. The agent learned to play tighter and more defensively — fold bad situations, extract value when strong.

### Phase 4 — The Hybrid

We merged the two training runs. The GA weights were used as the base; the RL weights informed the conditional feature magnitudes (semi-bluff bonus, weak raise penalty, slow play bonus). Final weights were saved as `hybrid_weights.json`.

**Trained weights vs defaults:**

| Feature | Default | Trained | Change |
|---------|---------|---------|--------|
| hand_strength | +4.80 | +8.82 | ▲ +4.02 |
| pot_bucket | +0.90 | +3.96 | ▲ +3.06 |
| pressure_bucket | -1.00 | -4.00 | ▼ -3.00 |
| opp_aggression | -1.10 | -1.99 | ▼ -0.89 |
| pair_bonus | +1.50 | +3.93 | ▲ +2.43 |
| fold_bias | -0.15 | -0.77 | ▼ -0.62 |

### Phase 5 — The Bluff Layer

Late in development we added a rule-based bluff layer that fires *before* Minimax on each turn:

**Semi-bluff (~25% trigger rate):** On the flop or turn, if we have a flush or straight draw (4 to a suit or 4 consecutive ranks), raise. Even if called, drawing hands have ~35% equity to complete by river. Only fires against passive/balanced opponents — bluffing a calling station is pointless.

**Reverse bluff / slow play (~15%):** With a monster hand (strength > 0.85) on early streets against an aggressive opponent in a small pot, flat-call instead of raising. The goal: disguise strength and let the aggressive opponent keep betting into us.

**Pure river bluff (~20%):** On the river with a weak hand in a small pot against a passive or balanced opponent who checked (call amount = 0), raise. With ~27% fold equity against these opponents, the EV is positive in small pots.

The triggers use a deterministic hash of game state rather than `random.random()` to keep decisions reproducible.

### Phase 6 — Preflop Table & Opponent Modeling

**Preflop equity table (`memoization_table.json`):** Rather than running Monte Carlo on every preflop action, we precomputed win rates for all ~169 canonical starting hands. The agent looks up its hand in O(1) instead of burning 50–100ms on MC simulations preflop.

**Persistent opponent model:** Unlike the round state (which resets between rounds), our opponent counters accumulate across the entire game. We track raise/call/fold frequencies from the first hand to the last, flushing them into persistent counters at the end of each round before `action_histories` resets. After enough data (≥4 voluntary actions), the opponent gets classified as one of: `maniac`, `aggressive`, `balanced`, `passive`, or `calling_station`.

---

## Architecture Reference

```
declare_action()
    │
    ├─ Load hybrid_weights.json (once per game)
    ├─ Compute hand strength (preflop: table lookup; post-flop: Monte Carlo)
    ├─ Classify opponent profile from persistent action history
    │
    ├─ Bluff layer
    │   ├─ Semi-bluff check (flop/turn + draw + non-aggressive opp)
    │   ├─ Reverse bluff check (monster hand + aggressive opp + small pot)
    │   └─ Pure river bluff (weak hand + passive opp checked + small pot)
    │
    ├─ Heuristic pruning (hand < 0.40 + no draw → fold or free check)
    │
    └─ Alpha-Beta Minimax (depth 3)
        ├─ Max nodes (our turn): standard alpha-beta
        ├─ Opponent nodes: profile-weighted expected value
        └─ Leaf nodes: φ(s) = Σ wⱼ · φⱼ(s)
```

---

## What Went Wrong (Post-Mortem)

After the tournament we did a structural review and found four bugs we should have caught during development:

**1. Minimax terminal values were not pot-scaled.**
When we fold, `terminal_value = 0.0`. When the opponent folds, `terminal_value = 2.0`. These are constants, completely disconnected from actual pot size. Folding a 300-chip pot and a 20-chip pot looked the same to the search. The fix is to scale terminal values by the actual pot amount.

**2. The opponent model reset every game.**
The tournament ran 200 games of 10 hands each. `receive_game_start_message` reset all opponent counters to zero. The classifier needs at least 4 voluntary actions before leaving "balanced" mode — which takes 2–3 hands. For the majority of every 10-hand game, the agent was flying blind. The fix: persist opponent stats in a file keyed by opponent name.

**3. Weights overfit to training opponents.**
The GA trained exclusively against StrongRulePlayer. The extreme weight values (pair_bonus: 3.93, pressure_bucket: -4.00) learned to exploit that specific opponent's tendencies. Against 16 diverse teams, these were miscalibrated. The fix: train against a diverse pool, including self-play and multiple rule-based opponents.

**4. Relative file path for weights.**
`_load_hybrid_weights("hybrid_weights.json")` uses a relative path. If the tournament runner's working directory differs from the submission folder, the agent silently falls back to default weights. The fix: use `os.path.dirname(os.path.abspath(__file__))` to locate the file relative to the script itself (already done in `memoization_table.json` lookup — should have been consistent).

The design was sound. The execution had four specific, fixable bugs. That's a better place to be than a fundamentally flawed approach.

---

## File Structure

```
AI-Poker-agent/
├── raise_player.py          # Final agent — the tournament submission
├── v2_player.py             # Experimental MCTS-Minimax hybrid (intermediate version)
├── train.py                 # 3-phase RL training: Random → StrongRule → self-play
├── genetic_train.py         # Genetic algorithm weight search
├── testperf.py              # Head-to-head evaluation harness
├── example.py               # Quick start example
│
├── hybrid_weights.json      # Final trained weights (GA + RL hybrid)
├── genetic_weights.json     # GA-only weights (for comparison)
├── memoization_table.json   # Precomputed preflop equity table (~169 hands)
│
├── randomplayer.py          # Baseline: random legal action
├── always_raise.py          # Baseline: always raise
├── strong_rule_player.py    # Baseline: tight-aggressive rule-based (used in training)
│
├── Archive/submission/      # Exact files submitted to the tournament
│   ├── custom_player.py
│   ├── hybrid_weights.json
│   └── memoization_table.json
│
└── pypokerengine/           # Poker engine (pypokerengine library)
```

---

## How to Run

**Quick sanity check — RaisedPlayer vs Random:**
```bash
python3 testperf.py -n1 "RaisedPlayer" -a1 RaisedPlayer -n2 "Random" -a2 RandomPlayer
```

**Benchmark vs tight-aggressive rule player:**
```bash
python3 testperf.py -n1 "RaisedPlayer" -a1 RaisedPlayer -n2 "StrongRule" -a2 StrongRulePlayer
```

**Self-play sanity check (should be ~50/50):**
```bash
python3 testperf.py -n1 "Agent1" -a1 RaisedPlayer -n2 "Agent2" -a2 RaisedPlayer
```

**Compare experimental V2 against final agent:**
```bash
python3 testperf.py -n1 "V2Player" -a1 V2Player -n2 "RaisedPlayer" -a2 RaisedPlayer
```

**Retrain weights with RL (outputs hybrid_weights.json):**
```bash
python3 train.py
```

**Retrain weights with Genetic Algorithm (outputs genetic_weights.json):**
```bash
python3 genetic_train.py
```

No external dependencies beyond Python 3.8+ and the included `pypokerengine` module.

---

## Key Design Decisions

**Why Minimax over pure MCTS?**
We prototyped an MCTS-Minimax hybrid in `v2_player.py`. MCTS is better in theory — it allocates search budget proportionally to promising branches rather than exploring uniformly. But it requires many rollouts to converge, and 0.5 seconds is not a lot of time. In practice, depth-3 Alpha-Beta with a good evaluation function outperformed the hybrid in head-to-head tests, because the evaluation function captured enough signal that deeper uniform search beat shallower adaptive search.

**Why Genetic Algorithm after RL?**
RL with policy gradient follows a single gradient direction. When the loss landscape has a local optimum that's not the global one (as happened with the pot-chasing trap), gradient descent stays stuck. GA maintains a *population* of solutions and recombines the best parts of multiple directions simultaneously. It's slower per iteration but much less likely to get stuck.

**Why deterministic bluff triggers?**
We used a hash of game state + salt instead of `random.random()`. This makes the agent's decisions reproducible (same hand → same bluff decision), which is useful for debugging and evaluation. The downside — which we recognized too late — is that a sufficiently adaptive opponent could reverse-engineer the trigger function and always call our bluffs. True mixed strategies require genuine randomness.

---

*COMPSCI 683 — Group 9 — Spring 2026, UMass Amherst*
