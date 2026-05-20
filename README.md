# GauntletAgent — AI Poker Agent

**COMPSCI 683: Artificial Intelligence | UMass Amherst | Spring 2026**

An AI agent for heads-up Limit Texas Hold'em — a game with roughly 10¹⁷ game-tree nodes that was only weakly solved at near-Nash accuracy in 2015. Brute-force search is impossible: the game has stochastic deals, hidden opponent cards, and a branching factor that compounds across four betting streets. Every practical agent must combine state abstraction, value estimation, and an opponent model.

This agent (**GauntletAgent**, implemented in `raise_player.py`) uses depth-limited adversarial search with a profile-weighted opponent model, a three-tier bluff pipeline, and a linear evaluation function whose weights were evolved by a Genetic Algorithm across a diverse 4-opponent training Gauntlet. It was selected as our group's tournament submission after an internal two-criteria benchmarking protocol against three other agents built independently by our teammates.

This README covers the full development arc: starting from a static threshold rule, through Alpha-Beta Minimax with Q-Learning weights (HybridAgent), hitting a local optimum, escaping it with a Genetic Algorithm, and arriving at a balanced tight-aggressive (TAG) generalist policy.

---

## The Journey

### Phase 0 — Where We Started

The course gave us a `BasePokerPlayer` interface and a `RandomPlayer` baseline. Our first working version computed a hand strength heuristic from the two hole cards — rank sum, pocket pair bonus, suited bonus — and used a fixed threshold: raise if strong, call if medium, fold if weak.

It beat the random agent. It lost to everything else.

The problem was obvious in hindsight: poker is a game of *hidden information and sequential decisions*. A static threshold ignores the opponent entirely, ignores what street you're on, and ignores pot size. You need to look ahead.

### Phase 1 — Adversarial Search: HybridAgent

The first real architecture introduced **depth-3 Alpha-Beta Minimax** over an abstract state space. Instead of making a decision in isolation, the agent searches a tree of possible action sequences and picks the move with the best expected outcome.

**State abstraction.** The real game state has continuous pot sizes, arbitrary stacks, and 52-card combinatorics. We compressed it into discrete features:
- Pot size → 5 buckets (`[20, 60, 120, 250]` chip thresholds)
- Betting pressure → 5 buckets
- Street → index 0–3
- Opponent profile → one of 5 categories

**Profile-weighted opponent nodes.** Pure Minimax assumes the opponent always plays the worst move for you. Real opponents don't. At each opponent node we compute a profile-weighted expected value:

```
V_opp(s) = Σ Pr(a | ρ) · V(s, a)
```

where ρ ∈ {aggressive, passive, balanced} is inferred from observed action frequencies:

| Profile | Raise | Call | Fold |
|---------|-------|------|------|
| Aggressive | 55% | 30% | 15% |
| Passive | 15% | 55% | 30% |
| Balanced | 33% | 44% | 23% |

**Linear evaluation function at leaf nodes.** At depth-0 cutoffs:

```
φ(s) = Σ wⱼ · φⱼ(s)
```

Eight base features: hand strength, normalised pot bucket, normalised pressure bucket, opponent aggression rate, normalised street index, and three boolean hole-card features (pair / suited / connected). Six conditional terms activate based on the most recent self-action: strong-raise bonus, semi-bluff bonus, weak-raise penalty, slow-play bonus, weak-call penalty, and fold bias. This version — **HybridAgent** — had weights trained via Q-Learning.

### Phase 2 — Reinforcement Learning (and its failure)

We wrote `train.py`: a 3-phase policy gradient RL loop.

**Phase 1 (iterations 0–59):** Agent vs RandomPlayer  
**Phase 2 (iterations 60–119):** Agent vs StrongRulePlayer (tight-aggressive rule-based)  
**Phase 3 (iterations 120–179):** Self-play

Update rule with decaying learning rate α (0.05 → 0.01):

```
wⱼ ← wⱼ + α · reward · φⱼ_avg · sign(wⱼ)
```

**What went wrong.** The agent learned quickly against RandomPlayer, but stalled in Phase 2. Worse, it drifted into a local optimum we called the **"additive trap"**: the pot bucket weight grew very large because winning big pots correlated with high rewards. The agent learned to *chase* large pots — calling with mediocre hands when the pot was already big, falling for the sunk-cost fallacy. Local gradient descent can't escape this: the gradient always points toward more pot-chasing because training data is biased toward hands the agent happened to win.

### Phase 3 — Genetic Algorithm on a 4-Opponent Gauntlet

To escape the local optimum we replaced gradient descent with a **population-based Genetic Algorithm** (`genetic_train.py`), with fitness evaluated across a 4-opponent Gauntlet: RandomPlayer, StrongRulePlayer, a rollout-based MC agent, and a co-evolutionary sibling.

**Why a Gauntlet instead of a single benchmark?**  
Evaluating against one fixed opponent produces a specialist — a policy that exploits that opponent's specific tendencies but is brittle everywhere else. The Gauntlet forces the GA to optimise four sub-objectives simultaneously, producing a *generalist* that is robust against diverse unknown opponents. This directly mirrors the unknown-cohort regime of the class tournament.

**GA setup:**
- Population of 8 agents with independently perturbed weight vectors
- Each generation: evaluate each agent on 5 games per Gauntlet opponent (20 games total), fitness = average chips won
- Selection: retain top 3 survivors, discard bottom 5
- Crossover: breed 5 children — each weight independently inherited from one of the top 3 parents (uniform crossover)
- Mutation: nudge weights randomly with probability p_mut = 0.30

**Result.** The GA converged to a qualitatively different weight landscape — a tight-aggressive (TAG) profile that HybridAgent's gradient descent could not reach. The five conditional bonus/penalty terms were frozen throughout to prevent the GA from unlearning foundational poker logic.

### Phase 4 — The Hybrid Weights

The final weights used both training runs: the GA provided the evolved base weights; the RL run validated the conditional feature magnitudes. The complete weight comparison:

| Feature | HybridAgent (Q-Learning) | GauntletAgent (GA) | Change |
|---------|--------------------------|---------------------|--------|
| hand_strength | +4.80 | +8.82 | ↑ +84% |
| pot_bucket | +0.90 | +3.96 | ↑ |
| pressure_bucket | −1.00 | −4.00 | ↓ 4× more negative |
| opp_aggression | −1.10 | −1.99 | ↓ |
| street_progress | +0.50 | +0.48 | — |
| pair_bonus | +1.50 | +3.93 | ↑ +162% |
| suited_bonus | +0.60 | +0.37 | — |
| connected_bonus | +0.40 | +0.75 | — |
| fold_bias | −0.15 | −0.77 | ↓ |
| strong_raise_bonus | +1.10 | +1.10 | frozen |
| semi_bluff_bonus | +0.80 | +0.80 | frozen |
| weak_raise_penalty | −2.00 | −2.00 | frozen |
| slow_play_bonus | +1.40 | +1.40 | frozen |
| weak_call_penalty | −1.40 | −1.40 | frozen |

The Gauntlet drove `fold_bias` strongly negative and amplified `hand_strength` and `pair_bonus`, producing a TAG profile: play tight (fold marginal spots), extract value when strong.

### Phase 5 — Three-Tier Bluff Pipeline

A rule-based bluff layer fires *before* Minimax on each turn. Three gates are checked in order:

**1. Semi-bluff (~25% trigger rate).** On the flop or turn, if we have a flush or straight draw (4 to a suit or 4 consecutive ranks among hole cards + community), raise. Drawing hands have ~35% equity to improve by river — the raise is justified even when called. Only fires against non-aggressive opponents; bluffing a calling station leaks EV at every opportunity.

**2. Slow play / reverse bluff (~15%).** With a monster hand (strength > 0.85) on early streets against an aggressive opponent in a small pot, flat-call instead of raising. Goal: disguise strength and let the aggressive opponent keep betting into us.

**3. Pure river bluff (~20%).** On the river with a weak hand in a small pot against a passive or balanced opponent who checked (call_amount = 0), raise. With ~27% fold equity against these profiles, EV is positive in small pots.

**Calling-station filter.** The bluff pipeline is disabled entirely against an opponent classified as a calling station (call rate ≥ 60%, fold rate < 15%). This filter costs only a memory counter and a comparison, and pays for itself within the first 20 rounds — spending bluff chips on a player who never folds is pure EV leakage.

All triggers use a deterministic hash of game state + a salt integer rather than `random.random()`, keeping decisions fully reproducible across runs.

**Heuristic pruning (trash filter).** Before running Minimax, hands with strength < 0.40 and no draw potential are handled immediately: fold if facing a bet, check for free otherwise. This saves search overhead for decisions that actually require lookahead.

### Phase 6 — Preflop Table & Persistent Opponent Modeling

**Preflop equity table (`memoization_table.json`).** Rather than running Monte Carlo on every preflop action, we precomputed win rates for all canonical starting hands offline. The agent looks up its hand in O(1) instead of spending 50–100ms on MC simulations. A lookup table of premium hand equities (AA, KK, …, KQo) is also consulted first to keep top hands precise without simulation overhead.

**Post-flop Monte Carlo.** The fast closed-form heuristic is replaced post-flop by time-bounded Monte Carlo rollouts inside a 0.10s budget (batches of 10 simulations until time runs out). This gives a sharper estimate of win probability once community cards are visible. Empirically, MAE is ~0.92% at 0.10s vs ground truth at 0.45s — accurate enough for decision-making without eating into the 0.5s turn limit.

**Persistent opponent model.** Unlike `action_histories` in the round state (which resets between rounds), our opponent counters accumulate across the entire game. We track raise/call/fold frequencies from the first hand to the last, flushing into persistent counters at the end of each round before the histories reset. After ≥4 voluntary actions the opponent is classified as one of: `maniac`, `aggressive`, `balanced`, `passive`, or `calling_station`. The expanded profile set (adding `maniac` and `calling_station` beyond the baseline three) handles the edge cases where standard profile distributions break down.

---

## Internal Benchmarks & Selection Protocol

Our group built four independent agents. We selected GauntletAgent as the tournament submission using a two-criteria protocol:

- **Criterion 1 (leaderboard gain):** performance against the class-wide cohort — a robust proxy for expected performance against an unknown opponent.
- **Criterion 2 (fixed-deal head-to-head):** controlled pairwise matchups under a fixed deal sequence so card-luck variance is eliminated and any margin reflects pure decision quality.

**Fixed-deal head-to-head results (avg chips/scenario):**

| Winner | Loser | Edge | N scenarios |
|--------|-------|------|-------------|
| GauntletAgent | MCCFRAgent | +16.16 | 2,000 |
| HybridAgent | GauntletAgent | +1.89 | 6,000 (partial) |
| HybridAgent | MCCFRAgent | +2.32 | 4,000 |
| PureMCAgent | HybridAgent | +13.54 | 2,000 |
| MCCFRAgent | PureMCAgent | +3.21 | 2,000 |

The four agents form a complete **non-transitive cycle** — a rock-paper-scissors structure where no single agent dominates across all opponent styles:

```
GauntletAgent ≻ MCCFRAgent ≻ PureMCAgent ≻ HybridAgent ≻ GauntletAgent
```

Tournament performance therefore depends critically on the opponent distribution. Across all internal matchups:

| Agent | Avg edge | Min edge | Robust? |
|-------|----------|----------|---------|
| **GauntletAgent** | **+7.14** | **−1.89** | **✓** |
| PureMCAgent | +5.17 | −3.21 | ✓ |
| HybridAgent | −3.11 | −13.54 | ✗ |
| MCCFRAgent | −5.09 | −16.16 | ✗ |

GauntletAgent achieves the best average cross-cohort edge and the smallest worst-case loss. HybridAgent's +1.89 edge over GauntletAgent comes specifically from GauntletAgent's large negative pressure weight making it susceptible to marginal raises collecting fold equity — an exploit that is specific to HybridAgent's architecture and unlikely to generalise to an unknown opponent pool.

---

## Architecture Reference

```
declare_action()
    │
    ├─ Load hybrid_weights.json (once per game)
    ├─ Compute hand strength
    │   ├─ Preflop: O(1) table lookup
    │   └─ Post-flop: time-bounded Monte Carlo (0.10s budget)
    ├─ Classify opponent from persistent cross-round action history
    │
    ├─ Three-tier bluff pipeline
    │   ├─ (1) Semi-bluff: draw + flop/turn + non-aggressive opp
    │   ├─ (2) Slow play: monster + aggressive opp + small pot
    │   └─ (3) River bluff: weak hand + passive opp checked + small pot
    │
    ├─ Trash filter (hand < 0.40, no draw → fold or free check)
    │
    └─ Alpha-Beta Minimax (depth 3)
        ├─ Max nodes (our turn): standard alpha-beta pruning
        ├─ Opponent nodes: profile-weighted expected value
        └─ Leaf nodes: φ(s) = Σ wⱼ · φⱼ(s)
```

---

## Generalist vs. Specialist

The four agents reveal a fundamental tension between generalist stability and specialist dominance.

PureMCAgent and MCCFRAgent are generalists by architecture — their rollout/equilibrium approaches are not tuned to any specific opponent style. GauntletAgent sits in the middle by design: the 4-opponent Gauntlet explicitly balanced multiple sub-objectives simultaneously. HybridAgent is inadvertently a specialist: its Q-Learning weights were tuned against a narrow benchmark, making it effective against GauntletAgent's pressure response but brittle everywhere else.

The two-criteria selection protocol addresses this directly: Criterion 1 captures the unknown-cohort regime the submission will actually face, while Criterion 2 verifies internal consistency. GauntletAgent satisfies both.

**Limitations.** GauntletAgent's opponent model assumes a stationary profile — an adversary who deliberately shifts style mid-match could degrade the classifier. A change-point detector or Bayesian mixture over profiles would address this. The linear evaluation function also cannot capture interaction effects (e.g., strong hand *and* high pressure simultaneously) that a shallow neural network leaf evaluator would handle naturally.

---

## File Structure

```
AI-Poker-agent/
├── raise_player.py          # GauntletAgent — tournament submission
├── train.py                 # 3-phase RL training: Random → StrongRule → self-play
├── genetic_train.py         # Genetic Algorithm weight search (4-opponent Gauntlet)
├── testperf.py              # Head-to-head evaluation harness
├── example.py               # Quick start example
│
├── hybrid_weights.json      # Final trained weights (GA Gauntlet)
├── genetic_weights.json     # GA-only weights (for comparison)
├── memoization_table.json   # Precomputed preflop equity table
│
├── randomplayer.py          # Baseline: random legal action
├── always_raise.py          # Baseline: always raise
├── strong_rule_player.py    # Baseline: tight-aggressive rule-based (used in Gauntlet)
│
├── Archive/submission/      # Exact files submitted to the class tournament
│   ├── custom_player.py
│   ├── hybrid_weights.json
│   └── memoization_table.json
│
└── pypokerengine/           # Poker engine — no external dependencies needed
```

---

## How to Run

**GauntletAgent vs Random:**
```bash
python3 testperf.py -n1 "GauntletAgent" -a1 RaisedPlayer -n2 "Random" -a2 RandomPlayer
```

**Benchmark vs tight-aggressive rule player:**
```bash
python3 testperf.py -n1 "GauntletAgent" -a1 RaisedPlayer -n2 "StrongRule" -a2 StrongRulePlayer
```

**Self-play sanity check (should be ~50/50):**
```bash
python3 testperf.py -n1 "Agent1" -a1 RaisedPlayer -n2 "Agent2" -a2 RaisedPlayer
```

**Retrain weights with RL (outputs hybrid_weights.json):**
```bash
python3 train.py
```

**Retrain weights with Genetic Algorithm on 4-opponent Gauntlet (outputs genetic_weights.json):**
```bash
python3 genetic_train.py
```

No external dependencies beyond Python 3.8+ and the included `pypokerengine` module.

---

## Key Design Decisions

**Why GA over RL?**  
RL with policy gradient follows a single gradient direction. When the loss landscape has a local optimum that isn't the global one — as happened with the pot-chasing trap — gradient descent stays stuck. GA maintains a population of 8 solutions and recombines the best features of multiple directions simultaneously via crossover. It's slower per iteration but far less likely to get stuck. The empirical result — convergence to a qualitatively different TAG weight profile that wins more chips — validates this concretely.

**Why a 4-opponent Gauntlet instead of a single benchmark?**  
Training against one fixed opponent produces a specialist: effective against that one style, brittle everywhere else. The Gauntlet evaluates fitness across four opponent types simultaneously, producing a generalist policy. This matches the class tournament regime, where the opponent distribution is unknown.

**Why deterministic bluff triggers?**  
We used a hash of game state + a salt integer instead of `random.random()`. This makes decisions fully reproducible across runs (same hand state → same bluff decision), which is useful for debugging and controlled evaluation. The tradeoff is that a sufficiently adaptive opponent could theoretically reverse-engineer the trigger function — true mixed strategies require genuine randomness.

---

*COMPSCI 683 — Group 9 — Spring 2026, UMass Amherst*
