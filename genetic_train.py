"""
genetic_train.py  —  Genetic Algorithm Weight Search for RaisedPlayer
COMPSCI 683 Artificial Intelligence — Spring 2026, UMass Amherst

================================================================
GENETIC ALGORITHM OVERVIEW
================================================================

How it works:
  Instead of nudging one weight vector via gradient-based RL,
  we maintain a POPULATION of 8 agents each with different weights
  and evolve them over 20 generations using natural selection.

Each generation:
  1. FITNESS   — each agent plays 5 games vs StrongRulePlayer (1v1)
                 fitness = average chips won per game
  2. SELECTION — rank agents by fitness, keep top 3 (survivors)
                 discard bottom 5
  3. CROSSOVER — breed 5 children from the top 3:
                 each weight randomly inherited from one parent
  4. MUTATION  — randomly nudge some weights in each child
                 prevents population from converging too early

Why 1v1 vs StrongRulePlayer for fitness?
  The final tournament is 1v1. Using a consistent benchmark opponent
  gives a reliable fitness signal that directly reflects tournament
  performance. Round-robin between siblings would only measure
  "who is slightly better than their near-identical siblings"
  — not how good the weight set actually is.

Why genetic algorithm vs RL?
  RL follows a single gradient direction — it can get stuck in local
  optima. Genetic algorithm explores 8 different directions simultaneously
  and recombines the best parts of each. Less likely to get stuck.

Output:
  genetic_weights.json  (compare against hybrid_weights.json from RL)

How to run:
  python3 genetic_train.py

How to use the genetic weights:
  cp genetic_weights.json hybrid_weights.json
  python3 testperf.py -n1 "RaisedPlayer" -a1 RaisedPlayer -n2 "StrongRule" -a2 StrongRulePlayer
================================================================
"""

import sys
import os
import json
import copy
import random
import time
from mcts_player import MCTSPlayer
from randomplayer import RandomPlayer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from raise_player import RaisedPlayer


# ============================================================
# BENCHMARK OPPONENT  (fitness evaluator)
# ============================================================

class StrongRulePlayer(BasePokerPlayer):
    """
    Tight-aggressive benchmark opponent.
    Used to evaluate fitness of each agent in the population.
    Folds ~40% of hands preflop, punishes weak raises postflop.
    Consistent and deterministic — gives a reliable fitness signal.
    """

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

POPULATION_SIZE  = 8     # Number of agents per generation
NUM_SURVIVORS    = 3     # Top agents kept each generation
NUM_GENERATIONS  = 10    # How many generations to run
GAMES_PER_EVAL   = 5     # Games per fitness evaluation (1v1 vs StrongRule)
MAX_ROUND        = 100  # Rounds per game
INITIAL_STACK    = 10000
SMALL_BLIND      = 10

MUTATION_RATE    = 0.3   # Probability of mutating each weight
MUTATION_STD     = 0.4   # Standard deviation of mutation noise

SAVE_PATH        = "genetic_weights.json"

# Only evolve the 8 base weights — leave conditional features at hand-tuned values
# Conditional features (strong_raise_bonus etc.) are strategically meaningful
# and were correctly hand-tuned — genetic search should not corrupt them
EVOLVABLE_KEYS = [
    "hand_strength",
    "pot_bucket",
    "pressure_bucket",
    "opp_aggression",
    "street_progress",
    "pair_bonus",
    "suited_bonus",
    "connected_bonus",
    "fold_bias",        # NEW: Let the GA dial in the optimal folding courage
    "slow_play_bonus",  # NEW: Let the GA optimize our trapping frequency
]


# ============================================================
# POPULATION INITIALIZATION
# ============================================================

def initialize_population(base_weights):
    """
    Create POPULATION_SIZE agents starting from base_weights
    with random variations.

    Agent 0: exact base weights (unmodified baseline)
    Agents 1-7: base weights + random perturbations

    Why start from base weights?
      The RL-trained weights are already better than random.
      We explore the neighbourhood of the best known solution
      rather than starting from scratch.

    Perturbation scale:
      Each evolvable weight perturbed by Gaussian(0, 0.5).
      This gives enough diversity to explore meaningfully
      without straying too far from the working region.
    """
    population = []

    for i in range(POPULATION_SIZE):
        individual = copy.deepcopy(base_weights)

        if i > 0:  # Agent 0 is the unmodified baseline
            for key in EVOLVABLE_KEYS:
                noise = random.gauss(0, 0.5)
                individual[key] = max(-10.0, min(10.0, individual[key] + noise))

        population.append(individual)

    return population


# ============================================================
# FITNESS EVALUATION
# ============================================================

# ============================================================
# FITNESS EVALUATION (The 4-Opponent Gauntlet)
# ============================================================

def evaluate_fitness(weights, current_population, agent_name="GeneticAgent"):
    """
    Evaluates fitness by playing a 4-opponent gauntlet:
    1. MCTSPlayer (Pure Math)
    2. RandomPlayer (Pure Chaos)
    3. StrongRulePlayer (Tight-Aggressive Baseline)
    4. Sibling (Self-Play / Arms Race)
    """
    agent = RaisedPlayer()
    agent._hybrid_weights = copy.deepcopy(weights)
    
    total_chips = 0
    games_played = 0

    # Opponent 1: MartinMCTS (The Math Test)
    config = setup_config(max_round=MAX_ROUND, initial_stack=INITIAL_STACK, small_blind_amount=SMALL_BLIND)
    config.register_player(name=agent_name, algorithm=agent)
    config.register_player(name="MCTS", algorithm=MCTSPlayer())
    try:
        result = start_poker(config, verbose=0)
        for p in result.get("players", []):
            if p.get("name") == agent_name:
                total_chips += p.get("stack", INITIAL_STACK) - INITIAL_STACK
        games_played += 1
    except Exception: pass

    # Opponent 2: RandomPlayer (The Chaos Test)
    config = setup_config(max_round=MAX_ROUND, initial_stack=INITIAL_STACK, small_blind_amount=SMALL_BLIND)
    config.register_player(name=agent_name, algorithm=agent)
    config.register_player(name="Random", algorithm=RandomPlayer())
    try:
        result = start_poker(config, verbose=0)
        for p in result.get("players", []):
            if p.get("name") == agent_name:
                total_chips += p.get("stack", INITIAL_STACK) - INITIAL_STACK
        games_played += 1
    except Exception: pass

    # Opponent 3: StrongRulePlayer (The Structured Baseline Test)
    config = setup_config(max_round=MAX_ROUND, initial_stack=INITIAL_STACK, small_blind_amount=SMALL_BLIND)
    config.register_player(name=agent_name, algorithm=agent)
    config.register_player(name="StrongRule", algorithm=StrongRulePlayer())
    try:
        result = start_poker(config, verbose=0)
        for p in result.get("players", []):
            if p.get("name") == agent_name:
                total_chips += p.get("stack", INITIAL_STACK) - INITIAL_STACK
        games_played += 1
    except Exception: pass

    # Opponent 4: Self-Play vs Sibling (The Arms Race)
    sibling_weights = random.choice(current_population)
    sibling_agent = RaisedPlayer()
    sibling_agent._hybrid_weights = copy.deepcopy(sibling_weights)

    config = setup_config(max_round=MAX_ROUND, initial_stack=INITIAL_STACK, small_blind_amount=SMALL_BLIND)
    config.register_player(name=agent_name, algorithm=agent)
    config.register_player(name="Sibling", algorithm=sibling_agent)
    try:
        result = start_poker(config, verbose=0)
        for p in result.get("players", []):
            if p.get("name") == agent_name:
                total_chips += p.get("stack", INITIAL_STACK) - INITIAL_STACK
        games_played += 1
    except Exception: pass

    return total_chips / max(1, games_played)

# ============================================================
# SELECTION
# ============================================================

def select_survivors(population, fitnesses):
    """
    Rank agents by fitness and keep top NUM_SURVIVORS.

    Returns: list of (weights, fitness) for top agents,
             sorted best first.
    """
    ranked = sorted(
        zip(population, fitnesses),
        key=lambda x: x[1],
        reverse=True  # highest fitness first
    )
    return ranked[:NUM_SURVIVORS]


# ============================================================
# CROSSOVER
# ============================================================

def crossover(parent1, parent2):
    """
    Produce a child by randomly inheriting each evolvable weight
    from either parent1 or parent2.

    Non-evolvable weights (conditional features) are always
    inherited from parent1 unchanged — they stay at hand-tuned values.

    Why uniform crossover?
      Each weight independently flips a coin (50/50 from each parent).
      This allows the best combination of weights from different
      parents rather than taking a contiguous segment from each.
      Good for weight vectors where each dimension is independent.
    """
    child = copy.deepcopy(parent1)

    for key in EVOLVABLE_KEYS:
        if random.random() < 0.5:
            child[key] = parent2[key]

    return child


# ============================================================
# MUTATION
# ============================================================

def mutate(weights):
    """
    Randomly nudge some weights with Gaussian noise.

    Each evolvable weight has MUTATION_RATE probability of being
    perturbed by Gaussian(0, MUTATION_STD).

    Why mutate?
      Without mutation, all children would be combinations of
      the same 3 survivors. The population would converge to
      a single point after a few generations, stopping exploration.
      Mutation keeps diversity alive.

    Why Gaussian noise?
      Small changes are more likely than large ones.
      The weight is unlikely to be flipped to a completely
      different value — it nudges gradually.
    """
    mutated = copy.deepcopy(weights)

    for key in EVOLVABLE_KEYS:
        if random.random() < MUTATION_RATE:
            noise = random.gauss(0, MUTATION_STD)
            mutated[key] = max(-10.0, min(10.0, mutated[key] + noise))

    return mutated


# ============================================================
# BREED NEW GENERATION
# ============================================================

def breed_new_generation(survivors, population_size):
    """
    Create population_size - NUM_SURVIVORS children from survivors.

    Strategy:
      - Keep all survivors unchanged (elitism)
      - Fill rest with children from crossover + mutation

    Elitism (keeping survivors unchanged) guarantees that the
    best solution found so far is never lost.
    """
    new_population = [w for w, f in survivors]  # keep survivors

    num_children = population_size - len(survivors)

    for _ in range(num_children):
        # Pick two random parents from survivors
        p1_weights, _ = random.choice(survivors)
        p2_weights, _ = random.choice(survivors)

        # Crossover then mutate
        child = crossover(p1_weights, p2_weights)
        child = mutate(child)
        new_population.append(child)

    return new_population


# ============================================================
# MAIN GENETIC LOOP
# ============================================================

def genetic_train():
    print("=" * 62)
    print("Genetic Algorithm Weight Search")
    print(f"  Population size  : {POPULATION_SIZE}")
    print(f"  Survivors / gen  : {NUM_SURVIVORS}")
    print(f"  Generations      : {NUM_GENERATIONS}")
    print(f"  Games / eval     : {GAMES_PER_EVAL}")
    print(f"  Mutation rate    : {MUTATION_RATE}")
    print(f"  Mutation std     : {MUTATION_STD}")
    print(f"  Fitness opponent : StrongRulePlayer (1v1)")
    print(f"  Evolvable keys   : {len(EVOLVABLE_KEYS)} base weights")
    print(f"  Output           : {SAVE_PATH}")
    print("=" * 62)
    print("  NOTE: Conditional features (bluff bonuses etc.) are")
    print("  kept at hand-tuned defaults — only base weights evolve.")
    print("=" * 62)

    # Load base weights — start from RL-trained weights if available,
    # otherwise fall back to hand-tuned defaults
    base_agent    = RaisedPlayer()
    base_weights  = base_agent._load_hybrid_weights("hybrid_weights.json")

    print(f"\n  Starting from: {'hybrid_weights.json' if os.path.exists('hybrid_weights.json') else 'hand-tuned defaults'}")

    # Initialize population
    population   = initialize_population(base_weights)
    best_weights = copy.deepcopy(base_weights)
    best_fitness = -float("inf")

    start_time = time.time()

    for generation in range(NUM_GENERATIONS):
        gen_start = time.time()

        # Evaluate fitness of every agent
        fitnesses = []
        for i, weights in enumerate(population):
            # FIXED: Passing the 'population' array in
            fitness = evaluate_fitness(weights, population, agent_name=f"Agent_{i}") 
            fitnesses.append(fitness)

        # Track overall best
        gen_best_idx     = fitnesses.index(max(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_avg_fitness  = sum(fitnesses) / len(fitnesses)

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_weights = copy.deepcopy(population[gen_best_idx])
            with open(SAVE_PATH, "w") as f:
                json.dump(best_weights, f, indent=2)

        gen_time = time.time() - gen_start
        elapsed  = time.time() - start_time

        print(
            f"  Gen {generation+1:3d}/{NUM_GENERATIONS} | "
            f"Best: {gen_best_fitness:+7.0f} chips | "
            f"Avg: {gen_avg_fitness:+7.0f} | "
            f"All-time best: {best_fitness:+7.0f} | "
            f"{gen_time:.0f}s/gen | "
            f"{elapsed:.0f}s total"
        )

        # Selection, crossover, mutation
        survivors   = select_survivors(population, fitnesses)
        population  = breed_new_generation(survivors, POPULATION_SIZE)

    # Final report
    total_time = time.time() - start_time
    print("=" * 62)
    print(f"Genetic training complete in {total_time:.1f}s")
    print(f"Best fitness achieved : {best_fitness:+.0f} chips vs StrongRule")
    print(f"Weights saved to      : {SAVE_PATH}")
    print("=" * 62)
    print("\nFinal evolved weights vs starting weights:")

    for k in EVOLVABLE_KEYS:
        orig   = base_weights.get(k, 0.0)
        evolved = best_weights.get(k, 0.0)
        change = evolved - orig
        arrow  = "▲" if change > 0.01 else ("▼" if change < -0.01 else "─")
        print(f"  {k:25s}: {evolved:+7.4f}  (start {orig:+.4f},  {arrow} {change:+.4f})")

    print("\nConditional features (unchanged — hand-tuned):")
    for k in best_weights:
        if k not in EVOLVABLE_KEYS:
            print(f"  {k:25s}: {best_weights[k]:+7.4f}")

    print("\nTo test genetic weights:")
    print("  cp genetic_weights.json hybrid_weights.json")
    print("  python3 testperf.py -n1 RaisedPlayer -a1 RaisedPlayer -n2 StrongRule -a2 StrongRulePlayer")

    return best_weights


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    genetic_train()