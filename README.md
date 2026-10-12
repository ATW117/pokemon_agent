# Pokémon Battle Agent: Double DQN with self-play

A reinforcement-learning agent that plays Gen-1 Pokémon battles. It started as a Boston University AI course assignment (Dec 2025, v1 below) and was later reworked (Oct 2026, v2). The rework fixed the training pipeline, several bugs in the course's battle framework, and the feature set.

## Results

All numbers are win rates on 3,000 fresh games per opponent with random teams. Team sizes are 1-6 Pokémon per side unless noted.

| Agent | vs RandomAgent | vs AggroAgent | vs Drac5290Agent (teams of 1-3) |
|---|---|---|---|
| v1 (course, `params.model`) | 49.6% | 44.2% | 40.0% |
| **v2 (`params_v4.model`)** | **78.9%** | **73.3%** | **62.3%** |

- In full 6v6 battles, v2 wins **89.7%** vs Random and **82.7%** vs Aggro.
- 1v1 battles are mostly decided by which two Pokémon get drawn: 64% vs Random.
- For reference, AggroAgent itself beats RandomAgent only 56% of the time.

## What changed in v2

**Training pipeline (`src/pas/pokemon/train/`)**
- `DQNTrainer` replaces the course's `ParallelTrain`:
  - Double DQN with 3-step returns, a Huber loss and a hard-synced target network.
  - Self-play against a pool of frozen past snapshots, mixed with Random, Aggro and the transition-model agent Drac5290.
  - Random team sizes from 1 to 6, and the learner plays both sides.
- The course trainer stored each action next to the *following* turn's transition, an off-by-one error. `DQNTrainer` records `(state the action was chosen in, action, resulting state)`.
- `FastMLP` is a plain-array network (458 → 512 → 256 → 1) with Adam. It is weight-compatible with the framework's `Sequential`/`Dense`, so saved `.model` files load in the original agent code.
- Evaluation plays every team pairing twice with sides swapped (paired evaluation), which cancels most of the team-draw luck.

**Agent (`PolicyAgent`, `CustomSensorArray`, `CustomRewardFunction`)**
- Training and evaluation now choose from the same actions. v1 allowed only attacks at evaluation time.
- Switch actions now describe the Pokémon being switched in. In v1 every switch looked like the same 1-power Normal move.
- New features:
  - Move identity (172 moves). v1 could not tell Swords Dance from Leer.
  - Type effectiveness.
  - An expected-damage estimate that mirrors the engine's damage formula, including STAB, crits, screens, accuracy/evasion stages, multi-hit and fixed-damage moves, and Fly/Dig.
  - Everything is encoded from the agent's own side, which makes self-play possible.
- Rewards are rescaled to roughly ±1 per step and computed per Pokémon slot. In v1, switching out a statused Pokémon counted as "curing" it.
- He weight initialization. The framework's default draws weights with standard deviation 1, which saturates a 300-input network.
- A guard against endless switch loops, applied the same way in acting and in training targets.

**Course framework bugs found and fixed (local patch jar, not published)**
- `CoreRegistry.readCSV` never closes its resource stream, and the framework rebuilds the registry for every battle and every Metronome use. Training leaked roughly 700 MB of native memory per cycle (45 GB RSS observed). `Battles.java` works around this in the trainer.
- In the transition model (`getPotentialEffects`), multi-hit moves expanded exponentially and Metronome → Mirror Move → Metronome recursed without bound. Drac5290Agent runs out of memory or never finishes because of this. The patch merges identical outcomes and adds a recursion guard.
- These fixes live in `patches/pokePA-patches.jar`, which is placed before `lib/*` on the classpath. Both are built from decompiled course code, so they are not included in this repo.

## Experiment: decision-time lookahead (`LookaheadAgent`)

`LookaheadAgent` adds a one-turn search on top of the trained network:
- It copies the battle and simulates every (my move, opponent reply) pair 4-8 times with the engine's own randomness.
- It scores each result as the immediate reward plus the network's value of the position that follows.
- It models the opponent as a mix of "plays like our network would" and uniform random.

It did **not** beat the plain greedy policy (1,000 games, same seeds):

| Agent | vs Random | vs Aggro | vs Drac (teams of 1-3) |
|---|---|---|---|
| Greedy (`PolicyAgent`) | 80.1% | 73.4% | 63.8% |
| Lookahead, mixed opponent | 77.7% | 69.8% | 65.8% |

The Q-network was already trained against this opponent mix, so it has learned the expected reply. A one-turn search with an inexact opponent model mostly adds noise. Assuming the opponent mirrors our network drops the Aggro result to 61.6%, and assuming it plays at random brings it back to 72.3%.

While building it, I found one more framework bug. `Team(TeamView)` never sets each Pokémon's team reference, so a `Battle` rebuilt from a view crashes when it applies a move, and its damage calculations ignore Reflect and Light Screen. This also affects Drac5290Agent's planning.

## Running

You need the course framework jars in `lib/` (not included). Run everything from the repo root.

```bash
./train_dqn.sh myrun                 # trains; logs and checkpoints go to runs/myrun/
./watchdog.sh myrun 18 &             # optional: stop the run if RSS exceeds 18 GB

# evaluate a model: <model> <games> <threads> <seed> <minTeam> <maxTeam> <opponents...>
java -cp "build:lib/*" src.pas.pokemon.train.Evaluate params_v4.model 3000 16 7 1 6 \
  edu.bu.pas.pokemon.agents.RandomAgent edu.bu.pas.pokemon.agents.AggroAgent
```

- `params_v4.model` holds the current weights.
- `params.model` holds the original v1 weights, kept for reference. They need the v1 code (git tag `v1-course`).
- The v2 run: 8 hours on 16 CPU threads of an NVIDIA DGX Spark, about 1M games.

---

## v1: original course version (Dec 2025)

### Project Overview
This repository contains a reinforcement learning agent trained to play Pokemon battles using Deep Q-Learning. The agent was trained on Boston University's Shared Computing Cluster (SCC) using a Double DQN architecture with epsilon-greedy exploration. The implementation includes custom reward shaping, extensive feature engineering, and strategic Pokemon switching logic.

### Dependencies

This project was built for a BU course and requires the following dependencies (not included):
- `pokePA-1.1.0.jar` - Custom Pokemon game framework (provided by course)
- `argparse4j-0.9.0.jar` - Command line argument parsing
- `junit-4.12.jar` - Unit testing framework
- `hamcrest-2.2.jar` - JUnit dependency


**Note**: This repository contains only my implementation of the Reinforcement Learning agent, sensory array, and reward function. it also includes the trained model params, and the script I used to train my model. To run this code, you would need the course framework.

### Architecture and Implementation
- Model Type: Double DQN (Deep Q-Network with target network)
- Input Features: 291-dimensional feature vector
- Network structure: 291-feature input layer, 2 hidden layers with 128 neurons and ReLU activations each, and 1-neuron output layer (Q-value for state-action pair)

### Feature Engineering (CustomSensorArray.java)
The sensor array generates a 291-dimensional feature vector encoding:
Active Pokemon Features (Both Teams):
- HP ratios (current/max)
- Last damage dealt (normalized)
- Non-volatile status conditions (one-hot: NONE, POISON, TOXIC, PARALYSIS, BURN, SLEEP, FREEZE)
- Status turn counters for SLEEP and FREEZE
- Stat stage multipliers (ATK, DEF, SPD, SPATK, SPDEF, ACC, EVASION) - normalized to [-1, 1]
- Speed comparison metrics (absolute speeds, relative difference, faster indicator)
- Pokemon types (one-hot encoding for both type slots)
- Volatile battle flags (CONFUSED, TRAPPED, SEEDED, FLINCHED, FOCUS_ENERGY, SUBSTITUTE)
- Height classification (one-hot)

Team-Wide Features:
- Team size and composition (6 slots per team)
- Per-Pokemon bench information:
1- Active status indicator
2- Fainted status
3- HP ratio
4- Non-volatile status (one-hot)
- Type distribution across team (normalized counts per type)
- Status condition histogram across team

Screen/Hazard Features:
- Reflect turns remaining (normalized)
- Light Screen turns remaining (normalized)

Move Features:
- Action category (PHYSICAL, SPECIAL, STATUS)
- Move type (one-hot)
- Power (normalized to 0-1, scaled by 150)
- Accuracy (normalized to 0-1)
- PP ratio
- Priority tier encoding (negative, neutral, positive)
- Critical hit ratio
- Height-targeting capability
- STAB (Same Type Attack Bonus) indicator
- Opponent's last move priority


### Reward Function Design (CustomRewardFunction.java)
The reward function uses a STATE_ACTION_STATE paradigm with carefully tuned components:
HP-Based Rewards:

Weight: 15.0× the change in HP differential between teams
Computed as: 15.0 × ((myHpAfter - oppHpAfter) - (myHpBefore - oppHpBefore))
Captures both damage dealt and damage taken

KO Rewards:
- +20.0 per opponent Pokemon fainted
- -20.0 per own Pokemon fainted
Encourages finishing off low-HP opponents

Status Condition Rewards:
- Curing own status: +3.0
- Inflicting SLEEP/FREEZE on opponent: +6.0
- Inflicting TOXIC on opponent: +5.0
- Inflicting other status on opponent: +2.5
- Receiving status has symmetric negative rewards
- Prioritizes debilitating status effects

Terminal State Rewards:
- Win: +40.0
- Loss: -40.0
- Draw: 0.0

Behavioral Penalties:
Switching penalty: -2.0
Encourages commitment and discourages excessive switching

Bounds: All rewards are clamped to [-50.0, 50.0]

### Pokemon Selection Strategy (PolicyAgent.java)
When forced to switch (e.g., after a faint), the agent uses a heuristic-based selection:
Scoring Components:

HP ratio: 1.8× weight
Speed (normalized): 0.4× weight
Offensive potential vs opponent: 0.9× weight
Defensive risk from opponent: -0.7× weight
Speed advantage bonus: +0.2
Status condition penalties (SLEEP/FREEZE: -0.9, PARALYSIS: -0.6, BURN: -0.5, POISON/TOXIC: -0.4)

Offensive Potential Estimation:

Evaluates type effectiveness multipliers
Considers STAB bonus
Factors in move power and accuracy
Accounts for stat stages

Defensive Risk Estimation:

Estimates incoming damage based on type weaknesses/resistances
Considers opponent's move power and accuracy
Adjusts for current stat stages

Double DQN Implementation
The agent uses Double DQN to reduce Q-value overestimation:

Main Network: Used for action selection and updated during training
Target Network: Used for Q-value estimation in TD-targets, updated every 10,000 steps
Bellman Update: Q(s,a) = reward + γ × max_a' Q_target(s', a')

Batch Processing Optimizations:

Batched forward passes for move evaluation (reduces computation)
Efficient replay buffer sampling
Fallback to individual processing on batch failures

Training Monitoring

Win rate logged every 100 games
Epsilon value logged every 100,000 training steps
Consecutive switch tracking to prevent degenerate policies

### Key Design Decisions
- Extensive Feature Engineering: 291 features capture nearly all relevant battle state information, including team composition, stat stages, status conditions, and move properties.
- Shaped Rewards: Rather than sparse +1/-1 win/loss rewards, the reward function provides dense feedback on HP trades, KOs, and status conditions to accelerate learning.
- Double DQN: Prevents Q-value overestimation that plagues vanilla DQN, leading to more stable training.
- Heuristic Pokemon Selection: Instead of learning switching strategy from scratch (which is difficult with limited episodes), uses domain knowledge to make reasonable switch decisions.
- Type Effectiveness Integration: Offensive/defensive estimations explicitly compute type matchup multipliers, incorporating core Pokemon battle mechanics.
- Status Prioritization: Higher rewards for SLEEP/FREEZE (most debilitating) and TOXIC (escalating damage) reflect competitive Pokemon strategy.

### Usage
This code is designed to work with the BU course framework and is provided for reference purposes.

Training the Agent
The agent was trained using the included train.sh script on BU's Shared Computing Cluster:
bashchmod +x train.sh
./train.sh
Training Configuration (from train.sh):

- Parallel Training: 28 worker threads (-j 28)
- Training Episodes: 5000 parallel games (-p 5000)
- Opponents: RandomAgent and AggroAgent (random selection via -r RANDOM)
- Replay Buffer Size: 50,000 transitions (-b 50000)
- Training Frequency: Update every 50 steps (-u 50)
- Batch Size: 256 samples (-m 256)
- Discount Factor: 0.99 (-g 0.99)
- Optimizer: Adam (-d adam)
- Learning Rate: 1e-4 (-n 1.0E-4)
- Validation Frequency: Every 5 training cycles (-v 5)

The script automatically logs training progress to logs/training_32core_<timestamp>.log and saves model parameters to params/model_params_32core.
SCC Setup

Training was performed on BU's Shared Computing Cluster with a remote desktop session:
- Requested compute resources (28+ CPU cores, sufficient memory)
- Set up remote desktop environment
- Compiled and ran the training script
- Monitored training logs for convergence and win rate improvements

### Future Improvements
Potential enhancements could include:
- Prioritized experience replay for more efficient learning
- Learned switching policy (instead of heuristic)
- Dueling DQN architecture for better value estimation
- Multi-step returns for improved credit assignment
- Ensemble models for robustness

### Acknowledgments
This project was completed as part of a BU AI course. The Pokemon battle framework was provided by the course instructors.



