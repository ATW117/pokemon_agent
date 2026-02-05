## Project Overview
This repository contains a reinforcement learning agent trained to play Pokemon battles using Deep Q-Learning. The agent was trained on Boston University's Shared Computing Cluster (SCC) using a Double DQN architecture with epsilon-greedy exploration. The implementation includes custom reward shaping, extensive feature engineering, and strategic Pokemon switching logic.

## Dependencies

This project was built for a BU course and requires the following dependencies (not included):
- `pokePA-1.1.0.jar` - Custom Pokemon game framework (provided by course)
- `argparse4j-0.9.0.jar` - Command line argument parsing
- `junit-4.12.jar` - Unit testing framework
- `hamcrest-2.2.jar` - JUnit dependency


**Note**: This repository contains only my implementation of the Reinforcement Learning agent, sensory array, and reward function. it also includes the trained model params, and the script I used to train my model. To run this code, you would need the course framework.

## Architecture and Implementation
- Model Type: Double DQN (Deep Q-Network with target network)
- Input Features: 291-dimensional feature vector
- Network structure: 291-feature input layer, 2 hidden layers with 128 neurons and ReLU activations each, and 1-neuron output layer (Q-value for state-action pair)

## Feature Engineering (CustomSensorArray.java)
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


## Reward Function Design (CustomRewardFunction.java)
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

## Pokemon Selection Strategy (PolicyAgent.java)
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

## Key Design Decisions
- Extensive Feature Engineering: 291 features capture nearly all relevant battle state information, including team composition, stat stages, status conditions, and move properties.
- Shaped Rewards: Rather than sparse +1/-1 win/loss rewards, the reward function provides dense feedback on HP trades, KOs, and status conditions to accelerate learning.
- Double DQN: Prevents Q-value overestimation that plagues vanilla DQN, leading to more stable training.
- Heuristic Pokemon Selection: Instead of learning switching strategy from scratch (which is difficult with limited episodes), uses domain knowledge to make reasonable switch decisions.
- Type Effectiveness Integration: Offensive/defensive estimations explicitly compute type matchup multipliers, incorporating core Pokemon battle mechanics.
- Status Prioritization: Higher rewards for SLEEP/FREEZE (most debilitating) and TOXIC (escalating damage) reflect competitive Pokemon strategy.

Usage
This code is designed to work with the BU course framework and is provided for reference purposes.



