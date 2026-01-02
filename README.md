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

