# Reinforcement Learning (CSCI 1410 – Assignment 5)

This project implements several variants of the on-policy reinforcement learning (RL) algorithm **SARSA** (State-Action-Reward-State-Action) to solve tasks using OpenAI's Gym environment.

## Algorithms Implemented

- **SARSA**: Updates the Q-function based on the current state, action, reward, next state, and next action.
- **SARSA(λ)**: Incorporates eligibility traces to provide a more efficient learning process.

- **Tabular SARSA(λ)**: Temporal-difference method with eligibility traces.
- **Epsilon-greedy policy**: Balances exploration and exploitation.
- **Policy visualization**: Displays the learned taxi behavior in the Gym environment.

## How to Run

The main entry point is `sarsa.py`. You can train a SARSA(λ) agent on the Taxi-v3 environment, plot learning curves, render the environment, or test a saved policy.

```bash
# Train the agent, save Q-values and policy, plot rewards, and render Taxi-v3
python sarsa.py
```

## Project Structure

```text
.
│── tabular_sarsa.py              # Core tabular SARSA(λ) class
│── sarsa.py                      # Main SARSA learner and runner script
│── qvalues_taxi_sarsa_lambda.npy # Saved Q-table after training (generated)
│── policy_taxi_sarsa_lambda.npy  # Saved policy after training (generated)
│── rewards_plot_lambda.png       # Reward curve visualization (generated)
