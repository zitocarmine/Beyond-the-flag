Gemini ha detto
This repository, Beyond-the-flag, features a comparative analysis and implementation of Reinforcement Learning (RL) agents trained to play Super Mario Bros. using different algorithmic approaches. The project explores the transition from "mechanical training" based on dense rewards to "curious intelligence" driven by intrinsic motivation.
+1

🚀 Overview
Traditional RL agents often struggle with sparse rewards, where feedback from the environment is infrequent. This project implements and compares three distinct methodologies to address this challenge:
+3


A2C (The Robot): A standard Advantage Actor-Critic agent using aggressive Reward Shaping (dense rewards for every step, killing enemies, etc.).
+1


A2C + ICM (The Confused): An A2C agent augmented with an Intrinsic Curiosity Module (ICM), which struggles with long-term stability and catastrophic forgetting.


PPO + ICM (The Child): A Proximal Policy Optimization agent combined with ICM and Curriculum Learning for robust, stable, and curious exploration.

🧠 The Architecture: Intrinsic Curiosity Module (ICM)
The core of the "curious" agents is the ICM, which rewards the agent for "surprise"—discovering states it cannot easily predict. The architecture consists of three neural subnets:
+2


Feature Extractor (ϕ): Compresses raw 84×84 pixel inputs into latent vectors using a CNN.
+1


Inverse Model: Learns to predict the action taken between two states, filtering out environmental changes the agent cannot control.
+1


Forward Model: Predicts the future state based on the current state and action. The prediction error between the estimated and actual future state becomes the intrinsic reward.
+2

📉 Methodology & Curriculum Learning
To achieve mastery, the PPO + ICM agent follows a 3-phase Curriculum Learning pipeline, gradually increasing task complexity:
+1


Phase 1: Survival & Curiosity (0–1M steps): Mario learns basic interactions, such as jumping and killing enemies, driven primarily by curiosity without external rewards for moving right.
+1


Phase 2: Direction & Expansion (1M–2.8M steps): Exploration is pushed forward with incentives for progression and horizontal movement.
+1


Phase 3: Mastery & Speed (2.8M–4M steps): The agent optimizes the path using time penalties and rewards for collecting coins and power-ups.
+2

📊 Key Results

A2C Baseline: Achieves the fastest convergence (~2.5M steps) but acts as a "Guided Robot". It optimizes for reward-maximization without truly understanding environment mechanics.
+2


A2C + ICM: Despite high intrinsic rewards, it suffers from catastrophic forgetting. Destructive weight updates cause it to lose the policies required to reach newly discovered areas.
+2


PPO + ICM: While slower initially, it is far more robust. It is capable of complex interactions, such as discovering and consistently using underground pipes, which pure reward shaping cannot replicate.
