# 🤖 Beyond the Flag: From Mechanical Training to Curious Intelligence

This repository features a comparative analysis and implementation of Reinforcement Learning (RL) agents trained to play **Super Mario Bros.** using different algorithmic approaches. The project explores the transition from "mechanical training" based on dense rewards to "curious intelligence" driven by intrinsic motivation.

---

## 🚀 Overview
Traditional RL agents often struggle with **sparse rewards**, where feedback from the environment is infrequent. This project implements and compares tre distinct methodologies to address this challenge:

1.  **A2C (The Robot):** A standard Advantage Actor-Critic agent using aggressive **Reward Shaping** (dense rewards for every step, killing enemies, etc.).
2.  **A2C + ICM (The Confused):** An A2C agent augmented with an **Intrinsic Curiosity Module (ICM)**, which struggles with long-term stability and catastrophic forgetting.
3.  **PPO + ICM (The Child):** A Proximal Policy Optimization agent combined with ICM and **Curriculum Learning** for robust, stable, and curious exploration.

---

## 🧠 The Architecture: Intrinsic Curiosity Module (ICM)
The core of the "curious" agents is the **ICM**, which rewards the agent for "surprise"—discovering states it cannot easily predict. The architecture consists of three neural subnets:

* **Feature Extractor ($\phi$):** Compresses raw $84 \times 84$ pixel inputs into latent vectors using a CNN.
* **Inverse Model:** Learns to predict the action taken between two states, filtering out environmental changes the agent cannot control (the "leaves-moving-in-the-wind" problem).
* **Forward Model:** Predicts the future state based on the current state and action. The **prediction error** between the estimated and actual future state becomes the **intrinsic reward**.

---

## 📉 Methodology & Curriculum Learning
To achieve mastery, the **PPO + ICM** agent follows a **3-phase Curriculum Learning pipeline**, gradually increasing task complexity:

* **Phase 1: Survival & Curiosity (0–1M steps):** Mario learns basic interactions, such as jumping and killing enemies, driven primarily by curiosity without external rewards for moving right.
* **Phase 2: Direction & Expansion (1M–2.8M steps):** Exploration is pushed forward with incentives for progression and horizontal movement.
* **Phase 3: Mastery & Speed (2.8M–4M steps):** The agent optimizes the path using time penalties and rewards for collecting coins and power-ups.

---

## 📂 Project Structure
To keep the repository clean, all core logic and scripts are organized within the `a2c/` directory:

```text
.
├── a2c/
│   ├── main_a2c.py          # Primary training script
│   ├── model_a2c.py         # Actor-Critic & ICM architectures
│   ├── env_wrapper.py       # Preprocessing & Frame stacking
│   ├── play_a2c_pure.py     # Script to test trained agents
│   ├── plot_a2c_pure.py     # Visualization of training metrics
│   └── requirements.txt     # Python dependencies
└── README.md                # Project documentation
