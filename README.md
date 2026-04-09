# 🤖 Beyond the Flag: From Mechanical Training to Curious Intelligence

This repository contains a comparative analysis and implementation of Reinforcement Learning (RL) agents trained to play **Super Mario Bros.** using three distinct algorithmic approaches. The project explores the shift from "mechanical training" based on dense, handcrafted rewards to "curious intelligence" driven by intrinsic motivation and stable policy updates.

---

## 🚀 The Challenge: Sparse vs. Dense Rewards
[cite_start]In classical Reinforcement Learning (RL), agents often struggle with **sparse rewards**—situations where feedback is provided only after a long sequence of actions, such as reaching the flag at the end of a level.

To address this, we explored three methodologies:
1.  **A2C (The Robot):** A baseline agent using aggressive **Reward Shaping** (points for every step, killing enemies, etc.) to brute-force the level.
2.  **A2C + ICM (The Confused):** An agent augmented with curiosity but lacking long-term memory stability, leading to "catastrophic forgetting".
3.  **PPO + ICM (The Child):** A robust agent using Policy Clipping and **Curriculum Learning** to explore the world through structured discovery.

---

## 🧠 The Solution: Intrinsic Curiosity Module (ICM)
To move beyond external prizes, we implemented the **ICM**, which rewards the agent for "surprise"—discovering states it cannot accurately predict. The architecture consists of three neural subnets:

* **Feature Extractor ($\phi$):** A CNN that compresses raw $84 \times 84$ pixel inputs into latent vectors to filter out environmental noise.
* **Inverse Model:** Predicts the action taken between two states, driving curiosity only toward things the agent can actually influence.
* **Forward Model:** Estimates the next state based on the current state and action. [cite_start]When its prediction fails, it triggers an **Intrinsic Reward ($r_i$)**.

The total optimization follows a unified learning objective:
$$\mathcal{L}_{total}=\mathcal{L}_{policy}+\mathcal{L}_{value}+\mathcal{L}_{forward}+\mathcal{L}_{inverse}$$

---

## 📂 Project Structure
The repository is organized into three main experimental directories:

```text
.
├── A2C/                        # Baseline: Dense Reward Shaping
│   ├── main_a2c.py             # Training script for standard A2C
│   ├── model_a2c.py            # Baseline Actor-Critic architecture
│   ├── graph_a2c_pure.png      # Results showing fast but robotic convergence
│   └── requirements.txt        # Dependencies
│
├── A2C ICM/                    # Experiment: Curiosity without stability
│   ├── main_a2c.py             # A2C training with ICM integration
│   ├── model_a2c.py            # ICM + A2C Neural Architectures
│   ├── graph_a2c_icm.png       # Results showing "comb-like" exploration gaps
│   └── requirements.txt        # Dependencies
│
├── PPO ICM/                    # Advanced: Robust & Intelligent Exploration
│   ├── main_ppo.py             # PPO training with Curriculum Learning
│   ├── model.py                # Stable PPO + ICM Architecture
│   ├── graph_ppo_icm.png       # Results showing stable, consistent mastery
│   └── requirements.txt        # Dependencies
│
└── README.md                   # Project documentation
