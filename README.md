# EEG Emotion Detection using Reinforcement Learning

## Overview

This project implements a reinforcement learning baseline for EEG emotion analysis using the DEAP dataset.

The RL agent learns to select emotionally informative EEG segments.

---

## Dataset

DEAP Dataset

Subjects: 32  
Trials per subject: 40  
EEG Channels: 32  
Sampling Rate: 128 Hz  

Data shape:

40 × 32 × 8064

---

## Pipeline

EEG signal
↓
Segmentation (1 second)
↓
Differential Entropy Features
↓
Prototype Learning (KMeans)
↓
Reinforcement Learning Agent
↓
Emotionally informative segment selection

---

## Neural Network

Architecture:

LSTM(128)
↓
FC(64)
↓
FC(32)
↓
Action probabilities

---

## Reward Function

Center Reward:

1 / (1 + dist)

dist = ||x − μ||

Inter/Intra Cluster Reward:

exp(-intra/inter)

Final reward:

R = center_reward + inter_intra_reward

---

## Installation

pip install -r requirements.txt

---

## Dataset Setup

Download DEAP dataset.

Place files inside:

data/

Example:

data/
s01.dat

---

## Running

python main.py

---

## Output

After training a model file will be saved:

eeg_rl_agent.zip

This model contains the trained RL policy.