# Prototyping Clinical Gaze Analysis: A Simulation Framework

This repository contains the reference implementation for the Computational Theory project: **"Unsupervised Recovery of Stochastic Clinical Protocols."**

## Abstract
We model the clinical expert's gaze as a **Probabilistic Finite Automaton (PFA)**. This codebase implements the **Bio-Saliency Generator** (to simulate valid input strings) and uses **Hidden Markov Models** to solve the inverse problem of recovering the state transition function $\delta$ from noisy observations.

## Project Structure
* `src/`: Core logic for the Automata and Generator.
* `validation/`: Statistical baselines (K-Means comparison).
* `main_reproduction.py`: The master script to reproduce the experiments.

## Formal Model
The system simulates an expert machine $\mathcal{M} = (Q, \Sigma, \delta, q_0, F)$ where observations are generated via:
$$O_t = \mu_L + \delta_{signal}(t) + \epsilon$$

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
