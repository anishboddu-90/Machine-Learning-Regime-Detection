# Signal Detection in Weakly Biased Stochastic Processes

  

## Problem

After establishing a successful rule-based strategy, this project explores whether a **Random Forest Classifier** can outperform manual human-made heuristics in identifying market regimes. The goal is to classify whether a specific step belongs to a "Biased Regime" based solely on historical price action features.

  

## Model: Random Forest Classifier

This iteration transitioned from using a simple sum to a multi-dimensional feature set to train the Random Forest model.

### Feature Engineering
To provide the model with "context", several rolling technical indicators were engineered:
- **Momentum:** Rolling cumulative sums over 5, 10, and 20-step windows. 
-  **Volatility:** 5-step rolling standard deviation. 
- **Persistence:** Streaks of consecutive positive or negative moves.
 - **Micro-Structure:** Rolling 5 and 20-step autocorrelation to detect "stickiness" in the noise.
  
  ## Results
  The model was trained on 10,000 simulated steps with a balanced class weight to account for the rarity of biased regimes (~25% of data)

| Metric | Adaptive Heuristic | Random Forest (ML) | Constant Strategy | 
| :--- | :--- | :--- | :--- |
 | **Logic Type** | Rule-Based Filter | ML Classification | Benchmark | | **Final Avg Bankroll** | **165.00** | 112.10 | 141.22 | 
 | **Signal Precision** | **39.10%** | 30.14% | N/A | 
 | **Annualized Sharpe** | **~1.00** | ~0.42 | 0.21 | 
 | **Win Rate** | 74.00% | 54.85% | **86.60%** |
 **Base Rate (Actual Bias):** 24.96%
 
## Why the Heuristic Won
While the model had a higher accuracy,  its precision is low. This project led to 3 critical conclusions :

1. **Signal-to-Noise Floor:** In a process where the bias is only 2% (52/48), the signal is so faint that a model like a Random Forest often mistakes random clusters of noise for structural regimes. 
2. **The Heuristic Edge:** The previous heuristic (`sum > 4`) acted as a high-pass filter. It had lower Recall (missed many regimes) but higher Precision. The ML model tried to find *every* regime, resulting in too many false positives. 
3. **Model Interpretability:** In low-edge environments, simple linear or rule-based models often perform better than ensemble methods, which are prone to overfitting in the training set.
