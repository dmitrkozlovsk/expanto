"""Constants for statistical functions."""

from __future__ import annotations

# -------- CONSTANTS FOR STATISTICAL FUNCTIONS --------

# Critical values for two-sided alpha levels in normal distribution
NORM_PPF_ALPHA_TWO_SIDED: dict[float, float] = {
    0.1: 1.6448536269514722,
    0.05: 1.959963984540054,
    0.01: 2.5758293035489004,
    0.001: 3.2905267314919255,
}

# Critical values for beta levels (Type II error rates) in normal distribution
NORM_PPF_BETA: dict[float, float] = {
    0.05: 1.6448536269514722,
    0.1: 1.2815515655446004,
    0.2: 0.8416212335729142,
}
