"""
Judge Calibration module for Apple Support AI Agent.
Calculates agreement metrics (Pearson r, MAE, Exact Agreement) between LLM-as-Judge and Human annotations.
"""

import numpy as np
from typing import List, Dict, Any

def calibrate_judge_vs_human(
    judge_scores: List[float], 
    human_scores: List[float]
) -> Dict[str, Any]:
    """
    Computes statistical alignment and calibration metrics between LLM Judge and Human annotations.
    """
    judge_arr = np.array(judge_scores, dtype=float)
    human_arr = np.array(human_scores, dtype=float)

    # 1. Pearson Correlation
    if np.std(judge_arr) == 0 or np.std(human_arr) == 0:
        pearson_r = 1.0
    else:
        pearson_r = float(np.corrcoef(judge_arr, human_arr)[0, 1])

    # 2. Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(judge_arr - human_arr)))

    # 3. Exact & Within 1-Point Agreement
    exact_match = float(np.mean(np.round(judge_arr) == np.round(human_arr)))
    within_one_point = float(np.mean(np.abs(judge_arr - human_arr) <= 1.0))

    return {
        "pearson_correlation": round(pearson_r, 4),
        "mean_absolute_error": round(mae, 4),
        "exact_agreement_rate": round(exact_match, 4),
        "within_1pt_agreement_rate": round(within_one_point, 4)
    }
