"""
Judge Calibration Script.
Validates alignment between LLM-as-Judge scores and Human Ground-Truth ratings on the Golden Set.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.pipeline import SupportAgentPipeline
from src.eval.judge import LLMSupportJudge
from src.eval.calibration import calibrate_judge_vs_human


console = Console()

def run_calibration():
    console.print(Panel.fit("[bold blue]LLM-as-Judge Calibration & Human Alignment Harness[/bold blue]"))

    with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    pipeline = SupportAgentPipeline()
    judge = LLMSupportJudge()

    human_scores = []
    judge_scores = []

    for item in golden_data:
        cust_text = item["customer_text"]
        human_score = item["human_quality_score"]

        res = pipeline.process(cust_text)
        
        judge_res = judge.evaluate_reply(
            customer_text=cust_text,
            predicted_intent=res["predicted_intent"],
            drafted_reply=res["drafted_reply"],
            reference_reply=item["reference_reply"],
            should_escalate=res["should_escalate"]
        )

        human_scores.append(human_score)
        judge_scores.append(judge_res["overall_score"])

    # Compute calibration metrics
    cal_res = calibrate_judge_vs_human(judge_scores, human_scores)

    table = Table(title="LLM-as-Judge Calibration & Agreement Matrix", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("Pearson Correlation (r)", f"{cal_res['pearson_correlation']:.4f}")
    table.add_row("Mean Absolute Error (MAE)", f"{cal_res['mean_absolute_error']:.4f}")
    table.add_row("Exact Score Agreement Rate", f"{cal_res['exact_agreement_rate']*100:.1f}%")
    table.add_row("Within +/- 1 Point Agreement Rate", f"{cal_res['within_1pt_agreement_rate']*100:.1f}%")

    console.print(table)
    r = cal_res["pearson_correlation"]
    if r >= 0.7:
        verdict = "[bold green]Strong judge-human alignment[/bold green]"
    elif r >= 0.4:
        verdict = "[bold yellow]Moderate judge-human alignment — see REPORT.md for limitations[/bold yellow]"
    else:
        verdict = "[bold red]Weak judge-human alignment — interpret judge scores cautiously[/bold red]"
    console.print(f"{verdict}. Within 1-point agreement: {cal_res['within_1pt_agreement_rate']*100:.1f}%.")


if __name__ == "__main__":
    run_calibration()
