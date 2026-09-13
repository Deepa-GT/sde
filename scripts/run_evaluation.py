"""
Comprehensive Evaluation Benchmark Harness script.
Evaluates Trivial Baseline, Simple ML Baseline, and Proposed AI Agent against the Golden Evaluation Set (200 cases).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.agent.pipeline import SupportAgentPipeline
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleMLBaselineAgent
from src.eval.metrics import evaluate_predictions
from src.eval.judge import LLMSupportJudge


console = Console()

def run_benchmark():
    console.print(Panel.fit("[bold green]Executing Evaluation Benchmark Suite[/bold green]\n[dim]Comparing Trivial Baseline vs Simple ML vs Proposed Agent[/dim]"))

    # Load Golden Evaluation Set
    with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    y_true_intent = [item["ground_truth_intent"] for item in golden_data]
    y_true_escalate = [item["should_escalate"] for item in golden_data]
    ref_replies = [item["reference_reply"] for item in golden_data]
    human_scores = [item["human_quality_score"] for item in golden_data]

    # Initialize Agents & Judge
    trivial_agent = TrivialBaselineAgent()
    simple_agent = SimpleMLBaselineAgent()
    proposed_agent = SupportAgentPipeline()
    judge = LLMSupportJudge()

    systems = [
        ("Baseline 1: Trivial", trivial_agent),
        ("Baseline 2: Simple ML", simple_agent),
        ("Proposed Agent: RAG + Policy Router", proposed_agent)
    ]

    summary_rows = []

    for name, agent in systems:
        console.print(f"\n[bold yellow]Running {name}...[/bold yellow]")
        y_pred_intent = []
        y_pred_escalate = []
        pred_replies = []
        judge_scores = []

        for item in golden_data:
            cust_text = item["customer_text"]
            res = agent.process(cust_text)
            
            p_intent = res["predicted_intent"]
            p_esc = res["should_escalate"]
            p_reply = res["drafted_reply"]

            y_pred_intent.append(p_intent)
            y_pred_escalate.append(p_esc)
            pred_replies.append(p_reply)

            # LLM Judge score
            judge_res = judge.evaluate_reply(
                customer_text=cust_text,
                predicted_intent=p_intent,
                drafted_reply=p_reply,
                reference_reply=item["reference_reply"],
                should_escalate=p_esc
            )
            judge_scores.append(judge_res["overall_score"])

        # Compute metric suite
        metrics = evaluate_predictions(
            y_true_intent=y_true_intent,
            y_pred_intent=y_pred_intent,
            y_true_escalate=y_true_escalate,
            y_pred_escalate=y_pred_escalate,
            reference_replies=ref_replies,
            predicted_replies=pred_replies
        )
        
        avg_judge = round(sum(judge_scores) / len(judge_scores), 2)

        summary_rows.append({
            "name": name,
            "intent_acc": f"{metrics['intent_accuracy']*100:.1f}%",
            "intent_f1": f"{metrics['intent_macro_f1']*100:.1f}%",
            "esc_f1": f"{metrics['escalation_f1']*100:.1f}%",
            "esc_rec": f"{metrics['escalation_recall']*100:.1f}%",
            "rouge_l": f"{metrics['mean_rouge_l']:.3f}",
            "cosine_sim": f"{metrics['mean_cosine_similarity']:.3f}",
            "judge_score": f"{avg_judge} / 5.0"
        })

    # Print Comparison Table
    table = Table(title="Headline Evaluation Results Comparison", show_header=True, header_style="bold cyan")
    table.add_column("System", style="bold white", width=35)
    table.add_column("Intent Acc", style="green", justify="right")
    table.add_column("Intent F1", style="green", justify="right")
    table.add_column("Escalation F1", style="magenta", justify="right")
    table.add_column("Escalation Recall", style="magenta", justify="right")
    table.add_column("ROUGE-L", style="yellow", justify="right")
    table.add_column("Cosine Sim", style="yellow", justify="right")
    table.add_column("LLM-Judge Score", style="bold red", justify="right")

    for row in summary_rows:
        table.add_row(
            row["name"],
            row["intent_acc"],
            row["intent_f1"],
            row["esc_f1"],
            row["esc_rec"],
            row["rouge_l"],
            row["cosine_sim"],
            row["judge_score"]
        )

    console.print(table)
    
    # Also print clear Markdown table
    console.print("\n[bold]Markdown Summary:[/bold]")
    md_header = "| System | Intent Acc | Intent F1 | Escalation F1 | Escalation Recall | ROUGE-L | Cosine Sim | LLM-Judge Score |"
    md_sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    print(md_header)
    print(md_sep)
    for row in summary_rows:
        print(f"| {row['name']} | {row['intent_acc']} | {row['intent_f1']} | {row['esc_f1']} | {row['esc_rec']} | {row['rouge_l']} | {row['cosine_sim']} | {row['judge_score']} |")

if __name__ == "__main__":
    run_benchmark()
