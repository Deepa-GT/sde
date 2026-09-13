"""
Interactive CLI & Single-Query runner for Apple Support AI Agent Pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.agent.pipeline import SupportAgentPipeline


console = Console()

def run_demo():
    console.print(Panel.fit("[bold blue]Apple Support AI Agent Pipeline[/bold blue]\n[dim]Hiver SDE Take-Home Assignment[/dim]"))
    
    pipeline = SupportAgentPipeline()
    
    sample_queries = [
        "@AppleSupport My iPhone 15 Pro battery drains 50% in 1 hour after updating to iOS 17.5!",
        "@AppleSupport Someone charged $500 to my stolen credit card on App Store! URGENT!",
        "@AppleSupport Dropped my iPad in the pool and the screen is completely dead.",
        "@AppleSupport How do I reset my Apple ID password?",
        "@AppleSupport Where is my order W123456789? Delivery status says delayed."
    ]

    for i, q in enumerate(sample_queries, 1):
        console.print(f"\n[bold yellow]--- Test Case {i} ---[/bold yellow]")
        console.print(f"[bold white]Customer Tweet:[/bold white] {q}")
        
        res = pipeline.process(q)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="white")
        
        table.add_row("Predicted Intent", res["predicted_intent"])
        table.add_row("Confidence", str(res["intent_confidence"]))
        table.add_row("Escalate?", f"[bold {'red' if res['should_escalate'] else 'green'}]{res['should_escalate']}[/bold {'red' if res['should_escalate'] else 'green'}]")
        table.add_row("Escalation Reason", res["escalation_reason"])
        table.add_row("Drafted Reply", f"[italic]{res['drafted_reply']}[/italic]")
        table.add_row("Latency", f"{res['processing_time_ms']} ms")
        
        console.print(table)

if __name__ == "__main__":
    run_demo()
