#!/usr/bin/env python
import json
from pathlib import Path


def generate_markdown_report(metrics_path: str, output_path: str):
    with open(metrics_path) as f:
        metrics = json.load(f)

    report = "# Model Evaluation Report\n\n"
    report += "## Metrics\n\n"
    for key, value in metrics.items():
        report += f"- **{key}**: {value:.4f}\n"

    with open(output_path, "w") as f:
        f.write(report)

    print(f"✓ Report generated: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generate_report.py <metrics.json> <output.md>")
        sys.exit(1)

    generate_markdown_report(sys.argv[1], sys.argv[2])
