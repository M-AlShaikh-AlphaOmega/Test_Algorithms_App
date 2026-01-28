#!/usr/bin/env python
from pathlib import Path


def setup_directories():
    dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "artifacts/models",
        "artifacts/reports",
        "artifacts/figures",
        "notebooks",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Project directories created")


def create_gitkeep_files():
    dirs = ["data/raw", "data/interim", "data/processed", "notebooks"]
    for d in dirs:
        gitkeep = Path(d) / ".gitkeep"
        gitkeep.touch()
    print("✓ .gitkeep files created")


if __name__ == "__main__":
    setup_directories()
    create_gitkeep_files()
    print("✓ Project setup complete")
