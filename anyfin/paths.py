"""Locations of the case data, saved experiments, and runtime cache in this checkout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = ROOT / "data" / "transactions.csv"
VALIDATION_CSV = ROOT / "data" / "validation.csv"
GEPA_PROMPT = ROOT / "prompts" / "gepa-selected.txt"
CACHE_DIR = ROOT / ".cache"
PAIRED_RESULTS = ROOT / "results" / "evaluation" / "c3-paired.csv"
SUBMISSIONS_DIR = ROOT / "results" / "submissions"


def resolve_input(path):
    """Allow earlier checkout-root input names while giving explicit existing paths priority."""
    path = Path(path)
    if path.exists() or path.resolve().parent != ROOT:
        return path
    original_paths = {
        "transactions.csv": TRAINING_CSV,
        "validation.csv": VALIDATION_CSV,
        "best_prompt.txt": GEPA_PROMPT,
        "challenger_prompt.txt": ROOT / "prompts" / "gepa-challenger.txt",
        "anyfin-track-a-validation-hybrid.csv": (
            SUBMISSIONS_DIR / "anyfin-track-a-validation-local-router-claude-haiku-5-5-hybrid.csv"
        ),
    }
    if path.name in original_paths:
        return original_paths[path.name]
    saved_submission = SUBMISSIONS_DIR / path.name
    if path.suffix == ".csv" and saved_submission.is_file():
        return saved_submission
    return path
