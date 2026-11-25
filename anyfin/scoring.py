"""Anyfin's Track A scoring rule, shared by the demo and offline evaluations."""


def row_score(label, truth):
    """Return +1 for a correct category, -1 for a wrong one, and 0 for an abstention."""
    if label == "unknown":
        return 0
    return 1 if label == truth else -1
