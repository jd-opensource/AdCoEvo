"""Allowed tCPA multipliers shared by agent schemas, Rule, and execution."""

TCPA_LEVELS = ("HOLD", "0.5", "0.6", "0.7", "0.8", "1.0", "1.2", "1.3")
TCPA_LEVEL_SET = frozenset(TCPA_LEVELS)

def is_allowed_tcpa_level(level: str) -> bool:
    return level in TCPA_LEVEL_SET
