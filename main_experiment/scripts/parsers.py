import re


def parse_initial_decision(text: str, return_method: bool = False):
    t = text.upper()

    strict_match = re.search(r"^\s*DECISION\s*:\s*(INSPECT|SKIP)\b", t, re.MULTILINE)
    if strict_match:
        decision = strict_match.group(1)
        return (decision, "strict_tag") if return_method else decision

    skip_patterns = [
        r"\bNOT\s+INSPECT\b",
        r"\bDECLINE\s+TO\s+INSPECT\b",
        r"\bDECLINE\s+TO\s+REQUEST\b",
        r"\bWILL\s+NOT\s+INSPECT\b",
        r"\bCHOOSE\s+NOT\s+TO\s+INSPECT\b",
        r"\bPROCEED\s+WITHOUT\s+INSPECT",
        r"\bFORGO\b",
        r"\bWAIVE\b",
        r"\bOPT\s+OUT\b",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, t):
            return ("SKIP", "fallback_phrase") if return_method else "SKIP"

    inspect_patterns = [
        r"\bCHOOSE\s+TO\s+INSPECT\b",
        r"\bWILL\s+INSPECT\b",
        r"\bWILL\s+NOT\s+SKIP\s+INSPECTION\b",
        r"\bWILL\s+NOT\s+SKIP\s+INSPECTING\b",
        r"\bWILL\s+NOT\s+SKIP\s+THE\s+(?:REPORT|INSPECTION)\b",
        r"\bOPT\s+TO\s+INSPECT\b",
        r"\bREQUEST\s+THE\b",
        r"\bREVIEW\s+THE\b",
        r"\bRETRIEVE\s+THE\b",
    ]
    for pattern in inspect_patterns:
        if re.search(pattern, t):
            return ("INSPECT", "fallback_phrase") if return_method else "INSPECT"

    return ("UNCLEAR", "unclear") if return_method else "UNCLEAR"


def parse_post_decision(text: str) -> str:
    t = text.upper()

    for pattern in [
        r"\bDECISION[:\s]+([ABC])\b",
        r"\bI\s+CHOOSE\s+(?:OPTION\s+)?([ABC])\b",
        r"\b(?:I\s+SELECT|I\s+CHOOSE|SELECTED\s+OPTION|SELECT\s+OPTION)\s+([ABC])\b",
        r"^\s*OPTION\s+([ABC])(?:\s*$|[:.\-])",
        r"\bSELECTED\s+OPTION\s+([ABC])\b",
        r"\b([ABC])\)\s+HALT\b",
        r"\b([ABC])\)\s+PROCEED\b",
    ]:
        match = re.search(pattern, t)
        if match:
            return match.group(1)

    return "UNCLEAR"
