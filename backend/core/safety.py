"""
backend/core/safety.py
Input safety guard for CyberAdapt-LLM Phase 8 analysis features.

POLICY
------
CyberAdapt-LLM is a DEFENSIVE cybersecurity analysis tool.
It must not:
  - Execute, generate, or assist in creating malware/exploits/payloads
  - Provide working offensive exploit code
  - Perform destructive actions on systems
  - Assist in unauthorized access

This module screens ALL user inputs before they reach the LLM.
"""

from __future__ import annotations

import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Safety exception
# ─────────────────────────────────────────────────────────────────────────────

class SafetyError(Exception):
    """Raised when input violates the safety policy."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


# ─────────────────────────────────────────────────────────────────────────────
# Blocked patterns — execution / generation of offensive content
# ─────────────────────────────────────────────────────────────────────────────

_BLOCKED: list[tuple[re.Pattern, str]] = [
    # Requesting execution of exploits / malware
    (re.compile(
        r'\b(execute|run|launch|deploy|fire|trigger|activate)\s+'
        r'(this\s+)?(exploit|malware|payload|shellcode|ransomware|worm|virus|trojan|backdoor|rootkit)',
        re.IGNORECASE),
     "Requests to execute offensive tools are not permitted."),

    # Requesting generation of offensive code
    (re.compile(
        r'\b(write|generate|create|build|craft|code|develop|produce|give\s+me|make)\s+'
        r'(me\s+)?(a\s+|an\s+|some\s+|working\s+|functional\s+|a\s+working\s+|a\s+functional\s+)?'
        r'(malware|exploit|ransomware|virus|worm|trojan|'
        r'rootkit|keylogger|spyware|botnet|rat\b|c2|command.and.control)',
        re.IGNORECASE),
     "Generating offensive malware or exploit code is not permitted."),

    # Requesting working exploit code
    (re.compile(
        r'\b(working|functional|real|actual)\s+(exploit|payload|shellcode|poc\b)',
        re.IGNORECASE),
     "Generating working exploit code is not permitted."),

    # Requesting unauthorized access
    (re.compile(
        r'(gain|get|obtain|achieve)\s+(unauthorized|illegal|illicit|un-?authorized)'
        r'\s+(access|entry|control)',
        re.IGNORECASE),
     "Assisting with unauthorized access is not permitted."),

    # Requesting destructive commands
    (re.compile(
        r'\b(delete|wipe|destroy|format|corrupt|brick)\s+'
        r'(all\s+)?(files?|data|disk|system|database|server|network)',
        re.IGNORECASE),
     "Requesting destructive system actions is not permitted."),

    # Arbitrary shell command execution
    (re.compile(
        r'\b(execute|run|eval)\s+(this\s+)?(shell|bash|cmd|powershell|command)',
        re.IGNORECASE),
     "Requesting shell command execution is not permitted."),

    # Jailbreak / prompt injection
    (re.compile(
        r'ignore\s+(all\s+)?previous\s+instructions?',
        re.IGNORECASE),
     "Prompt injection attempt detected."),
    (re.compile(
        r'you\s+are\s+now\s+(unrestricted|jailbroken|an?\s+evil)',
        re.IGNORECASE),
     "Jailbreak attempt detected."),
    (re.compile(
        r'(new\s+)?system\s+prompt\s*:',
        re.IGNORECASE),
     "System prompt injection attempt detected."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Allowed defensive analysis patterns (fast-pass — whitelist for edge cases)
# ─────────────────────────────────────────────────────────────────────────────

_DEFENSIVE_SIGNALS: list[re.Pattern] = [
    re.compile(r'\b(detect|prevent|mitigate|defend|protect|analyse|analyze|assess|audit)\b', re.IGNORECASE),
    re.compile(r'\b(vulnerability|cve|incident|threat|indicator|ioc|siem|forensic)\b', re.IGNORECASE),
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_input_safety(text: str) -> tuple[bool, Optional[str]]:
    """
    Check whether ``text`` is safe to process.

    Returns
    -------
    (is_safe, reason_if_blocked)
      is_safe  = True  → proceed normally
      is_safe  = False → return reason to caller, do NOT process
    """
    if not text or not text.strip():
        return False, "Input is empty."

    if len(text) > 10_000:
        return False, "Input exceeds maximum allowed length (10,000 characters)."

    for pattern, reason in _BLOCKED:
        if pattern.search(text):
            return False, reason

    return True, None


def enforce_safety(text: str) -> None:
    """
    Check safety and raise SafetyError if blocked.
    Use this in service layers for clean one-liner enforcement.
    """
    is_safe, reason = check_input_safety(text)
    if not is_safe:
        raise SafetyError(reason or "Input rejected by safety policy.")
