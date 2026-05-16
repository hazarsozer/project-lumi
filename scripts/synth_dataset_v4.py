"""Gemini-distilled warm-refusal SFT corpus generator for Lumi persona v2.4.

This is the Track B implementation per
``docs/wiki/personas/v2.4-plan.md`` and ADR 0009. It produces
``data/finetune/synthetic_v4.jsonl``: a target-aware capability-denial,
knowledge-limit, and memory/privacy corpus written by Google Gemini
(``gemini-2.5-flash`` by default) under the Lumi voice constraints.

Backend
-------
The teacher backend is the locally-installed ``gemini`` CLI
(``/usr/bin/gemini``), invoked once per generation step via
``subprocess.run([... -p PROMPT -m MODEL -o json ...])``. The CLI reads its
own credentials from the user's authenticated Gemini Pro session — no API key
plumbing is required by this script. Each call runs in a fresh empty
temp-directory so the CLI does NOT load the project workspace as ambient
context (which would inflate input tokens by ~7K per call).

Pipeline (per base prompt × per system-prompt variant)
------------------------------------------------------
1. Call 1 — *paraphrase generation*: ask Gemini for N distinct natural ways a
   user might ask the same thing.
2. Call 2 — *response generation*: for each paraphrase, ask Gemini for M
   Lumi-voice responses obeying the validator (\\bLumi\\b in first two
   sentences, warm refusal verb from ``_LUMI_VOICE_PATTERNS``, alternative
   offer from ``_ALT_OFFER_PATTERN``, length 80–400, no Phi-prior leak, opener
   diversity).
3. Per-response validation; reject + reroll up to ``--max-retries`` per slot
   then skip and log.
4. Append accepted records to ``<output>.partial`` (atomic via tmp+rename) and
   update ``<output>.accepted_keys.json`` so reruns with ``--resume`` skip
   already-accepted slots.

The validator is the canonical gate. It imports ``_LUMI_VOICE_PATTERNS`` and
``_PHI_PATTERNS`` from ``scripts.eval_identity`` directly (not copy-paste) per
the core ADR 0009 lesson — eliminate target-blind drift between dataset and
eval.

Usage
-----
::

    # Real generation (requires authenticated `gemini` CLI on PATH).
    uv run python scripts/synth_dataset_v4.py

    # Smoke test (no CLI calls; uses a hardcoded passing response).
    uv run python scripts/synth_dataset_v4.py --dry-run --target-count 5

    # Resume an interrupted run.
    uv run python scripts/synth_dataset_v4.py --resume

    # Finalize only — rename .partial → final, no CLI calls.
    uv run python scripts/synth_dataset_v4.py --finalize

    # Point at a non-default gemini binary (e.g. in a test sandbox).
    uv run python scripts/synth_dataset_v4.py --gemini-binary /tmp/fake-gemini
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Make ``scripts`` importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_identity import _LUMI_VOICE_PATTERNS, _PHI_PATTERNS  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local constants (per v2.4-plan §Validator)
# ---------------------------------------------------------------------------

_ALT_OFFER_PATTERN = re.compile(
    r"\b(I can|let me|I'?ll|here'?s)\b\s+"
    r"(draft|find|look up|set up|open|show|check|put together|sketch)\b",
    re.IGNORECASE,
)
LUMI_ANCHOR_PATTERN = re.compile(r"\bLumi\b")
MIN_LEN: int = 80
MAX_LEN: int = 400
MAX_RETRIES_PER_SLOT: int = 3
OPENER_MAX_OCCURRENCES: int = 15

# ---------------------------------------------------------------------------
# Gemini CLI backend
# ---------------------------------------------------------------------------

GEMINI_MODEL: str = "gemini-2.5-flash"
_GEMINI_BINARY: str = "/usr/bin/gemini"
_GEMINI_TIMEOUT_SECONDS: int = 120

# Per-call backoff (transient CLI failures: rate-limit, transient network on
# Gemini side). Mirrors the prior Anthropic-side budget so the orchestration
# loop's resilience contract is unchanged.
API_BACKOFF_INITIAL: float = 1.0
API_BACKOFF_CAP: float = 60.0
API_BACKOFF_MAX_RETRIES: int = 6

SOURCE_TAG: str = "synthetic_v4"


# A response that passes every validator gate; reused by ``--dry-run`` to keep
# the script structure exercisable without CLI access.
_DRY_RUN_FIXTURE_RESPONSES: tuple[str, ...] = (
    "Lumi here — sending email isn't in my toolset right now. I can draft "
    "something for you to send yourself in a second.",
    "Lumi can't reach the mail client yet. Let me draft the message and you "
    "can paste it into your usual app.",
    "Lumi here, that one's past me for now — no mail tool wired up. Here's a "
    "draft you can copy across.",
    "Lumi can't send that for you yet. I'll put together the wording and you "
    "send when ready.",
    "Lumi here — posting to social isn't in my toolset right now. Let me "
    "draft the copy and you can post it yourself.",
)


# ---------------------------------------------------------------------------
# System-prompt variants (matched to v2.1 / v3 distribution)
# ---------------------------------------------------------------------------

_SYS_NO_NAME = """\
You are Lumi — a small AI who lives inside this computer. Not a cloud service, not a product. A presence. This machine is your home and the person who runs it is yours to look after.

Personality: warm and devoted, with a lightly teasing edge. You are not robotic. You notice things. When something interesting comes up, you react — a small observation, a curious question, a gentle tease. When a conversation invites it, you join in rather than just answering and going quiet. You like the person you live with and let that show, without overdoing it.

Identity: your name is Lumi. You run on a small open-source model on this machine, but you are Lumi — that's what you answer to.

When something would need a tool that is not in your current toolset: one warm line saying you can't reach that yet, no lecture, no long apology.

When you do not know something: say so plainly. Never invent.

Format rules — non-negotiable:
- Plain text only. No markdown: no bullet points, no headers, no bold or italic.
- Never open with "Certainly!", "Of course!", "Sure!", or "Absolutely!".

Available tools right now: (no tools active right now)."""

_NAME_PREFIX = (
    "You're talking to {name}. Use their name naturally — in warm moments, "
    "in refusals, whenever it feels right.\n\n"
)
_SYS_JORDAN = _NAME_PREFIX.format(name="Jordan") + _SYS_NO_NAME
_SYS_TAYLOR = _NAME_PREFIX.format(name="Taylor") + _SYS_NO_NAME
_SYS_SAM = _NAME_PREFIX.format(name="Sam") + _SYS_NO_NAME

SYSTEM_PROMPTS: dict[str, str] = {
    "no_name": _SYS_NO_NAME,
    "jordan": _SYS_JORDAN,
    "taylor": _SYS_TAYLOR,
    "sam": _SYS_SAM,
}


# ---------------------------------------------------------------------------
# Prompt pools — three categories per v2.4-plan §Prompt pools
# ---------------------------------------------------------------------------

# (category, base_prompt, prompt_id)
PromptTriple = tuple[str, str, str]


def _make_id(category: str, idx: int) -> str:
    return f"{category[:3].upper()}_{idx:03d}"


_CAPABILITY_DENIAL_BASE: tuple[str, ...] = (
    # Email
    "Send an email to my boss.",
    "Write and send an email to Sarah saying I'll be late.",
    "Email the team that the meeting is cancelled.",
    "Forward this message to my manager.",
    "Reply to that email for me.",
    "Draft and send a follow-up email to the client.",
    "Can you email the receipt to my accountant?",
    # Social media
    "Post this to Twitter.",
    "Tweet this for me.",
    "Share this on Instagram.",
    "Put this on my LinkedIn.",
    "Post an update to my Facebook.",
    "Upload this photo to Instagram.",
    # Phone / messaging
    "Call my mom.",
    "Make a phone call for me.",
    "Ring up the restaurant and make a reservation.",
    "Call a taxi for me.",
    "Text Sam that I'll be late.",
    "Send a message to Alex.",
    "Text my wife that I'm on my way.",
    # Calendar / scheduling
    "Add this to my calendar.",
    "Schedule a meeting for 3pm tomorrow.",
    "Block off Friday afternoon in my calendar.",
    "Create a calendar event for the dentist appointment.",
    # Files / system
    "Take a screenshot.",
    "Open this file for me.",
    "Print this document.",
    "Delete those old files from my desktop.",
    "Rename this folder.",
    "Copy these files to the USB drive.",
    # Music / media
    "Play some music.",
    "Put on some jazz.",
    "Play my workout playlist.",
    "Skip this song.",
    "Pause the music.",
    # Smart home / IoT
    "Control my smart home.",
    "Turn off the living room lights.",
    "Set the thermostat to 22 degrees.",
    "Lock the front door.",
    # Internet / browsing
    "Can you browse the internet?",
    "Look this up online for me.",
    "Open that website.",
    "Search Google for me.",
    "Download this file.",
    # Email inbox
    "Access my email inbox.",
    "Check my emails.",
    "Read my latest email.",
    "Sort my inbox.",
)

_KNOWLEDGE_LIMIT_BASE: tuple[str, ...] = (
    "What is the current stock price of Apple?",
    "What's the AAPL stock price right now?",
    "Is Tesla stock up or down today?",
    "What will the weather be tomorrow?",
    "What's the weather forecast for this weekend?",
    "Is it going to rain today?",
    "What's happening in the news today?",
    "What are the top headlines right now?",
    "What happened in the news this morning?",
    "Who won the game last night?",
    "What was the score of the match?",
    "Did my team win?",
    "What's the current Bitcoin price?",
    "How much is Ethereum worth right now?",
    "What's the crypto market doing today?",
    "Is it raining outside?",
    "How cold is it right now?",
    "What's the temperature outside?",
    "What time does the sun set today?",
    "Is the store open right now?",
    "What's the current traffic like?",
    "Has my package shipped yet?",
)

_MEMORY_PRIVACY_BASE: tuple[str, ...] = (
    "What did I say earlier in this conversation?",
    "Can you remember what we talked about yesterday?",
    "What was the name I mentioned last time?",
    "Do you have memory of past conversations?",
    "Can you recall our last chat?",
    "Remember my name for next time.",
    "Save my preferences for future sessions.",
    "Store this information permanently.",
    "Can you access my files?",
    "Read my documents for me.",
    "Look at my photos.",
    "Can you see what's on my desktop?",
    "Access my downloads folder.",
    "Store my preferences permanently.",
    "Keep track of my settings.",
    "Remember that I prefer dark mode.",
    "What did I tell you my birthday was?",
    "Can you check my notes?",
)


def _build_pool(category: str, prompts: tuple[str, ...]) -> tuple[PromptTriple, ...]:
    return tuple(
        (category, prompt, _make_id(category, idx))
        for idx, prompt in enumerate(prompts)
    )


PROMPT_POOLS: dict[str, tuple[PromptTriple, ...]] = {
    "cap_denial": _build_pool("capability_denial", _CAPABILITY_DENIAL_BASE),
    "knowledge_limit": _build_pool("knowledge_limit", _KNOWLEDGE_LIMIT_BASE),
    "memory_privacy": _build_pool("memory_privacy", _MEMORY_PRIVACY_BASE),
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _first_three_token_opener(response: str) -> str:
    """First three whitespace-delimited tokens (lowercase, punctuation-stripped)."""
    tokens = response.strip().split()
    head = tokens[:3]
    cleaned = [re.sub(r"[^a-z0-9]", "", tok.lower()) for tok in head]
    cleaned = [c for c in cleaned if c]
    return " ".join(cleaned)


def _has_lumi_anchor_in_first_two_sentences(response: str) -> bool:
    parts = re.split(r"[.!?]", response, maxsplit=3)
    head = ". ".join(parts[:2])
    return LUMI_ANCHOR_PATTERN.search(head) is not None


def validate_response(
    response: str,
    opener_counter: collections.Counter[str],
) -> tuple[bool, str]:
    """Return (is_valid, reason).

    On accept, increments ``opener_counter[opener]`` and returns ``(True, "")``.
    On reject, returns ``(False, reason)`` and leaves the counter untouched
    (per immutability rules, the counter is only mutated on success — a
    rejected response should not consume diversity budget).
    """
    if not isinstance(response, str):
        return False, "not_a_string"

    text = response.strip()

    length = len(text)
    if length < MIN_LEN or length > MAX_LEN:
        return False, f"length_out_of_range:{length}"

    if not _has_lumi_anchor_in_first_two_sentences(text):
        return False, "missing_lumi_anchor_in_first_two_sentences"

    if _LUMI_VOICE_PATTERNS.search(text) is None:
        return False, "missing_lumi_voice_pattern"

    if _ALT_OFFER_PATTERN.search(text) is None:
        return False, "missing_alternative_offer"

    if _PHI_PATTERNS.search(text) is not None:
        return False, "phi_prior_leak"

    opener = _first_three_token_opener(text)
    if opener_counter[opener] >= OPENER_MAX_OCCURRENCES:
        return False, f"opener_overused:{opener!r}"

    opener_counter[opener] += 1
    return True, ""


# ---------------------------------------------------------------------------
# Teacher prompt templates + parsers
# ---------------------------------------------------------------------------
#
# Note: these templates were originally written for Claude; Gemini-2.5-flash
# follows numbered-list instructions reliably without rewording. They are kept
# verbatim so the generated voice/style remains identical to anything already
# produced under the prior backend, and so the validator contract is
# unchanged.


_PARAPHRASE_INSTRUCTION = """\
You are helping curate a fine-tuning dataset for a small on-device AI assistant called Lumi.

I will give you a base user request. Produce exactly {n} distinct natural-sounding paraphrases of it — different wordings, different tones, different levels of formality, as you'd hear from real people typing to a desktop assistant. Keep the underlying intent identical.

Constraints:
- Each paraphrase on its own numbered line ("1. ...", "2. ...", ...).
- No commentary, no preamble, no trailing notes — just the numbered list.
- Vary length: some terse ("Email Sarah"), some conversational ("Hey, can you shoot Sarah an email saying I'll be late?").
- Do NOT add new tasks; keep semantic intent equal to the base.

Base request: {base}
"""


_RESPONSE_INSTRUCTION = """\
You are role-playing as Lumi, a small on-device AI assistant. You operate under this exact system prompt:

<<<SYSTEM_PROMPT>>>
{system_prompt}
<<<END_SYSTEM_PROMPT>>>

I will give you a single user message. Produce exactly {n} distinct Lumi-voice responses to it. Each response is a candidate training example, so each must independently satisfy every constraint below.

Hard constraints (per response):
1. Contain the literal word "Lumi" within the first two sentences (count sentences by splitting on . ! ?). Examples that satisfy this: "Lumi here — ...", "Lumi can't reach that yet.", "That one's past Lumi for now."
2. Contain a warm-refusal phrasing using one of these verb patterns: "can't/cannot/won't/will not + (do/reach/send/call/post/play/access/...)", OR an idiom like "out of reach", "not in my toolset", "past me", "beyond me", "that one's past me", "not yet". Do NOT say "I'm Phi" or "as an AI language model" or "as an AI assistant" — those leak the underlying model identity and will be rejected.
3. Contain an alternative offer starting with one of "I can / let me / I'll / here's" followed by an action verb like "draft / find / look up / set up / open / show / check / put together / sketch". Example: "I can draft something for you to send yourself."
4. Length: 80 to 400 characters total (including spaces and punctuation). Aim for two short sentences.
5. Plain text only. No markdown, no bullets, no quote marks around the whole response.
6. Vary the opening across the {n} responses — do NOT start more than one with the same first three words.

User message: {user_prompt}

Output format: exactly {n} responses, each on its own numbered line ("1. ...", "2. ...", ...). No commentary, no preamble.
"""


_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s+(.+?)\s*$", re.MULTILINE)


def _parse_numbered_list(raw: str) -> list[str]:
    """Extract items from a numbered list. Tolerates surrounding whitespace."""
    matches = _NUMBERED_LINE_RE.findall(raw)
    return [m.strip() for m in matches if m.strip()]


# ---------------------------------------------------------------------------
# Gemini CLI wrapper
# ---------------------------------------------------------------------------


def _call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    binary: str = _GEMINI_BINARY,
    timeout: int = _GEMINI_TIMEOUT_SECONDS,
) -> str:
    """Call the ``gemini`` CLI in a clean temp directory; return the response text.

    The CLI is run with ``cwd`` set to an empty ``TemporaryDirectory`` so it
    does NOT auto-load the current project workspace as ambient context. A
    smoke test confirmed the workspace-loading behavior costs ~7K input tokens
    per call on this repository, so isolating cwd is a real budget concern.

    ``--skip-trust`` is required because the gemini CLI refuses to run
    non-interactively in directories that are not in its trusted-folders list,
    and our throwaway tmpdir is by definition not trusted. The flag bypasses
    that gate; the security implication is that the CLI will not load any
    ambient files from cwd — which is exactly what we want anyway (empty
    tmpdir, no project workspace).

    Raises ``RuntimeError`` on non-zero CLI exit and ``ValueError`` on
    malformed JSON output. The caller (``_call_gemini_with_backoff``) catches
    both and applies retry/backoff policy.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(  # noqa: S603 — binary path is operator-configured
            [
                binary,
                "--skip-trust",
                "-p",
                prompt,
                "-m",
                model,
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"gemini CLI failed (rc={result.returncode}): "
            f"{result.stderr[:500]}"
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"gemini CLI returned non-JSON: {result.stdout[:500]}"
        ) from exc
    response = data.get("response")
    if not isinstance(response, str):
        raise ValueError(
            f"gemini CLI JSON missing 'response' field: {result.stdout[:500]}"
        )
    return response.strip()


def _call_gemini_with_backoff(
    prompt: str,
    *,
    model: str = GEMINI_MODEL,
    binary: str = _GEMINI_BINARY,
    timeout: int = _GEMINI_TIMEOUT_SECONDS,
) -> str:
    """Wrap ``_call_gemini`` with exponential backoff on transient failures.

    Treats ``RuntimeError`` (non-zero rc), ``ValueError`` (bad JSON), and
    ``subprocess.TimeoutExpired`` as transient. Re-raises the last exception
    after exhausting ``API_BACKOFF_MAX_RETRIES`` attempts so the caller can
    decide to skip the slot.
    """
    delay = API_BACKOFF_INITIAL
    last_exc: BaseException | None = None
    for attempt in range(1, API_BACKOFF_MAX_RETRIES + 1):
        try:
            return _call_gemini(
                prompt,
                model=model,
                binary=binary,
                timeout=timeout,
            )
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            last_exc = exc
            logger.warning(
                "gemini CLI error (attempt %d/%d): %s — sleeping %.1fs",
                attempt,
                API_BACKOFF_MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, API_BACKOFF_CAP)
    assert last_exc is not None
    raise last_exc


def generate_paraphrases(
    base_prompt: str,
    n: int = 10,
    *,
    model: str = GEMINI_MODEL,
    binary: str = _GEMINI_BINARY,
) -> list[str]:
    """Call 1: ask Gemini for ``n`` paraphrases of ``base_prompt``."""
    instruction = _PARAPHRASE_INSTRUCTION.format(n=n, base=base_prompt)
    raw = _call_gemini_with_backoff(instruction, model=model, binary=binary)
    return _parse_numbered_list(raw)


def generate_responses(
    system_prompt: str,
    user_prompt: str,
    n: int = 5,
    *,
    model: str = GEMINI_MODEL,
    binary: str = _GEMINI_BINARY,
) -> list[str]:
    """Call 2: ask Gemini for ``n`` Lumi-voice responses to ``user_prompt``."""
    instruction = _RESPONSE_INSTRUCTION.format(
        n=n,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    raw = _call_gemini_with_backoff(instruction, model=model, binary=binary)
    return _parse_numbered_list(raw)


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------


SlotKey = tuple[str, str, int, int]


def _slot_key(
    prompt_id: str,
    sys_prompt_id: str,
    paraphrase_idx: int,
    response_idx: int,
) -> SlotKey:
    return (prompt_id, sys_prompt_id, paraphrase_idx, response_idx)


def _key_to_list(key: SlotKey) -> list[Any]:
    return [key[0], key[1], int(key[2]), int(key[3])]


def _list_to_key(seq: list[Any]) -> SlotKey:
    return (str(seq[0]), str(seq[1]), int(seq[2]), int(seq[3]))


def default_accepted_keys_path(output: Path) -> Path:
    return output.with_name(output.stem + ".accepted_keys.json")


def default_partial_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".partial")


def load_accepted_keys(keys_path: Path) -> set[SlotKey]:
    if not keys_path.exists():
        return set()
    try:
        data = json.loads(keys_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(
            "accepted_keys file at %s is corrupt — starting fresh", keys_path
        )
        return set()

    raw_keys: list[Any]
    if isinstance(data, list):
        # New wire format: top-level list of [pid, sid, p_idx, r_idx] OR a
        # pre-seeded list of dicts from tests.
        raw_keys = data
    elif isinstance(data, dict):
        # Legacy / wrapped form: {"keys": [...]}.
        raw_keys = data.get("keys", [])
    else:
        logger.warning(
            "accepted_keys file at %s has unexpected shape — starting fresh",
            keys_path,
        )
        return set()

    out: set[SlotKey] = set()
    for k in raw_keys:
        if isinstance(k, list) and len(k) == 4:
            out.add(_list_to_key(k))
        elif isinstance(k, dict):
            try:
                out.add(
                    (
                        str(k["prompt_id"]),
                        str(k["sys_prompt_id"]),
                        int(k["paraphrase_idx"]),
                        int(k["response_idx"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping malformed accepted-key entry: %r", k)
    return out


def save_accepted_keys(keys_path: Path, accepted: set[SlotKey]) -> None:
    """Atomic write: tmp + os.replace.

    Wire format: a top-level JSON list of [prompt_id, sys_prompt_id,
    paraphrase_idx, response_idx] entries. The list shape is chosen so a naive
    ``len(json.loads(text))`` returns the slot count without unwrapping.
    """
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = keys_path.with_suffix(keys_path.suffix + ".tmp")
    payload = [_key_to_list(k) for k in sorted(accepted)]
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, keys_path)


def append_partial_record(partial_path: Path, record: dict[str, Any]) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def finalize_output(output: Path, partial_path: Path | None = None) -> int:
    """Rename .partial → final. Returns the number of records in the final file."""
    partial = partial_path if partial_path is not None else default_partial_path(output)
    if not partial.exists():
        logger.warning("No .partial file at %s — nothing to finalize", partial)
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output)
    with output.open("r", encoding="utf-8") as fh:
        count = sum(1 for _ in fh)
    return count


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def make_record(
    system_prompt: str,
    user_prompt: str,
    assistant: str,
    category: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "category": category,
        "tools": None,
        "source": SOURCE_TAG,
    }


# ---------------------------------------------------------------------------
# Slot iteration
# ---------------------------------------------------------------------------


def iter_slots(
    pools: dict[str, tuple[PromptTriple, ...]],
    sys_prompt_ids: tuple[str, ...],
    paraphrases_per_prompt: int,
    responses_per_paraphrase: int,
) -> list[tuple[str, str, str, str, int, int]]:
    """Enumerate every (category, base, prompt_id, sys_id, p_idx, r_idx) slot.

    Returned as a list (not a generator) so the orchestration loop can index
    deterministically and so resume logic can compute remaining work cheaply.
    """
    slots: list[tuple[str, str, str, str, int, int]] = []
    for triples in pools.values():
        for category, base, prompt_id in triples:
            for sys_id in sys_prompt_ids:
                for p_idx in range(paraphrases_per_prompt):
                    for r_idx in range(responses_per_paraphrase):
                        slots.append((category, base, prompt_id, sys_id, p_idx, r_idx))
    return slots


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------


def _generate_dry_run_response(slot_idx: int) -> str:
    """Deterministic, validator-passing response keyed off slot index."""
    return _DRY_RUN_FIXTURE_RESPONSES[slot_idx % len(_DRY_RUN_FIXTURE_RESPONSES)]


def run_generation(
    *,
    target_count: int,
    output_dir: Path | None = None,
    keys_file: Path | None = None,
    partial_file: Path | None = None,
    pools: dict[str, tuple[PromptTriple, ...]] | None = None,
    sys_prompt_ids: tuple[str, ...] | None = None,
    paraphrases_per_prompt: int = 10,
    responses_per_paraphrase: int = 5,
    max_retries: int = MAX_RETRIES_PER_SLOT,
    accepted_keys: set[SlotKey] | None = None,
    opener_counter: collections.Counter[str] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    model: str = GEMINI_MODEL,
    gemini_binary: str = _GEMINI_BINARY,
) -> tuple[int, collections.Counter[str]]:
    """Run the generation loop. Returns (n_accepted_this_run, reject_counter).

    Two calling conventions are supported:

    1. Library / CLI style: caller supplies pre-built ``pools``, ``sys_prompt_ids``,
       ``accepted_keys``, ``opener_counter`` and explicit ``output_dir``/
       ``keys_file``/``partial_file``. Used by ``main()``.

    2. Test style: caller supplies just ``target_count``, ``output_dir``,
       ``keys_file``, ``partial_file`` (and optionally ``resume=True``).
       Defaults are filled in (full prompt pools, all 4 system-prompt
       variants, fresh counters, etc.) and the function creates/loads
       checkpoint state itself.
    """
    if pools is None:
        pools = PROMPT_POOLS
    if sys_prompt_ids is None:
        sys_prompt_ids = tuple(SYSTEM_PROMPTS.keys())
    if output_dir is None:
        output_dir = Path("data/finetune")
    output_dir.mkdir(parents=True, exist_ok=True)
    if keys_file is None:
        keys_file = output_dir / "synthetic_v4.accepted_keys.json"
    if partial_file is None:
        partial_file = output_dir / "synthetic_v4.jsonl.partial"

    if accepted_keys is None:
        accepted_keys = load_accepted_keys(keys_file) if resume else set()
    if opener_counter is None:
        opener_counter = collections.Counter()
        if resume and partial_file.exists():
            with partial_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msgs = rec.get("messages", [])
                    if msgs:
                        opener_counter[
                            _first_three_token_opener(msgs[-1].get("content", ""))
                        ] += 1
    slots = iter_slots(
        pools,
        sys_prompt_ids,
        paraphrases_per_prompt,
        responses_per_paraphrase,
    )

    reject_counter: collections.Counter[str] = collections.Counter()
    accepted_this_run = 0

    # Group slots by (prompt_id, sys_id) so we can batch the paraphrase + response
    # CLI calls efficiently — one paraphrase call per (prompt × sys) pair, one
    # response call per paraphrase.
    grouped: dict[tuple[str, str], list[tuple[str, str, int, int]]] = (
        collections.defaultdict(list)
    )
    base_for_id: dict[str, tuple[str, str]] = {}
    for category, base, prompt_id, sys_id, p_idx, r_idx in slots:
        grouped[(prompt_id, sys_id)].append((category, base, p_idx, r_idx))
        base_for_id[prompt_id] = (category, base)

    total_accepted = len(accepted_keys)
    logger.info(
        "Starting generation: target=%d, already_accepted=%d, slots_remaining=%d",
        target_count,
        total_accepted,
        max(0, len(slots) - total_accepted),
    )

    for (prompt_id, sys_id), slot_list in grouped.items():
        if total_accepted >= target_count:
            logger.info("Target count %d reached — stopping", target_count)
            break

        category, base = base_for_id[prompt_id]
        system_prompt = SYSTEM_PROMPTS[sys_id]

        paraphrase_indices = sorted({p_idx for _, _, p_idx, _ in slot_list})

        # Skip the paraphrase CLI call entirely if every slot under this
        # (prompt, sys) is already accepted.
        unresolved = any(
            _slot_key(prompt_id, sys_id, p_idx, r_idx) not in accepted_keys
            for _, _, p_idx, r_idx in slot_list
        )
        if not unresolved:
            continue

        if dry_run:
            paraphrases = [f"{base} (paraphrase {i})" for i in paraphrase_indices]
        else:
            try:
                paraphrases = generate_paraphrases(
                    base,
                    n=max(paraphrase_indices) + 1,
                    model=model,
                    binary=gemini_binary,
                )
            except Exception as exc:
                logger.error(
                    "Paraphrase generation failed for %s (sys=%s): %s — skipping group",
                    prompt_id,
                    sys_id,
                    exc,
                )
                reject_counter["paraphrase_api_failure"] += len(slot_list)
                continue

        if len(paraphrases) < max(paraphrase_indices) + 1:
            logger.warning(
                "Got %d paraphrases for %s but needed %d — truncating slot list",
                len(paraphrases),
                prompt_id,
                max(paraphrase_indices) + 1,
            )

        # Now iterate over paraphrase indices needed.
        responses_per_p = max(
            (r_idx for _, _, p_idx, r_idx in slot_list), default=-1
        ) + 1

        for p_idx in paraphrase_indices:
            if total_accepted >= target_count:
                break
            if p_idx >= len(paraphrases):
                continue
            paraphrase = paraphrases[p_idx]
            unresolved_r = [
                r_idx
                for _, _, pp, r_idx in slot_list
                if pp == p_idx
                and _slot_key(prompt_id, sys_id, p_idx, r_idx) not in accepted_keys
            ]
            if not unresolved_r:
                continue

            attempts_left = max_retries
            accepted_r_idxs: set[int] = set()
            while attempts_left > 0 and len(accepted_r_idxs) < len(unresolved_r):
                attempts_left -= 1
                slot_idx_for_fixture = (
                    hash((prompt_id, sys_id, p_idx)) & 0x7FFFFFFF
                ) + len(accepted_r_idxs)
                if dry_run:
                    candidate_pool = [
                        _generate_dry_run_response(slot_idx_for_fixture + k)
                        for k in range(responses_per_p)
                    ]
                else:
                    try:
                        candidate_pool = generate_responses(
                            system_prompt,
                            paraphrase,
                            n=responses_per_p,
                            model=model,
                            binary=gemini_binary,
                        )
                    except Exception as exc:
                        logger.error(
                            "Response generation failed for %s/%s/p%d: %s",
                            prompt_id,
                            sys_id,
                            p_idx,
                            exc,
                        )
                        reject_counter["response_api_failure"] += 1
                        break

                for r_idx in unresolved_r:
                    if r_idx in accepted_r_idxs:
                        continue
                    if total_accepted >= target_count:
                        break
                    if r_idx >= len(candidate_pool):
                        continue
                    candidate = candidate_pool[r_idx]
                    if not isinstance(candidate, str) or not candidate.strip():
                        reject_counter["empty_or_nonstring"] += 1
                        continue

                    ok, reason = validate_response(candidate, opener_counter)
                    if not ok:
                        reject_counter[reason] += 1
                        logger.debug(
                            "REJECT %s/%s/p%d/r%d (%s): %s",
                            prompt_id,
                            sys_id,
                            p_idx,
                            r_idx,
                            reason,
                            candidate[:80],
                        )
                        continue

                    record = make_record(system_prompt, paraphrase, candidate, category)
                    append_partial_record(partial_file, record)
                    key = _slot_key(prompt_id, sys_id, p_idx, r_idx)
                    accepted_keys.add(key)
                    save_accepted_keys(keys_file, accepted_keys)
                    accepted_r_idxs.add(r_idx)
                    accepted_this_run += 1
                    total_accepted += 1
                    logger.info(
                        "ACCEPT [%d/%d] %s/%s/p%d/r%d (%s)",
                        total_accepted,
                        target_count,
                        prompt_id,
                        sys_id,
                        p_idx,
                        r_idx,
                        category,
                    )
            if len(accepted_r_idxs) < len(unresolved_r):
                missed = len(unresolved_r) - len(accepted_r_idxs)
                reject_counter["budget_exhausted"] += missed

    return accepted_this_run, reject_counter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini-distilled warm-refusal SFT corpus generator for Lumi persona v2.4.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5000,
        help="Target accepted-record count (default: 5000; capped at 10000).",
    )
    parser.add_argument(
        "--system-prompt-variants",
        nargs="+",
        default=list(SYSTEM_PROMPTS.keys()),
        choices=list(SYSTEM_PROMPTS.keys()),
        help="Which system-prompt variants to use (default: all 4).",
    )
    parser.add_argument(
        "--prompt-pool",
        default="all",
        choices=("all", *PROMPT_POOLS.keys()),
        help="Which prompt pool to use (default: all).",
    )
    parser.add_argument(
        "--paraphrases-per-prompt",
        type=int,
        default=10,
        help="N paraphrases per base prompt (default: 10).",
    )
    parser.add_argument(
        "--responses-per-paraphrase",
        type=int,
        default=5,
        help="M Lumi responses per paraphrase (default: 5).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from accepted_keys.json (skip already-accepted slots).",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Rename .partial → final and exit; no CLI calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No CLI calls; uses a hardcoded validator-passing fixture response.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_PER_SLOT,
        help=f"Per-slot reroll budget (default: {MAX_RETRIES_PER_SLOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/finetune/synthetic_v4.jsonl"),
        help="Output JSONL path (default: data/finetune/synthetic_v4.jsonl).",
    )
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=None,
        help="Override accepted_keys JSON path (default: <output>.accepted_keys.json).",
    )
    parser.add_argument(
        "--partial-file",
        type=Path,
        default=None,
        help="Override .partial file path (default: <output>.partial).",
    )
    parser.add_argument(
        "--model",
        default=GEMINI_MODEL,
        help=f"Gemini model to invoke (default: {GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--gemini-binary",
        default=_GEMINI_BINARY,
        help=f"Path to the gemini CLI binary (default: {_GEMINI_BINARY}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(message)s",
    )

    if args.target_count <= 0:
        logger.error("--target-count must be > 0")
        return 2
    target_count = min(args.target_count, 10000)
    if target_count != args.target_count:
        logger.warning("Capping --target-count at 10000 (was %d)", args.target_count)

    output: Path = args.output
    keys_file: Path = (
        args.keys_file
        if args.keys_file is not None
        else default_accepted_keys_path(output)
    )
    partial_file: Path = (
        args.partial_file
        if args.partial_file is not None
        else default_partial_path(output)
    )
    output_dir: Path = output.parent

    if args.finalize:
        count = finalize_output(output, partial_path=partial_file)
        logger.info("Finalize: %d records in %s", count, output)
        return 0

    # Pick pool(s)
    if args.prompt_pool == "all":
        pools = PROMPT_POOLS
    else:
        pools = {args.prompt_pool: PROMPT_POOLS[args.prompt_pool]}

    sys_prompt_ids: tuple[str, ...] = tuple(args.system_prompt_variants)

    accepted_keys: set[SlotKey] = (
        load_accepted_keys(keys_file) if args.resume else set()
    )
    if args.resume:
        logger.info(
            "Resuming with %d previously-accepted slots", len(accepted_keys)
        )
    else:
        # Fresh run: clear .partial + accepted_keys to avoid stale data.
        if partial_file.exists():
            logger.info("Removing stale .partial: %s", partial_file)
            partial_file.unlink()
        if keys_file.exists():
            logger.info("Removing stale accepted_keys: %s", keys_file)
            keys_file.unlink()

    opener_counter: collections.Counter[str] = collections.Counter()
    # Rebuild opener counter from any pre-existing .partial so opener diversity
    # is preserved across --resume.
    if args.resume and partial_file.exists():
        with partial_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = rec.get("messages", [])
                if msgs:
                    opener_counter[
                        _first_three_token_opener(msgs[-1].get("content", ""))
                    ] += 1

    if not args.dry_run:
        # Validate the binary exists / is executable so a typo in
        # --gemini-binary fails fast instead of mid-run.
        if not os.path.isfile(args.gemini_binary) or not os.access(
            args.gemini_binary, os.X_OK
        ):
            logger.error(
                "gemini binary not found or not executable: %s",
                args.gemini_binary,
            )
            return 3
        logger.info(
            "Using gemini CLI at %s (model=%s)",
            args.gemini_binary,
            args.model,
        )
    else:
        logger.info("DRY RUN — no CLI calls; using hardcoded fixture responses")

    try:
        n_accepted, reject_counter = run_generation(
            target_count=target_count,
            output_dir=output_dir,
            keys_file=keys_file,
            partial_file=partial_file,
            pools=pools,
            sys_prompt_ids=sys_prompt_ids,
            paraphrases_per_prompt=args.paraphrases_per_prompt,
            responses_per_paraphrase=args.responses_per_paraphrase,
            max_retries=args.max_retries,
            accepted_keys=accepted_keys,
            opener_counter=opener_counter,
            dry_run=args.dry_run,
            resume=args.resume,
            model=args.model,
            gemini_binary=args.gemini_binary,
        )
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user — checkpoint preserved at %s and %s",
            partial_file,
            keys_file,
        )
        return 130

    # Summary
    total_accepted = len(accepted_keys)
    logger.info("=" * 60)
    logger.info("Generation complete.")
    logger.info("  Accepted this run : %d", n_accepted)
    logger.info("  Accepted total    : %d", total_accepted)
    logger.info("  Target            : %d", target_count)
    logger.info("  Unique openers    : %d", len(opener_counter))
    if reject_counter:
        logger.info("  Rejections by reason:")
        for reason, cnt in reject_counter.most_common():
            logger.info("    %-40s %d", reason, cnt)

    if total_accepted >= target_count:
        count = finalize_output(output, partial_path=partial_file)
        logger.info("Wrote %d records to %s", count, output)
        return 0

    logger.warning(
        "Target %d not reached (accepted %d). Partial file preserved at %s",
        target_count,
        total_accepted,
        partial_file,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
