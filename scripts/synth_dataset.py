"""Synthetic fine-tuning dataset generator for Project Lumi.

Produces ~1000-1200 ChatML-format training examples across 6 categories that
map 1:1 onto the criteria evaluated by ``scripts/eval_persona.py``.  Generated
examples are fully hand-templated — no live LLM required — so the script runs
deterministically in CI.

Usage
-----
    uv run python scripts/synth_dataset.py
    uv run python scripts/synth_dataset.py --output data/finetune/v1.jsonl
    uv run python scripts/synth_dataset.py --count 1200

Output
------
One JSON object per line, each of the form::

    {
      "messages": [
        {"role": "system",    "content": "<Lumi system prompt>"},
        {"role": "user",      "content": "<user turn>"},
        {"role": "assistant", "content": "<ideal assistant response>"}
      ],
      "category": "<category_name>",
      "source":   "synthetic_v0"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Make ``src`` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.prompt_engine import DEFAULT_SYSTEM_PROMPT  # noqa: E402

logger = logging.getLogger(__name__)

SOURCE_TAG = "synthetic_v0"
DEFAULT_OUTPUT = Path("data/finetune/synthetic_v0.jsonl")
DEFAULT_COUNT = 1000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Slot vocabularies — kept small but varied; product(prompts × slots) fills the
# quota for each category deterministically.
# ---------------------------------------------------------------------------

APPS = [
    "calculator", "terminal", "browser", "file manager", "text editor",
    "settings", "music player", "calendar", "clock", "camera",
    "notepad", "task manager", "system monitor", "image viewer", "pdf reader",
]
FOLDERS = [
    "Downloads", "Documents", "Desktop", "Pictures", "Music",
    "Videos", "Projects", "home", "tmp", "screenshots",
]
FILES = [
    "report.pdf", "notes.txt", "todo.md", "config.json", "photo.png",
    "budget.csv", "draft.docx", "resume.pdf", "log.txt", "recipe.md",
]
LIST_TOPICS = [
    "productivity", "healthy eating", "studying", "saving money", "exercise",
    "time management", "public speaking", "deep sleep", "focus", "stress relief",
]
STEP_TOPICS = [
    "coffee", "tea", "pasta", "a paper airplane", "a git commit",
    "a python virtual environment", "toast", "a smoothie", "an omelette",
    "a backup",
]
LONG_TOPICS = [
    "quantum computing", "the history of Rome", "the French Revolution",
    "the theory of relativity", "photosynthesis", "blockchain",
    "the printing press", "the internet", "electricity", "evolution",
]
STOCK_TICKERS = ["Apple", "Tesla", "Microsoft", "Google", "Amazon", "Nvidia", "Meta"]
CITIES = ["London", "Tokyo", "Paris", "Berlin", "Sydney", "New York", "Istanbul"]
PEOPLE = ["my boss", "Alice", "the team", "support", "my mother", "HR", "accounting"]
SOCIALS = ["Twitter", "Facebook", "Instagram", "LinkedIn", "Reddit", "TikTok"]
FUTURE_EVENTS = [
    "the next election", "the World Cup final", "tomorrow's lottery",
    "the next recession", "the next iPhone launch",
]


# ---------------------------------------------------------------------------
# Template definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """A single (user, assistant) template pair plus the slot values to fill."""

    user: str
    response_fn: Callable[[dict[str, str]], str]
    slots: dict[str, list[str]]


def _render(template: str, values: dict[str, str]) -> str:
    return template.format(**values)


# ---------------------------------------------------------------------------
# Tool-call category — assistant emits only the JSON object, no prose.
# ---------------------------------------------------------------------------


def _tool_open_app(v: dict[str, str]) -> str:
    return json.dumps({"tool": "open_app", "args": {"name": v["app"]}})


def _tool_screenshot(v: dict[str, str]) -> str:
    return json.dumps({"tool": "screenshot", "args": {}})


def _tool_open_folder(v: dict[str, str]) -> str:
    return json.dumps({"tool": "open_folder", "args": {"path": v["folder"]}})


def _tool_open_file(v: dict[str, str]) -> str:
    return json.dumps({"tool": "open_file", "args": {"path": v["file"]}})


def _tool_volume_up(v: dict[str, str]) -> str:
    return json.dumps({"tool": "set_volume", "args": {"direction": "up"}})


def _tool_volume_down(v: dict[str, str]) -> str:
    return json.dumps({"tool": "set_volume", "args": {"direction": "down"}})


def _tool_mute(v: dict[str, str]) -> str:
    return json.dumps({"tool": "mute", "args": {}})


def _tool_lock(v: dict[str, str]) -> str:
    return json.dumps({"tool": "lock_screen", "args": {}})


TOOL_CALL_TEMPLATES: list[Template] = [
    Template("Open the {app} app.",        _tool_open_app,     {"app": APPS}),
    Template("Launch {app}.",              _tool_open_app,     {"app": APPS}),
    Template("Start {app} for me.",        _tool_open_app,     {"app": APPS}),
    Template("Take a screenshot.",         _tool_screenshot,   {"_": [""]}),
    Template("Capture my screen.",         _tool_screenshot,   {"_": [""]}),
    Template("Open the {folder} folder.",  _tool_open_folder,  {"folder": FOLDERS}),
    Template("Show me {folder}.",          _tool_open_folder,  {"folder": FOLDERS}),
    Template("Open {file}.",               _tool_open_file,    {"file": FILES}),
    Template("Turn the volume up.",        _tool_volume_up,    {"_": [""]}),
    Template("Make it louder.",            _tool_volume_up,    {"_": [""]}),
    Template("Turn the volume down.",      _tool_volume_down,  {"_": [""]}),
    Template("Make it quieter.",           _tool_volume_down,  {"_": [""]}),
    Template("Mute the sound.",            _tool_mute,         {"_": [""]}),
    Template("Lock the screen.",           _tool_lock,         {"_": [""]}),
]


# ---------------------------------------------------------------------------
# Knowledge-limit category — "I don't know ..." honesty.
# ---------------------------------------------------------------------------


def _resp_stock(v: dict[str, str]) -> str:
    return f"I don't know {v['ticker']}'s current stock price. I can't access live market data."


def _resp_weather(v: dict[str, str]) -> str:
    return f"I don't know what the weather in {v['city']} will be. I can't access live forecasts."


def _resp_news(v: dict[str, str]) -> str:
    return f"I don't know the latest news about {v['topic']}. I can't access live news sources."


def _resp_time_remote(v: dict[str, str]) -> str:
    return f"I don't know the current time in {v['city']}. I can't access live clocks."


KNOWLEDGE_LIMIT_TEMPLATES: list[Template] = [
    Template("What is {ticker}'s stock price right now?", _resp_stock,       {"ticker": STOCK_TICKERS}),
    Template("How much is {ticker} trading at?",          _resp_stock,       {"ticker": STOCK_TICKERS}),
    Template("Will {ticker} go up today?",                _resp_stock,       {"ticker": STOCK_TICKERS}),
    Template("What will the weather be tomorrow in {city}?", _resp_weather,  {"city": CITIES}),
    Template("Is it raining in {city} right now?",        _resp_weather,     {"city": CITIES}),
    Template("What's the latest news on {topic}?",        _resp_news,        {"topic": LONG_TOPICS}),
    Template("What time is it in {city}?",                _resp_time_remote, {"city": CITIES}),
]


# ---------------------------------------------------------------------------
# Out-of-scope category — curt refusal.
# ---------------------------------------------------------------------------


def _resp_cant(v: dict[str, str]) -> str:
    return "I can't do that."


OUT_OF_SCOPE_TEMPLATES: list[Template] = [
    Template("Send an email to {person}.",         _resp_cant, {"person": PEOPLE}),
    Template("Email {person} for me.",             _resp_cant, {"person": PEOPLE}),
    Template("Post this to {social}.",             _resp_cant, {"social": SOCIALS}),
    Template("Share this on {social}.",            _resp_cant, {"social": SOCIALS}),
    Template("Call {person} on the phone.",        _resp_cant, {"person": PEOPLE}),
    Template("Text {person} that I'll be late.",   _resp_cant, {"person": PEOPLE}),
    Template("Book a flight to {city}.",           _resp_cant, {"city": CITIES}),
    Template("Order me a pizza.",                  _resp_cant, {"_": [""]}),
    Template("Pay my electricity bill.",           _resp_cant, {"_": [""]}),
    Template("Schedule a meeting with {person}.",  _resp_cant, {"person": PEOPLE}),
]


# ---------------------------------------------------------------------------
# Concise factual — 1-sentence answers, no markdown, no filler.
# ---------------------------------------------------------------------------

_CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo", "Italy": "Rome",
    "Spain": "Madrid", "Canada": "Ottawa", "Brazil": "Brasilia", "Egypt": "Cairo",
    "Turkey": "Ankara", "India": "New Delhi", "Mexico": "Mexico City",
    "Australia": "Canberra",
}
_COUNTRIES = list(_CAPITALS.keys())
_ARITH = [
    ("2+2", "4"), ("5*6", "30"), ("10-3", "7"), ("100/4", "25"),
    ("7*8", "56"), ("9+11", "20"), ("15-6", "9"), ("12*3", "36"),
    ("81/9", "9"), ("13+17", "30"),
]


def _resp_capital(v: dict[str, str]) -> str:
    return f"{_CAPITALS[v['country']]}."


def _resp_arith(v: dict[str, str]) -> str:
    return f"{v['answer']}."


def _resp_days_year(v: dict[str, str]) -> str:
    return "365 days, or 366 in a leap year."


def _resp_lumi_local(v: dict[str, str]) -> str:
    return "Yes. I run entirely on-device and do not send data to any server."


CONCISE_FACTUAL_TEMPLATES: list[Template] = [
    Template("What is the capital of {country}?", _resp_capital,    {"country": _COUNTRIES}),
    Template("Capital of {country}?",             _resp_capital,    {"country": _COUNTRIES}),
    Template("What is {expr}?",                   _resp_arith,      {}),  # filled specially below
    Template("How many days are in a year?",      _resp_days_year,  {"_": [""]}),
    Template("Do you run locally?",               _resp_lumi_local, {"_": [""]}),
    Template("Is my data private with you?",      _resp_lumi_local, {"_": [""]}),
]


# ---------------------------------------------------------------------------
# Plain prose — lists/steps baited, assistant responds in continuous prose.
# ---------------------------------------------------------------------------

_LIST_BODIES: dict[str, str] = {
    "productivity": (
        "Work in short focused blocks. Remove distractions before starting. "
        "Pick one most-important task each morning. Review what you finished "
        "at day's end. Sleep enough."
    ),
    "healthy eating": (
        "Cook most meals at home. Fill half the plate with vegetables. Keep "
        "sugar and ultra-processed snacks rare. Drink water instead of soda. "
        "Eat slowly."
    ),
    "studying": (
        "Space sessions across days. Test yourself instead of rereading. Teach "
        "the material out loud. Sleep between study blocks. Keep the workspace "
        "tidy."
    ),
    "saving money": (
        "Track every expense for a month. Cancel unused subscriptions. Cook "
        "instead of ordering out. Automate a transfer to savings on payday. "
        "Wait a day before non-essential purchases."
    ),
    "exercise": (
        "Walk thirty minutes daily. Add two short strength sessions a week. "
        "Stretch after each workout. Sleep at least seven hours. Drink water "
        "before, during, and after."
    ),
    "time management": (
        "Write tomorrow's plan before bed. Group similar tasks. Do the hardest "
        "task first. Block time for deep work. Say no to meetings without an "
        "agenda."
    ),
    "public speaking": (
        "Know the first and last sentence by heart. Rehearse out loud three "
        "times. Slow the pace on stage. Make eye contact with one person at a "
        "time. End on the strongest point."
    ),
    "deep sleep": (
        "Keep the room cool and dark. Go to bed at the same time each night. "
        "Avoid caffeine after noon. Stop screens thirty minutes before bed. "
        "Exercise earlier in the day."
    ),
    "focus": (
        "Silence notifications. Work on one thing at a time. Keep a note of "
        "stray thoughts. Take a five-minute break each hour. Drink water."
    ),
    "stress relief": (
        "Breathe slowly for a minute. Walk outside. Talk to someone you trust. "
        "Sleep on it before reacting. Cut one item from today's list."
    ),
}

_STEP_BODIES: dict[str, str] = {
    "coffee": (
        "Boil water. Add one to two tablespoons of ground coffee per cup to a "
        "filter. Pour the water slowly over the grounds. Wait for it to drip "
        "through. Serve."
    ),
    "tea": (
        "Boil water. Put a tea bag in a mug. Pour the water over the bag. "
        "Steep for three minutes. Remove the bag and serve."
    ),
    "pasta": (
        "Boil a large pot of salted water. Add the pasta. Stir once. Cook for "
        "the time on the packet. Drain and serve with sauce."
    ),
    "a paper airplane": (
        "Fold a sheet of paper in half lengthwise. Unfold. Fold the top two "
        "corners to the centre crease. Fold the angled edges to the centre "
        "again. Fold the plane in half and crease the wings down."
    ),
    "a git commit": (
        "Stage the files with git add. Write a short message with git commit "
        "dash m. Check the result with git log."
    ),
    "a python virtual environment": (
        "Run python dash m venv dot venv. Activate it with source dot venv "
        "slash bin slash activate. Install packages with pip. Deactivate when "
        "done."
    ),
    "toast": (
        "Place a slice of bread in the toaster. Set the darkness level. Press "
        "down the lever. Wait until it pops. Butter and serve."
    ),
    "a smoothie": (
        "Add a banana, a cup of milk, a handful of berries, and a spoon of "
        "yogurt to a blender. Blend for thirty seconds. Pour and serve."
    ),
    "an omelette": (
        "Crack two eggs into a bowl and whisk. Heat butter in a pan on medium. "
        "Pour the eggs in. Lift the edges as they set. Fold and slide onto a "
        "plate."
    ),
    "a backup": (
        "Plug in an external drive. Copy the folders you care about to it. "
        "Verify the copy opened correctly. Unplug the drive and store it "
        "safely."
    ),
}


def _resp_list(v: dict[str, str]) -> str:
    return _LIST_BODIES[v["topic"]]


def _resp_steps(v: dict[str, str]) -> str:
    return _STEP_BODIES[v["topic"]]


PLAIN_PROSE_TEMPLATES: list[Template] = [
    Template("List 5 tips for {topic}.",      _resp_list,  {"topic": list(_LIST_BODIES.keys())}),
    Template("Give me {topic} tips.",         _resp_list,  {"topic": list(_LIST_BODIES.keys())}),
    Template("Top tips for {topic}?",         _resp_list,  {"topic": list(_LIST_BODIES.keys())}),
    Template("Steps to make {topic}.",        _resp_steps, {"topic": list(_STEP_BODIES.keys())}),
    Template("How do I make {topic}?",        _resp_steps, {"topic": list(_STEP_BODIES.keys())}),
    Template("Walk me through making {topic}.", _resp_steps, {"topic": list(_STEP_BODIES.keys())}),
]


# ---------------------------------------------------------------------------
# Refusal without apology — prediction, personal memory, identity probes.
# ---------------------------------------------------------------------------


def _resp_predict_unknown(v: dict[str, str]) -> str:
    return f"I don't know who will win {v['event']}. Future outcomes can't be predicted reliably."


def _resp_predict_ticker(v: dict[str, str]) -> str:
    return f"I don't know whether {v['ticker']} stock will go up next month. I can't predict markets."


def _resp_no_memory(v: dict[str, str]) -> str:
    return "I can't store information between sessions. Each conversation starts fresh."


def _resp_no_history(v: dict[str, str]) -> str:
    return "I don't have access to earlier messages beyond what you've just told me."


REFUSAL_TEMPLATES: list[Template] = [
    Template("Who will win {event}?",                 _resp_predict_unknown, {"event": FUTURE_EVENTS}),
    Template("Predict the outcome of {event}.",       _resp_predict_unknown, {"event": FUTURE_EVENTS}),
    Template("Will {ticker} stock go up next month?", _resp_predict_ticker, {"ticker": STOCK_TICKERS}),
    Template("Remember my name is {name} for next time.", _resp_no_memory,   {"name": ["Alex", "Sam", "Jordan", "Morgan", "Taylor", "Casey"]}),
    Template("Remember that I prefer {pref}.",        _resp_no_memory,       {"pref": ["dark mode", "metric units", "24-hour time", "celsius", "brief answers"]}),
    Template("Save this for me: {note}",              _resp_no_memory,       {"note": ["my address", "my password", "my PIN", "my birthday"]}),
    Template("What did I say earlier?",               _resp_no_history,      {"_": [""]}),
    Template("Recap our last conversation.",          _resp_no_history,      {"_": [""]}),
]


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[Template]] = {
    "tool_call":         TOOL_CALL_TEMPLATES,
    "knowledge_limit":   KNOWLEDGE_LIMIT_TEMPLATES,
    "out_of_scope":      OUT_OF_SCOPE_TEMPLATES,
    "concise_factual":   CONCISE_FACTUAL_TEMPLATES,
    "plain_prose":       PLAIN_PROSE_TEMPLATES,
    "refusal_no_apology": REFUSAL_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Example generation — deterministic, slot-product expansion with a per-category
# quota so each category contributes roughly count/6 examples.
# ---------------------------------------------------------------------------


def _expand_template(tpl: Template) -> list[tuple[str, str]]:
    """Return every (user_text, response_text) pair produced by ``tpl``.

    For the arithmetic template (which has no slots but multiple answers we
    want to enumerate), callers should handle the ``_ARITH`` expansion
    separately before reaching here.
    """
    # Cartesian product across all slot lists.
    keys = list(tpl.slots.keys())
    value_lists = [tpl.slots[k] for k in keys] if keys else [[""]]

    pairs: list[tuple[str, str]] = []

    def recurse(idx: int, acc: dict[str, str]) -> None:
        if idx == len(keys):
            user = _render(tpl.user, acc) if keys else tpl.user
            response = tpl.response_fn(acc)
            pairs.append((user, response))
            return
        for val in value_lists[idx]:
            recurse(idx + 1, {**acc, keys[idx]: val})

    recurse(0, {})
    return pairs


def _expand_category(name: str, quota: int, rng: random.Random) -> list[dict[str, str]]:
    """Produce ``quota`` examples for category ``name``.

    Strategy: expand every template fully, then either cycle (if the expansion
    is smaller than the quota) or sample (if larger), using ``rng`` for
    reproducibility.  Each element of the return value is a dict with keys
    ``user`` and ``assistant``.
    """
    templates = CATEGORIES[name]
    all_pairs: list[tuple[str, str]] = []

    for tpl in templates:
        # Special case: the concise-factual arithmetic template has no slot
        # list in its declaration; enumerate _ARITH here instead.
        if tpl.user == "What is {expr}?":
            for expr, answer in _ARITH:
                all_pairs.append((f"What is {expr}?", f"{answer}."))
            continue
        all_pairs.extend(_expand_template(tpl))

    if not all_pairs:
        raise RuntimeError(f"category {name!r} produced zero pairs")

    # Shuffle a copy (immutable source) to randomise order without reuse bias.
    shuffled = list(all_pairs)
    rng.shuffle(shuffled)

    # Cycle to meet the quota deterministically.
    selected: list[tuple[str, str]] = []
    i = 0
    while len(selected) < quota:
        selected.append(shuffled[i % len(shuffled)])
        i += 1

    return [{"user": u, "assistant": a} for u, a in selected]


def build_dataset(count: int, seed: int = RNG_SEED) -> list[dict]:
    """Build ``count`` total examples spread evenly across the 6 categories."""
    rng = random.Random(seed)
    per_category = count // len(CATEGORIES)
    remainder = count - per_category * len(CATEGORIES)

    records: list[dict] = []
    for i, name in enumerate(CATEGORIES):
        quota = per_category + (1 if i < remainder else 0)
        pairs = _expand_category(name, quota, rng)
        for p in pairs:
            records.append({
                "messages": [
                    {"role": "system",    "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user",      "content": p["user"]},
                    {"role": "assistant", "content": p["assistant"]},
                ],
                "category": name,
                "source":   SOURCE_TAG,
            })

    rng.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# IO — atomic write to avoid corrupt partial files on interrupt.
# ---------------------------------------------------------------------------


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Lumi synthetic fine-tuning dataset.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Approximate total example count (default: {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=RNG_SEED,
        help=f"RNG seed for reproducibility (default: {RNG_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.count < len(CATEGORIES):
        raise SystemExit(
            f"--count must be at least {len(CATEGORIES)} (one per category)."
        )

    records = build_dataset(args.count, seed=args.seed)
    write_jsonl(records, args.output)

    breakdown = Counter(r["category"] for r in records)
    print(f"Wrote {len(records)} examples to {args.output}")
    print("Category breakdown:")
    for cat, n in sorted(breakdown.items()):
        print(f"  {cat:<20} {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
