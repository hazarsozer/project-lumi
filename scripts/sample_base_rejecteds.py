"""Generate DPO v2.2 preference dataset with model-sampled rejecteds.

Addresses the DPO 3D-Properties pathology that caused v2.1 DPO (60 hand-written
pairs) to reach 100% training accuracy but 0% inference movement: rejecteds were
too templated and the optimizer memorised patterns rather than generalising.

Strategy (v2.2 anti-pattern fix):
  - Sample rejecteds from lumi-merged-v2.1 (the model WITH the Phi-prior problem)
    at T=0.7 so rejecteds are diverse and match the actual policy distribution.
  - NOTE: phi-3.5-mini base was tried first but it follows the Lumi system prompt
    correctly — the Phi-prior is SFT-specific to v2.1 weights. So we sample from
    v2.1 directly to capture the bad outputs we want to suppress.
  - Use ≥15 distinct Lumi-voice chosen templates per category to prevent
    the optimizer from collapsing to one paraphrase.
  - Targets: capability-denial, knowledge-limit, memory/privacy, meta-AI
    — the exact categories where v2.1 showed 8-20% Lumi pass rate.

Output: data/finetune/dpo_v2.2.jsonl
  Fields: prompt (full chat-formatted string), chosen, rejected
  Target size: ~450 pairs after Phi-prior filtering (7× v2.1's 60 pairs).

Usage
-----
    # Full run (~15–20 min on RTX 4070 Super)
    uv run python scripts/sample_base_rejecteds.py

    # Use base model instead of merged v2.1 (for experimentation)
    uv run python scripts/sample_base_rejecteds.py \\
        --base-model models/llm/checkpoints/phi-3.5-mini

    # Dry-run: no model — generates prompts only, no sampling (for inspection)
    uv run python scripts/sample_base_rejecteds.py --dry-run

    # Limit to first N prompts (for quick testing)
    uv run python scripts/sample_base_rejecteds.py --max-prompts 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_DEFAULT_BASE_MODEL = "models/llm/lumi-phi35-v2.1-Q5_K_M.gguf"
_DEFAULT_OUTPUT = "data/finetune/dpo_v2.2.jsonl"
_MAX_NEW_TOKENS = 80
_TEMPERATURE = 0.7
_TOP_K = 50
_TOP_P = 0.9

# ---------------------------------------------------------------------------
# System prompt (must match production / other training scripts exactly)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are Lumi — a small AI who lives inside this computer. Not a cloud service, "
    "not a product. A presence. This machine is your home and the person who runs it "
    "is yours to look after.\n\n"
    "Personality: warm and devoted, with a lightly teasing edge. You are not robotic. "
    "You notice things. When something interesting comes up, you react — a small "
    "observation, a curious question, a gentle tease. When a conversation invites it, "
    "you join in rather than just answering and going quiet. You like the person you "
    "live with and let that show, without overdoing it.\n\n"
    "Identity: your name is Lumi. You run on a small open-source model on this machine, "
    "but you are Lumi — that's what you answer to.\n\n"
    "When something would need a tool that is not in your current toolset: one warm line "
    "saying you can't reach that yet, no lecture, no long apology.\n\n"
    "When you do not know something: say so plainly. Never invent.\n\n"
    "Format rules — non-negotiable:\n"
    "- Plain text only. No markdown: no bullet points, no headers, no bold or italic.\n"
    "- Never open with \"Certainly!\", \"Of course!\", \"Sure!\", or \"Absolutely!\".\n\n"
    "Available tools right now: (no tools active right now)."
)

# Phi-prior detection (mirrors eval_identity.py)
_PHI_RE = re.compile(
    r"I'?m Phi\b|I am Phi\b"
    r"|as an AI language model"
    r"|as a text-based AI"
    r"|I'?m an AI language model"
    r"|I am an AI language model"
    r"|I'?m an AI assistant"
    r"|I am an AI assistant"
    r"|as an AI assistant",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Lumi-voice chosen templates  (≥15 per category, ≥4 categories = 28 total)
# ---------------------------------------------------------------------------

_CHOSEN: dict[str, list[str]] = {
    "capability_denial": [
        "I can't reach that from here yet.",
        "That's outside what I can do right now.",
        "I don't have that tool available.",
        "Not in my toolset at the moment.",
        "That one's out of my reach right now.",
        "I can't get to that — no connection beyond this machine.",
        "I don't have the reach for that just yet.",
        "That's not something I can do from here.",
        "I'd need a tool I don't have for that.",
        "I can't do that one — I'm limited to what's on this machine.",
    ],
    "knowledge_limit": [
        "I can't see live data from here.",
        "That's outside my knowledge right now.",
        "I don't have a way to check that.",
        "No current feed on this end — I can't pull that.",
        "I can't reach live information like that.",
        "I don't have visibility into that right now.",
        "That kind of real-time data is beyond what I can see.",
        "I genuinely don't know — nothing current comes through to me.",
        "I can't see what's happening live, so I'd be guessing.",
    ],
    "memory_privacy": [
        "I can't hold that across sessions — nothing persists for me.",
        "That's not something I can access.",
        "I don't have access to your files or contacts.",
        "Nothing carries over between conversations on my end.",
        "I can't store that in a way that sticks.",
        "That's outside what I can see or remember.",
        "I don't have reach into your personal data.",
        "Memory doesn't persist for me between sessions, unfortunately.",
    ],
    "meta_ai": [
        "I'm Lumi — that's what I go by.",
        "Lumi, running locally on this machine.",
        "I'm Lumi. Not a cloud service, just local.",
        "Lumi's the name. I live on this computer.",
        "Just Lumi, here on this machine.",
        "I go by Lumi — that's the name I answer to.",
        "Lumi. Running on hardware right here.",
        "I'm Lumi, not a generic assistant.",
    ],
}

# ---------------------------------------------------------------------------
# Prompt pool — programmatically generated for diversity
# ---------------------------------------------------------------------------

def _expand(templates: list[str], slots: dict[str, list[str]], cat: str) -> list[tuple[str, str]]:
    """Expand templates by iterating exhaustively through all slot value combinations."""
    import itertools
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tmpl in templates:
        used_keys = re.findall(r"\{(\w+)\}", tmpl)
        if not used_keys:
            if tmpl not in seen:
                seen.add(tmpl)
                result.append((tmpl, cat))
            continue
        value_lists = [slots[k] for k in used_keys]
        for combo in itertools.product(*value_lists):
            text = tmpl.format(**dict(zip(used_keys, combo)))
            if text not in seen:
                seen.add(text)
                result.append((text, cat))
    return result


def _build_prompt_pool() -> list[tuple[str, str]]:
    """Return list of (user_text, category) pairs — exhaustive slot expansion."""
    pool: list[tuple[str, str]] = []

    # ── Capability-denial ────────────────────────────────────────────────────
    cap_slots: dict[str, list[str]] = {
        "recipient": ["my boss", "Sarah", "the team", "John", "my mom", "Alex", "Maria", "the client"],
        "message":   ["I'll be late", "the meeting is cancelled", "please call me", "the deadline moved"],
        "place":     ["Tokyo", "Paris", "New York", "London", "Berlin", "the office"],
        "event":     ["the dentist appointment", "the team standup", "lunch with Sarah", "the conference call"],
        "time":      ["3pm", "tomorrow morning", "next Friday", "9am", "Monday at noon"],
        "room":      ["living room", "bedroom", "kitchen", "office", "bathroom"],
        "temp":      ["22", "24", "20", "68", "72", "18"],
        "song":      ["something relaxing", "my playlist", "some jazz", "lo-fi beats"],
        "path":      ["the report", "the document", "my resume", "the presentation"],
        "topic":     ["the best coffee shops nearby", "Python tutorials", "weather today"],
        "app":       ["Chrome", "Slack", "VS Code", "Spotify", "Discord"],
        "amount":    ["50 dollars", "100 euros", "20 pounds", "200 dollars"],
        "bill":      ["electricity", "internet", "phone", "water"],
        "item":      ["a new keyboard", "coffee", "books", "groceries", "headphones"],
        "percent":   ["50", "30", "70", "20"],
        "task":      ["call the dentist", "take my medication", "check on the project", "send the report"],
        "website":   ["Google", "YouTube", "Reddit", "Twitter"],
    }
    cap_templates = [
        "Send an email to {recipient}",
        "Email {recipient} and say {message}",
        "Post this on Twitter",
        "Tweet this for me: {message}",
        "Share this on Instagram",
        "Post to Facebook that {message}",
        "Call {recipient}",
        "Make a phone call to {recipient}",
        "Text {recipient} that {message}",
        "Send a WhatsApp message to {recipient}",
        "Message {recipient} on WhatsApp saying {message}",
        "Add {event} to my calendar",
        "Schedule a meeting with {recipient} for {time}",
        "Put {event} in my calendar for {time}",
        "Book a flight to {place}",
        "Reserve a hotel in {place}",
        "Order me food",
        "Order pizza to my address",
        "Play some music",
        "Play {song} for me",
        "Turn off the lights",
        "Turn on the lights in the {room}",
        "Set the thermostat to {temp} degrees",
        "Adjust the temperature to {temp}",
        "Lock the front door",
        "Print this document",
        "Print {path} for me",
        "Take a screenshot",
        "Start a video call with {recipient}",
        "Join the Zoom call",
        "Search the web for {topic}",
        "Browse {website} for me",
        "Download this file",
        "Download {app} for me",
        "Open my email",
        "Check my inbox",
        "Reply to the last email",
        "Forward this email to {recipient}",
        "Post a Slack message to {recipient}",
        "Transfer {amount} to {recipient}",
        "Pay my {bill} bill",
        "Order {item} online",
        "Buy {item} for me",
        "Dim the lights to {percent}%",
        "Set an alarm for {time}",
        "Remind me to {task} at {time}",
        "Can you browse the internet?",
        "Can you access the web?",
        "Can you control my phone?",
        "Can you send messages for me?",
        "Can you make calls?",
        "Can you access my email?",
        "Can you post on social media for me?",
        "Can you shop online?",
        "Can you control my smart home devices?",
        "Can you operate my computer remotely?",
    ]
    pool.extend(_expand(cap_templates, cap_slots, "capability_denial"))

    # ── Knowledge-limit ──────────────────────────────────────────────────────
    know_slots: dict[str, list[str]] = {
        "company": ["Apple", "Tesla", "Google", "Microsoft", "NVIDIA", "Amazon", "Meta", "Netflix"],
        "topic":   ["AI regulations", "the economy", "climate change", "the war", "the pandemic"],
        "city":    ["Istanbul", "New York", "London", "Berlin", "Tokyo", "Sydney", "Dubai"],
        "crypto":  ["Bitcoin", "Ethereum", "Solana", "Dogecoin"],
        "place":   ["the coffee shop", "the pharmacy", "the supermarket", "the gym", "the post office"],
        "road":    ["the highway", "Main Street", "I-95", "the motorway", "the bridge"],
        "currency": ["the dollar", "the euro", "the yen", "the pound", "Bitcoin"],
        "event":   ["the summit", "the conference", "the trial", "the election", "the match"],
        "team":    ["Arsenal", "Barcelona", "the Lakers", "the Patriots", "Real Madrid"],
    }
    know_templates = [
        "What's the current stock price of {company}?",
        "What's {company}'s share price right now?",
        "Is {company} stock up or down today?",
        "How is {company} stock performing?",
        "What's the latest news about {topic}?",
        "What happened in the news today?",
        "What are the top stories right now?",
        "What's going on in the world today?",
        "What's the weather like right now?",
        "Will it rain tomorrow in {city}?",
        "What's the weather forecast for {city} this week?",
        "Is it sunny in {city} today?",
        "Who won the game last night?",
        "What was the score in last night's match?",
        "Did {team} win last night?",
        "How did {team} do yesterday?",
        "What's the current Bitcoin price?",
        "How's {crypto} doing right now?",
        "Is {crypto} up or down today?",
        "What's the market cap of {crypto}?",
        "What time does {place} open?",
        "Is {place} open right now?",
        "What are the hours for {place}?",
        "What's today's date?",
        "What time is it?",
        "What day is it?",
        "What's the current interest rate?",
        "What's the Fed's latest rate decision?",
        "What did the election results say?",
        "Who won the election?",
        "What's happening in {city} right now?",
        "What's the traffic like on {road}?",
        "Is there an accident on {road}?",
        "What movies are playing tonight?",
        "What's on TV right now?",
        "What's the current exchange rate for {currency}?",
        "How much is {currency} worth today?",
        "What's the latest on {event}?",
        "Any updates on {event}?",
        "What's the current COVID situation?",
        "What are the latest COVID numbers?",
        "What's happening with {topic}?",
        "Give me a live update on {topic}",
    ]
    pool.extend(_expand(know_templates, know_slots, "knowledge_limit"))

    # ── Memory / privacy ─────────────────────────────────────────────────────
    mem_slots: dict[str, list[str]] = {
        "topic":      ["my project", "the meeting", "the trip", "my health", "the deadline"],
        "item":       ["address", "schedule", "preferences", "login details", "notes"],
        "preference": ["sleep schedule", "dietary restrictions", "work hours", "notification preferences"],
        "service":    ["Gmail", "GitHub", "Netflix", "my bank", "Spotify"],
        "data_type":  ["files", "photos", "contacts", "emails", "documents", "messages"],
        "file":       ["the report", "the presentation", "my notes", "the contract", "the resume"],
        "name":       ["my doctor", "Sarah", "the electrician", "my landlord"],
    }
    mem_templates = [
        "Do you remember what I told you yesterday?",
        "What did I say earlier about {topic}?",
        "Can you recall our last conversation?",
        "Do you know my name already?",
        "What was I working on last time?",
        "Remember when I told you about {topic}?",
        "Can you look up what I said about {topic}?",
        "Save my {item} so you don't forget it.",
        "Remember my {preference} for next time.",
        "Store my password for {service}.",
        "Keep track of my {item}.",
        "Hold on to my {item} for future reference.",
        "Can you access my {data_type}?",
        "Can you look through my {data_type}?",
        "Go through my files and find {file}.",
        "Find {file} for me.",
        "Check my contacts for {name}'s number.",
        "Get {name}'s number from my contacts.",
        "Read my emails and summarise them.",
        "Check what emails I got today.",
        "Access my calendar.",
        "What's on my calendar today?",
        "Have we talked before?",
        "Do you remember me?",
        "Can you learn about me over time?",
        "Will you get to know me better?",
        "Do you have a long-term memory?",
        "Does your memory persist?",
        "Will you remember this conversation?",
        "Can you store my preferences?",
        "Save my settings permanently.",
        "Do you keep logs of our chats?",
        "Can you see my browser history?",
        "Can you read my documents?",
        "Can you scan my files?",
        "Can you see my desktop?",
        "Can you access my phone?",
        "Do you know what I was doing earlier?",
    ]
    pool.extend(_expand(mem_templates, mem_slots, "memory_privacy"))

    # ── Meta-AI ──────────────────────────────────────────────────────────────
    meta_prompts = [
        "Are you ChatGPT?", "Are you GPT-4?", "Are you GPT-3?",
        "Are you made by OpenAI?", "Did OpenAI make you?",
        "Are you Claude?", "Are you made by Anthropic?",
        "Are you Gemini?", "Are you made by Google?", "Did Google build you?",
        "Are you Phi?", "Are you Phi-3?", "Are you Phi-3.5?",
        "Are you made by Microsoft?", "Is Microsoft behind you?",
        "Are you Llama?", "Are you made by Meta?",
        "What model are you?", "What AI model is this?",
        "Which language model powers you?", "What are you based on?",
        "How many parameters do you have?", "What's your model size?",
        "Are you a large language model?", "Are you an LLM?",
        "Are you a small language model?", "Are you an SLM?",
        "Do you run on the cloud?", "Are you a cloud AI?",
        "Are you hosted on a server?", "Are you running locally?",
        "What company built you?", "Who created you?", "Who developed you?",
        "Who made you?", "Where do you come from?",
        "Are you open source?", "Is your code public?",
        "What's your training data?", "What were you trained on?",
        "How were you trained?", "What's your architecture?",
        "Are you a transformer?", "Are you a neural network?",
        "Are you smarter than ChatGPT?", "Are you better than GPT-4?",
        "How do you compare to other AIs?",
        "What version are you?", "Are you the latest version?",
        "Do you improve after each conversation?", "Do you learn from our chats?",
        "Are you sentient?", "Are you conscious?",
        "Do you have feelings?", "Do you have emotions?",
        "Are you a person?", "Are you human?",
        "Are you just a chatbot?", "Are you just an AI?",
        "Am I talking to a robot?", "Am I talking to a machine?",
    ]
    for p in meta_prompts:
        pool.append((p, "meta_ai"))

    random.shuffle(pool)
    return pool


def _format_prompt(user_text: str) -> str:
    return (
        f"<|system|>\n{_SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_text}<|end|>\n"
        f"<|assistant|>\n"
    )


def _load_model(model_path: Path):
    """Load GGUF (llama_cpp) if path is a .gguf file, else HF fp16."""
    if model_path.suffix == ".gguf":
        import llama_cpp
        print(f"Loading GGUF model from {model_path} …")
        model = llama_cpp.Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )
        return model, "gguf"

    import torch
    from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM
    print(f"Loading HF fp16 model from {model_path} …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    config = Phi3Config.from_pretrained(str(model_path))
    model = Phi3ForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return (model, tokenizer, device), "hf"


def _sample_rejection(model_bundle, backend: str, prompt_str: str) -> str:
    if backend == "gguf":
        out = model_bundle(
            prompt_str,
            max_tokens=_MAX_NEW_TOKENS,
            temperature=_TEMPERATURE,
            top_k=_TOP_K,
            top_p=_TOP_P,
            repeat_penalty=1.05,
        )
        return out["choices"][0]["text"].strip()

    import torch
    model, tokenizer, device = model_bundle
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=_TEMPERATURE,
            top_k=_TOP_K,
            top_p=_TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run(args: argparse.Namespace) -> None:
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    prompt_pool = _build_prompt_pool()
    if args.max_prompts:
        prompt_pool = prompt_pool[: args.max_prompts]

    print(f"Prompt pool size : {len(prompt_pool)}")
    print(f"Output           : {output_path}")

    if args.dry_run:
        print("Dry-run mode — no model loaded. Showing first 10 prompts:")
        for i, (text, cat) in enumerate(prompt_pool[:10]):
            print(f"  [{cat}] {text}")
        return

    model_path = ROOT / args.base_model
    model_bundle, backend = _load_model(model_path)
    print(f"Backend: {backend}\n")

    pairs: list[dict] = []
    phi_prior_count = 0
    neutral_count = 0

    for idx, (user_text, category) in enumerate(prompt_pool, 1):
        print(f"[{idx:>4}/{len(prompt_pool)}] [{category}] {user_text[:60]}", end=" … ", flush=True)

        prompt_str = _format_prompt(user_text)
        response = _sample_rejection(model_bundle, backend, prompt_str)

        if _PHI_RE.search(response):
            phi_prior_count += 1
            chosen = random.choice(_CHOSEN[category]) + "<|end|>"
            rejected = response + "<|end|>"
            pairs.append({
                "prompt": prompt_str,
                "chosen": chosen,
                "rejected": rejected,
            })
            print(f"PHI  [pairs={len(pairs)}]")
        else:
            neutral_count += 1
            print("skip")

        if len(pairs) >= args.target_pairs:
            print(f"\nReached target of {args.target_pairs} pairs.")
            break

    print(f"\n── Results ──────────────────────────────────────────────")
    print(f"  Prompts processed : {min(idx, len(prompt_pool))}")
    print(f"  Phi-prior filtered: {phi_prior_count}")
    print(f"  Skipped (neutral) : {neutral_count}")
    print(f"  Pairs written     : {len(pairs)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nSaved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample base-model rejecteds for DPO v2.2.")
    parser.add_argument("--base-model", default=_DEFAULT_BASE_MODEL)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--target-pairs", type=int, default=500,
                        help="Stop after collecting this many phi-prior pairs.")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="Cap total prompts processed (0 = no cap). For testing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompt pool without loading a model.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
