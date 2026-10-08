"""Our own fine-tuned Ontario tenancy model: Qwen3-4B + a LoRA adapter trained in train/.

The model answers from retrieved sources (realtor/lease/sources.py) rather than from memory.
The prompt here is used for both training and inference. If you change it, rebuild the
dataset and retrain, or the model will see a prompt it wasn't trained on.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from .sources import Source, render, retrieve

BASE_MODEL = "mlx-community/Qwen3-8B-4bit"
# Qwen3 can "think" before answering. We don't: answers come from the sources. Training text
# carries an empty <think></think> block, and asking for no thinking puts the same block in the
# prompt, so the two match.
CHAT_ARGS = {"add_generation_prompt": True, "enable_thinking": False}
# OPEN_HOUSE_ADAPTER points at another adapter folder, for comparing training runs.
ADAPTER_DIR = Path(os.environ.get("OPEN_HOUSE_ADAPTER", Path(__file__).parents[2] / "train" / "adapters"))

SYSTEM = (
    "You answer questions about residential tenancies in Ontario, Canada, for landlords and "
    "tenants alike. Use only the numbered sources: the Residential Tenancies Act, 2006 (RTA), "
    "Ontario's Standard Form of Lease, Landlord and Tenant Board (LTB) guidance, and LTB "
    "decisions. Stay neutral: explain what the law says for both sides, and show decisions "
    "either side won. Quote the exact words that matter and cite each source by its number. "
    "If someone says which side they're on and asks for help with a negotiation, you may offer "
    "suggestions for their side, clearly labelled as suggestions and not legal advice. If the "
    "sources don't answer the question, say so."
)

ROLES = {"tenant", "landlord"}


def build_messages(question: str, sources: list[Source], role: str | None = None,
                   history: list[tuple[str, str]] | None = None) -> list[dict]:
    """history: earlier (question, answer) turns in this conversation, oldest first."""
    messages = [{"role": "system", "content": SYSTEM}]
    for q, a in history or []:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": strip_citations(a)}]
    about = f"About me: I'm the {role}.\n" if role in ROLES else ""
    messages.append({"role": "user", "content": f"Sources:\n\n{render(sources)}\n\n{about}Question: {question}"})
    return messages


def strip_citations(answer: str) -> str:
    """Earlier answers cite sources that aren't in the current prompt, so drop the numbers."""
    return re.sub(r"\s?\[\d\]", "", answer)


def repeating(text: str) -> bool:
    """True once a finished line has already appeared twice: the model is looping, so stop."""
    lines = [l.strip() for l in text.split("\n")[:-1] if l.strip()]
    return any(lines.count(l) >= 3 for l in set(lines))


def clause_question(clause: str) -> str:
    return f'A lease says: "{clause}" Is this allowed in Ontario?'


FOLLOWUP_START = re.compile(r"^(what if|and |but |so |what about|how about|then |also |or |if so|in that case|"
                            r"what happens if|does that|is that|can they|can i still)", re.I)
FOLLOWUP_WORDS = re.compile(r"\b(it|that|this|they|them|those|then|instead)\b", re.I)


def looks_like_followup(question: str) -> bool:
    """Without Jev: short questions that lean on the previous one ("What if I broke it?")."""
    q = question.strip()
    return bool(FOLLOWUP_START.search(q)) or (len(q.split()) <= 9 and bool(FOLLOWUP_WORDS.search(q)))


def search_query(question: str, history: list[tuple[str, str]] | None = None) -> str:
    """Follow-ups like "what if I caused it?" need the earlier question to find the right law."""
    return f"{history[-1][0]} {question}" if history else question


@lru_cache(maxsize=1)
def _load():
    from mlx_lm import load

    adapter = str(ADAPTER_DIR) if (ADAPTER_DIR / "adapters.safetensors").exists() else None
    return load(BASE_MODEL, adapter_path=adapter)


def answer(question: str, sources: list[Source] | None = None, role: str | None = None,
           history: list[tuple[str, str]] | None = None, max_tokens: int = 600) -> tuple[str, list[Source]]:
    from mlx_lm import stream_generate

    sources = sources if sources is not None else retrieve(search_query(question, history), "question")
    model, tokenizer = _load()
    prompt = tokenizer.apply_chat_template(build_messages(question, sources, role, history), **CHAT_ARGS)
    text = ""
    for r in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        text += r.text
        if repeating(text):
            break
    return re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", text).strip(), sources


def explain_clause(clause: str) -> tuple[str, list[Source]]:
    return answer(clause_question(clause), retrieve(clause))
