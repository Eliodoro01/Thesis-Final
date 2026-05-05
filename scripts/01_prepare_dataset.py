"""
STEP 1 - Filtraggio OpenThoughts-114k
======================================
Requisiti:
    pip install datasets huggingface_hub
"""

import json
import re
import random
from datasets import load_dataset

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_SAMPLES   = 700
SEED             = 42
OUTPUT_FILE      = "data/reflection_train.json"
EVAL_FILE        = "data/reflection_eval.json"
EVAL_SIZE        = 80
MIN_THINKING_LEN = 300
MAX_THINKING_LEN = 6000
MIN_ANSWER_LEN   = 50
# ─────────────────────────────────────────────────────────────────────────────

print("Caricamento OpenThoughts-114k...")
ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
print(f"   Dataset caricato: {len(ds)} sample totali")


def extract_parts(example):
    convs = example.get("conversations", [])
    if len(convs) < 2:
        return None

    question = None
    raw_assistant = None

    for turn in convs:
        role = turn.get("from", "")
        # il dataset usa "user" (non "human")
        if role == "user":
            question = turn.get("value", "").strip()
        elif role == "assistant":
            raw_assistant = turn.get("value", "").strip()

    if not question or not raw_assistant:
        return None

    m_think = re.search(
        r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>",
        raw_assistant, re.DOTALL
    )
    m_sol = re.search(
        r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>",
        raw_assistant, re.DOTALL
    )

    if not m_think or not m_sol:
        return None

    thinking = m_think.group(1).strip()
    solution = m_sol.group(1).strip()
    return question, thinking, solution


def is_valid(question, thinking, solution):
    if not (MIN_THINKING_LEN <= len(thinking) <= MAX_THINKING_LEN):
        return False
    if len(solution) < MIN_ANSWER_LEN:
        return False
    if len(thinking.split()) < 50:
        return False
    return True


print("Parsing e filtraggio sample...")
valid = []
for ex in ds:
    parts = extract_parts(ex)
    if parts is None:
        continue
    question, thinking, solution = parts
    if is_valid(question, thinking, solution):
        valid.append((question, thinking, solution))

print(f"   Sample validi trovati: {len(valid)}")

if len(valid) < TARGET_SAMPLES + EVAL_SIZE:
    print(f"Attenzione: meno sample del previsto. Uso tutti i {len(valid)} disponibili.")
    TARGET_SAMPLES = max(0, len(valid) - EVAL_SIZE)

random.seed(SEED)
random.shuffle(valid)

train_raw = valid[:TARGET_SAMPLES]
eval_raw  = valid[TARGET_SAMPLES : TARGET_SAMPLES + EVAL_SIZE]


def format_sample(question, thinking, solution):
    assistant_content = f"<thinking>\n{thinking}\n</thinking>\n\n{solution}"
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "assistant", "value": assistant_content}
        ],
        "source": "OpenThoughts-114k"
    }


train_formatted = [format_sample(*s) for s in train_raw]
eval_formatted  = [format_sample(*s) for s in eval_raw]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(train_formatted, f, ensure_ascii=False, indent=2)

with open(EVAL_FILE, "w", encoding="utf-8") as f:
    json.dump(eval_formatted, f, ensure_ascii=False, indent=2)

print(f"\nDataset salvati:")
print(f"   Training : {OUTPUT_FILE}  ({len(train_formatted)} sample)")
print(f"   Eval     : {EVAL_FILE}    ({len(eval_formatted)} sample)")

if train_formatted:
    print("\n--- VERIFICA SAMPLE 0 ---")
    s = train_formatted[0]
    print("USER    :", s["conversations"][0]["value"][:120])
    av = s["conversations"][1]["value"]
    t_start = av.find("<thinking>") + 10
    t_end   = av.find("</thinking>")
    print("THINKING:", av[t_start:t_end][:250].strip())
    print("ANSWER  :", av[t_end+11:].strip()[:200])
