"""
STEP 1b — Distillation vera: Gemini Flash come teacher
=======================================================
Prende le STESSE domande di dataset_reflection_train.json,
le invia a Gemini Flash (teacher), e salva le risposte del teacher
come dataset di training per lo student (Llama 3.2:1B).

Questo è knowledge distillation:
  - Teacher: Gemini Flash (modello grande, capace)
  - Student: Llama 3.2:1B (modello piccolo, da addestrare)
  - Trasferimento: il teacher risponde alle domande, lo student impara a imitarlo

Costo stimato: ~1-3€ con Gemini Flash API (700 sample, risposte brevi/medie)
API key gratuita con quota limitata su: https://ai.google.dev

Requisiti:
    pip install google-genai

Esegui DOPO step1_filter_dataset.py
"""

import json
import time
import os
import re
from google import genai
from google.genai import types

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY       = "AIzaSyD_puPyCpPgqb_oUh-BVDDf22Sr4LGp8p4"   # da https://ai.google.dev
INPUT_FILE           = "dataset_reflection_train.json"
OUTPUT_FILE          = "dataset_distillation_train.json"
CHECKPOINT_FILE      = "dataset_distillation_checkpoint.json"  # resume se si interrompe
REQUESTS_PER_MINUTE  = 14        # quota free tier Gemini Flash: 15 RPM, stai sotto
SLEEP_BETWEEN_CALLS  = 60 / REQUESTS_PER_MINUTE   # ~4.3 secondi tra chiamate
MAX_RETRIES          = 3
# ─────────────────────────────────────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a helpful and precise assistant. 
Answer the user's question clearly and correctly.
Show your reasoning when solving problems, but be concise.
Do NOT use XML tags like <thinking> in your response."""

def call_teacher(question: str, retries: int = MAX_RETRIES) -> str | None:
    """Chiama Gemini Flash e restituisce la risposta. Gestisce retry e rate limit."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"User: {question}\n\nAssistant:",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"   Rate limit raggiunto, attendo {wait}s...")
                time.sleep(wait)
            elif "500" in err or "503" in err:
                print(f"   Errore server (attempt {attempt+1}/{retries}), attendo 10s...")
                time.sleep(10)
            else:
                print(f"   Errore non gestito: {err}")
                return None
    return None


# ── Carica dataset reflection (le domande) ────────────────────────────────────
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reflection_data = json.load(f)

questions = [s["conversations"][0]["value"] for s in reflection_data]
print(f"Domande da processare: {len(questions)}")

# ── Resume da checkpoint se esiste ───────────────────────────────────────────
completed = {}
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        completed = json.load(f)
    print(f"   Checkpoint trovato: {len(completed)} domande già processate, riprendo da lì.")

# ── Genera risposte del teacher ───────────────────────────────────────────────
failed = []

for i, question in enumerate(questions):
    key = str(i)

    if key in completed:
        continue   # già fatto, salta

    print(f"[{i+1}/{len(questions)}] {question[:70]}...")

    answer = call_teacher(question)

    if answer:
        completed[key] = {
            "question": question,
            "answer":   answer
        }
        print(f"   ok ({len(answer)} chars)")
    else:
        failed.append(i)
        print(f"   FALLITO, verrà saltato")

    # Salva checkpoint ogni 10 chiamate
    if (i + 1) % 10 == 0:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(completed, f, ensure_ascii=False, indent=2)
        print(f"   Checkpoint salvato ({len(completed)} completati)")

    time.sleep(SLEEP_BETWEEN_CALLS)

# ── Salva checkpoint finale ───────────────────────────────────────────────────
with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
    json.dump(completed, f, ensure_ascii=False, indent=2)

# ── Costruisce dataset nel formato conversations ──────────────────────────────
distillation_data = []
for key in sorted(completed.keys(), key=int):
    entry = completed[key]
    distillation_data.append({
        "conversations": [
            {"from": "human",     "value": entry["question"]},
            {"from": "assistant", "value": entry["answer"]}
        ],
        "source": "gemini-2.0-flash-teacher"
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(distillation_data, f, ensure_ascii=False, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDataset distillation salvato: {OUTPUT_FILE}")
print(f"   Sample generati: {len(distillation_data)}")
print(f"   Sample falliti:  {len(failed)}")
if failed:
    print(f"   Indici falliti: {failed[:10]}{'...' if len(failed)>10 else ''}")

print(f"\nTempo stimato per 700 domande al rate attuale:")
total_sec = len(questions) * SLEEP_BETWEEN_CALLS
print(f"   ~{total_sec/60:.0f} minuti ({total_sec/3600:.1f} ore)")
print(f"   Consiglio: lancia lo script e lascialo girare in background.")

# Verifica esempio
if distillation_data:
    print("\n--- ESEMPIO (distillation[0]) ---")
    s = distillation_data[0]
    print("USER:      ", s["conversations"][0]["value"][:100])
    print("ASSISTANT: ", s["conversations"][1]["value"][:300])

    print("\n--- CONFRONTO con reflection[0] ---")
    r = reflection_data[0]
    thinking = re.search(r"<thinking>(.*?)</thinking>", r["conversations"][1]["value"], re.DOTALL)
    answer   = re.search(r"</thinking>\s*\n*(.*)",       r["conversations"][1]["value"], re.DOTALL)
    if thinking:
        print("THINKING:  ", thinking.group(1).strip()[:200])
    if answer:
        print("ANSWER:    ", answer.group(1).strip()[:200])