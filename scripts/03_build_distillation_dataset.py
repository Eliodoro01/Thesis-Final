"""
STEP 1b — Crea dataset Distillation (Opzione A)
================================================
Prende dataset_reflection_train.json (già generato nello Step 1)
e rimuove i blocchi <thinking>...</thinking>, tenendo solo la risposta finale.

Questo produce un dataset identico per domande e risposte, con l'unica
differenza che il modello di distillation NON vede il ragionamento esplicito.

Variabile isolata: presenza del <thinking> nel training.

Esegui DOPO step1_filter_dataset.py
"""

import json
import re

INPUT_FILE  = "data/reflection_train.json"
OUTPUT_FILE = "data/distillation_train.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reflection_data = json.load(f)

distillation_data = []
skipped = 0

for sample in reflection_data:
    assistant_val = sample["conversations"][1]["value"]

    # Estrai solo il testo dopo </thinking>
    match = re.search(r"</thinking>\s*\n*(.*)", assistant_val, re.DOTALL)
    if not match:
        skipped += 1
        continue

    answer_only = match.group(1).strip()

    if len(answer_only) < 30:
        skipped += 1
        continue

    distillation_data.append({
        "conversations": [
            {"from": "human",     "value": sample["conversations"][0]["value"]},
            {"from": "assistant", "value": answer_only}
        ],
        "source": sample.get("source", "OpenThoughts-114k")
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(distillation_data, f, ensure_ascii=False, indent=2)

print(f"Dataset distillation salvato: {OUTPUT_FILE}")
print(f"   Sample totali:  {len(distillation_data)}")
print(f"   Sample scartati: {skipped}")
print(f"   (deve coincidere con data/reflection_train.json: {len(reflection_data)} sample)")

# Verifica: mostra un esempio
print("\n--- ESEMPIO (distillation[0]) ---")
s = distillation_data[0]
print("USER:      ", s["conversations"][0]["value"][:100])
print("ASSISTANT: ", s["conversations"][1]["value"][:200])
print("\n--- CONFRONTO con reflection[0] ---")
r = reflection_data[0]
print("ASSISTANT (reflection): ", r["conversations"][1]["value"][:300])
