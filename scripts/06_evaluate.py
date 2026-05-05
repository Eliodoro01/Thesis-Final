"""
STEP 3 — Evaluation comparativa: Reflection SFT vs Distillation
=================================================================
Questo script:
1. Carica entrambi i modelli (reflection fine-tuned e baseline/distillato)
2. Li fa rispondere a un set di 50 domande di ragionamento
3. Usa Gemini Flash come LLM judge per valutare le risposte
4. Produce una tabella di risultati 

Requisiti:
    pip install transformers accelerate bitsandbytes google-genai pandas

"""

import json
import os
import time
import torch
import pandas as pd
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")  # export GEMINI_API_KEY=your_key
BASE_MODEL_NAME     = "unsloth/Llama-3.2-1B-Instruct"
REFLECTION_LORA     = "models/reflection"                 # LoRA salvato nel step 2
DISTILLATION_LORA   = "models/distillation"              # LoRA salvato nel step 2b
EVAL_DATASET        = "data/reflection_eval.json"
OUTPUT_CSV          = "results/evaluation_results.csv"
MAX_NEW_TOKENS      = 512
# ─────────────────────────────────────────────────────────────────────────────

judge_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Domande di evaluation ─────────────────────────────────────────────────────
# Puoi aggiungere le tue o usare quelle dell'eval set filtrato nello Step 1
EVAL_QUESTIONS = [
    "A store sells apples for $0.75 each. If you buy 12 apples and pay with a $20 bill, how much change do you receive? Show your reasoning.",
    "A train travels 120 km in 1.5 hours. What is its average speed in km/h? Show your reasoning.",
    "If a rectangle has a perimeter of 36 cm and its length is twice its width, what are the dimensions?",
    "A class has 30 students. 40% are girls. How many boys are in the class?",
    "If 5 workers can complete a job in 8 days, how many days would 10 workers take?",
    "A shirt costs $45 after a 25% discount. What was the original price?",
    "If x + 3 = 10, what is 2x - 1?",
    "A car travels at 60 km/h for 2 hours, then at 90 km/h for 1 hour. What is the total distance?",
    "What is the sum of angles in a pentagon?",
    "If a number is divisible by both 4 and 6, what is the smallest such number greater than 1?",
    "A box contains 3 red and 5 blue balls. What is the probability of picking a red ball?",
    "If today is Wednesday, what day will it be in 100 days?",
    "A ladder 10 meters long leans against a wall. The base is 6 meters from the wall. How high does the ladder reach?",
    "Solve: 2(x + 4) = 3x - 1",
    "A recipe needs 2.5 cups of flour for 24 cookies. How much flour for 36 cookies?",
    "What is 15% of 240?",
    "If a square has an area of 144 cm², what is its perimeter?",
    "A tank fills in 6 hours and drains in 9 hours. How long to fill if drain is open?",
    "What is the next number in the sequence: 2, 6, 18, 54, ...?",
    "If f(x) = 3x² - 2x + 1, what is f(2)?",
]

# ── Carica modelli ────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

def load_model(model_path, lora_path=None):
    print(f"   Caricamento {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if lora_path:
        print(f"   Applicazione LoRA da {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def generate_response(model, tokenizer, question, max_new_tokens=MAX_NEW_TOKENS):
    prompt = CHAT_TEMPLATE.format(user=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── LLM Judge con Gemini Flash ────────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator comparing two AI responses to a reasoning question.

Question: {question}

Model A (Distillation) response:
{response_a}

Model B (Reflection) response:
{response_b}

Evaluate both responses on these criteria. For each, give a score from 1-5:
1. CORRECTNESS: Is the final answer correct?
2. REASONING_QUALITY: How well does the model show its reasoning steps?
3. CLARITY: How clear and well-structured is the response?

Then decide: which model is BETTER overall? (A, B, or TIE)

Respond ONLY in this JSON format, no other text:
{{
  "model_a": {{"correctness": 0, "reasoning_quality": 0, "clarity": 0}},
  "model_b": {{"correctness": 0, "reasoning_quality": 0, "clarity": 0}},
  "winner": "A|B|TIE",
  "reasoning": "brief explanation"
}}"""


def judge_responses(question, response_a, response_b):
    prompt = JUDGE_PROMPT.format(
        question=question,
        response_a=response_a[:1000],   # tronca per non superare token limit
        response_b=response_b[:1000],
    )
    try:
        result = judge_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = result.text.strip()
        # Pulizia JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"   Judge error: {e}")
        return None


# ── Esegui evaluation ─────────────────────────────────────────────────────────
print("Caricamento modello DISTILLATION (fine-tuned con LoRA)...")
baseline_model, baseline_tokenizer = load_model(BASE_MODEL_NAME, DISTILLATION_LORA)

print("Caricamento modello REFLECTION (fine-tuned con LoRA)...")
reflection_model, reflection_tokenizer = load_model(BASE_MODEL_NAME, REFLECTION_LORA)

results = []
print(f"\nAvvio evaluation su {len(EVAL_QUESTIONS)} domande...\n")

for i, question in enumerate(EVAL_QUESTIONS):
    print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {question[:60]}...")

    resp_baseline   = generate_response(baseline_model, baseline_tokenizer, question)
    resp_reflection = generate_response(reflection_model, reflection_tokenizer, question)

    time.sleep(1)   # rate limiting per Gemini API
    scores = judge_responses(question, resp_baseline, resp_reflection)

    row = {
        "question":           question,
        "baseline_response":  resp_baseline,
        "reflection_response": resp_reflection,
    }

    if scores:
        row.update({
            "baseline_correctness":        scores["model_a"]["correctness"],
            "baseline_reasoning_quality":  scores["model_a"]["reasoning_quality"],
            "baseline_clarity":            scores["model_a"]["clarity"],
            "reflection_correctness":      scores["model_b"]["correctness"],
            "reflection_reasoning_quality": scores["model_b"]["reasoning_quality"],
            "reflection_clarity":          scores["model_b"]["clarity"],
            "winner":                      scores["winner"],
            "judge_reasoning":             scores.get("reasoning", ""),
        })
    else:
        row.update({
            "baseline_correctness": None, "baseline_reasoning_quality": None, "baseline_clarity": None,
            "reflection_correctness": None, "reflection_reasoning_quality": None, "reflection_clarity": None,
            "winner": "ERROR", "judge_reasoning": "",
        })

    results.append(row)
    print(f"   → Winner: {row.get('winner', 'ERROR')}")

# ── Salva e stampa summary ────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nRisultati salvati in: {OUTPUT_CSV}")

# Summary per la tesi
valid = df[df["winner"] != "ERROR"]
print("\n" + "="*50)
print("          SUMMARY RISULTATI — TESI")
print("="*50)
wins = valid["winner"].value_counts()
print(f"  Distillation wins: {wins.get('A', 0)}")
print(f"  Reflection wins:   {wins.get('B', 0)}")
print(f"  Ties:            {wins.get('TIE', 0)}")
print()

metrics = ["correctness", "reasoning_quality", "clarity"]
for m in metrics:
    b_avg = valid[f"baseline_{m}"].mean()
    r_avg = valid[f"reflection_{m}"].mean()
    print(f"  {m.upper():<22} Baseline: {b_avg:.2f}  Reflection: {r_avg:.2f}  Δ={r_avg-b_avg:+.2f}")

print("="*50)
print("\nLa tabella completa è in:", OUTPUT_CSV)
