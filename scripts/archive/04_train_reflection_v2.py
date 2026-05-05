# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — QLoRA Fine-Tuning con Unsloth (Llama 3.2:1B-Instruct)        ║
# ║  Esegui su Google Colab con runtime GPU (T4 gratuita va bene)           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── CELLA 1: Installa dipendenze ──────────────────────────────────────────────

#!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#!pip install --no-deps --upgrade "trl>=0.9.0" peft accelerate bitsandbytes


# ── CELLA 2: Imports e configurazione ────────────────────────────────────────
import torch
import json
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import Dataset

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_NAME      = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.0
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 3
WARMUP_RATIO    = 0.05
OUTPUT_DIR      = "./reflection_lora_output"
DATASET_PATH    = "dataset_reflection_train.json"
# ─────────────────────────────────────────────────────────────────────────────

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NESSUNA'}")

# ── CELLA 3: Carica modello ───────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,
    load_in_4bit   = LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
)

print(f"Parametri trainabili: {model.num_parameters(only_trainable=True):,}")

# ── CELLA 4: Prepara dataset ──────────────────────────────────────────────────
LLAMA3_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{assistant}<|eot_id|>"
)

def format_for_training(example):
    user      = example["conversations"][0]["value"]
    assistant = example["conversations"][1]["value"]
    return {"text": LLAMA3_CHAT_TEMPLATE.format(user=user, assistant=assistant)}

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)

print(f"Dataset: {len(dataset)} sample")
print("Preview:", dataset[0]["text"][:200])

# ── CELLA 5: Training ─────────────────────────────────────────────────────────
trainer = UnslothTrainer(
    model           = model,
    processing_class   = tokenizer,
    train_dataset   = dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LENGTH,
    args = UnslothTrainingArguments(
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        num_train_epochs             = NUM_EPOCHS,
        warmup_ratio                 = WARMUP_RATIO,
        learning_rate                = LEARNING_RATE,
        fp16                         = not torch.cuda.is_bf16_supported(),
        bf16                         = torch.cuda.is_bf16_supported(),
        logging_steps                = 10,
        save_steps                   = 100,
        output_dir                   = OUTPUT_DIR,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",
        seed                         = 42,
        report_to                    = "none",
    ),
)

print("🚀 Avvio training...")
trainer_stats = trainer.train()

print(f"\n✅ Training completato!")
print(f"   Tempo: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"   Loss:  {trainer_stats.metrics['train_loss']:.4f}")

# ── CELLA 6: Salva modello ────────────────────────────────────────────────────
SAVE_PATH = "./reflection_model_final"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"✅ Modello salvato in: {SAVE_PATH}")

# Salva su Google Drive (decommenta):
from google.colab import drive
drive.mount('/content/drive')
model.save_pretrained("/content/drive/MyDrive/reflection_model_final")
tokenizer.save_pretrained("/content/drive/MyDrive/reflection_model_final")

# ── CELLA 7: Test rapido ──────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)

test_prompt = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "What is 15% of 240? Show your reasoning step by step."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512,
        temperature    = 0.6,
        top_p          = 0.9,
        do_sample      = True,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n--- OUTPUT DEL MODELLO ---")
print(response)