"""
STEP 2b — QLoRA Fine-Tuning modello DISTILLATION
=================================================
Identico a step2_train_qlora.py ma usa dataset_distillation_train.json.
Esegui su Colab con T4 GPU dopo aver eseguito step2_train_qlora.py.

L'unica differenza rispetto al training reflection è il dataset in input.
Tutti gli altri iperparametri sono identici — questo è fondamentale
per la validità del confronto.
"""

import torch
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ── CONFIG — identica a step2, cambia solo DATASET_PATH e OUTPUT_DIR ──────────
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
OUTPUT_DIR      = "models/distillation_lora_output"
DATASET_PATH    = "data/distillation_train.json"       # ← unica differenza
SAVE_PATH       = "models/distillation"
# ─────────────────────────────────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,
    load_in_4bit   = LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)

LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>"""

def format_for_training(example):
    return {
        "text": LLAMA3_CHAT_TEMPLATE.format(
            user      = example["conversations"][0]["value"],
            assistant = example["conversations"][1]["value"]
        )
    }

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)
print(f"Dataset distillation caricato: {len(dataset)} sample")

trainer = SFTTrainer(
    model              = model,
    processing_class   = tokenizer,
    train_dataset      = dataset,
    args = SFTConfig(
        dataset_text_field           = "text",
        max_seq_length               = MAX_SEQ_LENGTH,
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

print("Avvio training distillation...")
stats = trainer.train()
print(f"Training completato — Loss: {stats.metrics['train_loss']:.4f}")

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Modello salvato in: {SAVE_PATH}")

# Salva su Google Drive (solo in Colab)
# from google.colab import drive
# drive.mount('/content/drive')
# model.save_pretrained("/content/drive/MyDrive/distillation_model_final")
# tokenizer.save_pretrained("/content/drive/MyDrive/distillation_model_final")
