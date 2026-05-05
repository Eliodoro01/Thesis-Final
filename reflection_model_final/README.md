---
base_model: unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit
- lora
- qlora
- sft
- reflection-tuning
- chain-of-thought
- transformers
- trl
- unsloth
---

# Reflection Model — Llama 3.2 1B QLoRA

Fine-tuned LoRA adapter produced by **Reflection Tuning** — training on examples containing explicit `<thinking>` reasoning traces — as part of a Bachelor's thesis comparing Reflection Tuning vs Knowledge Distillation.

This model was evaluated against the companion Distillation model on mathematical reasoning tasks and showed focused strengths on certain multi-step algebra problems, while the distillation counterpart prevailed overall at this parameter scale.

## Model Details

- **Developed by:** Eliodoro Mascolo (Università degli Studi di Napoli "Parthenope")
- **Base model:** `unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit`
- **Model type:** Causal Language Model — LoRA adapter (QLoRA, 4-bit)
- **Language:** English
- **Fine-tuning method:** Reflection Tuning (explicit reasoning traces)
- **License:** Subject to [Meta Llama Community License](https://llama.meta.com/llama-downloads/)

## Training

| Hyperparameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit (bitsandbytes) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Batch size | 2 (grad. accum. 4) |
| Hardware | Google Colab T4 (16 GB) |

**Training data:** 700 examples from [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) retaining their original `<thinking>` blocks, encouraging the model to generate explicit intermediate reasoning steps before the final answer.

## Evaluation Results

Evaluated against Distillation model on 20 mathematical reasoning tasks using Gemini Flash as LLM-as-a-Judge (1–5 scale, n=19 valid judgments):

| Metric | This model | Distillation model |
|---|:---:|:---:|
| Correctness | 2.74 | **3.89** |
| Reasoning Quality | 3.05 | **3.74** |
| Clarity | 2.42 | **4.11** |
| Overall Average | 2.74 | **3.91** |

Win rate: 5/19 (26.3%). Wins concentrated on questions 1, 2, 3, 11, 12 — including select multi-step algebra problems where explicit reasoning traces provided a measurable advantage.

> **Note:** The thesis hypothesizes that Reflection Tuning benefits emerge more strongly at larger scales (7B+ parameters) and with larger datasets, where the model has sufficient capacity to leverage internal reasoning structures.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base      = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model     = PeftModel.from_pretrained(base, ".")

prompt = "Solve step by step: What is 15% of 240?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Framework versions

- PEFT 0.18.1
- Transformers >= 4.40
- Unsloth (training)
- TRL < 0.9.0
