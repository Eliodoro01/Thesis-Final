# Reflection Tuning vs Knowledge Distillation
### A Comparative Study in Efficient Fine-Tuning of Small Language Models

**Bachelor's Thesis — Università degli Studi di Napoli "Parthenope"**  
**Academic Year 2024/2025**

---

> *"Scaling laws have brought us far — but the future of AI may lie not in building bigger models, but in training smarter ones."*

---

## Abstract

This thesis investigates two contrasting fine-tuning paradigms for small language models operating under resource constraints:

- **Reflection Tuning (RT):** Training a model to generate explicit internal reasoning traces before producing an answer, fostering metacognitive self-correction.
- **Knowledge Distillation (KD):** Training a smaller *student* model to imitate the direct outputs of a larger, more capable *teacher* model.

Both strategies were applied to the same base model ([Llama 3.2 1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)) using identical hyperparameters and evaluated on 20 mathematical reasoning tasks judged by an LLM-as-a-Judge framework (Gemini Flash).

**Result:** At the 1B-parameter scale with 700 training samples, response-based Knowledge Distillation significantly outperforms Reflection Tuning — winning **14 out of 19** valid comparisons (73.7%).

---

## The Research Question

> At constrained model scale and limited data, which is more effective: teaching a model *how* to reason (Reflection Tuning), or teaching it *what* good answers look like (Knowledge Distillation)?

---

## Pipeline Architecture

```
                    OpenThoughts-114k (114k samples)
                               │
                               ▼
                  ┌─── Filter & Sample ───┐
                  │  700 train / 80 eval  │
                  └──────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    REFLECTION BRANCH              DISTILLATION BRANCH
              │                             │
    Use original data              Query Gemini Flash
    with <thinking> blocks         on 700 questions
    (reasoning traces)             (direct answers only)
              │                             │
              ▼                             ▼
    ┌─────────────────────────────────────────────────┐
    │   QLoRA Fine-tuning — Llama 3.2 1B-Instruct     │
    │   LoRA rank=16, α=32, 4-bit quantization        │
    │   Epochs=3, LR=2e-4, Google Colab T4 GPU        │
    └──────────┬───────────────────────────┬──────────┘
               │                           │
               ▼                           ▼
    [Reflection Model]             [Distillation Model]
               │                           │
               └─────────────┬─────────────┘
                             │
                             ▼
              LLM-as-a-Judge Evaluation
              (Gemini Flash, 20 questions)
              Metrics: Correctness, Reasoning, Clarity
                             │
                             ▼
              Distillation wins 14/19 (73.7%)
```

---

## Results

### Aggregate Scores (scale 1–5, n=19 valid judgments)

| Metric | Distillation | Reflection | Gap |
|---|:---:|:---:|:---:|
| Correctness | **3.89** | 2.74 | +1.16 |
| Reasoning Quality | **3.74** | 3.05 | +0.69 |
| Clarity | **4.11** | 2.42 | +1.69 |
| **Overall Average** | **3.91** | **2.74** | **+1.17** |

### Win Rate

```
Distillation  ████████████████████████████  73.7%  (14 wins)
Reflection    ██████████                    26.3%  ( 5 wins)
Ties                                         0.0%  ( 0 ties)
```

Reflection's 5 wins occurred on simpler problems and select multi-step algebra questions — suggesting that `<thinking>` blocks can help on specific problem types, but the model lacks the capacity to leverage them consistently at 1B scale.

### Evaluation Screenshots

| Question Sample | Results Overview |
|:---:|:---:|
| ![Eval 1](screen/evaluation%20(1).png) | ![Eval 2](screen/evaluation%20(2).png) |

---

## Key Findings

1. **At 1B parameters, direct imitation beats explicit reasoning.** Response-based KD produces more stable, correct, and clearly written responses than reflection-style fine-tuning on the same number of training samples.

2. **Reflection Tuning demands scale.** Existing literature suggests RT's benefits emerge more reliably above ~7B parameters — where the model has sufficient capacity to "think" before it can be trained to "think aloud" effectively.

3. **The teacher matters.** KD with a strong teacher (Gemini Flash) provides a reliable external signal of correctness; RT relies entirely on the base model's existing latent reasoning ability, which is limited at 1B.

4. **Feasibility on consumer hardware.** The complete pipeline — dataset preparation, teacher generation, training both models, evaluation — was executed on a free Google Colab T4 GPU, demonstrating the accessibility of modern fine-tuning techniques.

---

## Repository Structure

```
Thesis-Final/
│
├── Tesi/                           # LaTeX source for the thesis document
│   ├── thesis.tex                  # Main document
│   ├── Chapters/                   # 5 chapters + acknowledgments
│   │   ├── 1_introduzione.tex
│   │   ├── 2_tecnologie.tex
│   │   ├── 3_applicazione.tex
│   │   ├── 4_risultati.tex
│   │   └── 5_conclusioni.tex
│   ├── Figs/                       # Architecture diagrams & figures
│   └── References/references.bib  # Bibliography
│
├── prepare_dataset.py              # Step 1: Filter OpenThoughts-114k → 700 samples
├── generate_teacher_responses.py   # Step 1b: Query Gemini Flash (distillation branch)
├── build_distillation_dataset.py   # Step 1c: Assemble distillation training set
├── train_reflection.py             # Step 2: QLoRA training — Reflection model
├── train_distillation.py           # Step 2b: QLoRA training — Distillation model
├── evaluate.py                     # Step 3: LLM-as-a-Judge evaluation
│
├── dataset_reflection_train.json   # 700 training examples with <thinking> blocks
├── dataset_reflection_eval.json    # 80 evaluation examples
├── dataset_distillation_train.json # 700 question-answer pairs from Gemini Flash
├── dataset_distillation_checkpoint.json  # API checkpoint for resume support
│
├── evaluation_results.csv          # Full evaluation table (20 questions × 10 metrics)
├── screen/                         # Screenshots from evaluation runs
│
├── reflection_model_final/         # Trained LoRA adapter — Reflection model
└── distillation_model_final/       # Trained LoRA adapter — Distillation model
```

---

## Trained Models

Both fine-tuned LoRA adapters are included in this repository and can be loaded directly with the `peft` library.

| Model | Base | Method | Training Data | Overall Score |
|---|---|---|---|:---:|
| [`reflection_model_final`](./reflection_model_final/) | Llama 3.2 1B-Instruct | Reflection Tuning | 700 samples w/ `<thinking>` | 2.74 / 5 |
| [`distillation_model_final`](./distillation_model_final/) | Llama 3.2 1B-Instruct | Knowledge Distillation | 700 Gemini Flash responses | 3.91 / 5 |

### Quick Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit"
adapter_path   = "./distillation_model_final"   # or ./reflection_model_final

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base      = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model     = PeftModel.from_pretrained(base, adapter_path)

prompt = "Solve step by step: What is 15% of 240?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Reproducing the Experiments

### Prerequisites

```bash
pip install datasets huggingface_hub transformers accelerate bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "trl<0.9.0" peft google-genai pandas
```

A GPU is required for training. The experiments were run on a **Google Colab T4 (16 GB VRAM)**.  
A Gemini API key is required for generating distillation data and running evaluation.

### Step-by-step

```bash
# 1. Prepare base dataset (filters OpenThoughts-114k)
python prepare_dataset.py

# 2a. Generate teacher responses for distillation
#     Set GEMINI_API_KEY in the script first
python generate_teacher_responses.py

# 2b. Assemble distillation training set
python build_distillation_dataset.py

# 3a. Train reflection model (best run on Colab GPU)
python train_reflection.py

# 3b. Train distillation model
python train_distillation.py

# 4. Run evaluation
python evaluate.py
# → Outputs: evaluation_results.csv
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Base Model | [Llama 3.2 1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (unsloth 4-bit) |
| Fine-tuning | QLoRA — LoRA rank 16, α 32, 4-bit quantization |
| Training Library | [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) |
| Teacher Model | Gemini Flash (Google Generative AI) |
| Dataset | [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| Judge | Gemini Flash (LLM-as-a-Judge) |
| Hardware | Google Colab T4 GPU (16 GB VRAM) |
| Language | Python 3.10 |

---

## Thesis Structure

The written thesis (in Italian) is organized as follows:

| Chapter | Title | Content |
|:---:|---|---|
| 1 | Introduction | Historical evolution from CNNs to LLMs; scaling dilemma; motivation |
| 2 | Technologies | Transformers, self-attention, pre-training, fine-tuning, KD, RT |
| 3 | Application | Experimental design, pipeline implementation, dataset construction |
| 4 | Results | Evaluation methodology, quantitative results, qualitative analysis |
| 5 | Conclusions | Findings, limitations, and future research directions |

---

## Limitations

- **Scale:** T4 GPU constrained the base model to 1B parameters; RT literature primarily validates gains above 7B.
- **Data volume:** 700 training samples is small; most benchmark comparisons use 10k–100k examples.
- **Test set size:** 20 evaluation questions limits statistical significance.
- **Judge bias:** Gemini Flash served as both the distillation teacher and the evaluation judge, which may favor the distillation model's output style.
- **KD variant:** This is response-based distillation (API outputs only) — classical KD with soft target logits was not feasible.

---

## Author

**Eliodoro Mascolo**  
Bachelor's in Computer Science  
Università degli Studi di Napoli "Parthenope"  
Matricola: 0124002547

**Supervisor:** Prof. Emanuel Di Nardo  
**Co-supervisor:** Prof. Angelo Ciaramella

---

## License

The code in this repository is released for academic and research purposes.  
The trained model adapters are derived from [Meta's Llama 3.2](https://llama.meta.com/) and are subject to Meta's Llama Community License.
