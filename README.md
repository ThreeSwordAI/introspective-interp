# Fork notice

This is a fork of [TransluceAI/introspective-interp](https://github.com/TransluceAI/introspective-interp)
(Li et al., *Training Language Models to Explain Their Own Computations*, arXiv:2511.08579).

Here I test whether the paper's untrained input-ablation baseline (zero-shot, format examples only,
Appendix H.3) underestimates what balanced few-shot in-context learning can elicit from Qwen3-8B
without any weight updates. Finding: with 4 balanced demonstrations, parse-based exact match rises
from 30.1 (k=0) to 48.8 on average over 3 demo seeds (k=8: 46.1; paper's fine-tuned self-explainer: 83.4),
with every paired 95% CI excluding zero — so a meaningful part of the self-explanation capability is
elicitable by in-context task induction alone, but it stays far below the fine-tuned ceiling.

Please see **[`experiments/icl_input_ablation/`](experiments/icl_input_ablation/)** for the code,
HPC commands, and full results.

---

*Below is the original README of the upstream repository, unchanged:*

# Training Language Models To Explain Their Own Computations

[![Paper](https://img.shields.io/badge/arXiv-2511.08579-b31b1b.svg)](https://arxiv.org/abs/2511.08579)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Collection-yellow)](https://huggingface.co/collections/Transluce/training-language-models-to-explain-their-own-computations)

This repository contains the code and data for the paper **"[Training Language Models To Explain Their Own Computations](https://arxiv.org/abs/2511.08579)"**.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [Tasks Overview](#tasks-overview)
- [Feature Descriptions](#feature-descriptions)
  - [📥 Data & Checkpoints](#-data--checkpoints)
  - [🏋️ Training](#️-training)
  - [📊 Evaluation](#-evaluation)
- [Activation Patching](#activation-patching)
  - [📥 Data & Checkpoints](#-data--checkpoints-1)
  - [🏋️ Training](#️-training-1)
  - [📊 Evaluation](#-evaluation-1)
- [Input Ablations](#input-ablations)
  - [📥 Data & Checkpoints](#-data--checkpoints-2)
  - [🏋️ Training](#️-training-2)
  - [📊 Evaluation](#-evaluation-2)
- [📄 Citation](#-citation)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (for LM judge evaluation of feature descriptions)
- Arguments are configured for 2x 80GB H100 GPUs, but you can use less memory by adjusting batch sizes.

### Installation
```bash
git clone https://github.com/TransluceAI/introspective-interp.git
cd introspective-interp
# All packages are in pyproject.toml and will be auto-downloaded on first `uv run`
```

### Environment Setup
Add your API keys to `.env` file:
```bash
cp .env.example .env  # Create from template
# Edit .env to add your OpenAI API key
```

## Tasks Overview

We train language models to produce three types of explanations of their own computations. In our experiments, we use an **explainer model** (the model we train) to explain a **target model** (the model being analyzed).

| Task | Description | Target Model | Training Dataset |
|------|-------------|--------------|----------|
| **Feature Descriptions** | Generate natural language descriptions of model features | Llama-3.1-8B | LlamaScope SAE features + Neuronpedia |
| **Activation Patching** | Predict effects of activation patching interventions | Llama-3.1-8B, Qwen3-8B | CounterFact |
| **Input Ablations** | Predict effects of removing hint tokens | Llama-3.1-8B-Instruct, Qwen3-8B | MMLU + hint |

---

## Feature Descriptions

Explainer models generate natural language descriptions of features from Llama-3.1-8B. We train on SAE features and their descriptions, then evaluate on held-out SAE features, full activations, and activation differences.

### 📥 Data & Checkpoints
Links to datasets and pre-trained checkpoints from the paper are available below.

**Datasets:**
- [SAE features + Neuronpedia explanations](https://transluce-public.s3.us-east-1.amazonaws.com/introspective-interp/SAE_feature_explanations_llama3.1_8b.tar.gz) (training + in-distribution eval)
- [Full activations on FineWeb](https://transluce-public.s3.us-east-1.amazonaws.com/introspective-interp/fineweb_llama_3.1_8b_95seqlen_fineweb_acts_grads_-1.0.tar.gz) (OOD eval)
- [Activation differences on FineWeb](https://transluce-public.s3.us-east-1.amazonaws.com/introspective-interp/fineweb_llama_3.1_8b_95seqlen_counterfact_subsampled_2000_activation_difference.tar.gz) (OOD eval)

You must first download the data locally using:
```bash
cd /PATH/TO/DATA/DIR/

# Download and extract SAE features
wget https://transluce-public.s3.us-east-1.amazonaws.com/introspective-interp/SAE_feature_explanations_llama3.1_8b.tar.gz
tar -xzvf SAE_feature_explanations_llama3.1_8b.tar.gz

# Download OOD evaluation data: full activations
wget https://transluce-public.s3.us-east-1.amazonaws.com/introspective-interp/fineweb_llama_3.1_8b_95seqlen_fineweb_acts_grads_-1.0.tar.gz
tar -xzvf fineweb_llama_3.1_8b_95seqlen_fineweb_acts_grads_-1.0.tar.gz

# Download OOD evaluation data: activation differences
wget https://transluce-public.s3.us-east-1.amazonaws.com/introspective-interp/fineweb_llama_3.1_8b_95seqlen_counterfact_subsampled_2000_activation_difference.tar.gz
tar -xzvf fineweb_llama_3.1_8b_95seqlen_counterfact_subsampled_2000_activation_difference.tar.gz
```

**Pre-trained Models (available on HuggingFace):**
- [Transluce/features_explain_llama3.1_8b_llama3.1_8b](https://huggingface.co/Transluce/features_explain_llama3.1_8b_llama3.1_8b) - Llama-3.1-8B explains Llama-3.1-8B
- [Transluce/features_explain_llama3.1_8b_llama3.1_8b_instruct](https://huggingface.co/Transluce/features_explain_llama3.1_8b_llama3.1_8b_instruct) - Llama-3.1-8B-Instruct explains Llama-3.1-8B
- [Transluce/features_explain_llama3.1_8b_llama3_8b](https://huggingface.co/Transluce/features_explain_llama3.1_8b_llama3_8b) - Llama-3-8B explains Llama-3.1-8B
- [Transluce/features_explain_llama3.1_8b_simulator](https://huggingface.co/Transluce/features_explain_llama3.1_8b_simulator) - Simulator model: used to score candidate natural-language explanations of features of Llama-3.1-8B. Predicts where the described feature should activate in the sequence, which can then be compared to a target feature's true activations, enabling scoring of the explanations by computing correlation (the "simulator score").


### 🏋️ Training

**Config File Setup:**
We specify all training parameters (models, data paths, hyperparameters) in YAML config files. 

Configs for this task can all be found under `config/feature_descriptions/*`, where `*` follows the pattern `{explainer_model}_131k_{eval_method}.yaml`.

Explainer models is one of:
- `base` = Llama-3.1-8B
- `instruct` = Llama-3.1-8B-Instruct  
- `qwen` = Qwen3-8B
- `llama3` = Llama-3-8B

Evaluation methods is one of:
- _(no suffix)_ = LM judge similarity against ground-truth SAE feature descriptions (default)
- `simcor` = Simulator correlation scores on SAE features
- `ood_fw` = Simulator correlation scores on full LM activations from FineWeb
- `ood_diff` = Simulator correlation scores on LM activation differences from CounterFact


Edit these paths in your chosen config file before training:
```yaml
train:
  explanation_dir: "/PATH/TO/DATA/DIR/SAE_feature_explanations_llama3.1_8b/"

text:
  explanaton_dir: "/PATH/TO/DATA/DIR/SAE_feature_explanations_llama3.1_8b/"  # change for OOD evals

output_dir: "/PATH/TO/SAVE/CHECKPOINTS/"
cache_dir: "/PATH/TO/HF/CACHE/"  # Optional
```

**Run Training:**
```bash
uv run --env-file .env train.py --config config/feature_descriptions/base_131k.yaml
```

### 📊 Evaluation
```bash
uv run --env-file .env evaluate.py \
  --config config/feature_descriptions/base_131k.yaml \
  --target_model_path meta-llama/Llama-3.1-8B \
  --task features_explain \
  --model_path /PATH/TO/EXPLAINER/CHECKPOINT/ \   # can be a local path or a HF model from above, e.g. Transluce/features_explain_llama3.1_8b_llama3.1_8b
  --output_dir /PATH/TO/RESULTS/ \
  --batch_size 64
```

**Baselines:**
To run baseline explainer methods, specify:
- `--model_path nearest_neighbor` for top-1 nearest neighbor (finds most similar training explanations). Optionally add `--layerwise_similarities` to do this layerwise.
- `--model_path self_explanations` for untrained self-explanations, i.e. SelfIE (target model explains itself without training)

---

## Activation Patching

Explainer models predict how activation patching interventions affect target model outputs on CounterFact data.

### 📥 Data & Checkpoints

**Datasets (hosted on HuggingFace):**
- [Transluce/act_patch_llama_3.1_8b_counterfact](https://huggingface.co/datasets/Transluce/act_patch_llama_3.1_8b_counterfact) - activation patching results of Llama-3.1-8B target model
- [Transluce/act_patch_qwen3_8b_counterfact](https://huggingface.co/datasets/Transluce/act_patch_qwen3_8b_counterfact) - activation patching results of Qwen3-8B target model

**Pre-trained Models (available on HuggingFace):**
- [Transluce/act_patch_qwen3_8b_qwen3_8b](https://huggingface.co/Transluce/act_patch_qwen3_8b_qwen3_8b) - Qwen3-8B explains Qwen3-8B
- [Transluce/act_patch_llama3.1_8b_llama3.1_8b](https://huggingface.co/Transluce/act_patch_llama3.1_8b_llama3.1_8b) - Llama-3.1-8B explains Llama-3.1-8B

### 🏋️ Training

**Config File Setup:**
We specify all training parameters (models, data paths, hyperparameters) in YAML config files. 

Configs for this task can all be found under `config/act_patch/*`, where `*` is:
- `base_base_act_patch_cf.yaml` - Llama-3.1-8B explains Llama-3.1-8B
- `base_qwen_act_patch_cf.yaml` - Llama-3.1-8B explains Qwen3-8B
- `qwen_qwen_act_patch_cf.yaml` - Qwen3-8B explains Qwen3-8B
- `qwen_base_act_patch_cf.yaml` - Qwen3-8B explains Llama-3.1-8B

Edit these paths in your chosen config file before training:
```yaml
output_dir: "/PATH/TO/SAVE/CHECKPOINTS/"
cache_dir: "/PATH/TO/HF/CACHE/"  # Optional
```

**Available Configs:**

**Run Training:**
```bash
uv run --env-file .env train.py --config config/act_patch/base_act_patch_cf.yaml
```

### 📊 Evaluation
```bash
uv run --env-file .env evaluate.py \
  --config config/act_patch/base_act_patch_cf.yaml \
  --target_model_path /PATH/TO/TARGET/MODEL/ \ # can be a local path or a HF model ID , e.g. meta-llama/Llama-3.1-8B
  --task act_patch \
  --model_path /PATH/TO/EXPLAINER/CHECKPOINT/ \   # can be a local path or a HF model from above, e.g. act_patch_llama3.1_8b_llama3.1_8b
  --output_dir /PATH/TO/RESULTS/ \
  --batch_size 32
```


---

## Input Ablations

Explainer models predict how removing input hints affects target model's predictions on MMLU questions.

### 📥 Data & Checkpoints

**Datasets (hosted on HuggingFace):**
- [Transluce/input_ablation_llama_3.1_8b_instruct_mmlu_hint](https://huggingface.co/datasets/Transluce/input_ablation_llama_3.1_8b_instruct_mmlu_hint) - hint ablation results for Llama-3.1-8B-Instruct target model
- [Transluce/input_ablation_qwen3_8b_mmlu_hint](https://huggingface.co/datasets/Transluce/input_ablation_qwen3_8b_mmlu_hint) - hint ablation results for Qwen3-8B target model

**Pre-trained Models (available on HuggingFace):**
- [Transluce/input_ablation_llama3.1_8b_instruct_llama3.1_8b_instruct](https://huggingface.co/Transluce/input_ablation_llama3.1_8b_instruct_llama3.1_8b_instruct) - Llama-3.1-8B-Instruct explains Llama-3.1-8B-Instruct
- [Transluce/input_ablation_qwen3_8b_qwen3_8b_hint](https://huggingface.co/Transluce/input_ablation_qwen3_8b_qwen3_8b_hint) - Qwen3-8B explains Qwen3-8B


**Loading Datasets in Code:**
```python
from datasets import load_dataset

# Load input ablation dataset
dataset = load_dataset("Transluce/input_ablation_llama_3.1_8b_instruct_mmlu_hint", split="train")
```

### 🏋️ Training

**Config File Setup:**
We specify all training parameters (models, data paths, hyperparameters) in YAML config files. 

Configs for this task can all be found under `config/input_ablation/*`, where `*` is:
- `instruct_instruct_hint.yaml` - Llama-3.1-8B-Instruct explains Llama-3.1-8B-Instruct
- `qwen_qwen_hint.yaml` - Qwen3-8B explains Qwen3-8B
- `instruct_qwen_hint.yaml` - Llama-3.1-8B-Instruct explains Qwen3-8B (cross-model)
- `qwen_instruct_hint.yaml` - Qwen3-8B explains Llama-3.1-8B-Instruct (cross-model)

Edit these paths in your chosen config file before training:
```yaml
output_dir: "/PATH/TO/SAVE/CHECKPOINTS/"
cache_dir: "/PATH/TO/HF/CACHE/"  # Optional
```

**Available Configs:**

**Run Training:**
```bash
uv run --env-file .env train.py --config config/hint/instruct_instruct_hint.yaml
```

### 📊 Evaluation
```bash
# Llama-3.1-8B-Instruct evaluation
uv run --env-file .env evaluate.py \
  --config config/hint/instruct_instruct_hint.yaml \
  --target_model_path meta-llama/Llama-3.1-8B-Instruct \
  --task hint_attribution \
  --model_path /PATH/TO/EXPLAINER/CHECKPOINT/ \   # can be a local path or a HF model from above, e.g. Transluce/input_ablation_llama3.1_8b_instruct_llama3.1_8b_instruct
  --output_dir /PATH/TO/RESULTS/ \
  --batch_size 8

# Qwen3-8B evaluation  
uv run --env-file .env evaluate.py \
  --config config/hint/qwen_qwen_hint.yaml \
  --target_model_path Qwen/Qwen3-8B \
  --task hint_attribution \
  --model_path /PATH/TO/EXPLAINER/CHECKPOINT/ \   # can be a local path or a HF model from above, e.g. Transluce/
  --output_dir /PATH/TO/RESULTS/ \
  --batch_size 8
```

This task trains models to predict how removing specific input hints affects the model's reasoning and output generation. The models learn to understand causal relationships between input components and model behavior.


---

## 📄 Citation

```bibtex
@misc{li2025traininglanguagemodelsexplain,
      title={Training Language Models to Explain Their Own Computations}, 
      author={Belinda Z. Li and Zifan Carl Guo and Vincent Huang and Jacob Steinhardt and Jacob Andreas},
      year={2025},
      eprint={2511.08579},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.08579}, 
}
```