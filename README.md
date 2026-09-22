# Gemma 4 E2B VQA Fine-Tuning

Fine-tuning Google's Gemma 4 E2B multimodal model for Visual Question Answering (VQA) using QLoRA and LoRA.

## Project Overview

This project fine-tunes google/gemma-4-E2B on a Visual Question Answering dataset.

The model receives an image and a question and generates an answer based on the visual information in the image.

## Model

- Base model: google/gemma-4-E2B
- Fine-tuning method: QLoRA + LoRA
- Quantization: 4-bit NF4
- Compute dtype: FP16
- GPU: NVIDIA Tesla T4

## Dataset

Dataset: merve/vqav2-small

- Total examples: 1,000
- Training examples: 900
- Validation examples: 100

## Training Configuration

- Epochs: 1
- Training batch size: 1
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- Gradient checkpointing: Enabled
- FP16: Enabled

## LoRA Configuration

- Rank (r): 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Bias: None
- Task type: Causal Language Modeling

Trainable parameters:

- Trainable: 2,678,784
- Total: 5,106,976,288
- Trainable percentage: 0.0525%

## Results

Evaluation was performed on 100 validation examples.

- Evaluation loss: 11.8711
- Exact-match accuracy: 40.00%
- Correct predictions: 40 / 100

## Project Structure

gemma4-vqa/
├── gemma4-vqa-lora/
├── vqa_evaluation_results.json
├── requirements.txt
├── README.md
└── train_vqa.py

## Important

The full Gemma 4 E2B base model is NOT included in this repository because of its large size.

The LoRA adapter is included and can be loaded on top of the base model.

## Workflow

VQA Dataset
    ↓
Data Preprocessing
    ↓
Gemma 4 E2B
    ↓
4-bit QLoRA
    ↓
LoRA Fine-Tuning
    ↓
VQA Evaluation
    ↓
40% Exact-Match Accuracy

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- BitsAndBytes
- Google Colab
- NVIDIA Tesla T4



