# Gemma 4 E2B V2 Training

## Model
- Base model: google/gemma-4-E2B
- Fine-tuning method: QLoRA
- Quantization: 4-bit NF4
- Compute dtype: float16
- Double quantization: enabled

## Dataset
- Dataset: merve/vqav2-small
- Selected samples: 10,000
- Training: 8,000
- Validation: 1,000
- Test: 1,000

## LoRA
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: Language-model self-attention q/k/v/o projections
- Trainable parameters: 2,678,784
- Trainable percentage: 0.0525%

## Training
- Epochs: 1
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: 2e-4
- Weight decay: 0.01
- Optimizer: Trainer default
- FP16: enabled
- Gradient checkpointing: enabled

## Results
- Final training loss: 1.2144
- Final validation loss: 1.0792
- Best validation loss: 1.0784
