import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


MODEL_ID = "google/gemma-4-E2B"


# Load dataset
dataset = load_dataset(
    "merve/vqav2-small",
    split="validation[:1000]"
)

train_dataset = dataset.select(range(900))
eval_dataset = dataset.select(range(900, 1000))


# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID)


# 4-bit QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False
model.enable_input_require_grads()

for param in model.parameters():
    param.requires_grad = False


# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# Note:
# The VQA preprocessing and collator used in the experiment
# are documented in the project notebook/workflow.


training_args = TrainingArguments(
    output_dir="./gemma4-vqa-output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)


print("Model and training configuration ready.")
print("Training examples:", len(train_dataset))
print("Validation examples:", len(eval_dataset))
print("Start training after implementing the VQA dataset preprocessing and collator.")
