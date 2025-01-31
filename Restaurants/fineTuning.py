from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Loading model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/LLama2-7b-chat-hf",
    load_in_4bit=True,
    device_map="auto"
)

# configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# appying LoRA
model = get_peft_model(model, lora_config)

# training arguments
training_args = TrainingArguments(
    output_dir="./restaurant-reviews-qlora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    training_dataset=tokenized_datasets["train"]
    eval_dataset=tokenized_datastes["valid"]
)

# Train model
trainer.train()