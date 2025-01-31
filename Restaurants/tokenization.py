from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLama-2-7b-chat-hf")

# Load dataset
dataset = load_dataset("json", data_files={"train": "train.json", "valid": "valid.json"})

# Tokenize fucntion
def tokenize_function(examples):
    return tokenizer(examples["review_text"], padding="max_length", truncation=True)


# Tokenizer dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)