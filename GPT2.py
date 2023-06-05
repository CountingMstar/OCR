from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pretrained GPT model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Load the GPT tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare the training data
train_path = "train.txt"  # Path to your training dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128  # Adjust the block size as needed
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set mlm to True if you have masked language modeling data
)

# Fine-tuning configuration
training_args = TrainingArguments(
    output_dir="output_dir",  # Directory to save the fine-tuned model
    overwrite_output_dir=True,
    num_train_epochs=30000,  # Number of training epochs
    per_device_train_batch_size=4,
    save_steps=500,  # Save checkpoints every specified number of steps
    save_total_limit=2,  # Maximum number of checkpoints to save
)

# Create the Trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")