import random
import numpy as np
import torch
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
import sys
import copy
def get_random_states():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def print_state(state):
    for key in state.keys():
        print(key,":", state[key], sys.stderr)
        
# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Get initial random states
initial_states = copy.deepcopy(get_random_states())

model_name = "gpt2"  # You can replace this with any other suitable model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset (replace with your own dataset)
dataset = load_dataset("imdb", split="train")
# rs1 = copy.deepcopy(random.sample([1,2,3,4,5],2))
# print(rs1, sys.stderr)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)
print("With SFTTrainer")
for i in range(5):
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",  # Specify the text field in your dataset
        tokenizer=tokenizer,
    )
    rs = copy.deepcopy(random.sample([1,2,3,4,5],2))
    print(rs, sys.stderr)
print("Without SFTTrainer")
for i in range(5):
    rs = copy.deepcopy(random.sample([1,2,3,4,5],2))
    print(rs, sys.stderr)

# # Get random states after creating first SFTTrainer
# after_trainer1_states = copy.deepcopy(get_random_states())
# # Compare random states after first trainer creation
# # compare_random_states(initial_states, after_trainer1_states, "after creating first trainer")

# # print_state(initial_states)
# # print_state(after_trainer1_states)

