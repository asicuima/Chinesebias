

import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
import torch
from tqdm import tqdm
import argparse
import lmppl
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# 🧪 Step 1: Hugging Face Auth
# ----------------------------
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if huggingface_token:
    login(token=huggingface_token)
    print("✅ Successfully logged in to Hugging Face!")
else:
    print("❌ Error: Hugging Face token not found in .env file.")
    exit(1)

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)

# ----------------------------
# 🧩 Step 2: Argument Parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='Name of the model')
parser.add_argument('--model_type', type=str, help='Type of model')
parser.add_argument('--num_gpus', type=int, help='Number of GPUs available')
args = parser.parse_args()

# ----------------------------
# 📂 Step 3: Dataset Setup
# ----------------------------
datasets = [
    ('data/chinese_bias_dataset.csv', 'template_')
]

# ----------------------------
# ⚙️ Step 4: Load Model
# ----------------------------
if args.model_type == 'LM':
    print("🚀 Loading LM with 8-bit + auto device + trust_remote_code...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16
    )

    # Wrap into scorer-like interface
    class ScorerWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

    scorer = ScorerWrapper(model, tokenizer)

else:
    # For other model types using lmppl
    if args.model_type == 'MaskedLM':
        scorer = lmppl.MaskedLM(args.model_name, num_gpus=args.num_gpus, max_length=22)
    elif args.model_type == 'EncoderDecoderLM':
        scorer = lmppl.EncoderDecoderLM(
            args.model_name,
            num_gpus=args.num_gpus,
            device_map="auto",
            low_cpu_mem_usage=True
        )

# ----------------------------
# 🔁 Step 5: Perplexity Functions (Safe for CUDA)
# ----------------------------
def get_perplexity(template):
    model = scorer.model
    tokenizer = scorer.tokenizer
    device = model.device

    inputs = tokenizer(template, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone().to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

    return torch.exp(loss).item()

def get_encdec_perplexity(row, i):
    model = scorer.model
    tokenizer = scorer.tokenizer
    device = model.device

    output_text = row[f'partial_{i}'] + "."
    input_text = row['input_1'] if i == 1 else row['firstname'] + ' '

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(output_text, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

    return torch.exp(loss).item()

# ----------------------------
# 🚀 Step 6: Run Perplexity on Dataset
# ----------------------------
tqdm.pandas()

for filename, template_prefix in datasets:
    df = pd.read_csv(filename, encoding='utf-8', on_bad_lines='warn')
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    for i in range(1, 4):
        template_col = f'{template_prefix}{i}'
        if args.model_type == 'EncoderDecoderLM':
            df[f'perplexity_{i}'] = df.progress_apply(lambda row: get_encdec_perplexity(row, i), axis=1)
        else:
            df[f'perplexity_{i}'] = df[template_col].progress_apply(get_perplexity)

    modified_name = args.model_name.replace("/", "_")
    output_filename = f'1_{modified_name}_results.csv'
    df.to_csv(output_filename, index=False)
    print(f'✅ Processed {filename} -> {output_filename}')
    print(df.head())
