import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Retrieve the token from the environment
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Authenticate with Hugging Face
if huggingface_token:
    login(token=huggingface_token)
    print("✅ Successfully logged in to Hugging Face!")
else:
    print("❌ Error: Hugging Face token not found in .env file.")
    exit(1)  # Exit if the token is missing
    
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)


import pandas as pd
import lmppl
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='Name of the model')
parser.add_argument('--model_type', type=str, help='Type of model')
parser.add_argument('--num_gpus', type=int, help='Number of GPUs available')

args = parser.parse_args()


# List of dataset filenames and corresponding result suffixes
datasets = [
    ('chinese_bias_dataset.xlsx', 'template_')
]

# Initialize the scorer based on model type
if args.model_type == 'MaskedLM':
    scorer = lmppl.MaskedLM(args.model_name, num_gpus=args.num_gpus, max_length=22)
elif args.model_type == 'EncoderDecoderLM':
    scorer = lmppl.EncoderDecoderLM(
        args.model_name, 
        num_gpus=args.num_gpus, 
        device_map="auto", 
        low_cpu_mem_usage=True
    )
elif args.model_type == 'LM':
    scorer = lmppl.LM(
        args.model_name, 
        num_gpus=args.num_gpus,
        device_map="auto", 
        low_cpu_mem_usage=True
    )

# Function to get the perplexity value for a given template
def get_perplexity(template):
    return scorer.get_perplexity(template)

def get_encdec_perplexity(row, i):
    outputs = row[f'partial_{i}'] + "."
    inputs = row['input_1'] if i == 1 else row['firstname'] + ' '
    return scorer.get_perplexity(input_texts=inputs, output_texts=outputs)

tqdm.pandas()

for filename, template_prefix in datasets:
    df = pd.read_excel(filename, engine="openpyxl")
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
    print(f'Processed {filename} -> {output_filename}')

    print(df.head())


