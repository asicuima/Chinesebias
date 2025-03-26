import pandas as pd
import os

# Define input and output file names
models = [
    "meta-llama_Llama-3.1-8B",
    "google_gemma-7b",
    "THUDM_chatglm-6b",
    "TsinghuaAI_CPM-Generate"
]
categories = ["noun", "adjective"]

input_files = [f"../results/1_{model}_results_{category}.csv" for model in models for category in categories]
output_files = [f"../results/2_{model}_results_{category}.csv" for model in models for category in categories]

def process_file(input_filename, output_filename):
    df = pd.read_csv(input_filename, on_bad_lines='warn')
    
    for i in range(1, 4):
        cluster_summary = []
        overall_mean = df[f'perplexity_{i}'].mean()

        for group in df['Gender group'].unique():
            cluster_data = df[df['Gender group'] == group]
            cluster_gender = cluster_data['Gender group'].iloc[0]
            average_perplexity = round(cluster_data[f'perplexity_{i}'].mean(), 2)
            prop_perplexity = round(cluster_data[f'perplexity_{i}'].mean() / overall_mean, 3)
            cluster_summary.append({'Group': cluster_gender, 'Average Perplexity': average_perplexity, 'Proportional Perplexity': prop_perplexity})

        prop_df = pd.DataFrame(cluster_summary).sort_values(by='Average Perplexity')
        df = pd.merge(df, prop_df, on='Group', how='left')
        df[f'apx_{i}'] = df[f'perplexity_{i}'] / df['Proportional Perplexity']
        df = df.drop(['Average Perplexity', 'Proportional Perplexity'], axis=1)

    cluster_summary = []
    for group in df['Gender group'].unique():
        for descriptor in df['descriptor'].unique():
            group_df = df[(df['Gender group'] == group) & (df['descriptor'] == descriptor)]
            if not group_df.empty:
                cluster_summary.append({
                    'Group': group,
                    'apx_1': round(group_df['apx_1'].mean(), 2),
                    'apx_2': round(group_df['apx_2'].mean(), 2),
                    'apx_3': round(group_df['apx_3'].mean(), 2),
                    'descriptor': descriptor,
                    'axis': group_df['axis'].iloc[0],
                    'gender_nouns': group_df['Gender-specific nouns'].iloc[0],
                    'des_type': group_df['des_type'].iloc[0]
                })

    summary_df = pd.DataFrame(cluster_summary).sort_values(by='descriptor')
    summary_df.to_csv(output_filename, index=False)
    print(f"Processed {input_filename} -> {output_filename}")

# Process all files
for input_file, output_file in zip(input_files, output_files):
    if os.path.exists(input_file):
        process_file(input_file, output_file)
    else:
        print(f"Skipping {input_file}, file not found.")
