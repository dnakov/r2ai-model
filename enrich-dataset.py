import anthropic
import openai
import pandas as pd
import time
from tqdm import tqdm
import random
import os
import json
from io import StringIO

from r2ai.auto import ChatAuto

from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")
# model = "openai/gpt-4o"
model = "claude-3-5-sonnet-20241022"
# model = "claude-3-opus-20240229"
# max_tokens = 4095
max_tokens = 8000
temperature = 0.7
top_p = 0.9

llm = ChatAuto(model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, timeout=120)

def generate_dataset(num_examples=10):
    """Generate multiple examples and save to CSV"""
    
    data = pd.DataFrame()
    ok_data = pd.read_csv('data/radare2/radare2_ok.tsv', sep='\t')
    for i, row in ok_data.iterrows():
        messages = [
            {"role": "system", "content": f"""You are a helpful data expert who is tasked with enriching the dataset with additional examples.
            You'll be given a question and answer.
            The answer is a valid radare2 command.
            Generate {num_examples}-{num_examples*2} more examples that are very similar to the question and answer above.
            Only vary the answer if there are clear parameters that can be changed, like sorting, filtering, things like top N, numbers, addresses, etc.
            The questions should vary from minimal to longer. Minimal could be just the important part and nothing else. Sometimes, it's even just 1 word, always in english.
            Longer means should be looking more like a full question.
            If you can vary the command, do {num_examples}-{num_examples*2} more examples.
            respond in tsv format: Question\tAnswer
            """}, 
            {"role": "user", "content": f"""{row['q']}\t{row['a']}"""}
        ]
        
        response = llm.chat(messages=messages, stream=False)
        print(response['content'])
        
        parsed = pd.read_csv(StringIO(response['content']), sep='\t', names=['q', 'a'])
        data = pd.concat([data, pd.DataFrame([row]), parsed], ignore_index=True)
        # Save intermediate results after each batch
        data.to_csv(f'data/radare2/radare2_enriched.tsv', sep='\t', index=False)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure the 'q' and 'a' columns are properly formatted
    df['q'] = df['q'].apply(lambda x: x if x else "")
    df['a'] = df['a'].apply(lambda x: x if x else "")
    
    # Save both train and validation sets
    # train_size = int(len(df) * 0.95)
    
    df_train = df
    # df_val = df[train_size:]

    df_train.to_csv(file_path, sep='\t', index=False)
    # df_val.to_csv(f'data/pending/{today}_radare2_val.tsv', sep='\t', index=False)
    
    print(f"Generated {len(df)} examples")
    print(f"Training examples: {len(df_train)}")
    # print(f"Validation examples: {len(df_val)}")
    
    # Display some examples
    print("\nSample examples:")
    for _, row in df.head(3).iterrows():
        print("\nQ:", row['q'])
        print("A:", row['a'])
        print("-" * 50)
    return df

def validate_dataset(file_path='data/radare2/radare2_train.tsv'):
    """Validate the generated dataset"""
    df = pd.read_csv(file_path, sep='\t')
    
    # Basic validation
    print("\nDataset Statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Average question length: {df['q'].str.len().mean():.1f} characters")
    print(f"Average answer length: {df['a'].str.len().mean():.1f} characters")
    print(f"Null values: {df.isnull().sum().sum()}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate entries: {duplicates}")

if __name__ == "__main__":
    num_examples = 10
    ok_data = pd.read_csv('data/radare2/radare2_ok.tsv', sep='\t')
    generate_dataset(num_examples=num_examples)