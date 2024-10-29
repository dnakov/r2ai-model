import anthropic
import openai
import pandas as pd
import time
from tqdm import tqdm
import random
import os
import json
from r2ai.auto import ChatAuto

from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")
model = "openai/gpt-4o"
# model = "claude-3-5-sonnet-20241022"
# model = "claude-3-opus-20240229"
# max_tokens = 4095
max_tokens = 16000
temperature = 0.7
top_p = 0.9

llm = ChatAuto(model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, timeout=120)
def generate_pair(messages):
    commands = open("data/radare2/sources/all_commands.txt", "r").read()
    fortunes = open("data/radare2/sources/fortunes.tips", "r").read()
    prompt = f"""You're a helpful assistant who is extremely knowledgeable about the reverse engineering, malware analysis and security space in general. 
You're a pro at using radare2 for many different tasks. Your job is to enumerate all possible ways someone could use radare2 to answer a question. 
You should come up with a variety of different questions that utilize a variety of different commands. 
The radare2_command should be valid and be able to be run. 

<radare2_commands>
{commands}
</radare2_commands>

<radare2_fortunes>
{fortunes}
</radare2_fortunes>

<response_format>
[{{"q": "<question>", "a": "<radare2_command>"}}, ...]
</response_format>

<examples>
{json.dumps(examples())}
</examples>

Datetime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    text = ""

    try:
        response = llm.chat(messages=[{"role": "system", "content": prompt}, *messages], stream=False)
        # Parse response
        text = response['content']
        messages.append({"role": "assistant", "content": text})
        data = json.loads(text.replace("```json", "").replace("```", ""))
        if(len(data) > 0):
            print(data)
            return data
    except Exception as e:
        print('text:', text)
        print(f"Error generating pair: {e}")
        return []

def generate_dataset(file_path, num_examples=1000, messages=[], category=None):
    """Generate multiple examples and save to CSV"""
    
    data = []
    pbar = tqdm(total=num_examples, desc=f"Generating examples for {category}")
    lines = generate_pair(messages)
    while len(data) < num_examples:
        lines = generate_pair(messages)
        
        if len(lines) > 0:
            data.extend(lines)
            pbar.update(len(lines))
        
        # Sleep to respect rate limits
        time.sleep(0.5)
        messages.append({"role": "user", "content": "generate more"})
    print(data)
    pbar.close()
    
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

# read data/radare2_ok.tsv and convert to json array    
def examples():
    df = pd.read_csv('data/radare2/radare2_ok.tsv', sep='\t')
    return df.sample(n=10).to_dict('records')

if __name__ == "__main__":
    # categories = ["malware", "forensics", "crypto", "general", "vulnerability", "exploit", "reverse engineering", "binary analysis", "binary patching", "debugging"]
    categories = ["crypto", "general", "vulnerability", "exploit", "reverse engineering", "binary analysis", "binary patching", "debugging"]
    num_examples = 100
    for category in categories:
        messages = [{"role": "user", "content": f"""generate {num_examples} examples that would be applicable to this category: {category}. Respond in JSON format: [{{"q": "<question>", "a": "<radare2_command>"}}, ...] and NOTHING ELSE."""}]
        file_path = f'data/radare2/pending/{today}-{category.replace(" ", "_")}-{model.replace("/", ":")}-top_p-{top_p}-temp-{temperature}.tsv'
        generate_dataset(file_path=file_path, num_examples=num_examples, messages=messages, category=category)  # Generate 1500 examples (1425 train, 75 val)
        validate_dataset(file_path=file_path)
