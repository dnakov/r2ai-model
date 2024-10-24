import anthropic
import pandas as pd
import time
from tqdm import tqdm
import random
import os
import json
# Initialize Claude client
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

model = "claude-3-5-sonnet-20241022"
max_tokens = 8192
temperature = 0.9
messages = [
  {"role": "user", "content": """generate many examples and return only JSON in the following format: [{"q": "<question>", "a": "<radare2_command>"}, ...]"""}
]
def generate_pair():
    prompt = open("dataset-gen-prompt.txt", "r").read()
    text = ""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=prompt,
            messages=messages
        )
        print(response)        
        # Parse response
        messages.append({"role": "assistant", "content": response.content[0].text})
        text = response.content[0].text
        
        data = json.loads(text)
        if(len(data) > 0):
            print(data)
            return data
    except Exception as e:
        print('text:', text)
        print(f"Error generating pair: {e}")
        return []

def generate_dataset(num_examples=1000):
    """Generate multiple examples and save to CSV"""
    
    data = []
    pbar = tqdm(total=num_examples, desc="Generating examples")
    lines = generate_pair()
    while len(data) < num_examples:
        lines = generate_pair()
        
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
    train_size = int(len(df) * 0.95)
    
    df_train = df[:train_size]
    df_val = df[train_size:]
    
    df_train.to_csv('data/radare2_train.csv', index=False)
    df_val.to_csv('data/radare2_val.csv', index=False)
    
    print(f"Generated {len(df)} examples")
    print(f"Training examples: {len(df_train)}")
    print(f"Validation examples: {len(df_val)}")
    
    # Display some examples
    print("\nSample examples:")
    for _, row in df.head(3).iterrows():
        print("\nQ:", row['q'])
        print("A:", row['a'])
        print("-" * 50)
    return df

def validate_dataset(file_path='data/radare2_train.csv'):
    """Validate the generated dataset"""
    df = pd.read_csv(file_path)
    
    # Basic validation
    print("\nDataset Statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Average question length: {df['q'].str.len().mean():.1f} characters")
    print(f"Average answer length: {df['a'].str.len().mean():.1f} characters")
    print(f"Null values: {df.isnull().sum().sum()}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate entries: {duplicates}")

# def format_dataset(file_path='data/radare2_train.csv'):
#     df = pd.read_csv(file_path)
#     df = pd.DataFrame(df, columns=['q', 'a'])
    
#     # Save both train and validation sets
#     train_size = int(len(df) * 0.95)
    
#     df_train = df[:train_size]
#     # df_val = df[train_size:]
    
#     df_train.to_csv('data/radare2_train.csv', index=False)
#     # df_val.to_csv('data/radare2_val.csv', index=False)
    
#     print(f"Generated {len(df)} examples")
#     print(f"Training examples: {len(df_train)}")
#     # print(f"Validation examples: {len(df_val)}")
    
#     # Display some examples
#     print("\nSample examples:")
#     for _, row in df.head(3).iterrows():
#         print("\nQ:", row['q'])
#         print("A:", row['a'])
#         print("-" * 50)
#     return df


if __name__ == "__main__":
    # Generate dataset
    generate_dataset(num_examples=100)  # Generate 1500 examples (1425 train, 75 val)
    # format_dataset()
    # Validate the generated dataset
    validate_dataset()