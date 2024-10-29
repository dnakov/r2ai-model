import pandas as pd
import json

SYSTEM_PROMPT = """
***RADARE2 MODE: ON***
"""

df = pd.read_csv('data/radare2/radare2_enriched.tsv', sep='\t')
jsonl_data = []
for _, row in df.iterrows():
    q = row['q']
    a = row['a']
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a}
    ]
    jsonl_data.append(json.dumps({"messages": conversation}))

with open('data/radare2/radare2_train.jsonl', 'w') as f:
    for item in jsonl_data:
        f.write(item + '\n')