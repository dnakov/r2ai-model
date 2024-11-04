import pandas as pd
import datasets
import openai
import random
import time
import json
import re
client = openai.OpenAI()
def parse_usage():
  all_commands = open('data/radare2/sources/all_commands.txt').read().split('\n')
  usage_blocks = []
  usage_block = None
  for line in all_commands:
    if len(line.strip()) == 0:
      usage_block = None
      continue

    if line.startswith('Usage:'):
      raw_text = line
      line = line.replace('Usage:', '').strip()
      usage_parts = line.split('  ')
      usage_block = {
        'main_command': usage_parts[0],
        'description': usage_parts[1].strip() if len(usage_parts) > 1 else '',
        'commands': [],
        'raw_text': raw_text
      }
      usage_blocks.append(usage_block)
    elif usage_block:
      usage_block['raw_text'] += '\n' + line
      if line.startswith('| '):
        line = line[2:].strip()
        command_parts = line.split('  ')
        usage_block['commands'].append({
          'command': command_parts[0],
          'description': ' '.join(command_parts[1:]).strip()
        })
      else:
        usage_block['description'] += ' ' + line.strip()
        usage_block['description'] = usage_block['description'].strip()
    else:
      print(line)

  flat_usage_blocks = []
  
  for usage_block in usage_blocks:
    prev_command = None
    for command in usage_block['commands']:
      flat_usage_block = {}
      flat_usage_block['main_command'] = usage_block['main_command']
      flat_usage_block['main_description'] = usage_block['description']
      flat_usage_block['command'] = command['command']
      if 'as above' in command['description'] and prev_command:
        idx = command['description'].index('as above') + 8
        flat_usage_block['description'] = prev_command['description'] + ' ' + command['description'][idx:]
      else:
        flat_usage_block['description'] = command['description']
      flat_usage_blocks.append(flat_usage_block)
      prev_command = command

  df = pd.DataFrame(flat_usage_blocks)

  df.to_csv('data/radare2/sources/usage.tsv', sep='\t', index=False)
  for b in usage_blocks:
    b['commands'] = json.dumps(b['commands'])
  df = pd.DataFrame(usage_blocks)
  df.to_csv('data/radare2/sources/usage_blocks.tsv', sep='\t', index=False)
  # dataset = datasets.Dataset.from_list(flat_usage_blocks)
  # dataset.to_parquet('data/radare2/sources/usage.parquet')
  # dataset.to_json('data/radare2/sources/usage.jsonl', lines=True)

def get_price(response):
  pt = response.usage.prompt_tokens
  ct = response.usage.completion_tokens
  c_pt = 2.5 / 1000000
  c_ct = 10 / 1000000
  cost = (pt * c_pt) + (ct * c_ct)
  return f"${cost:.3f} pt: {pt}, ct: {ct}"

def create_prompts():
  # df = pd.read_csv('data/radare2/sources/usage.tsv', sep='\t')
  df = pd.read_csv('data/radare2/sources/usage_blocks.tsv', sep='\t')
  
  batch_requests = []
  for index, row in df.iterrows():
    commands = json.loads(row['commands'])
    system_prompt = f'''You are an expert at radare2 and have been tasked with figuring out all the questions anyone could ask that could be answered by a given command.
You'll be given an excerpt from the manual for the command in the format of:
Usage: <main_command_category>\t<help_text>
| <command>\t<help_text>
| <command_2>\t<help_text>
...

Use your knowledge of radare2 and binaries to formulate your responses targeting users not well familiar with these concepts.
Focus on the given commands and their help text.
If you see arguments, come up with potentially valid questions and values to use for them
You must respond in the following format:
<response>
  <n>1</n>
  <question>user question</question>
  <command>user command</command>
  <explanation>explanation</explanation>
  <r2cmd>radare2 command</r2cmd>
</response>
<response>
  <n>2</n>
  ...



You MUST NOT include the command or parts of it in the question. Pretend you are a less knowledgeable person trying to perform a task.
You must provide responses covering all individual commands. If there are arguments, vary them.
Vary the questions with different words that imply seeking information or asking to perform something

Example Response:
<response>
  <n>1</n>
  <question>How to find the distance between `sym.main` and `sym.hello`</question>
  <command>distance between sym.main and sym.hello</command>
  <explanation>We can subtract the addresses to find the distance between them. We can use ?vi to show the decimal value of the math expression `sym.main-sym.hello`.</explanation>
  <r2cmd>?vi sym.main-sym.hello</r2cmd>
</response>
'''
#     user_prompt = f'''Usage: {row['main_command']}\t{row['main_description']}
# | {row['command']}\t{row['description']}'''
    user_prompt = row['raw_text'] + '\n\n' + f"Provide {len(commands)*10} responses covering all individual commands above."
    request = {
      "custom_id": f'request-{index}',
      "method": 'POST',
      "url": '/v1/chat/completions',
      "body": {
        'messages': [
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_prompt}
        ],
        'model': 'gpt-4o',
        'temperature': 0.5,
        'top_p': 0.9,
        'max_tokens': 16000
      }
    }
    batch_requests.append(request)
  dataset = datasets.Dataset.from_list(batch_requests)
  dataset.to_json('data/radare2/sources/usage_batch.jsonl', lines=True)

  # randomly pick 1 request and do completion
  random.shuffle(batch_requests)
  for request in batch_requests[:1]:
    print(request['body']['messages'][0]['content'])
    print(request['body']['messages'][1]['content'])
    print('-' * 50)
    response = openai.chat.completions.create(**request['body'])
    print(response.choices[0].message.content)
    print('-'*5, get_price(response))


def create_batch():
  batch_input_file = client.files.create(file=open('data/radare2/sources/usage_batch.jsonl', 'rb'), purpose='batch')
  batch = client.batches.create(input_file_id=batch_input_file.id, endpoint='/v1/chat/completions', completion_window='24h', metadata={'description': 'testing'})
  return batch

def parse_batch_results(batch_id):
  while True:
    batch = client.batches.retrieve(batch_id)
    print(batch.status)
    match batch.status:
      case 'completed':
        break
      case 'expired':
        print('Batch expired')
        return
      case 'cancelled':
        print('Batch cancelled')
        return
      case 'validating':
        time.sleep(1)
      case 'failed':
        print('Batch failed')
        return
      case 'cancelling':
        print('Batch cancelling')
        return
      case 'in_progress':
        time.sleep(30)
      case 'finalizing':
        time.sleep(30)
      case _:
        print('Unknown batch status', batch)
        return
  output_file_id = batch.output_file_id
  output_file = client.files.content(output_file_id)
  print(output_file.text)
  responses = []
  pattern = r'<response>\s*<n>(.*?)</n>\s*<question>(.*?)</question>\s*<command>(.*?)</command>\s*<explanation>(.*?)</explanation>\s*<r2cmd>(.*?)</r2cmd>\s*</response>'
  for i, line in enumerate(output_file.text.split('\n')):
    if len(line) == 0:
      continue
    try: 
      res = json.loads(line)
      text = res['response']['body']['choices'][0]['message']['content']
      matches = re.finditer(pattern, text, re.DOTALL)
      for match in matches:
        responses.append({
          'n': match.group(1).strip(),
          'question': match.group(2).strip(),
          'command': match.group(3).strip(), 
          'r2cmd': match.group(4).strip(),
          'explanation': match.group(5).strip()
        })
    except Exception as e:
      print(line)
      print('num', i, len(line))
      print(e)

  return responses

if __name__ == '__main__':
  parse_usage()
  create_prompts()
  # batch = create_batch()
  # resp = parse_batch_results(batch)
  # df = pd.DataFrame(resp)
  # df.to_csv('data/radare2/pending/every_command_per_block_gpt4o.tsv', sep='\t', index=False)
