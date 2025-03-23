import os
import json
import glob
import random
from copy import deepcopy
from collections import defaultdict

import sys
sys.path.append('../')
from instances import *

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

data_path = 'check_problem_solution_tmp.jsonl'
problem_path = 'check_problem_solution_output.jsonl'

data = load_jsonl(data_path)
problem_check = load_jsonl(problem_path)

ids = [d['custom_id'] for d in problem_check]
data = [d for d in data if str(d['id']) in ids]

requests = []
counter = defaultdict(int)
for example, problem_check in zip(data, problem_check):
    response = problem_check['response']['body']['choices'][0]['message']['content']
    response = response.lower().split()[:30]
    response = ' '.join(response)
    correct_flg = 'yes' in response

    if correct_flg:
        custom_id = str(example['id'])
        problem = example['problem']
        solution = example['solution']
        domain = example['domain']
        counter[domain] += 1
        instance = example['instance']
        solution_prompt = f'**Problem**\n{problem}\n**Solution**\n{solution}\n\nIs this a correct solution to the problem and using the {domain} {instance}, yes or no?'
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": solution_prompt}
        ]
        requests.append({
            "custom_id": str(custom_id),
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": 
                {"model": "gpt-4o-mini", 
                "messages": messages,
                "max_tokens": 100},
        })


print(counter)
print(sum(list(counter.values())))

file_path = 'check_solution_input.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in requests:
        f.write(json.dumps(request) + '\n')

print(f"File saved to {file_path}")