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

file_path = 'check_problem_solution_input.jsonl'

# Get all matching files
file_paths = glob.glob("outputs/problem_solution_output_*.jsonl")
file_paths = sorted(file_paths)

data = [load_jsonl(path) for path in file_paths]

def separate(example):
    problem, solution = example.split("**Solution**:")
    problem = problem.replace("**Question**:",'')
    return problem.strip(), solution.strip()

def split_problem_solution(data):
    problems = []
    solutions = []
    for item in data:
        content = item['response']['body']['choices'][0]['message']['content']
        if ("**Question**:" not in content) or ("**Solution**:" not in content):
            problems.append(None)
            solutions.append(None)
        else:
            try:
                problem, solution = separate(content)
            except Exception as e:
                problem, solution = None, None
            problems.append(problem)
            solutions.append(solution)
    return problems, solutions

data = [split_problem_solution(d) for d in data]

meta = []
for domain in domains:
    list_instances = domains[domain]
    question_type = domain_question_mapping[domain]
    reference = example_datasets[domain]
    for instance in list_instances:
        meta.append({
            "domain": domain,
            "instance": instance,
            "question_type": question_type,
            "reference": reference,
        })

all_pairs = []
id_ = 0
for pairs in data:
    metas = deepcopy(meta)
    tmp = [pairs[0], pairs[1], metas]
    for problem, solution, meta_ in zip(*tmp):
        if problem!=None and solution!=None:
            meta_['problem'] = problem
            meta_['solution'] = solution
            id_ += 1
            meta_['id'] = id_
            all_pairs.append(meta_)

file_path = 'check_problem_solution_tmp.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for pair in all_pairs:
        f.write(json.dumps(pair) + '\n')

print(f"File saved to {file_path}")


requests=[]
problems_by_instance=defaultdict(list)
counter = defaultdict(int)
for example in all_pairs:
    custom_id = example['id']
    problem = example['problem']
    solution = example['solution']
    domain = example['domain']
    instance = example['instance']
    problems_by_instance[instance].append(problem)
    question_type = example['question_type']
    if question_type=='finance' and random.random() < 0.25:
        continue
    counter[domain] += 1
    problem_prompt = f'**Problem**\n{problem}\n\nIs this problem testing {domain} {instance}, yes or no?'
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": problem_prompt}
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

    
    
    # solution_prompt = f'**Problem**\n{problem}\n**Solution**\n{solution}\n\nIs this a correct solution to the problem and using the {domain} {instance}, yes or no?'

    # for i, prompt in enumerate([problem_prompt, solution_prompt]):

    #     tag = 'solution' if i==1 else 'problem'
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt}
    #     ]
    #     custom_id = f'{id_}_{tag}'
    #     if legacy==None or custom_id not in [x['custom_id'] for x in legacy]:
    #         requests.append({
    #             "custom_id": custom_id, 
    #             "method": "POST", 
    #             "url": "/v1/chat/completions", 
    #             "body": 
    #                 {"model": "gpt-4o-mini", 
    #                 "messages": messages,
    #                 "max_tokens": 100},
    #         })

print(counter)
print(sum(list(counter.values())))
# a = list(problems_by_instance.keys())[343]
# print(a,'\n')
# print(problems_by_instance[a][0])
# print('-----')
# print(problems_by_instance[a][1])
# print('-----')
# print(problems_by_instance[a][2])
# print('-----')
# print(problems_by_instance[a][3])
# print('-----')
# print(problems_by_instance[a][4])
# print('-----')
# print(problems_by_instance[a][5])
# print('-----')
# print(problems_by_instance[a][6])
# print('-----')
# print(problems_by_instance[a][7])
# assert 1==2


file_path = 'check_problem_solution_input.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in requests:
        f.write(json.dumps(request) + '\n')

print(f"File saved to {file_path}")



