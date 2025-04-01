import json
import re
import pickle
import argparse
from collections import defaultdict, Counter

import sys
sys.path.append('../')
from instances import *

parser = argparse.ArgumentParser(description='augmentation data generation')
args = parser.parse_args()

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

tmp_file = "./check_problem_solution_tmp.jsonl"
check_file = "./check_solution_output.jsonl"
definition_file = "../gen_instance_definition/instance_definition_output.jsonl"

tmp_data = load_jsonl(tmp_file)
check_data = load_jsonl(check_file)
def_data = load_jsonl(definition_file)

ids = [d['custom_id'] for d in check_data]
tmp_data = [d for d in tmp_data if str(d['id']) in ids]

definition_by_instance = {}
for domain in domains:
    list_instances = domains[domain]
    question_type = domain_question_mapping[domain]
    reference = example_datasets[domain]
    for instance in list_instances:
        item = def_data.pop(0)
        response = item['response']['body']['choices'][0]['message']['content']
        definition_by_instance[instance] = response

def find_yes(string):
    string = string.lower().split()[:30]
    return 'yes' in ' '.join(string)


# check_ids = [d['custom_id'].split('_')[0] for d in check_data]
# tmp_data = [d for d in tmp_data if str(d['id']) in check_ids]

print('\n\n',Counter([d['domain'] for d in tmp_data]),'\n\n')

all_pairs = []
for example, check_solution in zip(tmp_data,check_data):
    example['id'] = str(example['id'])
    assert example['id']==check_solution['custom_id'], f"{example['id']} {check_solution['custom_id']}"
    check = check_solution['response']['body']['choices'][0]['message']['content']
    yes_solution = find_yes(check)
    if yes_solution:
        all_pairs.append(example)


print(f'{len(tmp_data)} total --> {len(all_pairs)} correct')

n_counter = defaultdict(list)
for pair in all_pairs:
    domain = pair['domain']
    instance = pair['instance']
    n_counter[domain].append(instance)
print(Counter([pair['domain'] for pair in all_pairs]),'\n\n')

for domain in n_counter:
    count = Counter(n_counter[domain])
    average = sum(count.values()) / len(count)
    print(domain,' ',average)
print('\n')

def find_math(all_pairs,ty):
    instances=[pair['instance'] for pair in all_pairs if pair['domain']==ty]
    instances=list(set(instances))
    return len(instances)
print('Algorithm Coverage: ', find_math(all_pairs,'algorithm'), ' / ', find_math(tmp_data,'algorithm'))
print('Math Coverage: ', find_math(all_pairs,'math theorem'), ' / ', find_math(tmp_data,'math theorem'))
print('Physics Coverage: ', find_math(all_pairs,'physics theorem'), ' / ', find_math(tmp_data,'physics theorem'))
print('\n')


def extract_code(text):
    pattern = re.compile(r'```(python|java|cpp)?\n(.*?)```', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        return text, None
    
    return matches[0][1], matches[0][0]

from collections import defaultdict
pairs_by_instance = defaultdict(list)
problem_by_instance = defaultdict(list)
solution_by_instance = defaultdict(list)
for pair in all_pairs:
    instance = pair['instance']
    problem = pair['problem']
    pair['solution'], pair['language'] = extract_code(pair['solution'])
    domain = pair['domain']
    solution = pair['solution']
    pair['definition'] = definition_by_instance[instance]
    pairs_by_instance[instance].append(pair)
    problem_by_instance[instance].append(problem)
    solution_by_instance[instance].append(solution)

import pickle
with open("before_hard_negative_tmp.pkl", "wb") as file:
    pickle.dump({
        "pairs_by_instance": pairs_by_instance,
        "problem_by_instance": problem_by_instance,
        "solution_by_instance": solution_by_instance,
        "definition_by_instance": definition_by_instance
    }, file)

