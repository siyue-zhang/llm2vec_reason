import random
import json
from collections import Counter

import sys
sys.path.append('../')
from instances import *


def pick_language():

    def weighted_random_choice(instances, probabilities):
        return random.choices(instances, weights=probabilities, k=1)[0]

    instances = ['Python', 'Java', 'C++']
    probabilities = [0.6, 0.2, 0.2]
    chosen_instance = weighted_random_choice(instances, probabilities)

    return chosen_instance


def generate_problem_solution_prompt(question_type, domain, instance, ref):
    add = ' short' if random.random() < 0.5 else ''
    prompt = f"Your task is to create one{add} {question_type} problem with a correct solution.\n"
    if question_type=='coding':
        language = pick_language()
        add = '\nThe problem should be based on real life human activities.' if random.random() < 0.6 else ''
        prompt += f"The problem should be new and unique, not similar to common existing problems.{add}\n"
        prompt += f"The problem should test about the {domain}: {instance}, without explicitly mentioning {instance}.\n"
        if random.random() > 0.4:
            add = random.choice(ref)
            prompt += f"The problem should be as difficult as {add}.\n"
        prompt += f"The solution code should be written in the programming language {language}.\n"
    elif question_type!='math':
        add = '\nThe problem should be based on real life human activities.' if random.random() < 0.5 else ''
        prompt += f"The problem should be new and unique, not similar to common existing problems.{add}\n"
        prompt += f"The problem should test about the {domain}: {instance}, without explicitly mentioning {instance}.\n"
        if random.random() > 0.4:
            add = random.choice(ref)
            prompt += f"The problem should be as difficult as {add}.\n"
        if random.random() < 0.5:
            add = 'in multiple steps' if random.random() < 0.7 else f'by multiple {domain}s'
            prompt += f"The problem should be solved {add}.\n"
    else:
        options = [
            '\nThe problem should be based on real life human activities.',
            '\nThe problem should a multi-choice problem.',
            '\nThe problem should be theoretical and mathematical but not a proof problem.',
        ]
        options += options[-1]
        add = random.choice(options) 
        prompt += f"The problem should be new and unique, not similar to common existing problems.{add}\n"
        prompt += f"The problem should test about the {domain}: {instance}, without explicitly mentioning {instance}.\n"
        if random.random() > 0.2:
            add = random.choice(ref)
            prompt += f"The problem should be as difficult as {add}.\n"
        if random.random() < 0.5:
            add = 'in multiple steps' if random.random() < 0.7 else f'by multiple {domain}s'
            prompt += f"The problem should be solved {add}.\n"
        if random.random() < 0.5:
            prompt += f"The solution should not explicitly mention {instance}.\n"
    prompt += """
Format the response like this:

**Question**:
[Question here]

**Solution**:
[Solution here]
"""

    return prompt


def generate_requests():
    requests=[]
    count_domains=[]
    counter=0
    for domain in domains:
        list_instances = domains[domain]
        question_type = domain_question_mapping[domain]
        ref = example_datasets[domain]
        for instance in list_instances:
            counter += 1
            count_domains.append(domain)
            prompt = generate_problem_solution_prompt(question_type, domain, instance, ref)
            # if random.random() < 0.1:
            #     print(prompt)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            requests.append({
                "custom_id": f"request-{counter}", 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": 
                    {"model": "gpt-4o-mini",
                    # "temperature": 0.7,
                    "messages": messages,
                    "max_tokens": 2000},
            })
    print(Counter(count_domains))
    return requests


K = 16
for k in range(K):
    # if k<4:
    #     continue
    requests =  generate_requests()
    file_path = f'./inputs/problem_solution_input_{k+1}.jsonl'
    with open(file_path, 'w', encoding='utf-8') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')

    print(f"File saved to {file_path}")
