import json
import numpy as np
import re

subset = 'leetcode'

def load_json(file_path):
    """Load and return data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
        
doc_file = '/home/siyue/Projects/llm2vec_reason/results/in_domain/supervised_leetcode_all_v0.1_3500/BrightLeetcode_leetcode_predictions_doc.json'
file = '/home/siyue/Projects/llm2vec_reason/results/in_domain/supervised_leetcode_all_v0.1_3500/BrightLeetcode_leetcode_predictions.json'

doc_file = load_json(doc_file)
file = load_json(file)

from datasets import load_dataset

# "examples" is a DatasetDict (train/validation/test)
data_examples = load_dataset("xlangai/BRIGHT", "examples")
data_examples = data_examples[subset]

def sort_dict_by_value_desc(data):
    """Sort a dictionary by its values in descending order."""
    return dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

preproc_file = '/home/siyue/Projects/llm2vec_reason/preproc/aops_problems_output.jsonl'
preproc_file = load_jsonl(preproc_file)
new_query = []
for row in preproc_file:
    q = row['response']['body']['choices'][0]['message']['content']
    if '**Final Answer**' in q:
        q = q.split('**Final Answer**')[0]
    indicators = ["**Theorem**","**theorem**"," Theorems\n",]
    for sep in indicators:
        if sep in q:
            q = q.split(sep)[-1]
            q = q.split('\n')
            q = [s for s in q if len(s.strip())>0]
            q = [s for s in q if re.search(r'boxed', s)==None and re.match(r'^[\\\[\]\{\}\d]+$', s)==None and 'answer' not in s.lower()]
            q = q[0]
            break
    else:
        q = 'NO THEOREM FOUND'
        print('ALARM: No theorem found!')
    new_query.append(q)

topk = 5
count = 0
tops = 1
scount = 0
for example, preproc_query in zip(data_examples, new_query):
    scount+=1
    print('---------')
    query_id = example['id']
    query = example['query']
    print(query_id)
    print(query, '\n')
    retrieved_id_score = file[query_id]
    retrieved_id_score = sort_dict_by_value_desc(retrieved_id_score)
    retrieved = doc_file[query_id]
    gold_ids = retrieved['gold']
    # print('preproc_query: ', preproc_query, '\n')
    print('golds:\n', '\n '.join([k for k in gold_ids]), '\n')
    for doc_id in retrieved_id_score:
        count += 1
        print('doc id: ', doc_id, '  |  is gold' if doc_id in gold_ids else 'x' ,'\nscore: ', np.round(retrieved_id_score[doc_id],4))
        print(retrieved['retrieved'][doc_id]['content']['text'],'\n')

        if count>=topk:
            count=0
            break
    
    if scount>=tops:
        break
