import argparse
import json
import re
from datasets import load_dataset
import numpy as np
import pytrec_eval

from openai import OpenAI
client = OpenAI()

def prompt_gpt(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=150
    )
    return completion.choices[0].message.content


parser = argparse.ArgumentParser(description='augmentation data generation')
parser.add_argument('--subset', type=str, default='aops', help='Specify the subset of data to use')
parser.add_argument('--result_path', type=str, default='/home/siyue/Projects/llm2vec_reason/simcse_e5_aug_v3.5_aops')

args = parser.parse_args()


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

names = {
    'aops': 'BrightAops',
}
doc_file = f'{args.result_path}/{names[args.subset]}_{args.subset}_predictions_doc.json'
file = f'{args.result_path}/{names[args.subset]}_{args.subset}_predictions.json'
print(doc_file)

doc_file = load_json(doc_file)
file = load_json(file)

data_examples = load_dataset("xlangai/BRIGHT", "examples")
data_examples = data_examples[args.subset]

def sort_dict_by_value_desc(data):
    """Sort a dictionary by its values in descending order."""
    return dict(sorted(data.items(), key=lambda item: item[1], reverse=True))


topk = 10
ndcg_scores = []
old_ndcg_scores = []
for i, example in enumerate(data_examples):
    print('---------')
    query_id = example['id']
    query = example['query']
    reasoning = example['reasoning']
    print(query_id)
    # print(query, '\n')
    retrieved_id_score = file[query_id]
    retrieved_id_score = sort_dict_by_value_desc(retrieved_id_score)
    retrieved = doc_file[query_id]
    gold_ids = retrieved['gold']

    # print('golds:\n', '\n '.join([k for k in gold_ids]), '\n')
    retrieved_is_gold = []
    retrieved_is_relevant = []
    doc_ids = []
    scores = []
    for j, doc_id in enumerate(retrieved_id_score):
        if j==topk:
            break
        doc_ids.append(doc_id)
        scores.append(retrieved_id_score[doc_id])
        # print('doc id: ', doc_id, '  |  is gold' if doc_id in gold_ids else 'x' ,'\nscore: ', np.round(retrieved_id_score[doc_id],4))
        retrieved_is_gold.append(doc_id in gold_ids)
        retrieved_doc = retrieved['retrieved'][doc_id]['content']['text']
        prompt = f"""Here are two problems:
Problem 1:
{query}

Problem 2:
{retrieved_doc}

Your task is to analyze above two problems and tell if they are requiring the similar math theorems or formulas such as {reasoning}. First response yes or no, then explain the reason.
"""     
        response = prompt_gpt(prompt)
        relevance = 'yes' in response.lower()
        retrieved_is_relevant.append(relevance)

    run = {query_id:{d: score for d, score in zip(doc_ids, scores)}}

    qrel = {query_id:{d: 1 for d, f in zip(doc_ids, retrieved_is_relevant) if f}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg'})
    ndcg_score = evaluator.evaluate(run)
    ndcg_score = ndcg_score[query_id]['ndcg']
    ndcg_scores.append(ndcg_score)
    print(f"NDCG@10 for query {query_id}: {ndcg_score:.4f}")

    qrel = {query_id:{d: 1 for d in gold_ids}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg'})
    ndcg_score = evaluator.evaluate(run)
    ndcg_score = ndcg_score[query_id]['ndcg']
    old_ndcg_scores.append(ndcg_score)
    print(f"OLD GOLD: NDCG@10 for query {query_id}: {ndcg_score:.4f}\n")

    if i>5: 
        break

# If you want the average NDCG@10 for all queries
average_ndcg = np.mean(ndcg_scores)
print(f"==========")
print(f"Average NDCG@10: {average_ndcg:.4f}")

old_average_ndcg = np.mean(old_ndcg_scores)
print(f"\nAverage Old Gold NDCG@10: {old_average_ndcg:.4f}")