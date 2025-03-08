import json

# Function to load a JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    return data

# Example usage
file_path = "/home/siyue/Projects/llm2vec_reason/scale+_results/aops/BrightRetrieval_aops_predictions_doc.json"
json_data = load_json(file_path)

from datasets import load_dataset
data_examples = load_dataset("xlangai/BRIGHT", "examples")
data_examples = data_examples['aops']
query_map = {}
for ex in data_examples:
    qid = ex['id']
    query = ex['query']
    query_map[qid] = query

for query in json_data:
    print('query: ', query)
    gold = json_data[query]['gold']
    gold_ids = list(gold.keys())
    print('gold: ', gold_ids)
    retrieved = json_data[query]['retrieved']
    doc_map = {}
    for docid in retrieved:
        doc_map[docid] = retrieved[docid]['content']['text']
    retrieved_ids = [x for x in retrieved if retrieved[x]['score']>0]
    retrieved_ids = retrieved_ids[:10]
    for k, id in enumerate(retrieved_ids):
        flg = id in gold_ids
        print(f'top {k+1}: {id}, {flg}')
        if flg:
            print('\n',query_map[query], '\n', doc_map[id])
    input("next")
    print('----')

