import json
import random
import os
from collections import defaultdict

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

E5_EMBEDDING_PROMPTS = {
    "allnli": [
        "Given a premise, retrieve a hypothesis that is entailed by the premise",
        "Retrieve semantically similar text",
    ],
    # "dureader": "Given a Chinese search query, retrieve web passages that answer the question",
    "eli5_question_answer": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "miracl": "Given a question, retrieve Wikipedia passages that answer the question",
    "mrtydi": "Given a question, retrieve Wikipedia passages that answer the question",
    "msmarco_passage": "Given a web search query, retrieve relevant passages that answer the query",
    "msmarco_document": "Given a web search query, retrieve relevant documents that answer the query",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "quora_duplicates": [
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "Find questions that have the same meaning as the input question",
    ],
    "squad": "Retrieve Wikipedia passages that answer the question",
    # "t2ranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "trivia_qa": "Retrieve Wikipedia passages that answer the question",
}


class E5Mix(Dataset):
    def __init__(
        self,
        dataset_name: str = "E5Mix",
        split: str = "validation",
        file_path: str = "cache/echo-data",
        aug_file_path: str = "/home/siyue/Projects/llm2vec_reason/augmentation/output/augmentation_data.jsonl",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        domain: str = 'all',
        task: str = 'all',
        add_e5: bool = False
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        # NEW
        self.aug_file_path = aug_file_path
        self.domain = domain
        self.task = task
        self.add_e5 = add_e5

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        # logger.info(f"Loading E5 data from {file_path}...")
        # # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        if self.add_e5:
            for dataset in E5_EMBEDDING_PROMPTS:
                logger.info(f"Loading dataset {dataset}...")
                if dataset not in data_map:
                    data_map[dataset] = []
                with open(os.path.join(file_path, f"{dataset}.jsonl"), "r") as f:
                    dataset_samples = f.readlines()

                dataset_samples = [json.loads(d) for d in dataset_samples]

                for i, sample in enumerate(dataset_samples):
                    instruction = (
                        E5_EMBEDDING_PROMPTS[dataset]
                        if isinstance(E5_EMBEDDING_PROMPTS[dataset], str)
                        else E5_EMBEDDING_PROMPTS[dataset][i % 2]
                    )
                    query = f"{instruction}; " + self.separator + sample["query"]
                    if dataset in [
                        "allnli_split2",
                        "quora_duplicates_split1",
                        "quora_duplicates_split2",
                    ]:
                        pos = (
                            f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                            + self.separator
                            + sample["positive"]
                        )
                        neg = (
                            f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                            + self.separator
                            + sample["negative"]
                        )
                    else:
                        pos = self.separator + sample["positive"]
                        neg = self.separator + sample["negative"]

                    data_map[dataset].append(id_)

                    all_samples.append(
                        DataSample(
                            id_=id_,
                            query=query,
                            positive=pos,
                            negative=neg,
                            task_name=dataset,
                        )
                    )
                    id_ += 1

            # combine split1 and split2
            new_data_map = {}
            for dataset in data_map:
                new_dataset = dataset.replace("_split1", "").replace("_split2", "")
                if new_dataset not in new_data_map:
                    new_data_map[new_dataset] = []
                new_data_map[new_dataset] += data_map[dataset]
            data_map = new_data_map

            # equalize size for each one
            for dataset in data_map:
                if len(data_map[dataset])>20000:
                    data_map[dataset] = random.sample(data_map[dataset],20000)
                # print(dataset, len(data_map[dataset]))

            if self.shuffle_individual_datasets:
                for task, samples in data_map.items():
                    random.shuffle(samples)

            datasets = list(data_map.keys())

            logger.info(
                f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
            )
            all_batches = []
            for dataset in datasets:
                dataset_samples = data_map[dataset]
                for i in range(0, len(dataset_samples), self.effective_batch_size):
                    batch = dataset_samples[i : i + self.effective_batch_size]
                    if len(batch) == self.effective_batch_size:
                        all_batches.append(batch)
                    else:
                        logger.info(f"Skip 1 batch for dataset {dataset}.")
            random.shuffle(all_batches)


        ## NEW
        with open(self.aug_file_path, "r", encoding="utf-8") as f:
            augment_samples = f.readlines()

        augment_samples = [json.loads(d) for d in augment_samples]
 
        ##
        # datasets = list(set([s['task'] for s in augment_samples]))

        domain_map = {
            'aops': ['math'],
            'leetcode': ['coding'],
            'theoremqa': ['math','physics','finance'],
            'all':['math','physics','finance','coding','math']
        }
        domain_question_mapping = {
            'algorithm': 'coding',
            'data structure': 'coding',
            'math theorem': 'math',
            'physics theorem': 'physics',
            'finance formula': 'finance',
        }

        self.all_samples = defaultdict(lambda: [])

        for id_, augment_sample in enumerate(augment_samples):
            # domain filter
            sample_domain = augment_sample['domain']
            sample_domain_type = domain_question_mapping[sample_domain]
            if sample_domain_type not in domain_map[self.domain]:
                continue
            # task filter
            if self.task != 'all' and augment_sample['task_type'] != self.task:
                continue

            instruction = augment_sample["task"]
            query = f"{instruction}; " + self.separator + augment_sample["user_query"]
            pos = self.separator + augment_sample["positive_document"]
            neg = self.separator + augment_sample["hard_negative_document"]
            self.all_samples[instruction].append(DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name='augment',
                        pos_instance=augment_sample['instance'],
                        neg_instance=augment_sample['negative_instance']
                    ))
         
        self.data = []
        for dataset in self.all_samples:
            print(len(self.all_samples[dataset]), self.effective_batch_size)
            while len(self.all_samples[dataset])>=self.effective_batch_size:
                popped_items = []
                for _ in range(self.effective_batch_size):
                    random_index = random.randint(0, len(self.all_samples[dataset]) - 1)
                    popped_items.append(self.all_samples[dataset].pop(random_index))
                self.data.append(popped_items)
            print(dataset, ' : ', len(popped_items), ' samples.')

        random.shuffle(self.data)

        # self.data = random.sample(self.data, int(10240/self.effective_batch_size))
        
        logger.info(f"Loaded {len(self.data)*self.effective_batch_size} augmented samples.")
        
        if self.add_e5:
            e5 = random.sample(all_batches, int(40000/self.effective_batch_size))
            tmp = []
            for batch in e5:
                tmp.append([all_samples[idx] for idx in batch])
            e5 = tmp
            self.data += e5

        random.shuffle(self.data)
        self.data = [item for sublist in self.data for item in sublist]

        # final_idx_order = []
        # self.data = []
        # self.counter_augment = 0
        # for batch in all_batches:

            # replace_index = random.randint(0, len(batch)-1)

            # for i, idx in enumerate(batch):
            #     item = all_samples[idx]

                # if i==replace_index and len(augment_samples)>0:
                # if i==replace_index:
                #     # augment_sample = augment_samples.pop(0)
                #     augment_sample = random.choice(augment_samples)

                #     instruction = augment_sample["task"]
                #     query = f"{instruction}; " + self.separator + augment_sample["user_query"]
                #     pos = self.separator + augment_sample["positive_document"]
                #     neg = self.separator + augment_sample["hard_negative_document"]
                    
                #     item.query = query
                #     item.positive = pos
                #     item.negative = neg
                #     item.task_name = "augment"
                #     self.counter_augment += 1

                # self.data.append(item)

        # logger.info(f"Total {len(self.data)} samples.")
        # self.data = self.data[:30000]

        # self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

        # self.data += tmp
        # random.shuffle(self.data)
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "E5Mix does not have a validation split."
