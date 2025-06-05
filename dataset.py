import json
import pandas as pd
from tqdm import tqdm
import os, sys

data_dir = 'data/'
def process_json(data_path: str) -> pd.DataFrame:
    with open(data_path, "rb") as f:
        data = json.load(f)
    TASKS = set(data.keys())

    rows = []
    for task_name, questions_answers in tqdm(data.items()):
        for question, answer in questions_answers.items():
            rows.append({"task": task_name, "question": question, "answer": answer})

    df = pd.DataFrame(rows)
    df["id"] = df.index
    return df

def read_data(dataset_name: str) -> pd.DataFrame:
    df = process_json(os.path.join(data_dir, f'{dataset_name}.json'))
    return df