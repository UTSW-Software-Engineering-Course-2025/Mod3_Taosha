# %% Import files
from typing import List, Optional, Dict, Union

from collections import defaultdict
from dataclasses import dataclass
import json
import pandas as pd
from tqdm import tqdm
import ollama
import os, sys
import seaborn as sns
from matplotlib import pyplot as plt
import mlflow
import mlflow.tracking
import requests # For embedding API
from ollama import chat
from pydantic import BaseModel, Field
import pprint

# MLflow Configuration (ensure server is running and accessible)
mlflow.set_tracking_uri("http://198.215.61.34:8153/") # As in your original file

# Ensure this path is correct for your environment
os.chdir("/project/GCRB/Hon_lab/s440862/courses/se/MODULE_3_MATERIALS/mod3")
os.environ["NO_PROXY"] = "*"

from metrics import *

# %% Model configuration

# 2.1 Data Configuration
DATASET_NAME = "genehop"
data_path = f"data/{DATASET_NAME}.json"

# 2.2 Model Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "qwen3:4b"
EXPERIMENT_NAME = f"{DATASET_NAME}_{OLLAMA_MODEL_NAME}"
RESULT_SAVE_PATH = f"output/{EXPERIMENT_NAME}.csv"
OUTPUT_DIR = "output/"

# 2.3 Evaluation and Logging Configuration
MLFLOW_DIR_NAME = f"Taosha"
mlflow.set_experiment(MLFLOW_DIR_NAME)

# %% Data Loading

with open(data_path, "rb") as f:
    data = json.load(f)

# reformating to dataframes
TASKS = set(data.keys())

rows = []
for task_name, questions_answers in tqdm(data.items()):
    for question, answer in questions_answers.items():
        rows.append({"task": task_name, "question": question, "answer": answer})

df = pd.DataFrame(rows)
df["id"] = df.index


# %% Model Setups

### initiate
OllamaClient = ollama.Client(host=OLLAMA_BASE_URL)

# %% Model Setups

### system msg

system_message = [
    {
        "role": "system",
        "content": f"Hello. You are an expert in bioinformatics, and your job now is to use the NCBI Web APIs to give accurate and neat step-to-step answers to genomic questions in aspects of {list(data.keys())}. \
            For example, if I ask 'What are the aliases of the gene that contains this sequnece:ATTGTGAGAGTAACCAACGTGGGGTTACGGGGGAGAATCTGGAGAGAAGAGAAGAGGTTAACAACCCTCCCACTTCCTGGCCACCCCCCTCCACCTTTTCTGGTAAGGAGCCC. Let's decompose the question to sub-questions and solve them step by step.', your answer should be: 'Answer: ['FCGR3A', 'CD16', 'FCG3', 'CD16A', 'FCGR3', 'IGFR3', 'IMD20', 'FCR-10', 'FCRIII', 'CD16-II', 'FCGRIII', 'FCRIIIA', 'FcGRIIIA']'. \
                If it is a question in terms of 'sequence gene alias' or 'Disease gene location', you should answer with a list of the componens; \
                    if it is a question in terms of 'SNP gene function', you should answer with a short, summary sentence. Thanks.",
    }
]

### few-shot examples

example_messages = [
    {
        "role": "user", 
        "content": "Question: What are the aliases of the gene that contains this sequnece:ACTTCCAACATGGCGGCGGCCGGGGCGGCGGTGGCGCGCAGCCCGGGAATCGGAGCGGGACCTGCGCTGAGAGCCCGGCGCTCGCCCCCGCCGCGGGCCGCACGGCTGCCGCG. Let's decompose the question to sub-questions and solve them step by step."},
    {
        "role": "assistant", 
        "content": "Answer: ['QSOX2', 'SOXN', 'QSCN6L1']"},
    {
        "role": "user",
        "content": "Question: List chromosome locations of the genes related to Split-foot malformation with mesoaxial polydactyly. Let's decompose the question to sub-questions and solve them step by step.",
    },
    {
        "role": "assistant", 
        "content": "Answer: ['2q31.1']"},
    {
        "role": "user",
        "content": "Question: What is the function of the gene associated with SNP rs1218214598? Let's decompose the question to sub-questions and solve them step by step.",
    },
    {
        "role": "assistant", 
        "content": "Answer: ncRNA"},
    {
        "role": "user",
        "content": "Question: What is the function of the gene associated with SNP rs937944577? Let's decompose the question to sub-questions and solve them step by step.",
    },
    {
        "role": "assistant", 
        "content": "Answer: This gene is a member of the intermediate filament family. Intermediate filaments are proteins which are primordial components of the cytoskeleton and nuclear envelope. The proteins encoded by the members of this gene family are evolutionarily and structurally related but have limited sequence homology, with the exception of the central rod domain. Multiple alternatively spliced transcript variants encoding different isoforms have been found for this gene."},    
]

# %%  4.4 Implementing the model request function
### The function should return the response from the model and extract the answer (everything after 'Answer :' based on the format of the examples above)


def query_model(
    client: ollama.Client,
    system_message: dict,
    few_shot_examples: List[dict],
    user_query: str,
    model_name: str = OLLAMA_MODEL_NAME,
) -> str:
    messages = []
    messages.extend(system_message)
    messages.extend(few_shot_examples)
    messages.append({"role": "user", "content": f"Question: {user_query}"})

    try:
        response = client.chat(model=model_name, messages=messages)
        content = response["message"]["content"].strip()
        if "Answer:" in content:
            return content.split("Answer:")[-1].strip()
        return content
    except Exception as e:
        print(f"Error querying model: {e}")
        return None


# %% 5. Metrics

# --- Metric Map for GeneHop (YOU NEED TO CUSTOMIZE THIS) ---
metric_task_map = defaultdict(lambda: exact_match, 
    {
        "sequence gene alias": Levenshtein_score,
        'Disease gene location': Levenshtein_score,
        "SNP gene function": cos_similarity,
    }
)


# %%  6. Evaluation Loop & Result Saving Structure

@dataclass
class Result:
    id: int
    task: str
    question: str
    answer: str
    raw_prediction: Optional[str]
    processed_prediction: Optional[str]
    score: Optional[float]
    success: bool

def save_results(results: List[Result], results_csv_filename: str) -> None:
    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df.to_csv(results_csv_filename, index=False)
    print(f"Results saved.")
    return results


# %%  6. Evaluation Loop
with mlflow.start_run(run_name=f"{EXPERIMENT_NAME}") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # --- Log Parameters ---
    mlflow.log_param("dataset_name", DATASET_NAME)
    mlflow.log_param("ollama_model_name", OLLAMA_MODEL_NAME)
    mlflow.log_param("ollama_base_url", OLLAMA_BASE_URL)
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("system_prompt_preview", system_message[0]["content"][:250])
    mlflow.log_param("num_few_shot_examples", len(example_messages) // 2)

    # 6.2 Loop over the dataset with a progress bar

    # * Do not forget to add the results to our Result list, both successful and failed predictions
    # * API calls will not always work, so make sure we capture the exceptions from failed calls
    #    and add them to the Result list with a `status=False`
    results: List[Result] = []
    for index, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Evaluating {DATASET_NAME}"
    ):
        example_id = row["id"]
        task = row["task"]
        question = row["question"]
        true_answer = str(row["answer"])

        raw_prediction = None
        processed_prediction = None
        score = None
        success = False

        try:
            raw_prediction = query_model(
                OllamaClient, system_message, example_messages, question
            )
            if raw_prediction is not None:
                metric_func = metric_task_map[task]
                score = metric_func(raw_prediction, true_answer)
                success = True
            else:
                success = False
        except Exception as e:
            print(f"Error processing example {example_id}: {e}")
            success = False
            mlflow.log_dict(
                {
                    "error_processing_example": str(e),
                    "example_id": example_id,
                    "task": task,
                },
                f"processing_errors/error_{example_id}.json",
            )

        results.append(
            Result(
                id=example_id,
                task=task,
                question=question,
                answer=true_answer,
                raw_prediction=raw_prediction,
                processed_prediction=processed_prediction,
                score=score,
                success=success,
            )
        )

    save_results(results, RESULT_SAVE_PATH)

    # 7.1 Calculate the fraction of successful predictions
    results = pd.read_csv(RESULT_SAVE_PATH, index_col=0)
    success = results[results["success"] == True]
    success_fraction = len(success) / len(results)
    print(f"{success_fraction*100}% predictions were successfully runned.")
    mlflow.log_metric("success_fraction", success_fraction)

    # # 7.2 Calculate the overall score and the score by task
    # mean_total = round(success["score"].mean(), 3)
    # print(f"The average score for all successful cases are: {mean_total}.")
    # mlflow.log_metric("overall_mean_score", mean_total)

    df_avg_score_per_task = success.groupby(by="task")["score"].mean()
    print("For individual tasks:")
    df_avg_score_per_task_dict = df_avg_score_per_task.to_dict()
    for _task in df_avg_score_per_task_dict.keys():
        mlflow.log_metric(f"mean_score_{_task}", df_avg_score_per_task_dict[_task])

    # 7.3 Create a bar chart of the scores by task with a horizontal line for the overall score

    plt.figure(figsize=(25, 10))
    sns.barplot(df_avg_score_per_task)
    plt.ylim((0, 1))
    plt.axhline(y=mean_total, c="red")
    plt.text(y=mean_total + 0.01, s=f"total average score: {mean_total}", x=-0.5)
    plt.title(f"{EXPERIMENT_NAME}_average score")
    metrics_plot_save_path = os.path.join(
        OUTPUT_DIR, f"{EXPERIMENT_NAME}_average score.png"
    )
    plt.savefig(metrics_plot_save_path)
    mlflow.log_artifact(metrics_plot_save_path, artifact_path="plots")

    print(f"MLflow Run completed. Run ID: {run_id}")
    print(f"Viewable in MLflow UI. Tracking URI: {mlflow.get_tracking_uri()}")


# %%
