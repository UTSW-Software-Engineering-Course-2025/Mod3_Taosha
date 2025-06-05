# %%
import os

os.environ["NO_PROXY"] = "*"
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import copy
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
import mlflow
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://198.215.61.34:8153/")
import pprint as pp
from tqdm import tqdm
from dataclasses import dataclass

os.chdir("/project/GCRB/Hon_lab/s440862/courses/se/MODULE_3_MATERIALS/mod3")
from prompts import msg
from dataset import read_data
from metrics import *

load_dotenv("azure.env")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OLLAMA_ENDPOINT = "http://localhost:11434/v1"
OLLAMA_MODELS = ["qwen3:4b", "deepseek-r1:8b"]
MLFLOW_DIR_NAME = "Taosha"
mlflow.set_experiment(MLFLOW_DIR_NAME)

# MODEL_NAME = "gpt-4.1"
# DATASET_NAME = "genehop"
# system_message = msg[DATASET_NAME]["system_msg"]
# example_message = msg[DATASET_NAME]["example_msg"]

TEMPERATURE = 1.0


# %%
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


def get_client(model_name: str):
    if model_name.startswith("gpt"):
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        )
    elif model_name in OLLAMA_MODELS:
        client = OpenAI(base_url=OLLAMA_ENDPOINT, api_key="ollama")
    return client


def get_query_msg(question: str, dataset_name: str):
    messages = []
    system_message = msg[dataset_name]["system_msg"]
    example_message = msg[dataset_name]["example_msg"]
    messages.extend(system_message)
    messages.extend(example_message)
    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages


def run_query(clientModel, model_name: str, dataset_name: str, question: str, temperature: float = 1):
    input_msg = get_query_msg(question, dataset_name)
    try:
        response = clientModel.chat.completions.create(
            model=model_name,
            messages=input_msg,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if "Answer:" in content:
            return content.split("Answer:")[-1].strip()
        return content
    except Exception as e:
        print(f"Error querying model: {e}")
        print(
            f"Raw response content (if available): {response.get('message', {}).get('content', 'N/A')}"
        )
        return None

def run_expr(dataset_name: str, model_name: str, temperature: float = 1) -> dict:
    df = read_data(dataset_name)
    client = get_client(model_name)
    metric_task_map = call_metric_task_map(dataset_name)
    EXPERIMENT_NAME = f"{dataset_name}_{model_name}_{temperature}"
    METRICS_SAVE_PATH = os.path.join(f'output/{EXPERIMENT_NAME}.csv')

    with mlflow.start_run(run_name=f"{EXPERIMENT_NAME}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # --- Log Parameters ---
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_temperature", temperature)
        mlflow.log_param("ollama_base_url", OLLAMA_ENDPOINT)
        # mlflow.log_param("system_prompt_preview", system_message[0]["content"][:250])

        results: List[Result] = []
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Evaluating {EXPERIMENT_NAME}"
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
                raw_prediction = run_query(client, model_name, dataset_name, question, temperature)
                if raw_prediction is not None:
                    if dataset_name == "geneturing":
                        processed_prediction = get_answer(raw_prediction, task)
                        if task in [
                            "Gene disease association",
                            "Disease gene location",
                        ]:
                            true_answer = get_answer(true_answer, task)
                    else:
                        processed_prediction = raw_prediction
                    metric_func = metric_task_map[task]
                    score = metric_func(processed_prediction, true_answer)
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

        save_results(results, METRICS_SAVE_PATH)

        ### Result Analysis
        # 7.1 Calculate the fraction of successful predictions
        results = pd.read_csv(METRICS_SAVE_PATH, index_col=0)
        success = results[results["success"] == True]
        success_fraction = len(success) / len(results)
        print(f"{success_fraction*100}% predictions were successfully runned.")
        mlflow.log_metric("success_fraction", success_fraction)

        # 7.2 Calculate the overall score and the score by task
        mean_total = round(success["score"].mean(), 3)
        print(f"The average score for all successful cases are: {mean_total}.")
        if dataset_name == 'geneturing':
            mlflow.log_metric("overall_mean_score", mean_total)

        df_avg_score_per_task = success.groupby(by="task")["score"].mean()
        print("For individual tasks:")
        df_avg_score_per_task_dict = df_avg_score_per_task.to_dict()
        for _task in df_avg_score_per_task_dict.keys():
            mlflow.log_metric(f"mean_score_{_task}", df_avg_score_per_task_dict[_task])

        # 7.3 Create a bar chart of the scores by task with a horizontal line for the overall score

        plt.figure(figsize=(25, 10))
        sns.barplot(df_avg_score_per_task)
        if dataset_name == 'geneturing':
            plt.ylim((0, 1))
        plt.axhline(y=mean_total, c="red")
        plt.text(y=mean_total + 0.01, s=f"total average score: {mean_total}", x=-0.5)
        plt.title(f"{EXPERIMENT_NAME}_average score")
        metrics_plot_save_path = os.path.join(f"output/{EXPERIMENT_NAME}_average score.png")
        plt.savefig(metrics_plot_save_path)
        mlflow.log_artifact(metrics_plot_save_path, artifact_path="plots")

        print(f"MLflow Run completed. Run ID: {run_id}")
        print(f"Viewable in MLflow UI. Tracking URI: {mlflow.get_tracking_uri()}")
        return results



# %%

for d in ['genehop', 'geneturing']:
    for m in ["gpt-4.1", "qwen3:4b",]:
        for t in [0.1, 0.5, 1]:
            res = run_expr(d, m, t)

# %%

