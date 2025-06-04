# %% Import files
from typing import List, Optional

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

mlflow.set_tracking_uri("http://198.215.61.34:8153/")

os.chdir("/project/GCRB/Hon_lab/s440862/courses/se/MODULE_3_MATERIALS/mod3")
os.environ["NO_PROXY"] = "*"

# %% Model configuration

# 2.1 Data Configuration
data_path = "data/geneturing.json"
DATASET_NAME = "geneturing"

# 2.2 Model Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "qwen3:4b"
EXPERIMENT_NAME = f"{DATASET_NAME}_{OLLAMA_MODEL_NAME}"
RESULT_SAVE_PATH = f"output/{EXPERIMENT_NAME}.csv"
OUTPUT_DIR = "output/"

# 2.3 Evaluation and Logging Configuration
MLFLOW_EXPERIMENT_NAME = f"Taosha_{EXPERIMENT_NAME}_Experiment"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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

# %%  4. Model Specification

### 4.1 Setting up the large language model Ollama model client

OllamaClient = ollama.Client(host=OLLAMA_BASE_URL)

### 4.2 Setting up the system prompt

system_message = [
    {
        "role": "system",
        "content": f"Hello. You are an expertise in bioinformatics, and your job now is to use the NCBI Web APIs to give accurate and neat answers to genomic questions in aspects of {list(data.keys())}. For example, if I ask 'What is the official gene symbol of LMP10?', your answer should be: 'Answer: PSMB10'.",
    }
]

### 4.3 Setting up the few-shot examples

example_messages = [
    {"role": "user", "content": "Question: What is the official gene symbol of LMP10?"},
    {"role": "assistant", "content": "Answer: PSMB10"},
    {
        "role": "user",
        "content": "Question: Which gene is SNP rs1217074595 associated with?",
    },
    {"role": "assistant", "content": "Answer: LINC01270"},
    {
        "role": "user",
        "content": "Question: What are genes related to Meesmann corneal dystrophy?",
    },
    {"role": "assistant", "content": "Answer: KRT12, KRT3"},
    {
        "role": "user",
        "content": "Question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT",
    },
    {"role": "assistant", "content": "Answer: chr15:91950805-91950932"},
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
# reference: `evaluate.py` from the GeneGPT repository to find the metric functions and implement them here.
# Original functions: [evaluate.py](https://github.com/ncbi/GeneGPT/blob/main/evaluate.py)

# * **Default exact match** - The predicted answers and ground truth are both strings. The score is 1 if they are equal and 0 otherwise
# * **Gene disease association** - The predicted answers and ground truth are both lists of gene-disease associations. The score is the proportion of correct associations present in the prediction
# * **Disease gene location** - The predicted and true answers are lists (e.g., gene locations related to a disease), and the evaluation calculates the fraction of the correct items present in the prediction.
# * **Human genome DNA aligment** - 1 point for exact match, 0.5 point if chrX part matches, 0 otherwise


def exact_match(pred: str, true: str) -> float:
    return int(pred == true)


def gene_disease_association(pred: list, true: list) -> float:
    pred_set = set(pred.split(", "))
    true_set = set(true.split(", "))
    if not true_set:
        return 1 if not pred_set else 0
    return len(pred_set.intersection(true_set)) / len(true_set)


def disease_gene_location(pred: list, true: list) -> float:
    pred_set = set(pred.split(", "))
    true_set = set(true.split(", "))
    if not true_set:
        return 1 if not pred_set else 0
    return len(pred_set.intersection(true_set)) / len(true_set)


def human_genome_dna_alignment(pred: str, true: str) -> float:
    pred = str.lower(pred)
    true = str.lower(true)
    if pred == true:
        return 1.0
    elif true.split(":")[0] == pred.split(":")[0]:
        return 0.5
    return 0.0


metric_task_map = defaultdict(
    lambda: exact_match,
    {
        "gene_disease_association": gene_disease_association,
        "disease_gene_location": disease_gene_location,
        "human_genome_dna_alignment": human_genome_dna_alignment,
    },
)

# 5.2 Implement the answer mapping function


def get_answer(answer: str, task: str) -> str:

    mapper = {
        "Caenorhabditis elegans": "worm",
        "Homo sapiens": "human",
        "Danio rerio": "zebrafish",
        "Mus musculus": "mouse",
        "Saccharomyces cerevisiae": "yeast",
        "Rattus norvegicus": "rat",
        "Gallus gallus": "chicken",
    }

    if task == "SNP location":
        answer = answer.strip().split()[-1]
        if "chr" not in answer:
            answer = "chr" + answer

    elif task == "Gene location":
        answer = answer.strip().split()[-1]
        if "chr" not in answer:
            answer = "chr" + answer

    elif task == "Gene disease association":
        answer = answer.strip().replace("Answer: ", "")
        answer = answer.split(", ")

    elif task == "Disease gene location":
        answer = answer.strip().replace("Answer: ", "")
        answer = answer.split(", ")

    elif task == "Protein-coding genes":
        answer = answer.strip().replace("Answer: ", "")
        if answer == "Yes":
            answer = "TRUE"
        elif answer == "No":
            answer = "NA"

    elif task == "Multi-species DNA aligment":
        answer = answer.strip().replace("Answer: ", "")
        answer = mapper.get(answer, answer)

    else:
        answer = answer.strip().replace("Answer: ", "")

    return answer


# %%  6. Evaluation Loop
with mlflow.start_run(run_name=f"{MLFLOW_EXPERIMENT_NAME}") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # --- Log Parameters ---
    mlflow.log_param("dataset_name", DATASET_NAME)
    mlflow.log_param("ollama_model_name", OLLAMA_MODEL_NAME)
    mlflow.log_param("ollama_base_url", OLLAMA_BASE_URL)
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("system_prompt_preview", system_message[0]["content"][:250])
    mlflow.log_param("num_few_shot_examples", len(example_messages) // 2)

    # 6.1 Set up data structures for results

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
                processed_prediction = get_answer(raw_prediction, task)
                metric_func = metric_task_map[task]

                if task in ["Gene disease association", "Disease gene location"]:
                    score = metric_func(
                        pred=processed_prediction, true=get_answer(true_answer, task)
                    )
                else:
                    score = metric_func(pred=processed_prediction, true=true_answer)
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

    # 7.2 Calculate the overall score and the score by task
    mean_total = round(success["score"].mean(), 3)
    print(f"The average score for all successful cases are: {mean_total}.")
    mlflow.log_metric("overall_mean_score", mean_total)

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
