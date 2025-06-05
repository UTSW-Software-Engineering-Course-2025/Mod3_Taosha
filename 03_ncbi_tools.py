# %%
import os
import requests # For making HTTP requests to NCBI APIs
import xml.etree.ElementTree as ET # For parsing XML responses from Entrez
import time # For adding delays to respect API rate limits
import json # For parsing JSON arguments from LLM tool calls
import ast # For safely evaluating string representations of Python literals (like lists)
import re # For regex to extract sequences
from datetime import datetime

os.environ["NO_PROXY"] = "*"
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import copy
from pydantic import BaseModel, Field # Pydantic might not be strictly needed for this minimal change
from typing import List, Optional, Dict, Union, Any # Added Any for tool results
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
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")

OLLAMA_ENDPOINT = "http://localhost:11434/v1"
OLLAMA_MODELS = ["qwen3:4b", "deepseek-r1:8b"]
MLFLOW_DIR_NAME = "Taosha"
mlflow.set_experiment(MLFLOW_DIR_NAME)

TEMPERATURE = 1.0

# --- NEW: NCBI Entrez (E-utilities) Functions ---
ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def ncbi_esearch(db: str, term: str, retmax: int = 5) -> List[str]:
    """
    [TOOL] Performs an Entrez ESearch query to find UIDs (unique IDs) in a specified database.
    Args:
        db (str): The NCBI database to search (e.g., 'gene', 'pubmed', 'nuccore', 'protein', 'snp').
        term (str): The search query (e.g., gene name, accession number, SNP ID).
        retmax (int): Maximum number of UIDs to retrieve. Defaults to 5.
    Returns:
        List[str]: A list of UIDs (as strings) matching the search term.
    """
    params = {
        "db": db,
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "api_key": NCBI_API_KEY,
        "email": NCBI_EMAIL
    }
    try:
        response = requests.get(f"{ENTREZ_BASE_URL}esearch.fcgi", params=params)
        response.raise_for_status()
        data = response.json()
        return data["esearchresult"]["idlist"] if data and "esearchresult" in data and "idlist" in data["esearchresult"] else []
    except requests.exceptions.RequestException as e:
        return [f"ERROR: ESearch failed for {db}, term '{term}': {e}"]
    finally:
        time.sleep(0.2)

def ncbi_efetch_gene_info(gene_uid: str) -> Dict[str, Any]:
    """
    [TOOL] Retrieves and parses detailed gene information from NCBI Gene database using EFetch.
    Args:
        gene_uid (str): The UID of the gene to fetch.
    Returns:
        Dict[str, Any]: A dictionary with parsed gene details (summary, aliases, location, etc.).
    """
    if not gene_uid:
        return {"status": "error", "description": "No gene_uid provided."}

    params = {
        "db": "gene", "id": gene_uid, "rettype": "xml", "retmode": "xml",
        "api_key": NCBI_API_KEY, "email": NCBI_EMAIL
    }
    gene_info = {"status": "success", "gene_id": gene_uid}
    try:
        response = requests.get(f"{ENTREZ_BASE_URL}efetch.fcgi", params=params)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        gene_info["summary"] = root.findtext(".//Entrezgene_summary", default="N/A")
        gene_info["description"] = root.findtext(".//Entrezgene_descript", default="N/A")
        aliases = [node.text for node in root.findall(".//Entrezgene_gene/Gene-ref/Gene-ref_syn/Gene-ref_syn_E") if node.text]
        gene_info["aliases"] = aliases
        gene_info["chromosome_location"] = root.findtext(".//Entrezgene_location/Seq-loc/Seq-loc_int/Seq-interval/Seq-interval_id/Seq-id/Seq-id_genbank/Seq-id_genbank_accession", default="N/A")
    except (ET.ParseError, requests.exceptions.RequestException) as e:
        gene_info["status"] = f"error: {e}"
        gene_info["description"] = f"Failed to fetch/parse gene info for UID {gene_uid}."
    finally:
        time.sleep(0.2)
    return gene_info

def ncbi_blast_sequence(sequence: str, program: str = "blastn", database: str = "nt", expect: float = 1e-5) -> str:
    """
    [TOOL] Performs a web-based NCBI BLAST search and retrieves a summary of top hits.
    Args:
        sequence (str): The nucleotide or protein query sequence. Max length for web BLAST is usually limited.
        program (str): BLAST program (e.g., 'blastn' for nucleotide, 'blastp' for protein).
        database (str): Database to search against (e.g., 'nt' for nucleotide, 'nr' for non-redundant protein).
        expect (float): Expectation value (E-value) threshold.
    Returns:
        str: A summary of the top BLAST hits, or an error message.
    """
    if not sequence: return "Error: No sequence provided for BLAST."
    if len(sequence) > 1000: return "Error: Sequence too long for web BLAST (max 1000bp/aa recommended)."

    BLAST_BASE_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
    params_put = {
        "CMD": "Put", "PROGRAM": program, "DATABASE": database, "QUERY": sequence,
        "EXPECT": str(expect), "FORMAT_OBJECT": "Text", "FORMAT_TYPE": "Text",
        "NCBI_GI": "TRUE", "CLIENT": "web", "api_key": NCBI_API_KEY, "email": NCBI_EMAIL
    }
    rid = None
    try:
        response_put = requests.post(BLAST_BASE_URL, data=params_put)
        response_put.raise_for_status()
        rid_match = re.search(r"RID = (\w+)", response_put.text)
        if rid_match: rid = rid_match.group(1)
        else: return f"Error: Could not obtain BLAST RID. Raw response: {response_put.text[:500]}..."
    except requests.exceptions.RequestException as e: return f"Error submitting BLAST query: {e}"
    finally: time.sleep(1)

    if not rid: return "Error: Failed to obtain BLAST RID."

    params_get = {
        "CMD": "Get", "RID": rid, "FORMAT_OBJECT": "Text", "FORMAT_TYPE": "Text",
        "ALIGNMENT_VIEW": "Pairwise", "api_key": NCBI_API_KEY, "email": NCBI_EMAIL
    }
    max_attempts = 15
    attempts = 0
    while attempts < max_attempts:
        try:
            response_get = requests.get(BLAST_BASE_URL, params=params_get)
            response_get.raise_for_status()
            if "Status=WAITING" in response_get.text or "Status=UNKNOWN" in response_get.text:
                time.sleep(5 + attempts * 2)
                attempts += 1
                continue
            summary_lines = []
            capture = False
            for line in response_get.text.split('\n'):
                if 'Sequences producing significant alignments:' in line: capture = True; summary_lines.append(line); continue
                if capture:
                    if not line.strip() or "Description" in line or "Score" in line: break
                    summary_lines.append(line)
            return "\n".join(summary_lines) if summary_lines else "BLAST completed, no significant alignments."
        except requests.exceptions.RequestException as e: return f"Error retrieving BLAST results for RID {rid}: {e}"
        finally: time.sleep(0.5)
    return f"Error: Max attempts reached for RID {rid}. No BLAST results."


# --- NEW: Define Tools for LLM Function Calling ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "ncbi_esearch",
            "description": "Search NCBI databases (e.g., 'gene', 'pubmed', 'nuccore', 'protein', 'snp') for unique identifiers (UIDs) based on a search term. Use this to find IDs for genes, proteins, SNPs, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db": {"type": "string", "description": "The NCBI database to search, e.g., 'gene', 'snp', 'nuccore', 'protein', 'pubmed'."},
                    "term": {"type": "string", "description": "The search query, e.g., 'BRCA1 human', 'rs12345', 'TP53 mutation'."},
                    "retmax": {"type": "integer", "description": "Maximum number of UIDs to retrieve. Default is 5. Use 1 if you expect a specific ID.", "default": 1}
                },
                "required": ["db", "term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ncbi_efetch_gene_info",
            "description": "Retrieve comprehensive details for a specific gene from NCBI Gene database using its unique identifier (UID). Provides summary, aliases, description, and chromosome location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_uid": {"type": "string", "description": "The unique identifier (UID) of the gene to fetch (e.g., '672'). Obtained from ncbi_esearch('gene', ...)."},
                },
                "required": ["gene_uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ncbi_blast_sequence",
            "description": "Perform a sequence similarity search (BLAST) against NCBI databases (e.g., 'nt', 'nr'). Useful for identifying genes or proteins based on a DNA or protein sequence. Returns a summary of top hits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {"type": "string", "description": "The nucleotide (e.g., 'ATGC...') or protein (e.g., 'AGCT...') query sequence."},
                    "program": {"type": "string", "enum": ["blastn", "blastp", "blastx", "tblastn", "tblastx"], "description": "BLAST program to use, e.g., 'blastn' for DNA-DNA, 'blastp' for protein-protein. Default is 'blastn'.", "default": "blastn"},
                    "database": {"type": "string", "enum": ["nt", "nr", "refseq_rna", "refseq_protein", "swissprot"], "description": "Database to search against. Default is 'nt'.", "default": "nt"},
                    "expect": {"type": "number", "description": "Expectation value (E-value) threshold. Lower values mean more significant hits. Default is 1e-5.", "default": 1e-5}
                },
                "required": ["sequence"],
            },
        },
    },
]

# Map tool names to actual Python functions
available_functions = {
    "ncbi_esearch": ncbi_esearch,
    "ncbi_efetch_gene_info": ncbi_efetch_gene_info,
    "ncbi_blast_sequence": ncbi_blast_sequence,
}

# --- Minimal system message for tool use ---
# This OVERRIDES the system message from prompts.py for clarity.
# You might want to modify your prompts.py directly to include this.
TOOL_USE_SYSTEM_MESSAGE = [
    {
        "role": "system",
        "content": "You are a bioinformatics expert. You have access to tools to search NCBI databases (Entrez) and perform sequence alignments (BLAST). Utilize these tools to gather precise, up-to-date information to answer questions about genes, SNPs, sequences, and their functions or locations. Always provide a clear, concise answer based on the tool output. If a sequence is provided, consider using BLAST. If gene or SNP details are asked, use Entrez search and fetch tools."
    }
]


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
    print(f"Results saved to {results_csv_filename}.")


def get_client(model_name: str):
    if model_name.startswith("gpt"):
        # For Azure OpenAI, the model_name here is your deployment name
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        )
    elif model_name in OLLAMA_MODELS:
        # For Ollama, api_key can be 'ollama' or anything, as it's often not used for local setups
        client = OpenAI(base_url=OLLAMA_ENDPOINT, api_key="ollama")
    else:
        raise ValueError(f"Model '{model_name}' not configured for client setup.")
    return client


def get_query_msg(question: str, dataset_name: str):
    """
    Constructs the initial message list for the LLM.
    Now uses the specific TOOL_USE_SYSTEM_MESSAGE.
    """
    messages = []
    messages.extend(TOOL_USE_SYSTEM_MESSAGE) # Use the tool-aware system message
    # example_message from prompts.py is still included, ensure it's compatible
    messages.extend(msg[dataset_name]["example_msg"])
    messages.append({"role": "user", "content": question})
    return messages


def run_query(clientModel, model_name: str, dataset_name: str, question: str, temperature: float = 0.5, max_iterations: int = 5) -> Optional[str]:
    """
    Runs the LLM query in an agentic loop, incorporating tool calls.
    Args:
        clientModel: The initialized OpenAI or AzureOpenAI client.
        model_name (str): The name of the LLM model to use.
        dataset_name (str): Name of the dataset (e.g., 'genehop') to get example messages.
        question (str): The user's original question.
        temperature (float): The sampling temperature for the LLM.
        max_iterations (int): Maximum number of LLM-tool-LLM turns to prevent infinite loops.
    Returns:
        Optional[str]: The final answer from the LLM, or None if an error occurs.
    """
    messages = get_query_msg(question, dataset_name)
    current_response = None
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1
        try:
            # Pass tools and tool_choice="auto" to enable function calling
            response = clientModel.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto", # Model decides to call a tool or respond
                temperature=temperature,
            )
            current_response = response

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                # LLM wants to call a tool
                messages.append(response_message) # Add the tool call request to messages

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions.get(function_name)

                    if function_to_call:
                        try:
                            # Parse arguments from the tool call (JSON string)
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"DEBUG: LLM calling tool: {function_name} with args: {function_args}")

                            # Execute the tool and get its output
                            tool_output = function_to_call(**function_args)

                            # Append tool output to messages
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    # Ensure tool_output is JSON-serializable if it's a dict/list
                                    "content": json.dumps(tool_output) if isinstance(tool_output, (dict, list)) else str(tool_output),
                                }
                            )
                        except json.JSONDecodeError as e:
                            error_msg = f"Error decoding tool arguments for {function_name}: {e}. Raw args: {tool_call.function.arguments}"
                            print(error_msg)
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: {error_msg}"
                            })
                        except Exception as e:
                            error_msg = f"Error executing tool {function_name}: {e}"
                            print(error_msg)
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: {error_msg}"
                            })
                    else:
                        error_msg = f"Error: LLM requested to call unknown tool: {function_name}"
                        print(error_msg)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": error_msg
                        })
                # Continue loop to allow LLM to respond to tool output
                continue

            else:
                # LLM provided a direct content response (not a tool call)
                final_answer_content = response_message.content
                # Keep original parsing for "Answer: " if the model sometimes defaults to it
                if "Answer:" in final_answer_content:
                    return final_answer_content.split("Answer:")[-1].strip()
                return final_answer_content

        except Exception as e:
            print(f"Error during LLM chat completion loop: {e}")
            raw_content_info = "N/A"
            if current_response is not None and hasattr(current_response, 'choices') and \
               len(current_response.choices) > 0 and current_response.choices[0].message:
                raw_content_info = current_response.choices[0].message.content
            print(f"Raw LLM response content (if available): {raw_content_info}")
            return None

    print(f"Warning: Max iterations ({max_iterations}) reached without a final answer.")
    return "Error: Could not generate a complete answer within the allowed iterations."


def run_expr(dataset_name: str, model_name: str, temperature: float = 0.5) -> dict:
    df = read_data(dataset_name)
    client = get_client(model_name)
    metric_task_map = call_metric_task_map(dataset_name)
    EXPERIMENT_NAME = f"{dataset_name}_{model_name}_tool_use_temp{temperature}" # Indicate tool use in experiment name
    METRICS_SAVE_PATH = os.path.join(f'output/{EXPERIMENT_NAME}.csv')

    with mlflow.start_run(run_name=f"{EXPERIMENT_NAME}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # --- Log Parameters ---
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_temperature", temperature)
        mlflow.log_param("ollama_base_url", OLLAMA_ENDPOINT)
        # Use the global TOOL_USE_SYSTEM_MESSAGE content for logging
        mlflow.log_param("system_prompt_preview", TOOL_USE_SYSTEM_MESSAGE[0]["content"][:250])
        mlflow.log_param("ncbi_api_key_set", bool(NCBI_API_KEY))
        mlflow.log_param("ncbi_email_set", bool(NCBI_EMAIL))
        mlflow.log_param("tool_use_enabled", True)
        mlflow.log_param("num_tools_available", len(tools))

        log_filename = os.path.join(f'output/logs/{EXPERIMENT_NAME}.txt')
        log_file = open(log_filename, 'w+', encoding='utf-8')
        log_file.write(f"--- MLflow Run Log for Experiment: {EXPERIMENT_NAME} ---\n")
        log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")


        results: List[Result] = []
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Evaluating {EXPERIMENT_NAME}"
        ):
            example_id = row["id"]
            task = row["task"]
            question = row["question"]
            true_answer = str(row["answer"]) # Ensure this is a string

            raw_prediction = None
            processed_prediction = None
            score = None
            success = False

            try:
                # Call run_query which now incorporates agentic tool use
                llm_final_response_content = run_query(client, model_name, dataset_name, question, temperature)

                raw_prediction = str(llm_final_response_content) # Store the LLM's final text response

                if llm_final_response_content is not None:
                    # Metric compatibility helper (from previous versions)
                    # We need a way to ensure the LLM's raw_prediction and true_answer are comparable.
                    # This could involve ast.literal_eval if answers are list-like strings.
                    # For simplicity, we'll try to apply common transformations.
                    temp_processed_prediction = llm_final_response_content
                    temp_true_answer = true_answer

                    # Adapt based on dataset/task if true answers are sometimes list-like strings
                    if isinstance(llm_final_response_content, str):
                        try:
                            # Try to parse as a literal (list or string)
                            temp_processed_prediction = ast.literal_eval(llm_final_response_content)
                        except (ValueError, SyntaxError):
                            pass # Not a literal string, keep as is

                    if isinstance(true_answer, str):
                        try:
                            temp_true_answer = ast.literal_eval(true_answer)
                        except (ValueError, SyntaxError):
                            pass

                    # Ensure both are strings for metric calculation, e.g., by sorting lists
                    processed_prediction = str(sorted(temp_processed_prediction)) if isinstance(temp_processed_prediction, list) else str(temp_processed_prediction)
                    true_answer_for_metric_str = str(sorted(temp_true_answer)) if isinstance(temp_true_answer, list) else str(temp_true_answer)


                    metric_func = metric_task_map[task]
                    score = metric_func(processed_prediction, true_answer_for_metric_str)
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

            res = Result(
                    id=example_id,
                    task=task,
                    question=question,
                    answer=true_answer,
                    raw_prediction=raw_prediction,
                    processed_prediction=processed_prediction,
                    score=score,
                    success=success,
                )
            results.append(
                res 
            )

            log_file.write(f"{res}---\n")
            log_file.flush()
        save_results(results, METRICS_SAVE_PATH)
        log_file.close()

        ### Result Analysis
        # 7.1 Calculate the fraction of successful predictions
        results_df_loaded = pd.read_csv(METRICS_SAVE_PATH, index_col=0)
        success_df = results_df_loaded[results_df_loaded["success"] == True]
        success_fraction = len(success_df) / len(results_df_loaded)
        print(f"{success_fraction*100}% predictions were successfully runned.")
        mlflow.log_metric("success_fraction", success_fraction)

        # 7.2 Calculate the overall score and the score by task
        if not success_df.empty:
            mean_total = round(success_df["score"].mean(), 3)
            print(f"The average score for all successful cases are: {mean_total}.")
            # Log overall_mean_score regardless of dataset for general tracking
            mlflow.log_metric("overall_mean_score", mean_total)
        else:
            mean_total = 0.0
            print("No successful predictions to calculate overall mean score.")
            mlflow.log_metric("overall_mean_score", mean_total)

        df_avg_score_per_task = success_df.groupby(by="task")["score"].mean()
        print("For individual tasks:")
        df_avg_score_per_task_dict = df_avg_score_per_task.to_dict()
        for _task in df_avg_score_per_task_dict.keys():
            mlflow.log_metric(f"mean_score_{_task}", df_avg_score_per_task_dict[_task])

        # 7.3 Create a bar chart of the scores by task with a horizontal line for the overall score

        plt.figure(figsize=(25, 10))
        sns.barplot(x=df_avg_score_per_task.index, y=df_avg_score_per_task.values)
        # plt.ylim((0, 1)) # Keep if scores are strictly 0-1
        plt.axhline(y=mean_total, c="red")
        plt.text(y=mean_total + 0.01, s=f"total average score: {mean_total}", x=-0.5)
        plt.title(f"{EXPERIMENT_NAME}_average score")
        metrics_plot_save_path = os.path.join(f"output/{EXPERIMENT_NAME}_average score.png")
        plt.savefig(metrics_plot_save_path)
        mlflow.log_artifact(metrics_plot_save_path, artifact_path="plots")

        print(f"MLflow Run completed. Run ID: {run_id}")
        print(f"Viewable in MLflow UI. Tracking URI: {mlflow.get_tracking_uri()}")
        return results


# %% Main execution loop
if __name__ == "__main__":
    for d in ['genehop', 'geneturing']:
        for m in ["gpt-4.1"]:
            for t in [0.5, 0.9]: 
                res = run_expr(d, m, t)


# %%
run_expr('geneturing', 'gpt-4.1', 0.5)
# %%
