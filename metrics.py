# %%
import Levenshtein
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from collections import defaultdict

HF_HOME = "/project/GCRB/Hon_lab/s440862/courses/se/MODULE_3_MATERIALS/mod3/huggingFace"


def exact_match(pred: str, true: str) -> float:
    return int(pred == true)


# format answers of a list of genes/positions
def process_list(l: str) -> str:
    l = str.lower(l)
    if l.startswith("["):
        l = l[1:-2]
    l = ", ".join(sorted(l.split(",")))
    return l


### for gene hop dataset

### Levenshtein_score


def Levenshtein_score(s1: str, s2: str) -> float:
    dist = Levenshtein.distance(s1, s2)
    return dist


# %%
### cosine similarity


# https://huggingface.co/FremyCompany/BioLORD-2023
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# This is for the gene hop "SNP gene function"
# Sentences we want sentence embeddings for

# sentences = [
#     "This is not a protein coding gene. it is located on the 6th chromosome.",
#     "Located on the 6th chromosome, this gene is protein coding",
#     "This is a protein coding gene. it is located on the 6th chromosome.",
# ]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("FremyCompany/BioLORD-2023")
model = AutoModel.from_pretrained("FremyCompany/BioLORD-2023")


def cos_similarity(s1: str, s2: str) -> float:
    sentences = [s1, s2]

    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    cosine_sim = F.cosine_similarity(
        sentence_embeddings[0], sentence_embeddings[1], dim=0
    )
    return float(cosine_sim.detach().cpu())


# %%

### for gene turing dataset


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


### answer mapping func
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


# %%
### caller function
def call_metric_task_map(dataset_name: str):
    if dataset_name == "genehop":
        metric_task_map = defaultdict(
            lambda: exact_match,
            {
                "sequence gene alias": Levenshtein_score,
                "Disease gene location": Levenshtein_score,
                "SNP gene function": cos_similarity,
            },
        )
    elif dataset_name == "geneturing":
        metric_task_map = defaultdict(
            lambda: exact_match,
            {
                "gene_disease_association": gene_disease_association,
                "disease_gene_location": disease_gene_location,
                "human_genome_dna_alignment": human_genome_dna_alignment,
            },
        )
    return metric_task_map


# %%
