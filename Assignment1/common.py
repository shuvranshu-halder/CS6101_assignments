"""
Everything shared across run_parts1to4.py and run_part5.py lives here:
- dataset registry (add a new dataset -> everything else just works)
- tuned BM25 k1/b storage (json, auto-created)
- the one evaluation function used everywhere
- results_<dataset>.md writer

No yaml, no nested configs/ folder — one file.
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass

import ir_measures
from ir_measures import nDCG, R, RR, AP

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset registry — add an entry here (e.g. msmarco for extra credit) and
# every script below picks it up automatically via --dataset <name>.
# ---------------------------------------------------------------------------
DATASETS = {
    "scifact": {
        "ir_datasets_id": "beir/scifact/test",
        "corpus_ir_datasets_id": "beir/scifact",
    },
    "fever": {
        "ir_datasets_id": "beir/fever/test",
        "corpus_ir_datasets_id": "beir/fever",
    },
    "hotpotqa": {
        "ir_datasets_id": "beir/hotpotqa/test",
        "corpus_ir_datasets_id": "beir/hotpotqa",
    },
    # "msmarco": {  # extra credit — uncomment when ready
    #     "ir_datasets_id": "msmarco-passage/trec-dl-2019/judged",
    #     "corpus_ir_datasets_id": "msmarco-passage",
    # },
}

BM25_GRID = {"k1": [0.6, 0.9, 1.2, 1.5, 1.8], "b": [0.3, 0.4, 0.6, 0.8]}
BM25_DEFAULT = {"k1": 0.9, "b": 0.4}
BM25_PARAMS_PATH = ROOT / "bm25_params.json"

METRICS = [nDCG @ 10, R @ 100, RR @ 10, AP]
METRIC_NAMES = ["nDCG@10", "Recall@100", "MRR@10", "MAP"]


@dataclass
class Paths:
    dataset: str
    index_dir: Path
    splade_index_dir: Path
    runs_dir: Path
    qrels_path: Path
    results_md: Path
    log_path: Path
    hyde_jsonl: Path


def get_paths(dataset: str) -> Paths:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(DATASETS)}")
    p = Paths(
        dataset=dataset,
        index_dir=ROOT / "indexes" / dataset / "lucene",
        splade_index_dir=ROOT / "indexes" / dataset / "splade",
        runs_dir=ROOT / "runs" / dataset,
        qrels_path=ROOT / "runs" / dataset / "qrels.trec",
        results_md=ROOT / "results" / f"results_{dataset}.md",
        log_path=ROOT / "logs" / f"{dataset}.log",
        hyde_jsonl=ROOT / "runs" / dataset / "hyde_generations.jsonl",
    )
    for d in (p.index_dir.parent, p.splade_index_dir.parent, p.runs_dir,
              p.results_md.parent, p.log_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(dataset: str) -> logging.Logger:
    p = get_paths(dataset)
    logger = logging.getLogger(dataset)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(p.log_path)
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Tuned BM25 params — single json file, {dataset: {k1, b}}
# ---------------------------------------------------------------------------
def load_tuned_bm25(dataset: str):
    if not BM25_PARAMS_PATH.exists():
        return None
    data = json.loads(BM25_PARAMS_PATH.read_text())
    return data.get(dataset)


def save_tuned_bm25(dataset: str, k1: float, b: float):
    data = json.loads(BM25_PARAMS_PATH.read_text()) if BM25_PARAMS_PATH.exists() else {}
    data[dataset] = {"k1": k1, "b": b}
    BM25_PARAMS_PATH.write_text(json.dumps(data, indent=2))


def save_grid_results(dataset: str, grid_rows: list):
    """
    grid_rows: list of dicts, one per (k1, b) combination tried during tuning, e.g.
        {"k1": 0.6, "b": 0.3, "nDCG@10": 0.481, "Recall@100": 0.812, "MRR@10": 0.552, "MAP": 0.421}
    Writes the FULL grid (every combination, not just the winner) to:
      1. runs/<dataset>/bm25_grid_search.csv  -- raw data for your own plots/heatmaps
      2. results/results_<dataset>.md          -- a table under Part 2 so the report
         shows the whole search, not just the winning k1/b.
    """
    import csv
    p = get_paths(dataset)
    csv_path = p.runs_dir / "bm25_grid_search.csv"
    fieldnames = ["k1", "b"] + METRIC_NAMES
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grid_rows:
            writer.writerow(row)

    best = max(grid_rows, key=lambda r: r["nDCG@10"])
    with open(p.results_md, "a") as f:
        f.write(f"\n### Part 2 — BM25 k1/b Grid Search (full results)\n\n")
        f.write("| k1 | b | " + " | ".join(METRIC_NAMES) + " |\n")
        f.write("|---|---|" + "|".join(["---"] * len(METRIC_NAMES)) + "|\n")
        for row in sorted(grid_rows, key=lambda r: -r["nDCG@10"]):
            marker = " **<- winner**" if row is best else ""
            f.write(f"| {row['k1']} | {row['b']} | " +
                    " | ".join(f"{row[n]:.4f}" for n in METRIC_NAMES) + f" |{marker}\n")
        f.write(f"\nSelection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.\n")
        f.write(f"Winner: k1={best['k1']}, b={best['b']}. Full grid also saved to `{csv_path}`.\n")
    return best


# ---------------------------------------------------------------------------
# Evaluation — one function, used by every part
# ---------------------------------------------------------------------------
def evaluate_run(qrels_path: str, run_path: str) -> dict:
    qrels = ir_measures.read_trec_qrels(str(qrels_path))
    run = ir_measures.read_trec_run(str(run_path))
    res = ir_measures.calc_aggregate(METRICS, qrels, run)
    return {name: res[m] for name, m in zip(METRIC_NAMES, METRICS)}


# ---------------------------------------------------------------------------
# results_<dataset>.md writer
# ---------------------------------------------------------------------------
def init_results_file(dataset: str, corpus_size: int, num_queries: int):
    p = get_paths(dataset)
    with open(p.results_md, "w") as f:
        f.write(f"# Results — {dataset}\n\nCorpus size: {corpus_size} | Queries: {num_queries}\n")


def append_section(dataset: str, title: str, rows: dict = None, notes: str = ""):
    p = get_paths(dataset)
    with open(p.results_md, "a") as f:
        f.write(f"\n## {title}\n\n")
        if rows:
            f.write("| Method | " + " | ".join(METRIC_NAMES) + " |\n")
            f.write("|---|" + "|".join(["---"] * len(METRIC_NAMES)) + "|\n")
            for label, m in rows.items():
                f.write(f"| {label} | " + " | ".join(f"{m[n]:.4f}" for n in METRIC_NAMES) + " |\n")
        if notes:
            f.write(f"\n{notes}\n")
