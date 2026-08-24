"""
Part 5: SPLADE encoding + retrieval + 3-way expansion-term comparison
(SPLADE vs Rocchio/RM3 vs HyDE) for ONE dataset.

Requires run_parts1to4.py to have already been run for this dataset (needs
the Lucene index for query search fallback, plus runs/hyde_generations.jsonl
and Rocchio results for the comparison table).

Usage:
    python run_part5.py --dataset scifact --checkpoint naver/splade-cocondenser-ensembledistil
    python run_part5.py --dataset scifact --stage encode   # just re-encode
    python run_part5.py --dataset scifact --stage compare  # just the comparison table
"""
import argparse
import itertools
import json
import time

import ir_datasets
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import common

# so Part 4a's build_boosted_query / _run_rocchio are reused unchanged for the comparison
from run_parts1to4 import build_boosted_query, _run_rocchio  # noqa: F401


# =============================================================================
# SPLADE encoder — supports sharding a corpus across multiple GPUs
# (bash launcher spawns one process per GPU with --shard_id / --num_shards)
# =============================================================================
class SpladeEncoder:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, max_length=256):
        """w_j = max_i log(1 + ReLU(o_ij)) over vocab, per input text. Returns list of {term: weight}."""
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits  # [batch, seq_len, vocab_size]
        weights = torch.log1p(torch.relu(logits))
        attn_mask = inputs["attention_mask"].unsqueeze(-1)
        weights = weights * attn_mask
        pooled, _ = torch.max(weights, dim=1)  # [batch, vocab_size]
        out = []
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        for row in pooled.cpu():
            nonzero = row.nonzero().squeeze(-1).tolist()
            out.append({inv_vocab[i]: float(row[i]) for i in nonzero})
        return out


def encode_corpus_shard(dataset: str, checkpoint: str, shard_id: int, num_shards: int, batch_size: int):
    """Called once per GPU by scripts/run_part5.sh (each process pins its own CUDA device
    via CUDA_VISIBLE_DEVICES before this runs, so `cuda` here always means "my assigned GPU")."""
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    entry = common.DATASETS[dataset]

    encoder = SpladeEncoder(checkpoint)
    corpus_ds = ir_datasets.load(entry["corpus_ir_datasets_id"])

    out_path = p.splade_index_dir / f"shard_{shard_id}.jsonl"
    p.splade_index_dir.mkdir(parents=True, exist_ok=True)

    batch_ids, batch_texts = [], []

    def flush(f):
        if not batch_ids:
            return
        vectors = encoder.encode(batch_texts)
        for doc_id, vec in zip(batch_ids, vectors):
            f.write(json.dumps({"id": doc_id, "vector": vec}) + "\n")
        batch_ids.clear()
        batch_texts.clear()

    n = 0
    with open(out_path, "w") as f:
        for i, doc in enumerate(corpus_ds.docs_iter()):
            if i % num_shards != shard_id:
                continue
            text = f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}".strip()
            batch_ids.append(doc.doc_id)
            batch_texts.append(text)
            if len(batch_ids) >= batch_size:
                flush(f)
            n += 1
        flush(f)
    log.info(f"[Part5][shard {shard_id}/{num_shards}] encoded {n} docs -> {out_path}")


# =============================================================================
# Impact-index style retrieval over the SPLADE vectors: dot product of
# sparse query/doc weight dicts. For real BEIR-scale corpora, swap this loop
# for Pyserini's --impact Lucene index (see the assignment's IndexReader hint);
# this in-memory version is here so the pipeline runs end-to-end for grading /
# small-scale sanity checks without a second Lucene build step.
# =============================================================================
def load_shards(splade_index_dir):
    doc_vectors = {}
    for shard_path in sorted(splade_index_dir.glob("shard_*.jsonl")):
        with open(shard_path) as f:
            for line in f:
                rec = json.loads(line)
                doc_vectors[rec["id"]] = rec["vector"]
    return doc_vectors


def sparse_dot(vec_a: dict, vec_b: dict) -> float:
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(w * vec_b[t] for t, w in vec_a.items() if t in vec_b)


def part5_retrieve(dataset: str, checkpoint: str, top_k: int = 1000):
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    entry = common.DATASETS[dataset]

    log.info("[Part5] loading encoded corpus shards...")
    doc_vectors = load_shards(p.splade_index_dir)
    log.info(f"[Part5] loaded {len(doc_vectors)} doc vectors")

    encoder = SpladeEncoder(checkpoint)
    ds = ir_datasets.load(entry["ir_datasets_id"])
    run_path = p.runs_dir / "splade.trec"

    query_term_weights = {}  # saved for the expansion-term comparison table
    with open(run_path, "w") as f:
        for q in ds.queries_iter():
            qvec = encoder.encode([q.text])[0]
            query_term_weights[q.query_id] = qvec
            scores = [(doc_id, sparse_dot(qvec, dvec)) for doc_id, dvec in doc_vectors.items()]
            scores.sort(key=lambda x: -x[1])
            for rank, (doc_id, score) in enumerate(scores[:top_k], start=1):
                f.write(f"{q.query_id} Q0 {doc_id} {rank} {score:.6f} splade\n")

    (p.runs_dir / "splade_query_terms.json").write_text(json.dumps(query_term_weights, indent=2))
    metrics = common.evaluate_run(p.qrels_path, run_path)
    common.append_section(dataset, "Part 5 — SPLADE", {f"SPLADE ({checkpoint})": metrics})
    log.info(f"[Part5] retrieval done. nDCG@10={metrics['nDCG@10']:.4f}")


# =============================================================================
# 3-way expansion-term comparison: SPLADE vs Rocchio/RM3 (4a) vs HyDE (4b)
# for >=10 queries
# =============================================================================
def part5_compare_expansion_terms(dataset: str, n_queries: int = 10, top_terms: int = 15):
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    entry = common.DATASETS[dataset]

    splade_terms_path = p.runs_dir / "splade_query_terms.json"
    if not splade_terms_path.exists():
        raise RuntimeError("Run --stage retrieve first (need splade_query_terms.json).")
    splade_query_terms = json.loads(splade_terms_path.read_text())

    hyde_records = {}
    if p.hyde_jsonl.exists():
        with open(p.hyde_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                hyde_records[rec["query_id"]] = rec

    ds = ir_datasets.load(entry["ir_datasets_id"])
    queries = list(itertools.islice(ds.queries_iter(), n_queries))

    rows = []
    overlap_scores = []
    for q in queries:
        query_word_set = set(q.text.lower().split())

        splade_vec = splade_query_terms.get(q.query_id, {})
        splade_top = set(sorted(splade_vec, key=lambda t: -splade_vec[t])[:top_terms]) - query_word_set

        hyde_rec = hyde_records.get(q.query_id)
        hyde_top = set()
        if hyde_rec:
            term_counts = {}
            for doc in hyde_rec["hyde_docs"]:
                for t in doc.lower().split():
                    term_counts[t] = term_counts.get(t, 0) + 1
            hyde_top = set(sorted(term_counts, key=lambda t: -term_counts[t])[:top_terms]) - query_word_set

        # NOTE: for a real Rocchio term set per query, run part4a's
        # _get_feedback_term_weights on this query and cache it there instead
        # of recomputing per-query here; left as a TODO hook since it depends
        # on your chosen (N, k) setting from Part 4a.
        rocchio_top = set()  # TODO: populate from your Part 4a run for this query_id

        overlap = splade_top & hyde_top & rocchio_top if rocchio_top else splade_top & hyde_top
        overlap_scores.append(len(overlap))
        rows.append({
            "query_id": q.query_id, "query": q.text,
            "splade_terms": sorted(splade_top), "hyde_terms": sorted(hyde_top),
            "rocchio_terms": sorted(rocchio_top), "overlap": sorted(overlap),
        })

    out_path = p.runs_dir / "part5_expansion_term_comparison.json"
    out_path.write_text(json.dumps(rows, indent=2))

    notes = f"Per-query term lists: `{out_path}`\n\n"
    notes += "| Query | SPLADE terms | HyDE terms | Rocchio terms | Overlap |\n|---|---|---|---|---|\n"
    for r in rows:
        notes += (f"| {r['query'][:40]} | {', '.join(r['splade_terms'][:5])} | "
                  f"{', '.join(r['hyde_terms'][:5])} | {', '.join(r['rocchio_terms'][:5])} | "
                  f"{', '.join(r['overlap'][:5])} |\n")
    notes += ("\nTODO: fill in rocchio_terms (hook noted in code), then discuss 2-3 disagreement "
              "cases where the three sources pick different expansion terms and why.")

    common.append_section(dataset, "Part 5 — Expansion Term Comparison (SPLADE vs Rocchio/RM3 vs HyDE)",
                           rows=None, notes=notes)
    log.info(f"[Part5] expansion-term comparison written for {len(rows)} queries.")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(common.DATASETS))
    parser.add_argument("--checkpoint", default="naver/splade-cocondenser-ensembledistil")
    parser.add_argument("--stage", nargs="+", default=["encode", "retrieve", "compare"],
                         choices=["encode", "retrieve", "compare"])
    parser.add_argument("--shard_id", type=int, default=0, help="set by bash launcher, one process per GPU")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_compare_queries", type=int, default=10)
    args = parser.parse_args()

    if "encode" in args.stage:
        encode_corpus_shard(args.dataset, args.checkpoint, args.shard_id, args.num_shards, args.batch_size)
    if "retrieve" in args.stage:
        part5_retrieve(args.dataset, args.checkpoint)
    if "compare" in args.stage:
        part5_compare_expansion_terms(args.dataset, n_queries=args.n_compare_queries)
