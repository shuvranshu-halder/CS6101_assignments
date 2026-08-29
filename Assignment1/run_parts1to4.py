"""
Runs Part 1 (index), Part 2 (BM25 tune + baselines), Part 3 (vocab mismatch),
Part 4a (Rocchio/RM3), Part 4b (HyDE) for ONE dataset.

Usage:
    python run_parts1to4.py --dataset scifact                  # runs everything, in order
    python run_parts1to4.py --dataset scifact --stage part2    # runs just one stage
    python run_parts1to4.py --dataset scifact --stage part1 part2 part3

Stages: part1, part2, part3, part4a, part4b_generate, part4b_run (default: all, in order)
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import querybuilder
import common
import torch

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import multiprocessing as mp

DF_CUTOFF_FRACTION = 0.10  # Part 4a: drop terms appearing in >10% of corpus


# =============================================================================
# Generic pre-download helper (works for any BEIR dataset, not just hotpotqa)
# =============================================================================
def ensure_beir_source_downloaded(ir_datasets_id: str, tries: int = 3, timeout: int = 30):
    """
    Some BEIR mirrors (e.g. the TU Darmstadt host) are flaky and ir_datasets'
    own downloader has no retry cap, so a bad connection can hang forever.

    This looks up the *real* download URL, expected md5, and target cache
    path directly from ir_datasets' own registry for whichever dataset id is
    passed in, then fetches it with `wget -c --tries=N` (bounded retries,
    resumable) straight into the exact path ir_datasets expects. If the file
    is already there (fully or partially downloaded), ir_datasets.load(...)
    will just pick it up and skip downloading entirely.

    No-ops for any dataset id that isn't under the 'beir/' namespace.
    """
    import subprocess
    from ir_datasets.util import home_path
    from ir_datasets.util.download import DownloadConfig, LocalDownload

    parts = ir_datasets_id.split("/")
    if len(parts) < 2 or parts[0] != "beir":
        return  # not a BEIR dataset -- nothing to do here
    namespace, subset = parts[0], parts[1]

    dlc_ctxt = DownloadConfig.context(namespace, home_path() / namespace)
    d = dlc_ctxt[subset]

    # Where ir_datasets will ultimately look for/save the file
    if d._cache_path is not None:
        target_path = Path(d._cache_path)
    else:
        local_mirror = next((m for m in d.mirrors if isinstance(m, LocalDownload)), None)
        if local_mirror is None:
            return  # no stable local path to pre-populate; let ir_datasets handle it
        target_path = Path(local_mirror._path)

    if target_path.exists():
        return  # already downloaded (fully or the resumable partial ir_datasets left behind)

    remote_mirror = next((m for m in d.mirrors if hasattr(m, "url")), None)
    if remote_mirror is None:
        return  # e.g. instructions-only dataset; nothing to wget

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    print(f"[predownload] fetching {remote_mirror.url} -> {tmp_path}", flush=True)
    subprocess.run(
        [
            "wget", "-c",
            f"--tries={tries}",
            "--retry-connrefused",
            f"--timeout={timeout}",
            remote_mirror.url,
            "-O", str(tmp_path),
        ],
        check=True,
    )

    if d.expected_md5:
        import hashlib
        h = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        if h.hexdigest() != d.expected_md5:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"[predownload] checksum mismatch for {remote_mirror.url}: "
                f"expected {d.expected_md5}, got {h.hexdigest()} "
                f"(download was likely interrupted/corrupted -- deleted, please retry)"
            )
        print(f"[predownload] checksum OK, moving into place", flush=True)

    tmp_path.rename(target_path)


# =============================================================================
# PART 1 — Build Lucene index
# =============================================================================
def part1_build_index(dataset: str, threads: int = 8):
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    entry = common.DATASETS[dataset]

    jsonl_dir = p.index_dir.parent / "corpus_jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / "corpus.jsonl"

    log.info(f"[Part1] dumping corpus for {dataset}...")
    ensure_beir_source_downloaded(entry["corpus_ir_datasets_id"])
    corpus_ds = ir_datasets.load(entry["corpus_ir_datasets_id"])
    n_docs = 0
    with open(jsonl_path, "w") as f:
        for doc in corpus_ds.docs_iter():
            text = getattr(doc, "text", "") or ""
            title = getattr(doc, "title", "") or ""
            f.write(json.dumps({"id": doc.doc_id, "contents": f"{title}\n{text}".strip()}) + "\n")
            n_docs += 1

    log.info(f"[Part1] dumping qrels/queries...")
    ensure_beir_source_downloaded(entry["ir_datasets_id"])
    query_ds = ir_datasets.load(entry["ir_datasets_id"])
    with open(p.qrels_path, "w") as f:
        for qrel in query_ds.qrels_iter():
            f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
    n_queries = sum(1 for _ in query_ds.queries_iter())

    log.info(f"[Part1] building Lucene index over {n_docs} docs...")
    import subprocess
    t0 = time.time()
    subprocess.run([
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(jsonl_dir),
        "--index", str(p.index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ], check=True)
    build_time_s = time.time() - t0
    index_size_mb = sum(f.stat().st_size for f in p.index_dir.rglob("*") if f.is_file()) / (1024 ** 2)

    common.init_results_file(dataset, n_docs, n_queries)
    common.append_section(
        dataset, "Part 1 — Index Build", rows=None,
        notes=f"Build time: {build_time_s:.1f}s | Index size: {index_size_mb:.1f} MB "
              f"| Corpus docs: {n_docs} | Queries: {n_queries}",
    )
    log.info(f"[Part1] done. build_time={build_time_s:.1f}s size={index_size_mb:.1f}MB")


# =============================================================================
# PART 2 — BM25 grid tune + default/tuned/TF-IDF baselines
# =============================================================================
def _load_queries(dataset):
    entry = common.DATASETS[dataset]
    ds = ir_datasets.load(entry["ir_datasets_id"])
    return [(q.query_id, q.text) for q in ds.queries_iter()]


def _run_searcher_to_trec(searcher, queries, run_path, tag, top_k=1000):
    with open(run_path, "w") as f:
        for qid, qtext in queries:
            hits = searcher.search(qtext, k=top_k)
            for rank, hit in enumerate(hits, start=1):
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {tag}\n")


def part2_bm25_baselines(dataset: str):
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    queries = _load_queries(dataset)
    searcher = LuceneSearcher(str(p.index_dir))

    # --- grid search: tries EVERY (k1, b) combo, saves all of them ---
    log.info(f"[Part2] grid search over k1={common.BM25_GRID['k1']} b={common.BM25_GRID['b']}")
    grid_rows = []
    for k1, b in itertools.product(common.BM25_GRID["k1"], common.BM25_GRID["b"]):
        searcher.set_bm25(k1, b)
        run_path = p.runs_dir / f"_grid_k1{k1}_b{b}.trec"
        _run_searcher_to_trec(searcher, queries, run_path, f"grid_k1{k1}_b{b}")
        metrics = common.evaluate_run(p.qrels_path, run_path)
        log.info(f"  k1={k1} b={b} -> nDCG@10={metrics['nDCG@10']:.4f}")
        grid_rows.append({"k1": k1, "b": b, **metrics})

    best = common.save_grid_results(dataset, grid_rows)  # writes CSV + full table + returns winner
    common.save_tuned_bm25(dataset, best["k1"], best["b"])
    log.info(f"[Part2] winner: k1={best['k1']} b={best['b']} (saved to bm25_params.json)")

    # --- final comparison table: default BM25 vs tuned BM25 vs TF-IDF ---
    results = {}

    # 1. Default BM25 (k1=1.2, b=0.75)
    default_k1, default_b = 1.2, 0.75
    searcher.set_bm25(default_k1, default_b)
    run_path = p.runs_dir / "bm25_default.trec"
    _run_searcher_to_trec(searcher, queries, run_path, "bm25_default")
    results[f"BM25 (default k1={default_k1} b={default_b})"] = \
        common.evaluate_run(p.qrels_path, run_path)

    # 2. Tuned BM25
    searcher.set_bm25(best["k1"], best["b"])
    run_path = p.runs_dir / "bm25_tuned.trec"
    _run_searcher_to_trec(searcher, queries, run_path, "bm25_tuned")
    results[f"BM25 (tuned k1={best['k1']} b={best['b']})"] = common.evaluate_run(p.qrels_path, run_path)

    # 3. TF-IDF (ClassicSimilarity)
    try:
        if hasattr(searcher, "set_tfidf"):
            searcher.set_tfidf()
        else:
            from pyserini.pyclass import autoclass
            JClassicSimilarity = autoclass("org.apache.lucene.search.similarities.ClassicSimilarity")
            # In newer Anserini versions, access similarity through searcher.object.searcher
            if hasattr(searcher.object, "set_similarity"):
                searcher.object.set_similarity(JClassicSimilarity())
            elif hasattr(searcher.object, "searcher"):
                searcher.object.searcher.setSimilarity(JClassicSimilarity())
            else:
                raise AttributeError("Could not find setSimilarity method on Java searcher object.")
    except Exception as e:
        log.error(f"[Part2] Failed to set TF-IDF similarity: {e}")
        raise e  # Fail fast instead of silently copying BM25 metrics

    run_path = p.runs_dir / "tfidf.trec"
    _run_searcher_to_trec(searcher, queries, run_path, "tfidf")
    results["TF-IDF"] = common.evaluate_run(p.qrels_path, run_path)

    common.append_section(
        dataset, "Part 2 — BM25 vs TF-IDF (final comparison)", results,
        notes="k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 "
              "(see grid table above). TF-IDF used Lucene's ClassicSimilarity.",
    )
    log.info(f"[Part2] done.")

# =============================================================================
# PART 3 — Vocabulary mismatch analysis
# =============================================================================
def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize(text: str) -> set:
    return set(text.lower().split())  # TODO: swap for a real tokenizer + stopword removal


def generate_part3_deliverables(dataset: str, succ_j: list, fail_j: list) -> dict:
    p = common.get_paths(dataset)
    
    stats = {
        "succ_mean": float(np.mean(succ_j)) if succ_j else 0.0,
        "succ_median": float(np.median(succ_j)) if succ_j else 0.0,
        "fail_mean": float(np.mean(fail_j)) if fail_j else 0.0,
        "fail_median": float(np.median(fail_j)) if fail_j else 0.0,
    }

    # Plot Jaccard overlap distribution dynamically
    plt.figure(figsize=(8, 5))
    if succ_j:
        plt.hist(succ_j, bins=20, alpha=0.6, label=f"BM25 Successes (N={len(succ_j)})", color="green", density=True)
    if fail_j:
        plt.hist(fail_j, bins=20, alpha=0.6, label=f"BM25 Failures (N={len(fail_j)})", color="red", density=True)
        
    plt.xlabel("Jaccard Overlap")
    plt.ylabel("Density")
    plt.title(f"[{dataset.upper()}] Jaccard Overlap Distribution: BM25 Success vs Failure")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_path = p.runs_dir / f"{dataset}_jaccard_overlap_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    stats["plot_path"] = str(plot_path)
    return stats


def part3_vocab_mismatch(dataset: str, top_k: int = 10):
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    entry = common.DATASETS[dataset]
    tuned = common.load_tuned_bm25(dataset)
    if tuned is None:
        raise RuntimeError("Run part2 first (need tuned k1/b).")

    searcher = LuceneSearcher(str(p.index_dir))
    searcher.set_bm25(tuned["k1"], tuned["b"])

    query_ds = ir_datasets.load(entry["ir_datasets_id"])
    docstore = ir_datasets.load(entry["corpus_ir_datasets_id"]).docs_store()
    qrels = {}
    for qrel in query_ds.qrels_iter():
        qrels.setdefault(qrel.query_id, []).append(qrel.doc_id)

    failures = []
    succ_jaccards = []
    fail_jaccards = []

    for q in query_ds.queries_iter():
        gold_ids = qrels.get(q.query_id, [])
        if not gold_ids:
            continue
        hits = searcher.search(q.text, k=top_k)
        retrieved_ids = {h.docid for h in hits}
        
        # Store retrieved documents info for failure analysis
        retrieved_docs = []
        for h in hits:
            rdoc = docstore.get(h.docid)
            rtext = f"{getattr(rdoc, 'title', '')} {getattr(rdoc, 'text', '')}".strip()
            retrieved_docs.append({"docid": h.docid, "score": float(h.score), "text": rtext})

        for gold_id in gold_ids:
            gold_doc = docstore.get(gold_id)
            gold_text = f"{getattr(gold_doc, 'title', '')} {getattr(gold_doc, 'text', '')}".strip()
            overlap = _jaccard(_tokenize(q.text), _tokenize(gold_text))
            
            if gold_id in retrieved_ids:
                succ_jaccards.append(overlap)
            else:
                fail_jaccards.append(overlap)
                record = {
                    "query_id": q.query_id,
                    "query": q.text,
                    "gold_id": gold_id,
                    "gold_text": gold_text,
                    "retrieved_docs": retrieved_docs,
                    "jaccard": overlap
                }
                failures.append(record)

    # 1. Save ONLY failures to JSON for manual inspection
    out_path = p.runs_dir / "part3_failures.json"
    out_path.write_text(json.dumps(failures, indent=2))

    # 2. Generate stats & plot using in-memory Jaccard arrays
    stats = generate_part3_deliverables(dataset, succ_jaccards, fail_jaccards)

    common.append_section(
        dataset, "Part 3 — Vocabulary Mismatch", rows=None,
        notes=(f"n_success={len(succ_jaccards)} avg_jaccard={stats['succ_mean']:.4f} | "
               f"n_failure={len(failures)} avg_jaccard={stats['fail_mean']:.4f}\n"
               f"Plot saved to: `{stats['plot_path']}`\n"
               f"Failure cases saved to: `{out_path}`"),
    )
    log.info(f"[Part3] done. success_avg={stats['succ_mean']:.4f} fail_avg={stats['fail_mean']:.4f}")

# =============================================================================
# PART 4a — Rocchio & RM3
# =============================================================================
def _get_feedback_term_weights(index_reader, feedback_doc_ids, num_docs_total,
                                alpha=1.0, beta=0.75, query_terms=None, k=10):
    """Rocchio: w_t = alpha*f(q)[t] + (beta/N) * sum_{d in feedback} f~(d)[t]"""
    query_terms = query_terms or {}
    N = len(feedback_doc_ids) or 1
    term_scores = {}
    for docid in feedback_doc_ids:
        tf_vector = index_reader.get_document_vector(docid)
        if not tf_vector:
            continue
        doc_len = sum(tf_vector.values())
        for term, tf in tf_vector.items():
            df, _ = index_reader.get_term_counts(term, analyzer=None)
            if not df or df / num_docs_total > DF_CUTOFF_FRACTION:
                continue
            term_scores[term] = term_scores.get(term, 0.0) + (beta / N) * (tf / doc_len if doc_len else 0)
    for term, qtf in query_terms.items():
        term_scores[term] = term_scores.get(term, 0.0) + alpha * qtf
    return dict(sorted(term_scores.items(), key=lambda x: -x[1])[:k])


def build_boosted_query(original_query_text: str, expansion_terms: dict):
    """Query Builder API — required instead of string concatenation.
    Skips any term the Lucene analyzer reduces to nothing (stopwords, bare
    punctuation, etc.) — get_term_query() throws IndexError on those.
    IMPORTANT: qb.add(...) return value must be captured/reassigned —
    Anserini's builder does not reliably mutate in place."""
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    analyzer = Analyzer(get_lucene_analyzer())  # <-- must wrap in Python Analyzer, not use raw JAnalyzer

    def analyzes_to_something(term):
        try:
            return len(analyzer.analyze(term)) > 0
        except Exception:
            return False

    should = querybuilder.JBooleanClauseOccur["should"].value
    qb = querybuilder.get_boolean_query_builder()

    added = 0
    for term in original_query_text.lower().split():
        if not analyzes_to_something(term):
            continue
        qb = qb.add(querybuilder.get_term_query(term), should)
        added += 1
    for term, weight in expansion_terms.items():
        if not analyzes_to_something(term):
            continue
        qb = qb.add(querybuilder.get_boost_query(querybuilder.get_term_query(term), float(weight)), should)
        added += 1

    if added == 0:
        raise ValueError(f"No usable terms in query '{original_query_text}' + {list(expansion_terms)}")

    return qb.build()

def _run_rocchio(dataset, N, k, alpha=1.0, beta=0.75):
    p = common.get_paths(dataset)
    tuned = common.load_tuned_bm25(dataset)
    searcher = LuceneSearcher(str(p.index_dir))
    searcher.set_bm25(tuned["k1"], tuned["b"])
    index_reader = LuceneIndexReader(str(p.index_dir))
    num_docs_total = index_reader.stats()["documents"]

    entry = common.DATASETS[dataset]
    ds = ir_datasets.load(entry["ir_datasets_id"])
    run_path = p.runs_dir / f"rocchio_N{N}_k{k}.trec"
    drift_log = []

    with open(run_path, "w") as f:
        for q in ds.queries_iter():
            feedback_doc_ids = [h.docid for h in searcher.search(q.text, k=N)]
            query_terms = {t: q.text.lower().split().count(t) for t in set(q.text.lower().split())}
            expansion_terms = _get_feedback_term_weights(
                index_reader, feedback_doc_ids, num_docs_total,
                alpha=alpha, beta=beta, query_terms=query_terms, k=k,
            )
            boosted_query = build_boosted_query(q.text, expansion_terms)
            for rank, hit in enumerate(searcher.search(boosted_query, k=1000), start=1):
                f.write(f"{q.query_id} Q0 {hit.docid} {rank} {hit.score:.6f} rocchio_N{N}_k{k}\n")
            new_terms = set(expansion_terms) - set(query_terms)
            if new_terms:
                drift_log.append({"query_id": q.query_id, "query": q.text, "added_terms": list(new_terms)})

    return common.evaluate_run(p.qrels_path, run_path), drift_log

def _run_rm3(dataset, N, k):
    p = common.get_paths(dataset)
    tuned = common.load_tuned_bm25(dataset)
    searcher = LuceneSearcher(str(p.index_dir))
    searcher.set_bm25(tuned["k1"], tuned["b"])
    searcher.set_rm3(fb_docs=N, fb_terms=k, original_query_weight=0.5)

    entry = common.DATASETS[dataset]
    ds = ir_datasets.load(entry["ir_datasets_id"])
    run_path = p.runs_dir / f"rm3_N{N}_k{k}.trec"
    with open(run_path, "w") as f:
        for q in ds.queries_iter():
            for rank, hit in enumerate(searcher.search(q.text, k=1000), start=1):
                f.write(f"{q.query_id} Q0 {hit.docid} {rank} {hit.score:.6f} rm3_N{N}_k{k}\n")
    return common.evaluate_run(p.qrels_path, run_path)


def part4a_rocchio_rm3(dataset: str, N_values=(10, 20), k_values=(10, 20)):
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    results, all_drift = {}, []

    # 1. Load dataset-specific tuned BM25 parameters
    tuned = common.load_tuned_bm25(dataset)
    if tuned is None:
        raise RuntimeError(f"Run part2 first for dataset '{dataset}' to obtain tuned k1/b parameters.")

    tuned_k1, tuned_b = tuned["k1"], tuned["b"]

    # 2. Add Part 2 Baselines for comparison
    default_run = p.runs_dir / "bm25_default.trec"
    tuned_run = p.runs_dir / "bm25_tuned.trec"

    if default_run.exists():
        results["BM25 (Default k1=1.2, b=0.75)"] = common.evaluate_run(p.qrels_path, default_run)
    if tuned_run.exists():
        results[f"BM25 (Tuned k1={tuned_k1}, b={tuned_b})"] = common.evaluate_run(p.qrels_path, tuned_run)

    # 3. Run Rocchio & RM3 grid
    for N, k in itertools.product(N_values, k_values):
        rocchio_metrics, drift = _run_rocchio(dataset, N, k)
        results[f"Rocchio (N={N}, k={k})"] = rocchio_metrics
        results[f"RM3 (N={N}, k={k})"] = _run_rm3(dataset, N, k)
        all_drift.extend(drift)
        log.info(f"[Part4a] N={N} k={k} completed.")

    # 4. Format query drift notes for report extraction
    notes = "Query Drift Candidates (pick at least 2 concrete examples for report):\n"
    for d in all_drift[:10]:
        notes += f"- QID {d['query_id']}: '{d['query']}' | Added: {d['added_terms']}\n"

    common.append_section(dataset, f"Part 4a — Rocchio & RM3 ({dataset})", results, notes=notes)
    log.info(f"[Part4a] Done for dataset: {dataset}")


# =============================================================================
# PART 4b — HyDE (reuses part4a's build_boosted_query unchanged)
# =============================================================================
# =============================================================================
# PART 4b — HyDE
# =============================================================================

def _hyde_worker(
    gpu_id,
    query_items,
    model_name,
    n_samples,
    batch_size,
    output_path,
):
    """
    One complete model copy on one GPU.

    batch_size = queries PER GPU.
    """

    # Each worker uses its own visible/logical GPU.
    # gpu_id is the LOGICAL GPU index inside this process.
    # CUDA_VISIBLE_DEVICES in the .sh file controls which physical GPU
    # each logical index refers to.
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(
        f"[Part4b][GPU {gpu_id}] "
        f"Using logical cuda:{gpu_id} - "
        f"{torch.cuda.get_device_name(gpu_id)}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -------------------------------------------------------------------------
    # COMPLETE model copy on this GPU
    # -------------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    print(
        f"[Part4b][GPU {gpu_id}] Model loaded on cuda:{gpu_id}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Prompt
    # -------------------------------------------------------------------------
    prompt_tmpl = (
        "Write a short passage that answers the following question. "
        "Write as if it is a factual excerpt from a document.\n\n"
        "Question: {q}\n\n"
        "Passage:"
    )

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------
    with open(output_path, "w") as f:

        for local_start in range(0, len(query_items), batch_size):

            batch_items = query_items[
                local_start:local_start + batch_size
            ]

            batch = [item[1] for item in batch_items]

            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": prompt_tmpl.format(q=q.text),
                    }
                ]
                for q in batch
            ]

            # -------------------------------------------------------------
            # LEFT-PADDED batch
            # -------------------------------------------------------------
            inputs = tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
            )

            input_ids = inputs["input_ids"].to(device)

            attention_mask = inputs.get("attention_mask")

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # IMPORTANT:
            # With left padding, generated tokens begin after the full
            # padded input width.
            prompt_length = input_ids.shape[1]

            # -------------------------------------------------------------
            # Generate n_samples per query
            # -------------------------------------------------------------
            with torch.inference_mode():

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=n_samples,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # -------------------------------------------------------------
            # Save results
            # -------------------------------------------------------------
            for i, q in enumerate(batch):

                start_i = i * n_samples
                end_i = start_i + n_samples

                docs = [
                    tokenizer.decode(
                        out[prompt_length:],
                        skip_special_tokens=True,
                    ).strip()
                    for out in outputs[start_i:end_i]
                ]

                original_index = batch_items[i][0]

                record = {
                    "_index": original_index,
                    "query_id": q.query_id,
                    "query": q.text,
                    "hyde_docs": docs,
                }

                f.write(
                    json.dumps(record) + "\n"
                )

            f.flush()

            completed = min(
                local_start + len(batch),
                len(query_items),
            )

            print(
                f"[Part4b][GPU {gpu_id}] "
                f"Progress: {completed}/{len(query_items)} "
                f"(batch_size={len(batch)})",
                flush=True,
            )

    del model
    torch.cuda.empty_cache()

    print(
        f"[Part4b][GPU {gpu_id}] Worker complete.",
        flush=True,
    )


def part4b_generate_hyde_docs(
    dataset: str,
    n_samples: int = 4,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    batch_size: int = 8,
    num_gpus: int = 0,
):
    """
    Multi-GPU data-parallel HyDE generation.

    batch_size means BATCH SIZE PER GPU.

    Example:
        3 GPUs + batch_size=64

        GPU 0 -> 64 queries
        GPU 1 -> 64 queries
        GPU 2 -> 64 queries

        Effective batch = 192 queries.
    """

    log = common.get_logger(dataset)
    p = common.get_paths(dataset)

    # -------------------------------------------------------------------------
    # Determine visible GPUs WITHOUT initializing CUDA.
    #
    # CUDA_VISIBLE_DEVICES is controlled by the .sh file.
    # -------------------------------------------------------------------------
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible:
        physical_gpu_ids = [
            int(x.strip())
            for x in cuda_visible.split(",")
            if x.strip()
        ]

        # IMPORTANT:
        # CUDA_VISIBLE_DEVICES renumbers these GPUs logically.
        #
        # Example:
        # CUDA_VISIBLE_DEVICES=0,2,3
        #
        # becomes:
        #   cuda:0 -> physical GPU 0
        #   cuda:1 -> physical GPU 2
        #   cuda:2 -> physical GPU 3
        #
        # Therefore workers must receive 0,1,2.
        visible_gpu_ids = list(range(len(physical_gpu_ids)))

    else:
        visible_gpu_ids = list(range(torch.cuda.device_count()))

    if not visible_gpu_ids:
        raise RuntimeError(
            "No GPUs available. Check CUDA_VISIBLE_DEVICES and CUDA setup."
        )

    # -------------------------------------------------------------------------
    # Number of GPUs
    # -------------------------------------------------------------------------
    if num_gpus <= 0:
        num_gpus = len(visible_gpu_ids)

    if num_gpus > len(visible_gpu_ids):
        raise ValueError(
            f"Requested {num_gpus} GPUs, but only "
            f"{len(visible_gpu_ids)} GPUs are visible: "
            f"{visible_gpu_ids}"
        )

    # Use only the requested number.
    gpu_ids = visible_gpu_ids[:num_gpus]

    effective_batch_size = batch_size * num_gpus

    log.info("[Part4b] Multi-GPU data-parallel generation")
    log.info(f"[Part4b] Model={model_name}")
    log.info(f"[Part4b] GPUs={gpu_ids}")
    log.info(f"[Part4b] Batch size PER GPU={batch_size}")
    log.info(
        f"[Part4b] Effective global batch size={effective_batch_size}"
    )
    log.info(f"[Part4b] Samples/query={n_samples}")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    entry = common.DATASETS[dataset]
    ds = ir_datasets.load(entry["ir_datasets_id"])

    queries_list = list(ds.queries_iter())
    total_queries = len(queries_list)

    log.info(
        f"[Part4b] Total queries={total_queries}"
    )

    # -------------------------------------------------------------------------
    # Preserve original query order
    # -------------------------------------------------------------------------
    indexed_queries = [
        (idx, q)
        for idx, q in enumerate(queries_list)
    ]

    # -------------------------------------------------------------------------
    # Distribute queries across GPUs
    # -------------------------------------------------------------------------
    gpu_query_lists = [
        indexed_queries[gpu_index::num_gpus]
        for gpu_index in range(num_gpus)
    ]

    for gpu_id, gpu_queries in zip(gpu_ids, gpu_query_lists):

        log.info(
            f"[Part4b] GPU {gpu_id}: "
            f"{len(gpu_queries)} queries"
        )

    # -------------------------------------------------------------------------
    # Temporary files
    # -------------------------------------------------------------------------
    p.hyde_jsonl.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_paths = []

    for gpu_id in gpu_ids:

        temp_path = (
            p.hyde_jsonl.parent
            / f"hyde_generations_gpu{gpu_id}.jsonl"
        )

        if temp_path.exists():
            temp_path.unlink()

        temp_paths.append(temp_path)

    # -------------------------------------------------------------------------
    # CUDA multiprocessing
    # -------------------------------------------------------------------------
    ctx = mp.get_context("spawn")

    processes = []

    for gpu_id, gpu_queries, temp_path in zip(
        gpu_ids,
        gpu_query_lists,
        temp_paths,
    ):

        process = ctx.Process(
            target=_hyde_worker,
            args=(
                gpu_id,
                gpu_queries,
                model_name,
                n_samples,
                batch_size,
                str(temp_path),
            ),
        )

        process.start()
        processes.append(process)

    # -------------------------------------------------------------------------
    # Wait for workers
    # -------------------------------------------------------------------------
    failed = False

    for gpu_id, process in zip(gpu_ids, processes):

        process.join()

        if process.exitcode != 0:

            failed = True

            log.error(
                f"[Part4b] GPU {gpu_id} worker failed "
                f"with exit code {process.exitcode}"
            )

    if failed:

        raise RuntimeError(
            "[Part4b] One or more GPU workers failed. "
            "Check the worker logs above."
        )

    # -------------------------------------------------------------------------
    # Merge outputs
    # -------------------------------------------------------------------------
    log.info(
        "[Part4b] All GPU workers completed. "
        "Merging results..."
    )

    all_records = []

    for temp_path in temp_paths:

        if not temp_path.exists():

            raise RuntimeError(
                f"[Part4b] Missing worker output: {temp_path}"
            )

        with open(temp_path) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                all_records.append(
                    json.loads(line)
                )

    # -------------------------------------------------------------------------
    # Verify count
    # -------------------------------------------------------------------------
    if len(all_records) != total_queries:

        raise RuntimeError(
            f"[Part4b] Expected {total_queries} records, "
            f"but found {len(all_records)}."
        )

    # -------------------------------------------------------------------------
    # Restore original query order
    # -------------------------------------------------------------------------
    all_records.sort(
        key=lambda x: x["_index"]
    )

    # -------------------------------------------------------------------------
    # Write final output
    # -------------------------------------------------------------------------
    with open(p.hyde_jsonl, "w") as f:

        for record in all_records:

            record.pop("_index", None)

            f.write(
                json.dumps(record) + "\n"
            )

    # -------------------------------------------------------------------------
    # Remove temporary files
    # -------------------------------------------------------------------------
    for temp_path in temp_paths:

        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    log.info(
        f"[Part4b] COMPLETE: generated HyDE documents "
        f"for {total_queries} queries."
    )

    log.info(
        f"[Part4b] GPUs used={num_gpus}"
    )

    log.info(
        f"[Part4b] Batch size per GPU={batch_size}"
    )

    log.info(
        f"[Part4b] Effective batch size={effective_batch_size}"
    )

    log.info(
        f"[Part4b] Output={p.hyde_jsonl}"
    )

def _run_naive_hyde_concat(
    dataset: str,
    hyde_full_records: dict,
    run_path,
):
    """
    Variant 1: Naive HyDE concatenation.

    The original query and all generated HyDE documents are concatenated
    into one long query string and submitted directly to Lucene/BM25.
    """

    p = common.get_paths(dataset)

    # Load the same tuned BM25 parameters used elsewhere.
    tuned = common.load_tuned_bm25(dataset)

    if tuned is None:
        raise RuntimeError(
            "BM25 parameters not found. Run Part 2 first."
        )

    # Open the existing Lucene index.
    searcher = LuceneSearcher(str(p.index_dir))

    # Use the tuned BM25 parameters.
    searcher.set_bm25(
        tuned["k1"],
        tuned["b"],
    )

    # Write TREC run.
    with open(run_path, "w") as f:

        for qid, rec in hyde_full_records.items():

            # Original query + all generated HyDE documents.
            expanded_text = (
                rec["query"]
                + " "
                + " ".join(rec["hyde_docs"])
            )

            # Retrieve top 1000 documents.
            hits = searcher.search(
                expanded_text,
                k=1000,
            )

            # Write standard TREC format.
            for rank, hit in enumerate(hits, start=1):

                f.write(
                    f"{qid} Q0 {hit.docid} "
                    f"{rank} {hit.score:.6f} hyde_naive\n"
                )


                
def _run_rocchio_on_hyde(dataset: str, hyde_full_records: dict, run_path, N: int = 4, k: int = 10):
    """Variant 2: reuses Part 4a's build_boosted_query() unchanged — term weights come
    from HyDE doc word frequencies instead of an IndexReader lookup over corpus docs."""
    p = common.get_paths(dataset)
    tuned = common.load_tuned_bm25(dataset)
    searcher = LuceneSearcher(str(p.index_dir))
    searcher.set_bm25(tuned["k1"], tuned["b"])

    with open(run_path, "w") as f:
        for qid, rec in hyde_full_records.items():
            term_counts = {}
            for doc in rec["hyde_docs"]:
                for t in doc.lower().split():
                    term_counts[t] = term_counts.get(t, 0) + 1
            top_terms = dict(sorted(term_counts.items(), key=lambda x: -x[1])[:k])
            boosted_query = build_boosted_query(rec["query"], top_terms)
            for rank, hit in enumerate(searcher.search(boosted_query, k=1000), start=1):
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} hyde_rocchio\n")


def part4b_run_hyde(dataset: str, N: int = 4, k: int = 10):
    log = common.get_logger(dataset)
    p = common.get_paths(dataset)
    results = {}

    # 1. Load generated HyDE documents — keep the FULL record (query + hyde_docs),
    # not just hyde_docs, since both helpers below need the original query text.
    hyde_path = p.runs_dir / "hyde_generations.jsonl"
    if not hyde_path.exists():
        raise FileNotFoundError(f"HyDE docs not found at {hyde_path}. Run --stage part4b_generate first.")

    hyde_records = {}
    with open(hyde_path) as f:
        for line in f:
            rec = json.loads(line)
            hyde_records[rec["query_id"]] = rec  # keep full record: {"query_id","query","hyde_docs"}

    # 2. Variant 1: Naive Concatenation (Query + Concatenated HyDE Documents)
    naive_run_path = p.runs_dir / "hyde_naive_concat.trec"
    _run_naive_hyde_concat(dataset, hyde_records, naive_run_path)
    results["1. Naive Concatenation (Query + HyDE)"] = common.evaluate_run(p.qrels_path, naive_run_path)

    # 3. Variant 2: Rocchio/RM3-Weighted HyDE (HyDE as feedback docs)
    weighted_run_path = p.runs_dir / "hyde_rocchio_weighted.trec"
    _run_rocchio_on_hyde(dataset, hyde_records, weighted_run_path, N=N, k=k)
    results[f"2. HyDE + Rocchio (N={N}, k={k})"] = common.evaluate_run(p.qrels_path, weighted_run_path)

    # 4. Variant 3: 4a Corpus PRF Baseline (Best Rocchio from Part 4a)
    corpus_prf_path = p.runs_dir / f"rocchio_N{N}_k{k}.trec"
    if corpus_prf_path.exists():
        results[f"3. Part 4a Corpus PRF (N={N}, k={k})"] = common.evaluate_run(p.qrels_path, corpus_prf_path)
    else:
        log.warning("Part 4a TREC run not found. Run part4a first to include Corpus PRF.")

    # 5. Append results table to results log
    common.append_section(
        dataset,
        "Part 4b — HyDE vs Corpus PRF Comparison",
        results,
        notes="Comparison across Naive Concatenation, Rocchio-Weighted HyDE, and Corpus PRF."
    )
    log.info(f"[Part4b] Evaluation complete for {dataset}.")


def _part4b_run_with_best_rocchio(ds, args):
    N, k = common.get_best_rocchio_setting(ds)
    part4b_run_hyde(ds, N=N, k=k)
# =============================================================================
# CLI
# =============================================================================
STAGES = {
    "part1": lambda ds, args: part1_build_index(ds),
    "part2": lambda ds, args: part2_bm25_baselines(ds),
    "part3": lambda ds, args: part3_vocab_mismatch(ds),
    "part4a": lambda ds, args: part4a_rocchio_rm3(ds, N_values=args.N, k_values=args.k),
    "part4b_generate": lambda ds, args: part4b_generate_hyde_docs(
        ds,
        n_samples=args.hyde_samples,
        model_name=args.hyde_model,
        batch_size=args.hyde_batch_size,
        num_gpus=args.hyde_num_gpus,
    ),
    "part4b_run":  _part4b_run_with_best_rocchio,
}
DEFAULT_ORDER = ["part1", "part2", "part3", "part4a", "part4b_generate", "part4b_run"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(common.DATASETS)
    )

    parser.add_argument(
        "--stage",
        nargs="+",
        default=DEFAULT_ORDER,
        choices=list(STAGES)
    )

    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[10, 20],
        help="Part 4a: feedback doc counts"
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10, 20],
        help="Part 4a: expansion term counts"
    )

    parser.add_argument(
        "--hyde_samples",
        type=int,
        default=4,
        help="Part 4b: hypothetical docs per query"
    )

    parser.add_argument(
        "--hyde_batch_size",
        type=int,
        default=8,
        help="Part 4b: queries PER GPU"
    )

    parser.add_argument(
        "--hyde_num_gpus",
        type=int,
        default=0,
        help="Part 4b: number of GPUs; 0 = all visible GPUs"
    )

    parser.add_argument(
        "--hyde_model",
        default="Qwen/Qwen2.5-7B-Instruct"
    )

    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Execute requested stages
    # -------------------------------------------------------------------------
    for stage in args.stage:

        print(
            f"\n=== [{args.dataset}] Running stage: {stage} ===",
            flush=True,
        )

        STAGES[stage](args.dataset, args)

    print(
        f"\n=== [{args.dataset}] All requested stages completed ===",
        flush=True,
    )