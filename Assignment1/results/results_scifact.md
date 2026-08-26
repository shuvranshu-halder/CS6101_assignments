# Results — scifact

Corpus size: 5183 | Queries: 300

## Part 1 — Index Build


Build time: 5.0s | Index size: 10.1 MB | Corpus docs: 5183 | Queries: 300

### Part 2 — BM25 k1/b Grid Search (full results)

| k1 | b | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|---|
| 1.8 | 0.8 | 0.6903 | 0.9282 | 0.6558 | 0.6495 | **<- winner**
| 1.5 | 0.8 | 0.6889 | 0.9282 | 0.6548 | 0.6490 |
| 1.8 | 0.4 | 0.6857 | 0.9249 | 0.6530 | 0.6469 |
| 1.5 | 0.6 | 0.6844 | 0.9282 | 0.6496 | 0.6437 |
| 1.8 | 0.6 | 0.6842 | 0.9282 | 0.6490 | 0.6433 |
| 0.9 | 0.8 | 0.6841 | 0.9280 | 0.6514 | 0.6452 |
| 1.8 | 0.3 | 0.6835 | 0.9216 | 0.6504 | 0.6454 |
| 1.2 | 0.8 | 0.6833 | 0.9282 | 0.6475 | 0.6409 |
| 0.6 | 0.8 | 0.6823 | 0.9280 | 0.6499 | 0.6444 |
| 1.5 | 0.4 | 0.6822 | 0.9249 | 0.6495 | 0.6451 |
| 1.5 | 0.3 | 0.6821 | 0.9249 | 0.6487 | 0.6432 |
| 0.9 | 0.6 | 0.6814 | 0.9280 | 0.6484 | 0.6427 |
| 1.2 | 0.6 | 0.6810 | 0.9282 | 0.6463 | 0.6399 |
| 1.2 | 0.4 | 0.6803 | 0.9276 | 0.6487 | 0.6436 |
| 1.2 | 0.3 | 0.6802 | 0.9220 | 0.6485 | 0.6432 |
| 0.6 | 0.6 | 0.6800 | 0.9247 | 0.6480 | 0.6423 |
| 0.9 | 0.4 | 0.6789 | 0.9253 | 0.6457 | 0.6401 |
| 0.9 | 0.3 | 0.6786 | 0.9287 | 0.6453 | 0.6394 |
| 0.6 | 0.3 | 0.6776 | 0.9280 | 0.6473 | 0.6399 |
| 0.6 | 0.4 | 0.6773 | 0.9280 | 0.6464 | 0.6411 |

Selection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.
Winner: k1=1.8, b=0.8. Full grid also saved to `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/bm25_grid_search.csv`.

## Part 2 — BM25 vs TF-IDF (final comparison)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (default k1=0.9 b=0.4) | 0.6789 | 0.9253 | 0.6457 | 0.6401 |
| BM25 (tuned k1=1.8 b=0.8) | 0.6903 | 0.9282 | 0.6558 | 0.6495 |
| TF-IDF | 0.6903 | 0.9282 | 0.6558 | 0.6495 |

k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 (see grid table above). TF-IDF used Lucene's ClassicSimilarity.

## Part 3 — Vocabulary Mismatch


n_success=272 avg_jaccard=0.0497 | n_failure=67 avg_jaccard=0.0303

Full per-query records: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part3_failures.json`

TODO: categorize failures (synonymy / paraphrase / abbrev-expansion / other), pick 2-3 examples per category, write the overlap-vs-failure verdict.


### Part 2 — BM25 k1/b Grid Search (full results)

| k1 | b | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|---|
| 1.8 | 0.8 | 0.6903 | 0.9282 | 0.6558 | 0.6495 | **<- winner**
| 1.5 | 0.8 | 0.6889 | 0.9282 | 0.6548 | 0.6490 |
| 1.8 | 0.4 | 0.6857 | 0.9249 | 0.6530 | 0.6469 |
| 1.5 | 0.6 | 0.6844 | 0.9282 | 0.6496 | 0.6437 |
| 1.8 | 0.6 | 0.6842 | 0.9282 | 0.6490 | 0.6433 |
| 0.9 | 0.8 | 0.6841 | 0.9280 | 0.6514 | 0.6452 |
| 1.8 | 0.3 | 0.6835 | 0.9216 | 0.6504 | 0.6454 |
| 1.2 | 0.8 | 0.6833 | 0.9282 | 0.6475 | 0.6409 |
| 0.6 | 0.8 | 0.6823 | 0.9280 | 0.6499 | 0.6444 |
| 1.5 | 0.4 | 0.6822 | 0.9249 | 0.6495 | 0.6451 |
| 1.5 | 0.3 | 0.6821 | 0.9249 | 0.6487 | 0.6432 |
| 0.9 | 0.6 | 0.6814 | 0.9280 | 0.6484 | 0.6427 |
| 1.2 | 0.6 | 0.6810 | 0.9282 | 0.6463 | 0.6399 |
| 1.2 | 0.4 | 0.6803 | 0.9276 | 0.6487 | 0.6436 |
| 1.2 | 0.3 | 0.6802 | 0.9220 | 0.6485 | 0.6432 |
| 0.6 | 0.6 | 0.6800 | 0.9247 | 0.6480 | 0.6423 |
| 0.9 | 0.4 | 0.6789 | 0.9253 | 0.6457 | 0.6401 |
| 0.9 | 0.3 | 0.6786 | 0.9287 | 0.6453 | 0.6394 |
| 0.6 | 0.3 | 0.6776 | 0.9280 | 0.6473 | 0.6399 |
| 0.6 | 0.4 | 0.6773 | 0.9280 | 0.6464 | 0.6411 |

Selection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.
Winner: k1=1.8, b=0.8. Full grid also saved to `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/bm25_grid_search.csv`.

## Part 2 — BM25 vs TF-IDF (final comparison)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (default k1=1.2 b=0.75) | 0.6826 | 0.9282 | 0.6464 | 0.6398 |
| BM25 (tuned k1=1.8 b=0.8) | 0.6903 | 0.9282 | 0.6558 | 0.6495 |
| TF-IDF | 0.6903 | 0.9282 | 0.6558 | 0.6495 |

k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 (see grid table above). TF-IDF used Lucene's ClassicSimilarity.

### Part 2 — BM25 k1/b Grid Search (full results)

| k1 | b | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|---|
| 1.8 | 0.8 | 0.6903 | 0.9282 | 0.6558 | 0.6495 | **<- winner**
| 1.5 | 0.8 | 0.6889 | 0.9282 | 0.6548 | 0.6490 |
| 1.8 | 0.4 | 0.6857 | 0.9249 | 0.6530 | 0.6469 |
| 1.5 | 0.6 | 0.6844 | 0.9282 | 0.6496 | 0.6437 |
| 1.8 | 0.6 | 0.6842 | 0.9282 | 0.6490 | 0.6433 |
| 0.9 | 0.8 | 0.6841 | 0.9280 | 0.6514 | 0.6452 |
| 1.8 | 0.3 | 0.6835 | 0.9216 | 0.6504 | 0.6454 |
| 1.2 | 0.8 | 0.6833 | 0.9282 | 0.6475 | 0.6409 |
| 0.6 | 0.8 | 0.6823 | 0.9280 | 0.6499 | 0.6444 |
| 1.5 | 0.4 | 0.6822 | 0.9249 | 0.6495 | 0.6451 |
| 1.5 | 0.3 | 0.6821 | 0.9249 | 0.6487 | 0.6432 |
| 0.9 | 0.6 | 0.6814 | 0.9280 | 0.6484 | 0.6427 |
| 1.2 | 0.6 | 0.6810 | 0.9282 | 0.6463 | 0.6399 |
| 1.2 | 0.4 | 0.6803 | 0.9276 | 0.6487 | 0.6436 |
| 1.2 | 0.3 | 0.6802 | 0.9220 | 0.6485 | 0.6432 |
| 0.6 | 0.6 | 0.6800 | 0.9247 | 0.6480 | 0.6423 |
| 0.9 | 0.4 | 0.6789 | 0.9253 | 0.6457 | 0.6401 |
| 0.9 | 0.3 | 0.6786 | 0.9287 | 0.6453 | 0.6394 |
| 0.6 | 0.3 | 0.6776 | 0.9280 | 0.6473 | 0.6399 |
| 0.6 | 0.4 | 0.6773 | 0.9280 | 0.6464 | 0.6411 |

Selection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.
Winner: k1=1.8, b=0.8. Full grid also saved to `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/bm25_grid_search.csv`.

## Part 2 — BM25 vs TF-IDF (final comparison)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (default k1=1.2 b=0.75) | 0.6826 | 0.9282 | 0.6464 | 0.6398 |
| BM25 (tuned k1=1.8 b=0.8) | 0.6903 | 0.9282 | 0.6558 | 0.6495 |
| TF-IDF | 0.6903 | 0.9282 | 0.6558 | 0.6495 |

k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 (see grid table above). TF-IDF used Lucene's ClassicSimilarity.

### Part 2 — BM25 k1/b Grid Search (full results)

| k1 | b | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|---|
| 1.8 | 0.8 | 0.6903 | 0.9282 | 0.6558 | 0.6495 | **<- winner**
| 1.5 | 0.8 | 0.6889 | 0.9282 | 0.6548 | 0.6490 |
| 1.8 | 0.4 | 0.6857 | 0.9249 | 0.6530 | 0.6469 |
| 1.5 | 0.6 | 0.6844 | 0.9282 | 0.6496 | 0.6437 |
| 1.8 | 0.6 | 0.6842 | 0.9282 | 0.6490 | 0.6433 |
| 0.9 | 0.8 | 0.6841 | 0.9280 | 0.6514 | 0.6452 |
| 1.8 | 0.3 | 0.6835 | 0.9216 | 0.6504 | 0.6454 |
| 1.2 | 0.8 | 0.6833 | 0.9282 | 0.6475 | 0.6409 |
| 0.6 | 0.8 | 0.6823 | 0.9280 | 0.6499 | 0.6444 |
| 1.5 | 0.4 | 0.6822 | 0.9249 | 0.6495 | 0.6451 |
| 1.5 | 0.3 | 0.6821 | 0.9249 | 0.6487 | 0.6432 |
| 0.9 | 0.6 | 0.6814 | 0.9280 | 0.6484 | 0.6427 |
| 1.2 | 0.6 | 0.6810 | 0.9282 | 0.6463 | 0.6399 |
| 1.2 | 0.4 | 0.6803 | 0.9276 | 0.6487 | 0.6436 |
| 1.2 | 0.3 | 0.6802 | 0.9220 | 0.6485 | 0.6432 |
| 0.6 | 0.6 | 0.6800 | 0.9247 | 0.6480 | 0.6423 |
| 0.9 | 0.4 | 0.6789 | 0.9253 | 0.6457 | 0.6401 |
| 0.9 | 0.3 | 0.6786 | 0.9287 | 0.6453 | 0.6394 |
| 0.6 | 0.3 | 0.6776 | 0.9280 | 0.6473 | 0.6399 |
| 0.6 | 0.4 | 0.6773 | 0.9280 | 0.6464 | 0.6411 |

Selection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.
Winner: k1=1.8, b=0.8. Full grid also saved to `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/bm25_grid_search.csv`.

## Part 2 — BM25 vs TF-IDF (final comparison)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (default k1=1.2 b=0.75) | 0.6826 | 0.9282 | 0.6464 | 0.6398 |
| BM25 (tuned k1=1.8 b=0.8) | 0.6903 | 0.9282 | 0.6558 | 0.6495 |
| TF-IDF | 0.6744 | 0.9032 | 0.6390 | 0.6281 |

k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 (see grid table above). TF-IDF used Lucene's ClassicSimilarity.

### Part 2 — BM25 k1/b Grid Search (full results)

| k1 | b | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|---|
| 1.8 | 0.8 | 0.6903 | 0.9282 | 0.6558 | 0.6495 | **<- winner**
| 1.5 | 0.8 | 0.6889 | 0.9282 | 0.6548 | 0.6490 |
| 1.8 | 0.4 | 0.6857 | 0.9249 | 0.6530 | 0.6469 |
| 1.5 | 0.6 | 0.6844 | 0.9282 | 0.6496 | 0.6437 |
| 1.8 | 0.6 | 0.6842 | 0.9282 | 0.6490 | 0.6433 |
| 0.9 | 0.8 | 0.6841 | 0.9280 | 0.6514 | 0.6452 |
| 1.8 | 0.3 | 0.6835 | 0.9216 | 0.6504 | 0.6454 |
| 1.2 | 0.8 | 0.6833 | 0.9282 | 0.6475 | 0.6409 |
| 0.6 | 0.8 | 0.6823 | 0.9280 | 0.6499 | 0.6444 |
| 1.5 | 0.4 | 0.6822 | 0.9249 | 0.6495 | 0.6451 |
| 1.5 | 0.3 | 0.6821 | 0.9249 | 0.6487 | 0.6432 |
| 0.9 | 0.6 | 0.6814 | 0.9280 | 0.6484 | 0.6427 |
| 1.2 | 0.6 | 0.6810 | 0.9282 | 0.6463 | 0.6399 |
| 1.2 | 0.4 | 0.6803 | 0.9276 | 0.6487 | 0.6436 |
| 1.2 | 0.3 | 0.6802 | 0.9220 | 0.6485 | 0.6432 |
| 0.6 | 0.6 | 0.6800 | 0.9247 | 0.6480 | 0.6423 |
| 0.9 | 0.4 | 0.6789 | 0.9253 | 0.6457 | 0.6401 |
| 0.9 | 0.3 | 0.6786 | 0.9287 | 0.6453 | 0.6394 |
| 0.6 | 0.3 | 0.6776 | 0.9280 | 0.6473 | 0.6399 |
| 0.6 | 0.4 | 0.6773 | 0.9280 | 0.6464 | 0.6411 |

Selection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.
Winner: k1=1.8, b=0.8. Full grid also saved to `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/bm25_grid_search.csv`.

## Part 2 — BM25 vs TF-IDF (final comparison)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (default k1=1.2 b=0.75) | 0.6826 | 0.9282 | 0.6464 | 0.6398 |
| BM25 (tuned k1=1.8 b=0.8) | 0.6903 | 0.9282 | 0.6558 | 0.6495 |
| TF-IDF | 0.6744 | 0.9032 | 0.6390 | 0.6281 |

k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 (see grid table above). TF-IDF used Lucene's ClassicSimilarity.

## Part 3 — Vocabulary Mismatch


n_success=272 avg_jaccard=0.0497 | n_failure=67 avg_jaccard=0.0303
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/scifact_jaccard_overlap_distribution.png`
Full per-query records: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part3_failures.json`

## Part 3 — Vocabulary Mismatch


n_success=272 avg_jaccard=0.0497 | n_failure=67 avg_jaccard=0.0303
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/scifact_jaccard_overlap_distribution.png`
Full per-query records: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part3_failures.json`

## Part 3 — Vocabulary Mismatch


n_success=272 avg_jaccard=0.0497 | n_failure=67 avg_jaccard=0.0303
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/scifact_jaccard_overlap_distribution.png`
Failure cases saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part3_failures.json`

## Part 3 — Vocabulary Mismatch


n_success=272 avg_jaccard=0.0497 | n_failure=67 avg_jaccard=0.0303
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/scifact_jaccard_overlap_distribution.png`
Failure cases saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part3_failures.json`

## Part 3 — Vocabulary Mismatch


n_success=272 avg_jaccard=0.0497 | n_failure=67 avg_jaccard=0.0303
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/scifact_jaccard_overlap_distribution.png`
Failure cases saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part3_failures.json`

## Part 3 — Vocabulary Mismatch


n_success=272 avg_jaccard=0.0497 | n_failure=67 avg_jaccard=0.0303
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/scifact_jaccard_overlap_distribution.png`
Failure cases saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part3_failures.json`


## Part 4a — Rocchio & RM3 (scifact)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (Default k1=1.2, b=0.75) | 0.6826 | 0.9282 | 0.6464 | 0.6398 |
| BM25 (Tuned k1=1.8, b=0.8) | 0.6903 | 0.9282 | 0.6558 | 0.6495 |
| Rocchio (N=10, k=10) | 0.6824 | 0.9164 | 0.6485 | 0.6400 |
| RM3 (N=10, k=10) | 0.6519 | 0.9227 | 0.6012 | 0.5949 |
| Rocchio (N=10, k=20) | 0.6820 | 0.9149 | 0.6461 | 0.6413 |
| RM3 (N=10, k=20) | 0.6565 | 0.9277 | 0.6074 | 0.6013 |
| Rocchio (N=20, k=10) | 0.6808 | 0.9164 | 0.6463 | 0.6379 |
| RM3 (N=20, k=10) | 0.6273 | 0.9293 | 0.5766 | 0.5721 |
| Rocchio (N=20, k=20) | 0.6824 | 0.9149 | 0.6466 | 0.6418 |
| RM3 (N=20, k=20) | 0.6468 | 0.9260 | 0.6008 | 0.5971 |

Query Drift Candidates (pick at least 2 concrete examples for report):
- QID 1: '0-dimensional biomaterials show inductive properties.' | Added: ['stem', 'epitheli', '0', 'emt', 'properti']
- QID 5: '1/2000 in UK have abnormal PrP positivity.' | Added: ['strain', 'prion', 'infect']
- QID 36: 'A deficiency of vitamin B12 increases blood levels of homocysteine.' | Added: ['homocystein']
- QID 49: 'ADAR1 binds to Dicer to cleave pre-miRNA.' | Added: ['mirna', 'rna', 'sirna', 'microrna']
- QID 50: 'AIRE is expressed in some skin tumors.' | Added: ['air', 'antigen', 'autoimmun']
- QID 51: 'ALDH1 expression is associated with better breast cancer outcomes.' | Added: ['pd']
- QID 70: 'Activation of PPM1D suppresses p53 function.' | Added: ['suppress', 'tumour', 'mutat', 'mdm2']
- QID 72: 'Activator-inhibitor pairs are provided dorsally by Admpchordin.' | Added: ['dorsal', 'organ', 'ventral']
- QID 94: 'Albendazole is used to treat lymphatic filariasis.' | Added: ['macrophag', 'antigen', 'lymphat']
- QID 100: 'All hematopoietic stem cells segregate their chromosomes randomly.' | Added: ['segreg', 'chromosom']


## Part 5 — SPLADE

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| SPLADE (naver/splade-v3) | 0.7096 | 0.9343 | 0.6831 | 0.6754 |

## Part 5 — SPLADE

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| SPLADE (naver/splade-v3) | 0.7096 | 0.9343 | 0.6831 | 0.6754 |




## Part 4b — HyDE vs Corpus PRF Comparison

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| 1. Naive Concatenation (Query + HyDE) | 0.7027 | 0.9582 | 0.6599 | 0.6519 |
| 2. HyDE + Rocchio (N=20, k=20) | 0.6804 | 0.9332 | 0.6452 | 0.6377 |
| 3. Part 4a Corpus PRF (N=20, k=20) | 0.6824 | 0.9149 | 0.6466 | 0.6418 |

Comparison across Naive Concatenation, Rocchio-Weighted HyDE, and Corpus PRF.

## Part 5 — SPLADE

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| SPLADE (naver/splade-v3) | 0.7096 | 0.9343 | 0.6831 | 0.6754 |

## Part 5 — Expansion Term Comparison (SPLADE vs Rocchio/RM3 vs HyDE)


Per-query term lists: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part5_expansion_term_comparison.json` | avg 3-way overlap: 0.10 terms/query

| Query | SPLADE terms | HyDE terms | Rocchio terms | Overlap |
|---|---|---|---|---|
| 0-dimensional biomaterials show inductiv | ##mate, ##rial, ##tive, ##uc, 0 | and, biological, in, materials, of | 0, breast, dimension, emt, epitheli |  |
| 1,000 genomes project enables mapping of | ##rance, ##s, 000, gene, genome | a, and, human, the, to |  |  |
| 1/2000 in UK have abnormal PrP positivit | ##ivity, ##p, ##sit, %, 20 | according, and, disease, is, of | 0, breast, hpv, infect, prion |  |
| 5% of perinatal mortality is due to low  | ##ina, ##tal, ##weight, %, baby | a, according, and, as, can | 95, mortal, pregnanc, weight, women | weight |
| A deficiency of vitamin B12 increases bl | ##2, ##cy, ##stein, b, b1 | amino, an, and, health, homocysteine | acid, d, folat, folic, homocystein |  |
| A high microerythrocyte count raises vul | ##cy, ##emia, ##ery, ##lass, ##ro | alpha-globin, and, are, blood, by |  |  |
| A total of 1,000 people in the UK are as | ##j, ##ym, as, britain, carrier | and, been, disease, have, health | hiv, infect |  |
| ADAR1 binds to Dicer to cleave pre-miRNA | ##1, ##ea, ##na, ##r, ##ve | a, and, cleavage, enzyme, for | microrna, mir, mirna, rna, sequenc |  |
| AIRE is expressed in some skin tumors. | ##e, air, antigen, cancer, cancers | a, and, expression, immune, its | air, antigen, autoimmun, defici, melanoma |  |
| ALDH1 expression is associated with bett | ##1, ##dh, al, breasts, cancers | a, and, in, of, outcomes | 95, ci, diabet, metabol, pd |  |

TODO: discuss 2-3 disagreement cases where the three sources pick different expansion terms and why (e.g. SPLADE finding semantically related but lexically distant terms vs. Rocchio/RM3's corpus-cooccurrence terms vs. HyDE's LLM-hallucinated-but-plausible terms).


## Part 5 — SPLADE

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| SPLADE (naver/splade-v3) | 0.7096 | 0.9343 | 0.6831 | 0.6754 |

## Part 5 — Expansion Term Comparison (SPLADE vs Rocchio/RM3 vs HyDE)


Per-query term lists: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/scifact/part5_expansion_term_comparison.json` | avg 3-way overlap: 0.20 terms/query

| Query | SPLADE terms | HyDE terms | Rocchio terms | Overlap |
|---|---|---|---|---|
| 0-dimensional biomaterials show inductiv | 0, 3d, bio, biology, characteristics | ability, applications, biological, exhibit, interactions | 0, breast, dimension, emt, epitheli |  |
| 1,000 genomes project enables mapping of | 000, 1, 1000, dna, gen | aimed, human, individuals, international, our |  |  |
| 1/2000 in UK have abnormal PrP positivit | %, 20, 2000, britain, hepatitis | 1, according, data, disease, diseases | 0, breast, hpv, infect, prion |  |
| 5% of perinatal mortality is due to low  | %, 5, baby, casualties, death | according, approximately, attributed, care, defined | 95, mortal, pregnanc, weight, women | weight |
| A deficiency of vitamin B12 increases bl | b, b1, cause, compound, def | acid, amino, cardiovascular, crucial, health | acid, d, folat, folic, homocystein |  |
| A high microerythrocyte count raises vul | +, antigen, blood, deficiency, disease | +)-thalassemia, alpha-globin, blood, cells, chains |  |  |
| A total of 1,000 people in the UK are as | britain, carrier, disease, hiv, infection | according, creutzfeldt-jakob, data, disease, health | hiv, infect |  |
| ADAR1 binds to Dicer to cleave pre-miRNA | ad, ada, bind, binding, cl | cleavage, enzyme, gene, interaction, key | microrna, mir, mirna, rna, sequenc | rna |
| AIRE is expressed in some skin tumors. | air, antigen, appearance, cancer, cancers | autoimmune, certain, development, expression, function | air, antigen, autoimmun, defici, melanoma |  |
| ALDH1 expression is associated with bett | al, antigen, benefit, breasts, cancers | 1, aldehyde, biomarker, dehydrogenase, higher | 95, ci, diabet, metabol, pd |  |

TODO: discuss 2-3 disagreement cases where the three sources pick different expansion terms and why (e.g. SPLADE finding semantically related but lexically distant terms vs. Rocchio/RM3's corpus-cooccurrence terms vs. HyDE's LLM-hallucinated-but-plausible terms).
