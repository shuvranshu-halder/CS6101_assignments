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

## Part 4a — Rocchio & RM3

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| Rocchio (N=10, k=10) | 0.6813 | 0.9116 | 0.6488 | 0.6411 |
| RM3 (N=10, k=10) | 0.6519 | 0.9227 | 0.6012 | 0.5949 |
| Rocchio (N=10, k=20) | 0.6832 | 0.9149 | 0.6477 | 0.6430 |
| RM3 (N=10, k=20) | 0.6565 | 0.9277 | 0.6074 | 0.6013 |
| Rocchio (N=20, k=10) | 0.6813 | 0.9116 | 0.6488 | 0.6411 |
| RM3 (N=20, k=10) | 0.6273 | 0.9293 | 0.5766 | 0.5721 |
| Rocchio (N=20, k=20) | 0.6832 | 0.9149 | 0.6477 | 0.6430 |
| RM3 (N=20, k=20) | 0.6468 | 0.9260 | 0.6008 | 0.5971 |

Query-drift candidates (pick 2 concrete examples for the report):
- q='0-dimensional biomaterials show inductive properties.' added_terms=['stem', 'properti', 'emt', '0', 'epitheli']
- q='1/2000 in UK have abnormal PrP positivity.' added_terms=['strain', 'prion', 'infect']
- q='A deficiency of vitamin B12 increases blood levels of homocysteine.' added_terms=['homocystein']
- q='ADAR1 binds to Dicer to cleave pre-miRNA.' added_terms=['sirna', 'rna', 'microrna', 'mirna']
- q='AIRE is expressed in some skin tumors.' added_terms=['air', 'autoimmun', 'antigen']

