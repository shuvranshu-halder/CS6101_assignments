# Results — hotpotqa

Corpus size: 5233329 | Queries: 7405

## Part 1 — Index Build


Build time: 329.1s | Index size: 2322.4 MB | Corpus docs: 5233329 | Queries: 7405

### Part 2 — BM25 k1/b Grid Search (full results)

| k1 | b | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|---|
| 0.9 | 0.4 | 0.6330 | 0.7957 | 0.8032 | 0.5502 | **<- winner**
| 0.9 | 0.3 | 0.6329 | 0.7956 | 0.8030 | 0.5491 |
| 0.6 | 0.4 | 0.6321 | 0.7963 | 0.8045 | 0.5485 |
| 1.2 | 0.3 | 0.6302 | 0.7944 | 0.7981 | 0.5470 |
| 0.6 | 0.3 | 0.6288 | 0.7947 | 0.8008 | 0.5449 |
| 1.2 | 0.4 | 0.6284 | 0.7947 | 0.7963 | 0.5454 |
| 0.6 | 0.6 | 0.6282 | 0.7930 | 0.8007 | 0.5457 |
| 1.5 | 0.3 | 0.6242 | 0.7924 | 0.7915 | 0.5414 |
| 0.9 | 0.6 | 0.6231 | 0.7897 | 0.7942 | 0.5411 |
| 1.5 | 0.4 | 0.6214 | 0.7906 | 0.7891 | 0.5385 |
| 1.8 | 0.3 | 0.6161 | 0.7885 | 0.7822 | 0.5329 |
| 0.6 | 0.8 | 0.6152 | 0.7876 | 0.7864 | 0.5331 |
| 1.2 | 0.6 | 0.6138 | 0.7867 | 0.7833 | 0.5314 |
| 1.8 | 0.4 | 0.6134 | 0.7866 | 0.7808 | 0.5308 |
| 1.5 | 0.6 | 0.6025 | 0.7803 | 0.7700 | 0.5201 |
| 0.9 | 0.8 | 0.6007 | 0.7796 | 0.7689 | 0.5186 |
| 1.8 | 0.6 | 0.5915 | 0.7733 | 0.7565 | 0.5089 |
| 1.2 | 0.8 | 0.5835 | 0.7676 | 0.7486 | 0.5008 |
| 1.5 | 0.8 | 0.5661 | 0.7575 | 0.7285 | 0.4837 |
| 1.8 | 0.8 | 0.5491 | 0.7471 | 0.7091 | 0.4673 |

Selection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.
Winner: k1=0.9, b=0.4. Full grid also saved to `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/hotpotqa/bm25_grid_search.csv`.

## Part 2 — BM25 vs TF-IDF (final comparison)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (default k1=1.2 b=0.75) | 0.5915 | 0.7730 | 0.7576 | 0.5093 |
| BM25 (tuned k1=0.9 b=0.4) | 0.6330 | 0.7957 | 0.8032 | 0.5502 |
| TF-IDF | 0.4378 | 0.6849 | 0.5668 | 0.3598 |

k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 (see grid table above). TF-IDF used Lucene's ClassicSimilarity.

## Part 3 — Vocabulary Mismatch


n_success=9746 avg_jaccard=0.1335 | n_failure=5064 avg_jaccard=0.0885
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/hotpotqa/hotpotqa_jaccard_overlap_distribution.png`
Failure cases saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/hotpotqa/part3_failures.json`

## Part 4a — Rocchio & RM3 (hotpotqa)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (Default k1=1.2, b=0.75) | 0.5915 | 0.7730 | 0.7576 | 0.5093 |
| BM25 (Tuned k1=0.9, b=0.4) | 0.6330 | 0.7957 | 0.8032 | 0.5502 |
| Rocchio (N=10, k=10) | 0.6000 | 0.7654 | 0.7737 | 0.5154 |
| RM3 (N=10, k=10) | 0.5063 | 0.7022 | 0.6561 | 0.4214 |
| Rocchio (N=10, k=20) | 0.6262 | 0.7908 | 0.7963 | 0.5427 |
| RM3 (N=10, k=20) | 0.5301 | 0.7311 | 0.6805 | 0.4434 |
| Rocchio (N=20, k=10) | 0.5988 | 0.7667 | 0.7726 | 0.5146 |
| RM3 (N=20, k=10) | 0.4709 | 0.6804 | 0.6212 | 0.3910 |
| Rocchio (N=20, k=20) | 0.6258 | 0.7910 | 0.7962 | 0.5426 |
| RM3 (N=20, k=20) | 0.5001 | 0.7117 | 0.6525 | 0.4184 |

Query Drift Candidates (pick at least 2 concrete examples for report):
- QID 5a7bbb64554299042af8f7cc: 'Who is older, Annie Morton or Terry Richardson?' | Added: ['richardson', 'anni']
- QID 5abd259d55429924427fcf1a: 'Are both Dictyosperma, and Huernia described as a genus?' | Added: ['genu']
- QID 5ac23ff0554299636651994d: 'When was Poison's album "Shut Up, Make Love" released?' | Added: ['up']
- QID 5ae0361155429925eb1afc2c: 'Which  French ace pilot and adventurer fly L'Oiseau Blanc' | Added: ['ac']
- QID 5a8e068b5542995085b37384: 'Are Ferocactus and Silene both types of plant?' | Added: ['silen', 'plant']
- QID 5ae73acb5542991e8301cc07: 'D1NZ is a series based on what oversteering technique?' | Added: ['overst']
- QID 5a7320565542991f9a20c61d: 'who is younger Keith Bostic or Jerry Glanville ?' | Added: ['footbal']
- QID 5adc53f75542996e6852530a: 'Are both Cypress and Ajuga genera?' | Added: ['genera', 'famili', 'bugl', 'plant']
- QID 5a7be2595542997c3ec972ac: 'Who was born earlier, Emma Bull or Virginia Woolf?' | Added: ['woolf']
- QID 5a7759fc5542993569682d60: 'Where are Teide National Park and Garajonay National Park located?' | Added: ['locat', 'nation']


## Part 4b — HyDE vs Corpus PRF Comparison

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| 1. Naive Concatenation (Query + HyDE) | 0.5717 | 0.7853 | 0.6926 | 0.4930 |
| 2. HyDE + Rocchio (N=10, k=20) | 0.5583 | 0.7441 | 0.7129 | 0.4771 |
| 3. Part 4a Corpus PRF (N=10, k=20) | 0.6262 | 0.7908 | 0.7963 | 0.5427 |

Comparison across Naive Concatenation, Rocchio-Weighted HyDE, and Corpus PRF.
