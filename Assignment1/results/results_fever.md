# Results — fever

Corpus size: 5416568 | Queries: 6666

## Part 1 — Index Build


Build time: 768.9s | Index size: 4492.3 MB | Corpus docs: 5416568 | Queries: 6666

### Part 2 — BM25 k1/b Grid Search (full results)

| k1 | b | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|---|
| 0.9 | 0.3 | 0.6716 | 0.9226 | 0.6459 | 0.6196 | **<- winner**
| 0.6 | 0.3 | 0.6693 | 0.9177 | 0.6439 | 0.6187 |
| 1.2 | 0.3 | 0.6681 | 0.9249 | 0.6412 | 0.6158 |
| 1.5 | 0.3 | 0.6598 | 0.9253 | 0.6313 | 0.6060 |
| 0.6 | 0.4 | 0.6531 | 0.9149 | 0.6254 | 0.6019 |
| 0.9 | 0.4 | 0.6513 | 0.9185 | 0.6230 | 0.5991 |
| 1.8 | 0.3 | 0.6497 | 0.9248 | 0.6185 | 0.5935 |
| 1.2 | 0.4 | 0.6433 | 0.9191 | 0.6134 | 0.5902 |
| 1.5 | 0.4 | 0.6340 | 0.9158 | 0.6025 | 0.5791 |
| 1.8 | 0.4 | 0.6220 | 0.9137 | 0.5882 | 0.5657 |
| 0.6 | 0.6 | 0.6106 | 0.9021 | 0.5795 | 0.5603 |
| 0.9 | 0.6 | 0.5957 | 0.8996 | 0.5617 | 0.5428 |
| 1.2 | 0.6 | 0.5800 | 0.8923 | 0.5432 | 0.5247 |
| 1.5 | 0.6 | 0.5635 | 0.8876 | 0.5249 | 0.5072 |
| 0.6 | 0.8 | 0.5541 | 0.8805 | 0.5177 | 0.5028 |
| 1.8 | 0.6 | 0.5485 | 0.8814 | 0.5085 | 0.4917 |
| 0.9 | 0.8 | 0.5264 | 0.8698 | 0.4869 | 0.4733 |
| 1.2 | 0.8 | 0.5005 | 0.8589 | 0.4589 | 0.4468 |
| 1.5 | 0.8 | 0.4754 | 0.8493 | 0.4337 | 0.4233 |
| 1.8 | 0.8 | 0.4529 | 0.8408 | 0.4110 | 0.4015 |

Selection rule: highest nDCG@10 (primary BEIR metric); ties broken by Recall@100.
Winner: k1=0.9, b=0.3. Full grid also saved to `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/fever/bm25_grid_search.csv`.

## Part 2 — BM25 vs TF-IDF (final comparison)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (default k1=1.2 b=0.75) | 0.5214 | 0.8708 | 0.4812 | 0.4669 |
| BM25 (tuned k1=0.9 b=0.3) | 0.6716 | 0.9226 | 0.6459 | 0.6196 |
| TF-IDF | 0.2075 | 0.6637 | 0.1727 | 0.1789 |

k1/b grid shared across datasets; winner chosen per-dataset by highest nDCG@10 (see grid table above). TF-IDF used Lucene's ClassicSimilarity.

## Part 3 — Vocabulary Mismatch


n_success=5988 avg_jaccard=0.0554 | n_failure=1949 avg_jaccard=0.0350
Plot saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/fever/fever_jaccard_overlap_distribution.png`
Failure cases saved to: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/fever/part3_failures.json`
