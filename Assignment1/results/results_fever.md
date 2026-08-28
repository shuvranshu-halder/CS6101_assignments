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

## Part 4a — Rocchio & RM3 (fever)

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| BM25 (Default k1=1.2, b=0.75) | 0.5214 | 0.8708 | 0.4812 | 0.4669 |
| BM25 (Tuned k1=0.9, b=0.3) | 0.6716 | 0.9226 | 0.6459 | 0.6196 |
| Rocchio (N=10, k=10) | 0.6662 | 0.9190 | 0.6415 | 0.6153 |
| RM3 (N=10, k=10) | 0.6182 | 0.9025 | 0.5841 | 0.5592 |
| Rocchio (N=10, k=20) | 0.6681 | 0.9203 | 0.6432 | 0.6171 |
| RM3 (N=10, k=20) | 0.6330 | 0.9084 | 0.5999 | 0.5743 |
| Rocchio (N=20, k=10) | 0.6662 | 0.9188 | 0.6414 | 0.6151 |
| RM3 (N=20, k=10) | 0.6134 | 0.8972 | 0.5840 | 0.5590 |
| Rocchio (N=20, k=20) | 0.6682 | 0.9204 | 0.6431 | 0.6171 |
| RM3 (N=20, k=20) | 0.6305 | 0.9045 | 0.6019 | 0.5762 |

Query Drift Candidates (pick at least 2 concrete examples for report):
- QID 70041: '2 Hearts is a musical composition by Minogue.' | Added: ['song', 'minogu']
- QID 202314: 'The New Jersey Turnpike has zero shoulders.' | Added: ['rout', 'jersei', 'turnpik']
- QID 6032: 'Aruba is the only ABC Island.' | Added: ['antil', 'curaçao', 'island', 'caribbean']
- QID 130048: 'Burbank, California has always been completely void of industry.' | Added: ['burbank']
- QID 204575: 'Commodore is ranked above a rear admiral.' | Added: ['rank', 'admir', 'navi']
- QID 164883: 'Hezbollah received a type of training from Iran.' | Added: ['iran', 'al']
- QID 219675: 'Corsica belongs to Italy.' | Added: ['micropterix', 'franc', 'itali', 'island', 'italian', 'sardinia']
- QID 134850: 'Ice-T refused to ever make hip-hop music.' | Added: ['hop', 'music', 'hip']
- QID 124578: 'The Gettysburg Address is a speech.' | Added: ['abraham', 'nation', 'lincoln', 'speech']
- QID 134126: 'Jason Bourne removed Riz Ahmed from the movie's cast.' | Added: ['bourn']


## Part 4b — HyDE vs Corpus PRF Comparison

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| 1. Naive Concatenation (Query + HyDE) | 0.7375 | 0.9318 | 0.7241 | 0.6889 |
| 2. HyDE + Rocchio (N=20, k=20) | 0.6658 | 0.9126 | 0.6426 | 0.6126 |
| 3. Part 4a Corpus PRF (N=20, k=20) | 0.6682 | 0.9204 | 0.6431 | 0.6171 |

Comparison across Naive Concatenation, Rocchio-Weighted HyDE, and Corpus PRF.

## Part 5 — SPLADE

| Method | nDCG@10 | Recall@100 | MRR@10 | MAP |
|---|---|---|---|---|
| SPLADE (naver/splade-v3) | 0.7565 | 0.9525 | 0.7478 | 0.7063 |

## Part 5 — Expansion Term Comparison (SPLADE vs Rocchio/RM3 vs HyDE)


Per-query term lists: `/mnt/nas/shuvranshu/CS6101_assignments/Assignment1/runs/fever/part5_expansion_term_comparison.json` | avg 3-way overlap: 0.60 terms/query

| Query | SPLADE terms | HyDE terms | Rocchio terms | Overlap |
|---|---|---|---|---|
| Ukrainian Soviet Socialist Republic was  | communist, country, founded, founder, member | given, international, member, nations, republics | republ, ssr, ukrain, union |  |
| 2 Hearts is a musical composition by Min | artist, composed, composer, drums, feet | australian, her, kylie, language, large | album, her, kyli, minogu, music | song |
| The New Jersey Turnpike has zero shoulde | bridge, height, highway, interstate, lane | absence, along, design, high-speed, jersey's | 95, bridg, counti, i, jersei |  |
| Aruba is the only ABC Island. | 11, ab, aba, ac, ai | ar, bonaire, caribbean, geographical, historical | antil, bonair, caribbean, curaçao, dutch | island |
| Burbank, California has always been comp | bu, ca, city, devoid, economic | burbank, city, development, entertainment, history | art, burbank, high, luther, school |  |
| Commodore is ranked above a rear admiral | admiral, before, below, captain, category | admiral, authority, below, hierarchical, hierarchy | abov, admir, commodor, navi, offic | rank |
| Hezbollah received a type of training fr | army, cia, iran, iranian, islam | advanced, capabilities, enhancing, hezbollah's, includes | al, forc, group, iran, isra | iran |
| In states still employing the electric c | alternatives, chairs, electrical, executed, execution | availability, capital, execution, humane, method |  |  |
| Corsica belongs to Italy. | belong, country, france, geography, island | belong, france, french, geographical, island | corsican, di, ferri, franc, island | island |
| Ice-T refused to ever make hip-hop music | artist, failed, he, hip, hop | after, albums, career, genre, he | artist, hip, hop, ic, music | music |

TODO: discuss 2-3 disagreement cases where the three sources pick different expansion terms and why (e.g. SPLADE finding semantically related but lexically distant terms vs. Rocchio/RM3's corpus-cooccurrence terms vs. HyDE's LLM-hallucinated-but-plausible terms).
