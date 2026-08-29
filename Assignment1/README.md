# Part 1 to 4

nohup bash scripts/run_1to4.sh dataset_name --stage part1 --part2 --part3 --part4a --part4b_generate --part4b_run > log_file.log 2>&1 &

## arguments for "part4b_generate":
| Argument | Meaning |
| :--- | :--- |
| `--hyde_samples` | Number of HyDE documents generated per query |
| `--hyde_batch_size` | Queries per GPU in your modified multi-GPU implementation |
| `--hyde_num_gpus` | Number of GPUs to use |
| `--hyde_model` | Qwen model used for HyDE generation |

# Part 5 run
nohup bash scripts/run_part5.sh dataset_name --stage encode retrieve compare > log_file.log 2>&1 &
## arguments
| Parameter | Used for | GPU? |
| :--- | :--- | :---: |
| `--batch_size` | Corpus SPLADE encoding(encode) | ✅ Yes |
| `--query_batch_size` | Query SPLADE encoding during retrieve(retrieve) | ✅ Yes |
| `--index_threads` | Lucene indexing/search threads | ❌ CPU |
