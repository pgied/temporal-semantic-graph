# temporal-semantic-graph

Code accompanying the paper **_<Paper title>_** (<authors>, <venue> <year>). <!-- TODO: fill in -->

The pipeline turns a collection of time-stamped documents into a graph that shows how
topics develop over time. It has three stages, each a separate command:

```
documents ──encode──▶ embeddings ──map──▶ temporal semantic graph ──visualize──▶ figure
```

1. **encode**: embed every document with a sentence-transformers model.
2. **map**: build the graph with Mapper, using time as the lens. The time axis is covered
   with overlapping windows and the documents in each window are clustered. Each cluster
   becomes a node, and clusters that share documents are connected.
3. **visualize**: draw the graph with time on the x-axis and semantic similarity on the y-axis.

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repository>
cd temporal-semantic-graph
uv sync                      # creates .venv and installs the locked dependencies
```

The first `encode` run downloads the embedding model (about 90 MB) from the Hugging Face Hub.

## Dataset

We use the legislator tweets released with:

> Alexander Hoyle, Rupak Sarkar, Pranav Goel, Philip Resnik. 2023.
> *Natural Language Decompositions of Implicit Content Enable Better Text Representations.*
> EMNLP 2023. [doi:10.18653/v1/2023.emnlp-main.815](https://doi.org/10.18653/v1/2023.emnlp-main.815) ·
> [code and data](https://github.com/ahoho/inferential-decompositions)

> [!IMPORTANT]
> **We do not have the right to redistribute this dataset, so it is not part of this repository.**
> Download it from the original source. The terms of its authors apply.

Source: <https://github.com/ahoho/inferential-decompositions/tree/main/data>

| File | Content |
|---|---|
| `sampled_tweets_senate_115-117.jsonl` | Tweets by US senators (115th–117th Congress), one JSON object per line: `id`, `datetime`, `tweet`, `party`, `bioguide`, `account_type`, `topic_idx`, … |
| `gpt3.5_tweets_to_gen_all.jsonl` | Line *i* holds the GPT-3.5 "inferential decompositions" of the tweet on line *i* of the file above (a JSON list of short statements). |

### 1. Download the files into `data/raw/`

**Browser:** open the link above, open each of the two files, click *Download raw file*
and save it in `data/raw/`.

**Command line** (in Windows PowerShell, type `curl.exe` instead of `curl`):

```bash
curl -L -o data/raw/sampled_tweets_senate_115-117.jsonl https://raw.githubusercontent.com/ahoho/inferential-decompositions/main/data/sampled_tweets_senate_115-117.jsonl
curl -L -o data/raw/gpt3.5_tweets_to_gen_all.jsonl https://raw.githubusercontent.com/ahoho/inferential-decompositions/main/data/gpt3.5_tweets_to_gen_all.jsonl
```

Each file has 39,284 lines. Git ignores the contents of `data/raw/`, so the data cannot be
committed by accident.

### 2. Point `.env` to the files

```bash
cp .env.example .env         # Windows PowerShell: Copy-Item .env.example .env
```

If you saved the files somewhere else, change the paths in `.env`. Absolute paths and paths
relative to the project root both work:

```dotenv
TWEETS_PATH=data/raw/sampled_tweets_senate_115-117.jsonl
DECOMPOSITIONS_PATH=data/raw/gpt3.5_tweets_to_gen_all.jsonl
```

`.env` also sets the embedding model (`EMBEDDING_MODEL`) and the output folder (`OUTPUT_DIR`).
If a dataset file cannot be found, `encode` stops and prints which variable to set.

## Usage

Run the stages in order. Each stage reads the previous stage's output from `outputs/`.

```bash
uv run python main.py encode         # add --limit 3000 for a quick test run
uv run python main.py map
uv run python main.py visualize
```

`uv run python main.py <stage> --help` lists the options of a stage, for example:

| Stage | Options |
|---|---|
| `encode` | `--source tweet\|decompositions` (embed the tweet, or average its decompositions), `--model`, `--limit`, `--batch-size` |
| `map` | `--intervals` (time windows), `--overlap` (between windows), `--clusters` (per window), `--seed` |
| `visualize` | `--color-by` (document field, default `party`), `--out` (`.png`, `.pdf` or `.svg`) |

Each stage can also be run as a module (`python -m src.encoder`, `src.mapper`, `src.visualize`).
In VS Code, the launch configurations *Encode Documents*, *Map to Graph* and *Visualize Graph*
do this.

| Output | Written by |
|---|---|
| `outputs/embeddings.npy`: one L2-normalised vector per document | `encode` |
| `outputs/documents.jsonl`: the documents in the same order (id, time, text, metadata) | `encode` |
| `outputs/graph.json`: the graph in networkx node-link format | `map` |
| `outputs/graph.png`: the figure | `visualize` |

## Project structure

```
temporal-semantic-graph/
├── main.py              # entry point: encode | map | visualize
├── src/
│   ├── config.py        # paths and settings, read from .env
│   ├── dataset.py       # loader for the dataset (not included)
│   ├── encoder.py       # stage 1: documents -> embeddings
│   ├── mapper.py        # stage 2: embeddings -> temporal semantic graph (Mapper)
│   └── visualize.py     # stage 3: graph -> figure
├── data/raw/            # put the dataset files here (git-ignored)
├── outputs/             # generated files (git-ignored)
├── .env.example         # template for .env
├── pyproject.toml
└── uv.lock
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{TODO,
  title  = {<Paper title>},
  author = {<Authors>},
  year   = {<Year>},
}
```

If you use the dataset, please also cite its authors:

```bibtex
@inproceedings{hoyle-etal-2023-natural,
  title     = {Natural Language Decompositions of Implicit Content Enable Better Text Representations},
  author    = {Hoyle, Alexander and Sarkar, Rupak and Goel, Pranav and Resnik, Philip},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year      = {2023},
  pages     = {13188--13214},
  doi       = {10.18653/v1/2023.emnlp-main.815},
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The license does not
cover the dataset.
