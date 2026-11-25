
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from src.data.loader import load_decompositions_df
from src.utils import device_manager

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR_RAW = DATA_DIR / "raw"
TWEETS_FILE = "sampled_tweets_senate_115-117.jsonl"
DECOMPOSITIONS_FILE = "gpt3.5_tweets_to_gen_all.jsonl"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "decomposition"

def encode_documents(full_cfg: DictConfig):
    device = device_manager.get_device()
    
    model_id = OmegaConf.select(full_cfg.encoder, "model_id")
    batch_size = OmegaConf.select(full_cfg.encoder, "batch_size")
    text_column = OmegaConf.select(full_cfg.encoder, "text_column")
    created_at_column = OmegaConf.select(full_cfg.encoder, "created_at_column")
    partition_cols = OmegaConf.select(full_cfg.encoder, "partition_cols")

    # load data
    df = load_decompositions_df(DATA_DIR_RAW / TWEETS_FILE, DATA_DIR_RAW / DECOMPOSITIONS_FILE)
    df[partition_cols] = df[created_at_column].dt.strftime("%Y-%m-%d")
    
    # Compose SentenceTransformer
    model = SentenceTransformer(
        model_id,
        model_kwargs = {
            "torch_dtype": torch.bfloat16,"attn_implementation": "sdpa",  # We support flash_attention_2; sdpa; eager
            "trust_remote_code": True
        },
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )
    
    meta_data_columns = list(df.columns.values)

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df[i: i + batch_size]

        text_list = []
        for row in batch_df[text_column].tolist():
            concatenated_text = " ".join(item.strip() for item in row if item.strip())
            text_list.append(concatenated_text)
        
        # encode only the filtered narratives
        vecs = model.encode(
            text_list,
            normalize_embeddings=True,
            compression_ratio=0.3333,
            show_progress_bar=True,
            batch_size=batch_size,
            device=device,
        )

        _data = {}
        for meta_col in meta_data_columns:
            _data[meta_col] = [doc[meta_col] for _, doc in batch_df.iterrows()]

        _data['embedding'] = [vec.tolist() for vec in vecs]

        df_embedding_batch = pd.DataFrame(_data)

        table = pa.Table.from_pandas(df_embedding_batch)

        pq.write_to_dataset(table, root_path=EMBEDDINGS_DIR, partition_cols=partition_cols)

