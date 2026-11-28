from datetime import datetime
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pyarrow.dataset as ds
from sklearn.cluster import HDBSCAN
from tdamapper.cover import CubicalCover


def get_clusterer(mapper_cfg: DictConfig):    
    if mapper_cfg.clusterer.to_use == 'hdbscan':
        clust = HDBSCAN(
            min_samples=mapper_cfg.clusterer.hdbscan.min_samples,
            min_cluster_size=mapper_cfg.clusterer.hdbscan.min_cluster_size,
            n_jobs=mapper_cfg.clusterer.hdbscan.n_jobs,
            metric=mapper_cfg.clusterer.hdbscan.metric,
            cluster_selection_method="eom"
        )

        if "cluster_selection_epsilon" in mapper_cfg.clusterer.hdbscan:
            clust.cluster_selection_epsilon=mapper_cfg.clusterer.hdbscan.cluster_selection_epsilon

    else:
        raise ValueError('Unsupported clusterer type')

    min_required = max(
        mapper_cfg.clusterer.hdbscan.min_samples or 1,
        mapper_cfg.clusterer.hdbscan.min_cluster_size or 1,
    )
    return _SafeClusterer(clust, min_samples_required=min_required)


def get_cover(mapper_cfg):
    if mapper_cfg.cover.to_use == 'proximity_cubical_cover':
        cover = CubicalCover(
            n_intervals=mapper_cfg.cover.proximity_cubical_cover.n_intervals,
            overlap_frac=mapper_cfg.cover.proximity_cubical_cover.overlap_frac
        )
    else:
        raise ValueError('Unsupported cover type')

    return cover


def convert_graph_for_storage(graph, ids):
    nodes = {int(nid): [str(ids[i]) for i in graph.nodes[nid]['ids']] for nid in graph.nodes}
    edges = [(int(edge[0]), int(edge[1])) for edge in graph.edges]

    ograph = {
        'nodes': nodes,
        'edges': edges
    }
    return ograph


def get_embeddings_from_dataset_path(mapper_cfg: DictConfig):
    lens_attribute_name = OmegaConf.select(mapper_cfg, "lens.attribute.name", default="created_at")
    lens_attribute_type = OmegaConf.select(mapper_cfg, "lens.attribute.type", default="datetime")
    
    id_column = OmegaConf.select(mapper_cfg, "id_column", default="id")
    # encoding_text_column = OmegaConf.select(mapper_cfg, "encoding_text_column", default="decomposition")
    message_text_column = OmegaConf.select(mapper_cfg, "message_text_column", default="tweet")
    
    input_path = OmegaConf.select(mapper_cfg, "input_path", default="")
    # data_loader = OmegaConf.select(mapper_cfg, "data_loader", default="")

    cols = [
        id_column,
        "embedding",
        message_text_column,
    ]
    
    ids = []
    texts = []
    embeddings = []
    lens_attribute = None
    raw_times = None
    lens_attribute = []
    raw_times = []
    cols.append(lens_attribute_name)

    dataset = ds.dataset(input_path, format='parquet', partitioning='hive')
    
    scanner = dataset.scanner(
        columns=cols,
        batch_size=10000
    )

    # Process each batch (record batch) in a streaming manner.
    for record_batch in scanner.to_batches():
        # Convert the record batch to a table or a pandas DataFrame if needed.
        df_batch = record_batch.to_pandas()
        df_batch = df_batch[["embedding", id_column, message_text_column, lens_attribute_name]]

        ids.extend(df_batch[id_column])
        embeddings.extend(df_batch['embedding'])
        texts.extend(df_batch[message_text_column])

        if lens_attribute is not None:
            if lens_attribute_type == 'datetime':
                ts = []

                for x in df_batch[lens_attribute_name]:
                    # Case 1: pandas / python datetime
                    if isinstance(x, (pd.Timestamp, datetime)):
                        t = x

                    # Case 2: string → parse
                    elif isinstance(x, str):
                        t = pd.to_datetime(x, utc=True)

                    # Case 3: numeric timestamp already
                    elif isinstance(x, (int, float, np.integer, np.floating)):
                        ts.append(float(x))
                        continue

                    else:
                        raise TypeError(
                            f"Unsupported datetime type for lens: {type(x)}"
                        )

                    ts.append(t.timestamp())

                ts = np.asarray(ts, dtype=float)

                lens_attribute.extend(ts)
                raw_times.extend(ts)
            else:
                lens_attribute.extend(df_batch[lens_attribute_name])

    if lens_attribute is not None and lens_attribute_type == 'datetime':
        lens_attribute = np.array(lens_attribute)
        lens_attribute = (lens_attribute - lens_attribute.min()) / (lens_attribute.max() - lens_attribute.min())

    X = np.array(embeddings)
    print(X.shape)

    raw_times = np.array(raw_times, dtype=float)

    return X, texts, raw_times, ids, lens_attribute

    
def get_y(lens_attribute=None):
    attr = np.asarray(lens_attribute)

    # Case 1: string / object → try datetime
    if attr.dtype.kind in {"U", "S", "O"}:
        try:
            y = (
                pd.to_datetime(attr, utc=True)
                .astype("int64")      # nanoseconds
                .to_numpy()
                .reshape(-1, 1)
            ) / 1e9                  # seconds
        except Exception as e:
            raise ValueError(
                "Attribute lens is non-numeric and not parseable as datetime."
            ) from e

    # Case 2: already numeric
    else:
        y = attr.astype(float).reshape(-1, 1)
        
    return y


class _SafeClusterer:
    """Wraps a clusterer so bins with fewer samples than required return a single cluster.

    tdamapper calls clustering.fit(X).labels_, so we intercept fit() and expose labels_.
    """
    def __init__(self, clusterer, min_samples_required=2):
        self._clust = clusterer
        self._min = min_samples_required
        self.labels_ = None

    def fit(self, X):
        if len(X) < self._min:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self
        self._clust.fit(X)
        self.labels_ = self._clust.labels_
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def __getattr__(self, name):
        return getattr(self._clust, name)