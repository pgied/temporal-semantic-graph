import json
import os

from omegaconf import DictConfig, OmegaConf
from tdamapper.core import MapperAlgorithm

from src.mappers import mapper_utils


def create_mapper_graph(full_cfg: DictConfig):   

    # id_column = OmegaConf.select(full_cfg.mappers, "id_column", default="id")
    # created_at_column = OmegaConf.select(full_cfg.mappers, "created_at_column", default="datetime")
    # user_id_column = OmegaConf.select(full_cfg.mappers, "user_id_column", default="user_id")
    # run_simple_graph = OmegaConf.select(full_cfg.mappers, "run_simple_graph", default=True)

    # encoding_text_column = OmegaConf.select(full_cfg.mappers, "encoding_text_column", default="decompositions")
    # message_text_column = OmegaConf.select(full_cfg.mappers, "message_text_column", default="tweet")

    output_path = OmegaConf.select(full_cfg.mapper, "output_path", default="")

    mapper_cfg = OmegaConf.select(full_cfg, "mapper", default=None)

    clusterer = mapper_utils.get_clusterer(mapper_cfg)    
    cover = mapper_utils.get_cover(mapper_cfg)

    X, texts, raw_times, ids, lens_attribute = mapper_utils.get_embeddings_from_dataset_path(mapper_cfg=mapper_cfg)
    y = mapper_utils.get_y(lens_attribute=lens_attribute)
    
    mg_path = os.path.join(output_path, 'mapper_graph.json')
    os.makedirs(output_path, exist_ok=True)
    
    mapper_graph = MapperAlgorithm(cover=cover, clustering=clusterer).fit_transform(X, y)

    # convert to object
    ograph = mapper_utils.convert_graph_for_storage(mapper_graph, ids)  

    with open(mg_path, 'w', encoding='utf-8') as f:
        json.dump(ograph, f, indent=4)