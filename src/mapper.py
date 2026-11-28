import hydra
from omegaconf import DictConfig

from src.mappers import mapper_graph

@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(full_cfg: DictConfig):
    mapper_graph.create_mapper_graph(full_cfg)
        

if __name__ == "__main__":
    main()