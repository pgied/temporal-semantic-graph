import hydra
from omegaconf import DictConfig

from src import encoders


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(full_cfg: DictConfig):
    print('implement')
        

if __name__ == "__main__":
    main()