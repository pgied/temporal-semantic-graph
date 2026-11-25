import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.encoders import decomposition_encoder


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(full_cfg: DictConfig):
    encoder_name = HydraConfig.get().runtime.choices["encoder"] 
    if encoder_name == 'decomposition':
        decomposition_encoder.encode_documents(full_cfg)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")
        

if __name__ == "__main__":
    main()
