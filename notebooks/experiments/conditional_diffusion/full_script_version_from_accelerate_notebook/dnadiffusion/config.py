import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="None", config_path=".", config_name="config")
def main(conf: DictConfig) -> None:
    pass
