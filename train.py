from dnadiffusion.data.dataloader import LoadingData
from dnadiffusion.trainer import Trainer
from dnadiffusion.trainer_fabric import Trainer as TrainerFabric

if __name__ == "__main__":
    trainer = TrainerFabric()
    trainer.train()
