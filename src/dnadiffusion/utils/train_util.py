import copy
from typing import Dict, Tuple

import torch
from accelerate import Accelerator
from torch.optim import Adam
from tqdm import tqdm

from dnadiffusion.metrics.metrics import compare_motif_list, generate_similarity_using_train
from dnadiffusion.utils.sample_util import create_sample
from dnadiffusion.utils.utils import EMA


class TrainLoop:
    def __init__(
        self,
        data: Tuple[Dict[str, object], torch.utils.data.DataLoader],
        model: torch.nn.Module,
        accelerator: Accelerator,
        epochs: int = 10000,
        loss_show_epoch: int = 10,
        sample_epoch: int = 100,
        save_epoch: int = 500,
        model_name: str = "model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k",
        image_size: int = 200,
        num_sampling_to_compare_cells: int = 1000,
    ):
        self.encode_data, self.train_dl = data
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.accelerator = accelerator
        self.epochs = epochs
        self.loss_show_epoch = loss_show_epoch
        self.sample_epoch = sample_epoch
        self.save_epoch = save_epoch
        self.model_name = model_name
        self.image_size = image_size
        self.num_sampling_to_compare_cells = num_sampling_to_compare_cells

        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        # Metrics
        self.train_kl, self.test_kl, self.shuffle_kl = 1, 1, 1
        self.seq_similarity = 1

        # Prepare for training
        self.model, self.optimizer, self.train_dl = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dl
        )

    def train_loop(self):
        # Initialize wandb
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "dnadiffusion",
                init_kwargs={"wandb": {"notes": "testing wandb accelerate script"}},
            )

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            # Getting loss of current batch
            for _, batch in enumerate(self.train_dl):
                loss = self.train_step(batch)

            # Logging loss
            if epoch % self.loss_show_epoch == 0 and self.accelerator.is_main_process:
                self.log_step(loss, epoch)

            # Sampling
            if epoch % self.sample_epoch == 0 and self.accelerator.is_main_process:
                self.sample()

            # Saving model
            if epoch % self.save_epoch == 0:
                self.save_model(epoch)

    def train_step(self, batch):
        x, y = batch

        with self.accelerator.autocast():
            loss = self.model(x, y)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.wait_for_everyone()
        self.optimizer.step()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))

        self.accelerator.wait_for_everyone()
        return loss

    def log_step(self, loss, epoch):
        if self.accelerator.is_main_process:
            self.accelerator.log(
                {
                    "train": self.train_kl,
                    "test": self.test_kl,
                    "shuffle": self.shuffle_kl,
                    "loss": loss.item(),
                    "seq_similarity": self.seq_similarity,
                },
                step=epoch,
            )
            print(f" Epoch {epoch} Loss:", loss.item())

    def sample(self):
        self.model.eval()

        # Sample from the model
        print("saving")
        synt_df = create_sample(
            self.accelerator.unwrap_model(self.model),
            conditional_numeric_to_tag=self.encode_data["numeric_to_tag"],
            cell_types=self.encode_data["cell_types"],
            number_of_samples=int(self.num_sampling_to_compare_cells / 10),
        )
        self.seq_similarity = generate_similarity_using_train(self.encode_data["X_train"])
        self.train_kl = compare_motif_list(synt_df, self.encode_data["train_motifs"])
        self.test_kl = compare_motif_list(synt_df, self.encode_data["test_motifs"])
        self.shuffle_kl = compare_motif_list(
            synt_df, self.encode_data["shuffle_motifs"]
        )
        print("Similarity", self.seq_similarity, "Similarity")
        print("KL_TRAIN", self.train_kl, "KL")
        print("KL_TEST", self.test_kl, "KL")
        print("KL_SHUFFLE", self.shuffle_kl, "KL")

    def save_model(self, epoch):
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
        }
        torch.save(
            checkpoint_dict,
            f"dnadiffusion/checkpoints/epoch_{epoch}_{self.model_name}.pt",
        )
