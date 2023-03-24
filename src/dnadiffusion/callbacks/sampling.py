import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from dnadiffusion.metrics.sampling_metrics import (
    compare_motif_list,
    generate_similarity_using_train,
    sampling_to_metric,
)


class Sample(pl.Callback):
    def __init__(
        self,
        data_module: pl.LightningDataModule,
        image_size: int,
        num_sampling_to_compare_cells: int,
    ) -> None:
        self.data_module = data_module
        self.image_size = image_size
        self.number_sampling_to_compare_cells = num_sampling_to_compare_cells

    def on_train_start(self, *args, **kwargs) -> None:
        self.X_train = self.data_module.X_train
        self.train_motifs = self.data_module.train_motifs
        self.test_motifs = self.data_module.test_motifs
        self.shuffle_motifs = self.data_module.shuffle_motifs
        self.cell_types = self.data_module.cell_types
        self.numeric_to_tag = self.data_module.numeric_to_tag

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, L_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % 15 == 0:
            L_module.eval()
            additional_variables = {
                "model": L_module.model,
                "timesteps": L_module.timesteps,
                "device": L_module.device,
                "betas": L_module.betas,
                "sqrt_one_minus_alphas_cumprod": L_module.sqrt_one_minus_alphas_cumprod,
                "sqrt_recip_alphas": L_module.sqrt_recip_alphas,
                "posterior_variance": L_module.posterior_variance,
                "image_size": self.image_size,
            }

            synt_df = sampling_to_metric(
                self.cell_types,
                self.numeric_to_tag,
                additional_variables,
                int(self.number_sampling_to_compare_cells / 10),
            )
            seq_similarity = generate_similarity_using_train(self.X_train)
            train_kl = compare_motif_list(synt_df, self.train_motifs)
            test_kl = compare_motif_list(synt_df, self.test_motifs)
            shuffle_kl = compare_motif_list(synt_df, self.shuffle_motifs)
            L_module.train()

            trainer.logger.log_metrics(
                {
                    "train_kl": train_kl,
                    "test_kl": test_kl,
                    "shuffle_kl": shuffle_kl,
                    "seq_similarity": seq_similarity,
                },
                step=trainer.global_step,
            )
