import copy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dnadiffusion.data.dataloader import LoadingData, SequenceDataset
from dnadiffusion.metrics.metrics import (
    compare_motif_list,
    generate_heatmap,
    generate_similarity_using_train,
    kl_comparison_generated_sequences,
)
from dnadiffusion.models.diffusion import p_losses
from dnadiffusion.models.networks import Unet_lucas
from dnadiffusion.sample import sampling_to_metric
from dnadiffusion.utils.ema import EMA
from dnadiffusion.utils.scheduler import linear_beta_schedule
from dnadiffusion.utils.utils import one_hot_encode


class Trainer:
    def __init__(
        self,
        data_path: str = "K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        limit_total_sequences: int = 0,
        model_name: str = "model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k",
        load_saved_data: bool = True,
        save_model_by_epoch: bool = False,
        save_and_sample_every: int = 100,
        epochs: int = 10000,
        epochs_loss_show: int = 10,
        num_sampling_to_compare_cells: int = 1000,
        timesteps: int = 50,
        batch_size: int = 240,
        channels: int = 1,
        image_size: int = 200,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.load_saved_model = load_saved_data
        self.save_model_by_epoch = save_model_by_epoch
        self.save_and_sample_every = save_and_sample_every
        self.epochs = epochs
        self.epochs_loss_show = epochs_loss_show
        self.num_sampling_to_compare_cells = num_sampling_to_compare_cells
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.channels = channels
        self.image_size = image_size

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], split_batches=True, log_with=["wandb"])
        self.device = self.accelerator.device

        if load_saved_data:
            encode_data = np.load("dnadiffusion/data/encode_data.npy", allow_pickle=True).item()
        else:
            encode_data = LoadingData(
                data_path,
                change_component_index=False,
                subset_components=[
                    "GM12878_ENCLB441ZZZ",
                    "hESCT0_ENCLB449ZZZ",
                    "K562_ENCLB843GMH",
                    "HepG2_ENCLB029COU",
                ],
                limit_total_sequences=limit_total_sequences,
                number_of_sequences_to_motif_creation=self.number_of_sampling_to_compare_cells,
            )

        # Splitting encode data into train/test/shuffle
        self.df_results_seq_guime_count_train = encode_data.train["motifs"]
        self.df_results_seq_guime_count_test = encode_data.test["motifs"]
        self.df_results_seq_guime_count_shuffle = encode_data.train_shuffle["motifs"]

        self.final_comp_values_train = encode_data.train["motifs_per_components_dict"]
        self.final_comp_values_test = encode_data.test["motifs_per_components_dict"]
        self.final_comp_values_shuffle = encode_data.train_shuffle["motifs_per_components_dict"]

        # Dataset used for sequences
        df = encode_data.train["dataset"]
        self.cell_components = df.sort_values("TAG")["TAG"].unique().tolist()
        self.conditional_tag_to_numeric = {x: n + 1 for n, x in enumerate(df.TAG.unique())}
        self.conditional_numeric_to_tag = {n + 1: x for n, x in enumerate(df.TAG.unique())}
        self.cell_types = sorted(self.conditional_numeric_to_tag.keys())
        self.x_train_cell_type = torch.from_numpy(
            df["TAG"].apply(lambda x: self.conditional_tag_to_numeric[x]).to_numpy()
        )

        # Creating X_train for sequence similarity
        dna_alphabet = ["A", "C", "T", "G"]
        x_train_seq = np.array([one_hot_encode(x, dna_alphabet, 200) for x in tqdm(df["sequence"]) if "N" not in x])
        X_train = x_train_seq
        X_train = np.array([x.T.tolist() for x in X_train])
        X_train[X_train == 0] = -1
        self.X_train = X_train

        # Sequence dataset loading
        tf = T.Compose([T.ToTensor()])
        seq_dataset = SequenceDataset(seqs=X_train, c=self.x_train_cell_type, transform=tf)
        train_dl = DataLoader(seq_dataset, batch_size, shuffle=True, num_workers=48, pin_memory=True)

        # Preparing model/optimizer/EMA/dataloader
        self.model = Unet_lucas(dim=200, channels=1, dim_mults=(1, 2, 4), resnet_block_groups=4)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.start_epoch = 0
        self.train_kl, self.test_kl, self.shuffle_kl = 1, 1, 1
        self.seq_similarity = 0.38
        self.model, self.optimizer, self.train_dl = self.accelerator.prepare(self.model, self.optimizer, train_dl)

    # Saving model
    def save(self, epoch, results_path):
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
            "train_kl": self.train_kl,
            "test_kl": self.test_kl,
            "shuffle_kl": self.shuffle_kl,
            "seq_similarity": self.seq_similarity,
        }
        torch.save(checkpoint_dict, results_path)

    def load(self, model_path, model_name):
        # Loading the checkpoint
        checkpoint_dict = torch.load(model_path + model_name)

        # Recreating variables
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.start_epoch = checkpoint_dict["epoch"]

        # Recreating EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])

        self.train_kl = checkpoint_dict["train_kl"]
        self.test_kl = checkpoint_dict["test_kl"]
        self.shuffle_kl = checkpoint_dict["shuffle_kl"]
        self.seq_similarity = checkpoint_dict["seq_similarity"]
        print("saving")

        # Continue training
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.train()

    def create_samples(self, model_path, model_name):
        if self.accelerator.is_main_process:
            # define beta schedule
            betas = linear_beta_schedule(timesteps=self.timesteps, beta_end=0.2)
            betas.to(self.device)
            # define alphas
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
            # calculations for diffusion q(x_t | x_{t-1}) and others
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

            # Recreating model
            checkpoint_dict = torch.load(model_path + model_name)
            self.model = self.accelerator.unwrap_model(self.model)
            self.model.load_state_dict(checkpoint_dict["model"])
            self.model.eval()

            # Creating variables
            additional_variables = {
                "model": self.model,
                "timesteps": self.timesteps,
                "betas": betas,
                "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
                "sqrt_recip_alphas": sqrt_recip_alphas,
                "posterior_variance": posterior_variance,
                "image_size": self.image_size,
                "accelerator": self.accelerator,
                "device": self.device,
            }

            # Creating samples
            cell_num_list = list(self.conditional_tag_to_numeric.values())
            heat_new_sequences_train = kl_comparison_generated_sequences(
                cell_num_list,
                self.final_comp_values_train,
                additional_variables=additional_variables,
                conditional_numeric_to_tag=self.conditional_numeric_to_tag,
                number_of_sequences_sample_per_cell=self.num_sampling_to_compare_cells,
            )
            generate_heatmap(heat_new_sequences_train, "DNADIFFUSION", "Train", self.cell_components)

            heat_new_sequences_test = kl_comparison_generated_sequences(
                cell_num_list,
                self.final_comp_values_test,
                additional_variables=additional_variables,
                conditional_numeric_to_tag=self.conditional_numeric_to_tag,
                number_of_sequences_sample_per_cell=self.num_sampling_to_compare_cells,
            )

            generate_heatmap(heat_new_sequences_test, "DNADIFFUSION", "Test", self.cell_components)

            heat_new_sequences_shuffle = kl_comparison_generated_sequences(
                cell_num_list,
                self.final_comp_values_shuffle,
                additional_variables=additional_variables,
                conditional_numeric_to_tag=self.conditional_numeric_to_tag,
                number_of_sequences_sample_per_cell=self.num_sampling_to_compare_cells,
            )
            generate_heatmap(
                heat_new_sequences_shuffle,
                "DNADIFFUSION",
                "Shuffle",
                self.cell_components,
            )

    def train(self):
        # define beta schedule
        betas = linear_beta_schedule(timesteps=self.timesteps, beta_end=0.2)
        betas.to(self.device)
        # define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        """if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "dnadiffusion",
                init_kwargs={"wandb": {"notes": "testing wandb accelerate script"}},
            )
        """

        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            self.model.train()

            for step, batch in enumerate(self.train_dl):
                x, y = batch
                x = x.type(torch.float32)
                y = y.type(torch.long)
                batch_size = x.shape[0]
                t = torch.randint(0, self.timesteps, (batch_size,)).long()

                with self.accelerator.autocast():
                    loss = p_losses(
                        self.model,
                        x,
                        t,
                        y,
                        loss_type="huber",
                        sqrt_alphas_cumprod_in=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod_in=sqrt_one_minus_alphas_cumprod,
                        device=self.device,
                    )

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.accelerator.wait_for_everyone()
                self.optimizer.step()

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))

            print(f"\nEpoch {epoch} Loss:", loss.item())
            if (epoch % self.epochs_loss_show) == 0:
                if self.accelerator.is_main_process:
                    """self.accelerator.log(
                        {
                            "train": self.train_kl,
                            "test": self.test_kl,
                            "shuffle": self.shuffle_kl,
                            "loss": loss.item(),
                            "seq_similarity": self.seq_similarity,
                        },
                        step=epoch,
                    #)
                    print(f" Epoch {epoch} Loss:", loss.item())
                    """
            if epoch != 0 and epoch % self.save_and_sample_every == 0 and self.accelerator.is_main_process:
                self.model.eval()
                additional_variables = {
                    "model": self.model,
                    "timesteps": self.timesteps,
                    "device": self.device,
                    "betas": betas,
                    "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
                    "sqrt_recip_alphas": sqrt_recip_alphas,
                    "posterior_variance": posterior_variance,
                    "accelerator": self.accelerator,
                    "image_size": self.image_size,
                }
                synt_df = sampling_to_metric(
                    self.cell_types,
                    self.conditional_numeric_to_tag,
                    additional_variables,
                    int(self.num_sampling_to_compare_cells / 10),
                )
                self.seq_similarity = generate_similarity_using_train(self.X_train)
                self.train_kl = compare_motif_list(synt_df, self.df_results_seq_guime_count_train)
                self.test_kl = compare_motif_list(synt_df, self.df_results_seq_guime_count_test)
                self.shuffle_kl = compare_motif_list(synt_df, self.df_results_seq_guime_count_shuffle)
                print("Similarity", self.seq_similarity, "Similarity")
                print("KL_TRAIN", self.train_kl, "KL")
                print("KL_TEST", self.test_kl, "KL")
                print("KL_SHUFFLE", self.shuffle_kl, "KL")

            if epoch != 0 and epoch % 500 == 0 and self.accelerator.is_main_process:
                model_path = f"./models/epoch_{str(epoch)}_{self.model_name}.pt"
                self.save(epoch, model_path)
