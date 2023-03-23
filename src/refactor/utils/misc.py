import argparse
import importlib
import math
import os
import random
from typing import Any, Dict, Generator

import numpy as np
import torch


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--logdir", type=str, default="logs", help="where to save logs and ckpts")
    parser.add_argument("--name", type=str, default="dummy", help="postfix for logdir")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume training from given folder or checkpoint",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """ "
    Seed everything.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def extract_data_from_batch():
    return None


def cycle(dl):
    while True:
        yield from dl


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def one_hot_encode(seq, nucleotides, max_seq_len: int) -> np.ndarray:
    """
    One-hot encode a sequence of nucleotides.
    """
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(nucleotides)))
    for i in range(seq_len):
        seq_array[i, nucleotides.index(seq[i])] = 1
    return seq_array


def log(t: torch.Tensor, eps=1e-20) -> torch.Tensor:
    """
    Toch log for the purporses of diffusion time steps t.
    """
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def instantiate_from_config(config, **kwargs):
    if not "_target_" in config:
        raise KeyError("Expected key `_target_` to instantiate.")
    return get_obj_from_str(config["_target_"])(**config.get("params", {}), **kwargs)


def get_obj_from_str(string, reload=False):
    module, class_ = string.rsplit(".", 1)
    if reload:
        module_to_reload = importlib.import_module(module)
        importlib.reload(module_to_reload)
    return getattr(importlib.import_module(module, package=None), class_)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    From Perception Prioritized Training of Diffusion Models: https://arxiv.org/abs/2204.00227.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    from
    https://github.com/Erlemar/pytorch_tempest/blob/3d593b91fc025a2d0bea2342478f811961acf79a/src/utils/technical_utils.py#L11
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
