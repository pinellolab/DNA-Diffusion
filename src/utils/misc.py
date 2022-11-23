import math
import importlib

import torch

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


def cycle(dl):
    while True:
        for data in dl:
            yield data


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


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


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
