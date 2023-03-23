import numpy as np
import pytest
import torch

import src.models.networks.unet_lucas as unet


def test_layernorm_zero_dim():
    dim = 0
    with pytest.raises(ValueError):
        unet.LayerNorm(dim)


def test_layernorm_forward_negative_values_float64():
    test_layer = unet.LayerNorm(2)
    all_negs = torch.tensor([[-2, -8, -1], [-1, -6, -3], [-3, -9, -7]], dtype=torch.float64)
    # row_wise_mu = np.array([-11/3,-10/3,-19/3])
    # row_wise_var = np.array([9.555555555555557,4.222222222222222, 6.222222222222222])
    result = torch.tensor(
        [
            [
                [
                    [0.5391356561685612, -1.4017527060382593, 0.8626170498696979],
                    [1.1354154987771312, -1.2976177128881499, 0.16220221411101882],
                    [1.3361988407547418, -1.0689590726037936, -0.26723976815094846],
                ],
                [
                    [0.5391356561685612, -1.4017527060382593, 0.8626170498696979],
                    [1.1354154987771312, -1.2976177128881499, 0.16220221411101882],
                    [1.3361988407547418, -1.0689590726037936, -0.26723976815094846],
                ],
            ]
        ],
        dtype=torch.float64,
    )
    assert torch.eq(test_layer.forward(all_negs), result).all()


def test_layernorm_forward_negative_values_float32():
    test_layer = unet.LayerNorm(2)
    all_negs = torch.tensor([[-2, -8, -1], [-1, -6, -3], [-3, -9, -7]], dtype=torch.float32)
    # row_wise_mu = np.array([-11/3,-10/3,-19/3])
    # row_wise_var = np.array([9.555555555555557,4.222222222222222, 6.222222222222222])
    result = torch.tensor(
        [
            [
                [
                    [0.53916365, -1.4018253, 0.8626618],
                    [1.1355485, -1.2977698, 0.16222118],
                    [1.3363053, -1.069044, -0.26726097],
                ],
                [
                    [0.53916365, -1.4018253, 0.8626618],
                    [1.1355485, -1.2977698, 0.16222118],
                    [1.3363053, -1.069044, -0.26726097],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.eq(test_layer.forward(all_negs), result).all()


def test_layernorm_forward_positive_values_float64():
    test_layer = unet.LayerNorm(2)
    all_pos = torch.tensor([[2, 8, 1], [1, 6, 3], [3, 9, 7]], dtype=torch.float64)
    # row_wise_mu = np.array([-11/3,-10/3,-19/3])
    # row_wise_var = np.array([9.555555555555557,4.222222222222222, 6.222222222222222])
    result = torch.tensor(
        [
            [
                [
                    [-0.5391356561685612, 1.4017527060382593, -0.8626170498696979],
                    [-1.1354154987771312, 1.2976177128881499, -0.16220221411101882],
                    [-1.3361988407547418, 1.0689590726037936, 0.26723976815094846],
                ],
                [
                    [-0.5391356561685612, 1.4017527060382593, -0.8626170498696979],
                    [-1.1354154987771312, 1.2976177128881499, -0.16220221411101882],
                    [-1.3361988407547418, 1.0689590726037936, 0.26723976815094846],
                ],
            ]
        ],
        dtype=torch.float64,
    )
    assert torch.eq(test_layer.forward(all_pos), result).all()


def test_layernorm_forward_positive_values_float32():
    test_layer = unet.LayerNorm(2)
    all_pos = torch.tensor([[2, 8, 1], [1, 6, 3], [3, 9, 7]], dtype=torch.float32)
    # row_wise_mu = np.array([11/3,10/3,19/3])
    # row_wise_var = np.array([9.555555555555557,4.222222222222222, 6.222222222222222])
    result = torch.tensor(
        [
            [
                [
                    [-0.53916365, 1.4018253, -0.8626618],
                    [-1.1355485, 1.2977698, -0.16222118],
                    [-1.3363053, 1.069044, 0.26726097],
                ],
                [
                    [-0.53916365, 1.4018253, -0.8626618],
                    [-1.1355485, 1.2977698, -0.16222118],
                    [-1.3363053, 1.069044, 0.26726097],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.eq(test_layer.forward(all_pos), result).all()


def test_layernorm_forward_float64():
    test_layer = unet.LayerNorm(2)
    mixed_values = torch.tensor([[-2, 8, -1], [1, -6, 3], [-3, 9, -7]], dtype=torch.float64)
    # row_wise_mu = np.array([5/3,-2/3,-1/3])
    # row_wise_var = np.array([20.222222222222225,14.888888888888891, 46.22222222222222])
    result = torch.tensor(
        [
            [
                [
                    [-0.8153540887225921, 1.4083388805208406, -0.5929847917982488],
                    [0.43191970826789955, -1.3821430664572785, 0.950223358189379],
                    [-0.39222802744805746, 1.3727980960682014, -0.9805700686201437],
                ],
                [
                    [-0.8153540887225921, 1.4083388805208406, -0.5929847917982488],
                    [0.43191970826789955, -1.3821430664572785, 0.950223358189379],
                    [-0.39222802744805746, 1.3727980960682014, -0.9805700686201437],
                ],
            ]
        ],
        dtype=torch.float64,
    )
    assert torch.eq(test_layer.forward(mixed_values), result).all()


def test_layernorm_forward__float32():
    test_layer = unet.LayerNorm(2)
    mixed_values = torch.tensor([[-2, 8, -1], [1, -6, 3], [-3, 9, -7]], dtype=torch.float32)
    # row_wise_mu = np.array([5/3,-2/3,-1/3])
    # row_wise_var = np.array([20.222222222222225,14.888888888888891, 46.22222222222222])
    result = torch.tensor(
        [
            [
                [
                    [-0.8153741, 1.4083735, -0.5929993],
                    [0.43193412, -1.3821892, 0.95025504],
                    [-0.39223224, 1.3728127, -0.9805805],
                ],
                [
                    [-0.8153741, 1.4083735, -0.5929993],
                    [0.43193412, -1.3821892, 0.95025504],
                    [-0.39223224, 1.3728127, -0.9805805],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.eq(test_layer.forward(mixed_values), result).all()


def test_layernorm_dimensions():
    return
