# from: https://github.com/nerfstudio-project/nerfstudio.git

import torch
from torch import nn
import numpy as np
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
import tinycudann as tcnn

try:
    from typing import Literal, Optional
except ImportError:
    from typing_extensions import Literal, Optional


class HashEncoding(nn.Module):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
        dtype = None
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_levels = num_levels
        self.min_res = min_res
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.scalings = torch.floor(min_res * self.growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size

        self.tcnn_encoding = None
        self.hash_table = torch.empty(0)
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn":
            encoding_config = self.get_tcnn_encoding_config(
                num_levels=self.num_levels,
                features_per_level=self.features_per_level,
                log2_hashmap_size=self.log2_hashmap_size,
                min_res=self.min_res,
                growth_factor=self.growth_factor,
                interpolation=interpolation,
            )
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=in_dim,
                encoding_config=encoding_config,
                dtype=dtype
            )

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the hash encoding."""
        self.hash_table = torch.rand(size=(self.hash_table_size * self.num_levels, self.features_per_level)) * 2 - 1
        self.hash_table *= self.hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)


    @classmethod
    def get_tcnn_encoding_config(
        cls, num_levels, features_per_level, log2_hashmap_size, min_res, growth_factor, interpolation=None
    ) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": min_res,
            "per_level_scale": growth_factor,
        }
        if interpolation is not None:
            encoding_config["interpolation"] = interpolation
        return encoding_config


    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level


    def hash_fn(self, in_tensor: Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x


    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]


    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)