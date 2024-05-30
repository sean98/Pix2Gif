import torch.nn as nn


class DimensionUpscale(nn.Module):
    def __init__(self, dim_size, dim=2):
        """
        :param dim_size: the new dimension size
        :param dim: where the new dimension we want to upscale is in the current shape
        """
        super(DimensionUpscale, self).__init__()
        self.dim_size = dim_size
        self.dim = dim

    def forward(self, x):
        # Upscale by adding a new dimension
        return x.unsqueeze(dim=self.dim).expand(*x.shape[:self.dim], self.dim_size, *x.shape[self.dim:])
