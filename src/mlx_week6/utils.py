import torch


def is_bad(x):
    has_nan = torch.isnan(x).any()
    has_inf = torch.isinf(x).any()
    is_large = torch.abs(torch.max(x)) > 1e2

    return has_nan, has_inf, is_large
