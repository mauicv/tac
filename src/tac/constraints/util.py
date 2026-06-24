import torch

_EPS = 1e-6

def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(_EPS)


def _perp(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Component of v perpendicular to unit axis n."""
    return v - (v * n).sum(-1, keepdim=True) * n
