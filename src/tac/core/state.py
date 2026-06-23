import torch


class State:
    def __init__(self, points: torch.Tensor, inv_mass: torch.Tensor = None, batch_size: int = 1):
        assert points.shape[1] == 3, "Points must be a tensor of shape (N, 3)"
        self.x_current = points.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        self.x_previous = points.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        if inv_mass is None:
            inv_mass = torch.ones(points.shape[0])
        else:
            assert inv_mass.shape == (points.shape[0],), "Inv mass must be a tensor of shape (N,)"
            inv_mass = inv_mass
        self.inv_mass = inv_mass.unsqueeze(0).repeat(batch_size, 1)
        self.batch_size = batch_size

