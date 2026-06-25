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

    def add(self, points: torch.Tensor, inv_mass: torch.Tensor = None):
        assert points.shape[1] == 3, "Points must be a tensor of shape (N, 3)"
        assert inv_mass is None or inv_mass.shape == (points.shape[0],), "Inv mass must be a tensor of shape (N,)"
        points = points.repeat(self.batch_size, 1, 1)
        inv_mass = inv_mass.repeat(self.batch_size, 1)
        self.x_current = torch.cat([self.x_current, points], dim=1)
        self.x_previous = torch.cat([self.x_previous, points], dim=1)
        if inv_mass is None:
            inv_mass = torch.ones(points.shape[0])
        else:
            inv_mass = inv_mass.repeat(self.batch_size, 1)
        self.inv_mass = torch.cat([self.inv_mass, inv_mass], dim=1)

