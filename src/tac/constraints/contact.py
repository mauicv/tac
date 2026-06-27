from tac.core.state import State
from tac.constraints.base import Constraint
import torch


class FloorContact(Constraint):
    constraint_types = ["measured", "simulated"] 

    def __init__(self, state: State, point: int, floor_z: float, alpha: float):
        super().__init__(state, alpha)
        self.point = point
        self.floor_z = floor_z

    def to_dict(self):
        return {
            "class": "FloorContact",
            "params": {
                "point": self.point,
                "floor_z": self.floor_z,
                "alpha": self.alpha,
            },
        }

    @property
    def shape(self):
        return (1, )

    def get_residual(self, lam: torch.Tensor):
        x = self.state.x_current[:, self.point]
        z = x[:, 2]
        if z > self.floor_z:
            return torch.zeros(self.state.batch_size, 1)
        C = z - self.floor_z
        return C

    def _apply(self, lam, dt, **kwargs):
        contact = kwargs.get("contact", None)
        if contact is not None:
            raise NotImplementedError("Measured contact is not implemented")

        x = self.state.x_current[:, self.point]
        z = x[:, 2]
        if z < self.floor_z:
            z = self.floor_z
            self.state.x_previous[:, self.point, 0:2] = self.state.x_current[:, self.point, 0:2]
            self.state.x_current[:, self.point, 2] = z
        return lam