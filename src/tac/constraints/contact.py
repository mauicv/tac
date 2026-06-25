from tac.core.state import State


class FloorContact:
    constraint_types = ["measured", "simulated"] 

    def __init__(self, state: State, point: int, floor_z: float, alpha: float):
        self.state = state
        self.point = point
        self.floor_z = floor_z
        self.alpha = alpha

    def to_dict(self):
        return {
            "class": "FloorContact",
            "params": {
                "point": self.point,
                "floor_z": self.floor_z,
                "alpha": self.alpha,
            },
        }

    def apply(self, lam, dt, **kwargs):
        contact = kwargs.get("contact", None)
        if contact is not None:
            raise NotImplementedError("Measured contact is not implemented")

        x = self.state.x_current[:, self.point]
        z = x[:, 2]
        if z < self.floor_z:
            z = self.floor_z
        self.state.x_current[:, self.point, 2] = z
        return lam