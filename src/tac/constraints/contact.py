from tac.core.state import State


class FloorContact:
    def __init__(self, state: State, point: int, floor_z: float, alpha: float):
        self.state = state
        self.point = point
        self.floor_z = floor_z
        self.alpha = alpha

    def apply(self, lam, dt):
        x = self.state.x_current[:, self.point]
        z = x[:, 2]
        if z < self.floor_z:
            z = self.floor_z
        self.state.x_current[:, self.point, 2] = z
        return lam