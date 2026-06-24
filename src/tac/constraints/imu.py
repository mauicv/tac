from tac.constraints.util import _normalize, _perp, _EPS
from tac.core.state import State
from typing import List


class PitchRollIMU:
    constraint_types = ["measured"]

    def __init__(
            self,
            state: State,
            up: List[float],
            imu_point: int,
            points: List[int],
            alpha: float,
        ):
        assert len(points) > 2, "Points must be at least 3"
        self.state = state
        self.imu_point = imu_point
        self.points = points
        self.up = up
        v = state.x_current[:, self.points].mean(dim=-1) # (B, N)
        n = _normalize(v)
        self.n = n
        self.alpha = alpha

    def apply(self, lam, dt, **kwargs):
        pitch = kwargs.get("pitch", None)
        roll = kwargs.get("roll", None)
        if pitch is None or roll is None: return lam
        w = self.state.inv_mass     # (B, N) inverse masses
        x = self.state.x_current    # (B, N, 3)

        # TBD: Implement the IMU constraint

        return lam