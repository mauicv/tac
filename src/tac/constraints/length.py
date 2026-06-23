from tac.core.state import State
import torch


class Length:
    def __init__(self, state: State, i: int, j: int, alpha: float):
        self.state = state
        self.i = i
        self.j = j
        self.alpha = alpha
        dist = self.get_length()    # shape, (b, 1)
        self.L = dist[[0], :]     # shape, (1, 1)

    def get_length(self):
        x = self.state.x_current[:,self.i] 
        y = self.state.x_current[:,self.j]
        d = x - y
        dist = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        return dist # shape, (b, 1)

    def apply(self, lam, dt):
        assert lam.shape == (self.state.batch_size, 1), "Lambda must be a tensor of shape (b, 1)"
        x,y = self.state.x_current[:, self.i], self.state.x_current[:, self.j]
        delta = x - y                                                                   # shape, (b, 3)
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)                    # shape, (b, 1)
        n = delta / dist.clamp(min=1e-6)                                                # shape, (b, 3)
        a_tilde = self.alpha / (dt * dt)
        C = dist - self.L                                                               # shape, (b, 1)
        wa, wb = self.state.inv_mass[:, [self.i]], self.state.inv_mass[:, [self.j]]     # shape, (b, 1)
        denum = wa + wb + a_tilde                                                       # shape, (b, 1)
        dlam = - (C + a_tilde * lam) / denum                                            # shape, (b, 1)
        lam = lam + dlam                                                                # shape, (b, 1)
        x_update = wa * dlam * n
        y_update = - wb * dlam * n
        self.state.x_current[:, self.i] += x_update                                     # shape, (b, 3)
        self.state.x_current[:, self.j] += y_update                                     # shape, (b, 3)
        return lam
