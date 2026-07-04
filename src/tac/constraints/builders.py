from tac.core.state import State
from tac.constraints.autograd import AutogradConstraint
from typing import List, Literal, Callable
import jax.numpy as jnp


class ConstraintBuilder:
    def __init__(self, C: Callable, rest=None, name: str = None, angular_wrap: bool = False):
        self.C = C
        self.rest = rest
        self.name = name
        self.angular_wrap = angular_wrap
        self.constraint_sets = [{
            'points': [],
            'alphas': [],
            'covered_points': set(),
            'name': self.name,
            'controls': [],
        }]

    def add_constraint(self,
            points: List[int],
            alpha: float,
            control: int = -1,
        ):
        constraint_points = set(points)
        options = []
        for i, item in enumerate(self.constraint_sets):
            if not constraint_points.intersection(item['covered_points']):
                options.append(i)
        if len(options) == 0:
            self.constraint_sets.append({
                'points': [points],
                'alphas': [alpha],
                'controls': [control],
                'covered_points': constraint_points,
                'name': self.name,
            })
        else:
            ind = options[0]
            self.constraint_sets[ind]['points'].append(points)
            self.constraint_sets[ind]['alphas'].append(alpha)
            self.constraint_sets[ind]['controls'].append(control)
            self.constraint_sets[ind]['covered_points'].update(constraint_points)
        return self

    def build(self, state: State) -> List[AutogradConstraint]:
        constraints = []
        for cs in self.constraint_sets:
            constraint = AutogradConstraint.create(
                state,
                cs['points'],
                cs['alphas'],
                self.C,
                rest=self.rest,
                name=cs['name'],
                angular_wrap=self.angular_wrap,
                control=cs['controls']
            )
            constraints.append(constraint)
        return constraints



class LengthBuilder(ConstraintBuilder):
    def __init__(self):

        def C(points: jnp.ndarray) -> jnp.ndarray:
            return jnp.linalg.norm(points[..., 0, :] - points[..., 1, :], axis=-1)

        super().__init__(C, name='length')


class FloorContactBuilder(ConstraintBuilder):
    def __init__(self, floor_height: float):

        def C(points: jnp.ndarray) -> jnp.ndarray:
            return jnp.minimum(0.0, points[..., 0, 2] - floor_height)      # (B, n)

        super().__init__(C, rest=0.0, name='floor_contact')


class JointBuilder(ConstraintBuilder):
    # points: [axis0, axis1, armA, armB]; C is the signed hinge angle about the
    # axis. target=None holds the build-time angle (rigid hinge); pass a target
    # angle (radians) to actuate the DOF to a commanded position.
    def __init__(self, target: float = None):

        def C(points: jnp.ndarray) -> jnp.ndarray:
            p0 = points[..., 0, :]                                          # axis origin
            p1 = points[..., 1, :]                                          # axis end
            a = points[..., 2, :]                                           # arm on body A
            b = points[..., 3, :]                                           # arm on body B
            axis = p1 - p0
            axis = axis / (jnp.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8)
            ra = a - p0
            rb = b - p0
            ra = ra - (ra * axis).sum(-1, keepdims=True) * axis             # perp to axis
            rb = rb - (rb * axis).sum(-1, keepdims=True) * axis
            x = (ra * rb).sum(-1)                                           # ~cos(theta)
            y = (jnp.cross(ra, rb) * axis).sum(-1)                          # ~sin(theta), signed
            return jnp.arctan2(y, x)                                        # (B, n)

        super().__init__(C, rest=target, name='joint', angular_wrap=True)


class PlanarBuilder(ConstraintBuilder):
    # Holds a group of points coplanar. C == 0 when coplanar; rest defaults to
    # the build-time value (~0 for a face built flat).
    def __init__(self):

        def C(points: jnp.ndarray) -> jnp.ndarray:                          # (B, n, k, 3) -> (B, n)
            k = points.shape[-2]
            if k == 4:
                a = points[..., 0, :]
                b = points[..., 1, :]
                c = points[..., 2, :]
                d = points[..., 3, :]
                # signed volume of the tetra (scalar triple product); 0 iff coplanar
                return ((b - a) * jnp.cross(c - a, d - a)).sum(-1)          # (B, n)
            # k != 4: smallest eigenvalue of the point covariance = variance out
            # of the best-fit plane; 0 iff coplanar.
            centred = points - points.mean(-2, keepdims=True)              # (B, n, k, 3)
            cov = jnp.einsum("...ki,...kj->...ij", centred, centred)       # (B, n, 3, 3)
            return jnp.linalg.eigvalsh(cov)[..., 0]                         # (B, n)

        super().__init__(C, name='planar')
