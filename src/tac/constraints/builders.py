from tac.core.state import State
from tac.constraints.autograd import AutogradConstraint
from typing import List, Literal, Callable
import jax.numpy as jnp


class ConstraintBuilder:
    def __init__(self, C: Callable):
        self.C = C
        self.constraint_sets = [{
            'points': [],
            'alphas': [],
            'covered_points': set(),
        }]

    def add_constraint(self,
            points: List[int],
            alpha: float
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
                'covered_points': constraint_points,
            })
        else:
            ind = options[0]
            self.constraint_sets[ind]['points'].append(points)
            self.constraint_sets[ind]['alphas'].append(alpha)
            self.constraint_sets[ind]['covered_points'].update(constraint_points)
        return self

    def build(self, state: State) -> List[AutogradConstraint]:
        constraints = []
        for cs in self.constraint_sets:
            constraint = AutogradConstraint.create(state, cs['points'], cs['alphas'], self.C)
            constraints.append(constraint)
        return constraints



class LengthBuilder(ConstraintBuilder):
    def __init__(self):

        def C(points: jnp.ndarray) -> jnp.ndarray:
            d = jnp.linalg.norm(points[..., 0, :] - points[..., 1, :], axis=-1)
            return d

        super().__init__(C)


class FloorContactBuilder(ConstraintBuilder):
    def __init__(self, floor_height: float):

        def C(points: jnp.ndarray) -> jnp.ndarray:
            return jnp.maximum(0.0, points[..., 2] - floor_height)

        super().__init__(C)

