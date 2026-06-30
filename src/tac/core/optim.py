from tac.constraints.base import Constraint
from tac.constraints.length import Length, VectorizedLength
from tac.constraints.joint import Joint, VectorizedJoint
from tac.core.state import State
from typing import List, Literal
import torch

optimizable_class_map ={
    'joint': (Joint, VectorizedJoint),
    'length': (Length, VectorizedLength),
}

def _filter_for(constraints: List[Constraint], type: Literal['joint', 'length']):
    from_class, to_class = optimizable_class_map[type]
    alphas = [constraint.alpha for constraint in constraints if isinstance(constraint, from_class)]
    constraints = [constraint for constraint in constraints if isinstance(constraint, from_class)]
    return to_class, constraints, alphas

def _optimize_constraints(state: State, constraints: List[Constraint], optimization_type: Literal['Jacobi', 'Gauss-Seidel']):
    if optimization_type == 'Jacobi':
        return _optimize_constraints_jacobi(state, constraints)
    elif optimization_type == 'Gauss-Seidel':
        return _optimize_constraints_gauss_seidel(state, constraints)
    else:
        raise ValueError(f"Invalid optimization type: {optimization_type}")

def _optimize_constraints_jacobi(state: State, constraints: List[Constraint]):
    # TODO: this method assumes all joints have the same alpha.
    optimized_constraints = []
    to_class, from_constraints, from_alphas = _filter_for(constraints, 'joint')
    optimized_constraints.extend(_jacobi_optimise(state, from_constraints, from_alphas, to_class))
    to_class, from_constraints, from_alphas = _filter_for(constraints, 'length')
    optimized_constraints.extend(_jacobi_optimise(state, from_constraints, from_alphas, to_class))
    remaining_constraints = [constraint for constraint in constraints if not isinstance(constraint, Joint) and not isinstance(constraint, Length)]
    return [*optimized_constraints, *remaining_constraints]

def _optimize_constraints_gauss_seidel(state: State, constraints: List[Constraint]):
    # TODO: this method assumes all joints have the same alpha.
    optimized_constraints = []
    to_class, from_constraints, from_alphas = _filter_for(constraints, 'length')
    optimized_constraints.extend(_gauss_seidel_optimise(state, from_constraints, from_alphas, to_class))
    to_class, from_constraints, from_alphas = _filter_for(constraints, 'joint')
    optimized_constraints.extend(_gauss_seidel_optimise(state, from_constraints, from_alphas, to_class))
    remaining_constraints = [constraint for constraint in constraints if not isinstance(constraint, Joint) and not isinstance(constraint, Length)]
    return [*optimized_constraints, *remaining_constraints]

def _jacobi_optimise(state: State, constraints: List[Constraint], alphas, cls):
    optimized_constraints = []
    if len(alphas) > 0:
        vectorized_from_constraints = [cls.from_constraints(state, constraints, alphas[0])]
        optimized_constraints.extend(vectorized_from_constraints)
    return optimized_constraints

def _gauss_seidel_optimise(state: State, constraints: List[Constraint], alphas, cls):
    optimized_constraints = []
    cover = ConstraintPointCover()
    if len(alphas) > 0:
        for constraint, alpha in zip(constraints, alphas):
            cover.add_constraint(constraint, alpha)

    assert len(constraints) == sum(len(cs['constraints']) for cs in cover.constraint_sets)
    for cs in cover.constraint_sets:
        if len(cs['alphas']) > 0:
            alpha = cs['alphas'][0]
            optimized_constraints.append(cls.from_constraints(state, cs['constraints'], alpha))
    cover.pprint()
    return optimized_constraints


class ConstraintPointCover:
    def __init__(self):
        self.constraint_sets = [{
            'constraints': [],
            'alphas': [],
            'points': set(),
        }]

    def add_constraint(self, constraint: Constraint, alpha: float):
        constraint_points = set(constraint._get_points())
        options = []
        for i, item in enumerate(self.constraint_sets):
            if not constraint_points.intersection(item['points']):
                options.append(i)
        if len(options) == 0:
            self.constraint_sets.append({
                'constraints': [constraint],
                'alphas': [alpha],
                'points': constraint_points,
            })
        else:
            ind = options[0]
            self.constraint_sets[ind]['constraints'].append(constraint)
            self.constraint_sets[ind]['alphas'].append(alpha)
            self.constraint_sets[ind]['points'].update(constraint_points)

    def pprint(self):
        print(f"Constraint Point Cover: {self.constraint_sets[0]["constraints"][0].__class__.__name__}")
        for cs in self.constraint_sets:
            print(f"Constraints: {len(cs['constraints'])}")
            print(f"Points: {cs['points']}")
            print("-" * 10)