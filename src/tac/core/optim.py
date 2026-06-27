from tac.constraints.base import Constraint
from tac.constraints.length import Length, VectorizedLength
from tac.constraints.joint import Joint, VectorizedJoint
from tac.core.state import State
from typing import List, Literal
import torch


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
    joints = [constraint for constraint in constraints if isinstance(constraint, Joint)]
    joint_alphas = [constraint.alpha for constraint in joints]
    if len(joint_alphas) > 0:
        vectorized_joints = [VectorizedJoint.from_joints(state, joints, joint_alphas[0])]
        optimized_constraints.extend(vectorized_joints)
    lengths = [constraint for constraint in constraints if isinstance(constraint, Length)]
    length_alphas = [constraint.alpha for constraint in lengths]
    if len(length_alphas) > 0:
        vectorized_lengths = [VectorizedLength.from_lengths(state, lengths, length_alphas[0])]
        optimized_constraints.extend(vectorized_lengths)

    remaining_constraints = [constraint for constraint in constraints if not isinstance(constraint, Joint) and not isinstance(constraint, Length)]
    return [*optimized_constraints, *remaining_constraints]

def _optimize_constraints_gauss_seidel(state: State, constraints: List[Constraint]):
    pass
