from tac.constraints.base import Constraint
from tac.constraints.length import Length, VectorizedLength
from tac.constraints.joint import Joint, VectorizedJoint
from typing import List, Literal
import torch


def _optimize_constraints(constraints: List[Constraint], optimization_type: Literal['Jacobi', 'Gauss-Seidel']):
    if optimization_type == 'Jacobi':
        return _optimize_constraints_jacobi(constraints)
    elif optimization_type == 'Gauss-Seidel':
        return _optimize_constraints_gauss_seidel(constraints)
    else:
        raise ValueError(f"Invalid optimization type: {optimization_type}")

def _optimize_constraints_jacobi(constraints: List[Constraint]):
    joints = [constraint for constraint in constraints if isinstance(constraint, Joint)]
    lengths = [constraint for constraint in constraints if isinstance(constraint, Length)]
    vectorized_joints = VectorizedJoint.from_joints(state, joints, alpha, dt)
    vectorized_lengths = VectorizedLength.from_lengths(state, lengths, alpha, dt)
    return vectorized_joints, vectorized_lengths

def _optimize_constraints_gauss_seidel(constraints: List[Constraint]):
    pass
