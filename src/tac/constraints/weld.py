from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.base import Constraint
from typing import List


class SimpleWeld(Constraint):
    def __init__(self, state: State, points: List[int], alpha: float, dt: float):
        super().__init__(state, alpha, dt)
        self.points = points
        self.lengths = []

        for i in range(len(points)):
            for j in range(i+1, len(points)):
                self.lengths.append(Length(state, points[i], points[j], alpha, dt))

    def to_dict(self):
        return {
            "class": "SimpleWeld",
            "params": {
                "points": self.points,
                "alpha": self.alpha,
                "dt": self.dt,
            },
        }

    def get_constraint_set(self):
        return self.lengths


class FacetedWeld(Constraint):
    def __init__(self, state: State, faces: List[List[int]], alpha: float, dt: float):
        super().__init__(state, alpha, dt)
        self.faces = faces
        self.face_welds = []

        for face in faces:
            face_weld = SimpleWeld(state, face, alpha, dt)
            self.face_welds.append(face_weld)

    def to_dict(self):
        return {
            "class": "FacetedWeld",
            "params": {
                "faces": self.faces,
                "alpha": self.alpha,
                "dt": self.dt,
            },
        }

    def get_constraint_set(self):
        constraint_set = []
        for face_weld in self.face_welds:
            constraint_set.extend(face_weld.get_constraint_set())
        return constraint_set


class LengthSet(Constraint):
    def __init__(self, state: State, A: List[int], B: List[int], alpha: float, dt: float):
        super().__init__(state, alpha, dt)
        self.A = A
        self.B = B
        self.lengths = []

        for i in range(len(A)):
            for j in range(len(B)):
                self.lengths.append(Length(state, A[i], B[j], alpha, dt))

    def to_dict(self):
        return {
            "class": "LengthSet",
            "params": {
                "A": self.A,
                "B": self.B,
                "alpha": self.alpha,
                "dt": self.dt,
            },
        }

    def get_constraint_set(self):
        return self.lengths