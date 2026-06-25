from tac.core.state import State
from tac.constraints.length import Length
from typing import List


class SimpleWeld:
    def __init__(self, state: State, points: List[int], alpha: float):
        self.state = state
        self.points = points
        self.alpha = alpha
        self.lengths = []

        for i in range(len(points)):
            for j in range(i+1, len(points)):
                self.lengths.append(Length(state, points[i], points[j], alpha))

    def to_dict(self):
        return {
            "class": "SimpleWeld",
            "params": {
                "points": self.points,
                "alpha": self.alpha,
            },
        }

    def get_constraint_set(self):
        return self.lengths


class FacetedWeld:
    def __init__(self, state: State, faces: List[List[int]], alpha: float):
        self.state = state
        self.faces = faces
        self.alpha = alpha
        self.face_welds = []

        for face in faces:
            face_weld = SimpleWeld(state, face, alpha)
            self.face_welds.append(face_weld)

    def to_dict(self):
        return {
            "class": "FacetedWeld",
            "params": {
                "faces": self.faces,
                "alpha": self.alpha,
            },
        }

    def get_constraint_set(self):
        constraint_set = []
        for face_weld in self.face_welds:
            constraint_set.extend(face_weld.get_constraint_set())
        return constraint_set


class LengthSet:
    def __init__(self, state: State, A: List[int], B: List[int], alpha: float):
        self.state = state
        self.A = A
        self.B = B
        self.alpha = alpha
        self.lengths = []

        for i in range(len(A)):
            for j in range(len(B)):
                self.lengths.append(Length(state, A[i], B[j], alpha))

    def to_dict(self):
        return {
            "class": "LengthSet",
            "params": {
                "A": self.A,
                "B": self.B,
                "alpha": self.alpha,
            },
        }

    def get_constraint_set(self):
        return self.lengths