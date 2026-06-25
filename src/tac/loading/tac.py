import json
from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.constraints.joint import Joint
from tac.constraints.imu import PitchRollIMU
import torch


def save_tac(state, constraints, file_path):
    data = {
        "points": state.x_current[0].tolist(),
        "inv_mass": state.inv_mass[0].tolist(),
        "constraints": [constraint.to_dict() for constraint in constraints]
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)


class_map = {
    "Length": Length,
    "FloorContact": FloorContact,
    "FacetedWeld": FacetedWeld,
    "Joint": Joint,
    "PitchRollIMU": PitchRollIMU,
}


def load_tac(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    state = State(
        torch.tensor(data["points"]),
        inv_mass=torch.tensor(data["inv_mass"]),
    )
    joints = []
    welds = []
    for constraint in data["constraints"]:
        if constraint["class"] == "Joint":
            joints.append(class_map[constraint["class"]](state, **constraint["params"]))
        elif constraint["class"] == "FacetedWeld":
            welds.append(class_map[constraint["class"]](state, **constraint["params"]))
        else:
            raise ValueError(f"Unknown constraint class: {constraint['class']}")
    return state, welds, joints