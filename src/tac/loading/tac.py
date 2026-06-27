import json
from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.constraints.joint import Joint
from tac.constraints.imu import PitchRollIMU
import torch
from tac.loading.data import Data


class_map = {
    "Length": Length,
    "FloorContact": FloorContact,
    "FacetedWeld": FacetedWeld,
    "Joint": Joint,
    "PitchRollIMU": PitchRollIMU,
}

def save_tac(data: Data, file_path):
    data_dict = {
        "points": data.state.x_current[0].tolist(),
        "inv_mass": data.state.inv_mass[0].tolist(),
        "constraints": [constraint.to_dict() for constraint in data.constraints],
        "graphics": [graphic.to_dict() for graphic in data.graphics],
        "n": data.n
    }
    with open(file_path, 'w') as f:
        json.dump(data_dict, f)

def load_tac(file_path) -> Data:
    with open(file_path, 'r') as f:
        data = json.load(f)
    state = State(
        torch.tensor(data["points"]),
        inv_mass=torch.tensor(data["inv_mass"]),
    )
    constraints = [class_map[constraint["class"]](state, **constraint["params"]) for constraint in data["constraints"]]
    graphics = [class_map[graphic["class"]](state, **graphic["params"]) for graphic in data["graphics"]]
    return Data(state, constraints, graphics, data["n"])