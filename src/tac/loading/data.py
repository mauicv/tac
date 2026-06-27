from dataclasses import dataclass
from typing import List
from tac.core.state import State
from tac.constraints.base import Constraint
from tac.constraints.weld import FacetedWeld

@dataclass
class Data:
    state: State
    constraints: List[Constraint]
    graphics: List[FacetedWeld]
    n: int
