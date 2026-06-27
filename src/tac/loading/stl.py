import pyvista as pv
import numpy as np
import os
from tqdm import tqdm
import torch

from tac.core.state import State
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.weld import FacetedWeld
from tac.effects.inertial import Inertia, Gravity
from tac.constraints.joint import Joint


def faces_iter(faces):
    finished = False
    ind = 0
    while not finished:
        face_size = faces[ind]
        yield face_size, faces[ind+1:ind+face_size+1]
        ind += face_size + 1
        if ind >= len(faces):
            finished = True


class ParsedStl:
    def __init__(self):
        self.points = []
        self.faces = []
        self.names = []
        self.mesh_map = {}

    def _add_point(self, point):
        for j, other_point in enumerate(self.points):
            if np.allclose(point, other_point):
                return j, True
        self.points.append(point)
        return len(self.points) - 1, False

    def _get_point_index(self, point):
        for j, other_point in enumerate(self.points):
            if np.allclose(point, other_point):
                return j
        return None

    def add_mesh(self, name, mesh):
        self.names.append(name)
        mesh_index = len(self.points)
        for point in mesh.points:
            ind, is_dup = self._add_point(point)
            if self.mesh_map.get(ind) is None:
                self.mesh_map[ind] = []
            self.mesh_map[ind].append(name)

        for face_size, face in faces_iter(mesh.faces):
            face = [self._get_point_index(mesh.points[i]) for i in face]
            self.faces.extend([face_size, *face])

    def _get_joint_points(self, first_mesh_name, second_mesh_name):
        data = {ind:names for ind, names in self.mesh_map.items() if len(names) > 1}
        inds = []
        for ind, names in data.items():
            if first_mesh_name in names and second_mesh_name in names:
                inds.append(ind)
        return inds

    def _get_mesh_points(self, mesh_name):
        return [ind for ind, names in self.mesh_map.items() if mesh_name in names]                


def load_stl(file_path):
    stl_parsed = ParsedStl()
    for file in os.listdir(file_path):
        if file.endswith('.stl'):
            file = os.path.join(file_path, file)
            print(f'loading file: {file}')
            mesh = pv.read(file)
            name = file.split('/')[-1].split('.')[0]
            stl_parsed.add_mesh(name, mesh)

    state = State(
        torch.tensor(stl_parsed.points),
        inv_mass=torch.tensor([1.0 for _ in range(len(stl_parsed.points))]),
    )

    welds = []
    joints = []

    faces = [face for face_size, face in faces_iter(stl_parsed.faces)]
    welds = FacetedWeld(state, faces, 0.001, 0.005)
    
    for mesh_name_1 in stl_parsed.names:
        for mesh_name_2 in stl_parsed.names:
            if mesh_name_1 != mesh_name_2:
                joint_points = stl_parsed._get_joint_points(mesh_name_1, mesh_name_2)
                if len(joint_points) > 0:
                    first_mesh_point = stl_parsed._get_mesh_points(mesh_name_1)[0]
                    second_mesh_point = stl_parsed._get_mesh_points(mesh_name_2)[0]
                    
                    joint = Joint(
                        state,
                        axis_points=joint_points,
                        from_point=first_mesh_point,
                        to_point=second_mesh_point,
                        alpha=0.01,
                        dt=0.005
                    )
                    joints.append(joint)

    return stl_parsed, state, welds, joints
