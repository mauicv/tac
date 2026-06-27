import pyvista as pv
import numpy as np
import os
from tqdm import tqdm
import torch
from dataclasses import dataclass

from tac.core.state import State
from tac.loading.data import Data
from tac.constraints.length import Length
from tac.constraints.contact import FloorContact
from tac.constraints.base import Constraint
from tac.constraints.weld import FacetedWeld, LengthSet
from tac.effects.inertial import Inertia, Gravity
from tac.constraints.joint import Joint
from typing import List


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
        self.mesh_coms = {}

        self.constraints = []
        self.graphics = []
        self.state = None
        self.n = None

    def _add_point(self, point):
        point = point.tolist()
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

    def get_mesh_points(self, mesh_name):
        return [ind for ind, names in self.mesh_map.items() if mesh_name in names]

    def get_mesh_com(self, mesh_name):
        return self.mesh_coms[mesh_name]

    def add_mesh_coms(self):
        cur_i = len(self.points)
        for i, mesh_name in enumerate(self.names):
            point_inds = self.get_mesh_points(mesh_name)
            mesh_points = np.array(self.points)[point_inds]
            com = np.mean(mesh_points, axis=0).tolist()
            self.points.append(com)
            self.mesh_coms[mesh_name] = cur_i + i
        print(f"--> Added {len(self.mesh_coms)} coms")

    def add_floor(self, floor_z, x_length, y_length):
        cur_i = len(self.points)
        self.points.append([-x_length/2, -y_length/2, floor_z])
        self.points.append([x_length/2, -y_length/2, floor_z])
        self.points.append([-x_length/2, y_length/2, floor_z])
        self.points.append([x_length/2, y_length/2, floor_z])
        self.faces.extend([4, cur_i, cur_i+1, cur_i+3, cur_i+2])
        self.constraints.extend([FloorContact(self.state, i, floor_z, 1.0) for i in range(cur_i)])
        print(f"--> Added {len(self.constraints)} floor constraints")

    def build_state(self):
        self.state = State(
            torch.tensor(self.points),
            inv_mass=torch.ones(len(self.points)),
        )
        print(f"--> Added {len(self.points)} points to state")

    def build_face_constraints(self):
        faceted_weld = FacetedWeld(self.state, [face for face_size, face in faces_iter(self.faces)], 0.001)
        self.graphics.append(faceted_weld)
        self.constraints.extend(faceted_weld.get_constraint_set())
        print(f"--> Added {len(faceted_weld.get_constraint_set())} face constraints")

    def build_joint_constraints(self):
        joint_constraints = []
        for mesh_name_1 in self.names:
            for mesh_name_2 in self.names:
                if mesh_name_1 != mesh_name_2:
                    joint_points = self._get_joint_points(mesh_name_1, mesh_name_2)
                    if len(joint_points) != 2:
                        continue
                    print(f"--> Adding joint between {mesh_name_1} and {mesh_name_2}")
                    a = self.get_mesh_com(mesh_name_1)
                    b = self.get_mesh_com(mesh_name_2)
                    joint_constraints.append(Joint(self.state, joint_points, a, b, 0.0))
        self.constraints.extend(joint_constraints)

    def build_com_constraints(self):
        com_constraints = []
        for mesh_name in self.names:
            com = self.get_mesh_com(mesh_name)
            points = self.get_mesh_points(mesh_name)
            assert len(points) > 0, "Mesh must have at least one point"
            length_set = LengthSet(self.state, points, [com], 0.0)
            com_constraints.extend(length_set.get_constraint_set())
        print(f"--> Added {len(com_constraints)} com constraints")
        self.constraints.extend(com_constraints)

    def build(self):
        self.add_mesh_coms()
        self.n = len(self.points)
        floor_z = min([point[2] for point in self.points])
        self.add_floor(floor_z, 2.0, 2.0)
        self.build_state()
        self.build_face_constraints()
        self.build_com_constraints()
        self.build_joint_constraints()
        print(f"{len(self.constraints)} total constraints")
        return self

    def to_data(self):
        return Data(
            state=self.state,
            constraints=self.constraints,
            graphics=self.graphics,
            n=self.n
        )

def load_stl(file_path):
    stl_parsed = ParsedStl()
    for file in os.listdir(file_path):
        if file.endswith('.stl'):
            file = os.path.join(file_path, file)
            print(f'loading file: {file}')
            mesh = pv.read(file)
            name = file.split('/')[-1].split('.')[0]
            stl_parsed.add_mesh(name, mesh)
    return stl_parsed.build().to_data()
