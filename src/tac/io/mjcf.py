"""Minimal MJCF loader: robot geometry in world frame at the XML (all-joints-zero) pose.

Hand parser for the subset of MJCF that onshape-to-robot emits. Anything outside that
subset raises, rather than silently producing wrong geometry. Physical parameters
(masses, gains, damping, ...) are deliberately ignored: they are to be learned.

"Components" are the rigid parts of the robot: the root body plus every body that has a
hinge joint. A body with no joint is welded to its parent, so its geometry and sites fold
into the nearest component above it.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Body:
    name: str
    parent: str | None                 # parent component, None for the root
    verts: np.ndarray                  # (V, 3) unique mesh vertices, world frame
    meshes: list[str] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    parent: str                        # parent component
    child: str                         # component the joint moves
    anchor: np.ndarray                 # (3,) world frame
    axis: np.ndarray                   # (3,) unit, world frame
    range: tuple[float, float] | None  # radians


@dataclass
class Site:
    name: str
    body: str                          # component
    pos: np.ndarray                    # (3,) world frame


@dataclass
class Actuator:
    name: str
    joint: str


@dataclass
class RobotSpec:
    bodies: dict[str, Body]
    joints: dict[str, Joint]
    sites: dict[str, Site]
    actuators: list[Actuator]          # order = control column order


# --- supported subset: tag -> allowed attributes / children ---------------------------

_ATTRS = {
    'mujoco': {'model'},
    'compiler': {'angle', 'meshdir', 'autolimits'},
    'asset': set(),
    'mesh': {'name', 'file'},
    'worldbody': set(),
    'body': {'name', 'pos', 'quat'},
    'freejoint': {'name'},
    'joint': {'name', 'type', 'axis', 'range', 'pos'},
    'geom': {'type', 'mesh', 'pos', 'quat'},
    'site': {'name', 'pos'},
    'actuator': set(),
    'keyframe': set(),
    'key': {'name', 'qpos', 'ctrl'},
}
_CHILDREN = {
    'mujoco': {'compiler', 'asset', 'worldbody', 'actuator', 'keyframe'},
    'asset': {'mesh'},
    'worldbody': {'body'},
    'body': {'freejoint', 'joint', 'geom', 'site', 'body'},
    'keyframe': {'key'},
}
_ACTUATOR_TAGS = {'position', 'motor', 'velocity', 'general'}


def _check(el: ET.Element):
    """Raise on any tag/attribute outside the supported subset (recursive)."""
    if el.tag in _ACTUATOR_TAGS:
        # gains etc. are ignored (learned); only the joint mapping matters
        if 'joint' not in el.attrib:
            raise ValueError(f"<{el.tag}> without a joint= is not supported")
        return
    if el.tag not in _ATTRS:
        raise ValueError(f"unsupported MJCF element <{el.tag}>")
    extra = set(el.attrib) - _ATTRS[el.tag]
    if extra:
        raise ValueError(f"unsupported attribute(s) {sorted(extra)} on <{el.tag} {el.attrib}>")
    allowed = _ACTUATOR_TAGS if el.tag == 'actuator' else _CHILDREN.get(el.tag, set())
    for child in el:
        if child.tag not in allowed:
            raise ValueError(f"unsupported <{child.tag}> inside <{el.tag}>")
        _check(child)


# --- math helpers -----------------------------------------------------------------------

def _vec(el: ET.Element, key: str, default: str) -> np.ndarray:
    return np.array([float(s) for s in el.get(key, default).split()])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(4,) wxyz quaternion -> (3, 3) rotation matrix (normalised, as MuJoCo does)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _local(el: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """Element's (pos (3,), rot (3, 3)) relative to its parent frame."""
    return _vec(el, 'pos', '0 0 0'), _quat_to_mat(_vec(el, 'quat', '1 0 0 0'))


def load_stl(path: str | Path) -> np.ndarray:
    """Binary STL -> (V, 3) unique vertices in the file's frame."""
    data = Path(path).read_bytes()
    n = struct.unpack('<I', data[80:84])[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL (ASCII STL is not supported)")
    tri = np.dtype([('normal', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    v = np.frombuffer(data, dtype=tri, count=n, offset=84)['v'].reshape(-1, 3)
    return np.unique(v.astype(np.float64), axis=0)


# --- loader -----------------------------------------------------------------------------

def load_mjcf(path: str | Path) -> RobotSpec:
    """Parse an MJCF file into world-frame geometry at the all-joints-zero pose."""
    path = Path(path)
    root = ET.parse(path).getroot()
    _check(root)

    compiler = root.find('compiler')
    if compiler is None or compiler.get('angle') != 'radian':
        raise ValueError('only <compiler angle="radian"> is supported')
    meshdir = path.parent / compiler.get('meshdir', '')

    mesh_files = {}
    for m in root.iter('mesh'):
        name = m.get('name', Path(m.get('file')).stem)
        mesh_files[name] = meshdir / m.get('file')
    stl_cache: dict[str, np.ndarray] = {}

    bodies: dict[str, Body] = {}
    joints: dict[str, Joint] = {}
    sites: dict[str, Site] = {}
    hinge_order: list[str] = []

    def walk(el: ET.Element, parent_comp: str | None, P: np.ndarray, R: np.ndarray):
        # world frame of this body: x_world = R @ x_local + P
        p, r = _local(el)
        P, R = P + R @ p, R @ r
        name = el.get('name')
        if name is None:
            raise ValueError('every <body> needs a name')

        hinges = el.findall('joint')
        free = el.find('freejoint') is not None
        if len(hinges) > 1:
            raise ValueError(f"body {name}: multiple joints not supported")
        if free and parent_comp is not None:
            raise ValueError(f"body {name}: freejoint only supported on a root body")

        if parent_comp is None or hinges:
            comp = name
            bodies[comp] = Body(name=comp, parent=parent_comp, verts=np.zeros((0, 3)))
        else:
            comp = parent_comp   # welded: fold into parent component

        for j in hinges:
            if j.get('type', 'hinge') != 'hinge':
                raise ValueError(f"joint {j.get('name')}: only hinge joints are supported")
            axis = R @ _vec(j, 'axis', '0 0 1')
            rng = j.get('range')
            joints[j.get('name')] = Joint(
                name=j.get('name'),
                parent=parent_comp,
                child=comp,
                anchor=P + R @ _vec(j, 'pos', '0 0 0'),
                axis=axis / np.linalg.norm(axis),
                range=tuple(float(s) for s in rng.split()) if rng else None,
            )
            hinge_order.append(j.get('name'))

        for g in el.findall('geom'):
            if g.get('type') != 'mesh':
                raise ValueError(f"body {name}: only mesh geoms are supported")
            mesh = g.get('mesh')
            if mesh not in stl_cache:
                stl_cache[mesh] = load_stl(mesh_files[mesh])
            gp, gr = _local(g)
            v = stl_cache[mesh] @ (R @ gr).T + (P + R @ gp)       # (V, 3) world
            bodies[comp].verts = np.concatenate([bodies[comp].verts, v])
            bodies[comp].meshes.append(mesh)

        for s in el.findall('site'):
            sites[s.get('name')] = Site(
                name=s.get('name'), body=comp, pos=P + R @ _vec(s, 'pos', '0 0 0'))

        for child in el.findall('body'):
            walk(child, comp, P, R)

    for b in root.find('worldbody').findall('body'):
        walk(b, None, np.zeros(3), np.eye(3))

    for b in bodies.values():
        if len(b.verts) == 0:
            raise ValueError(f"component {b.name} has no mesh geometry")

    actuators = []
    act = root.find('actuator')
    for a in (act if act is not None else []):
        if a.get('joint') not in joints:
            raise ValueError(f"actuator {a.get('name')}: unknown joint {a.get('joint')}")
        actuators.append(Actuator(name=a.get('name', a.get('joint')), joint=a.get('joint')))

    # the geometry above is the all-joints-zero pose; refuse keyframes that disagree
    for key in root.iter('key'):
        if 'qpos' in key.attrib:
            qpos = _vec(key, 'qpos', '')
            n_free = 7 if root.find('.//freejoint') is not None else 0
            if len(qpos) != n_free + len(hinge_order):
                raise ValueError(f"keyframe {key.get('name')}: qpos has wrong length")
            if np.any(qpos[n_free:] != 0):
                raise ValueError(f"keyframe {key.get('name')}: non-zero joint angles not supported")

    return RobotSpec(bodies=bodies, joints=joints, sites=sites, actuators=actuators)
