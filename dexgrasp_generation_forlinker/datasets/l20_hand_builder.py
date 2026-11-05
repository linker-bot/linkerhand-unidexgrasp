"""Official L20 mesh builder using the shared named joint contract."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import trimesh
import torch
from pytorch3d.structures import Meshes
from l20.model import L20Model, OFFICIAL_URDF, INDEPENDENT_NAMES, origin, resolve_mesh

class L20HandBuilder:
    joint_names = list(L20Model().names)

    def __init__(self, mesh_dir=None, mjcf_path=None):
        self.model = L20Model(mjcf_path or OFFICIAL_URDF)
        self.joint_names = list(self.model.names)
        self.mesh = {}
        for link in self.model.root.findall('link'):
            parts = []
            for visual in link.findall('visual'):
                element = visual.find('geometry/mesh')
                if element is None:
                    raise ValueError('Unsupported official visual geometry')
                mesh = trimesh.load(str(resolve_mesh(element.get('filename'),self.model.path)),force='mesh',process=False)
                mesh.apply_scale(np.fromstring(element.get('scale','1 1 1'),sep=' '))
                mesh.apply_transform(origin(visual.find('origin')))
                parts.append(mesh)
            if parts:
                self.mesh[link.get('name')] = trimesh.util.concatenate(parts)

    def qpos_to_qpos_dict(self,qpos,hand_qpos_names=None):
        names = list(hand_qpos_names or self.joint_names)
        if len(qpos)!=len(names) or len(set(names))!=len(names):
            raise ValueError('Joint names and values must match uniquely')
        full = dict(zip(names,np.asarray(qpos,dtype=float)))
        if set(full)==set(INDEPENDENT_NAMES):
            full = self.model.expand([full[n] for n in INDEPENDENT_NAMES])
        self.model.validate(full)
        return full

    def qpos_dict_to_qpos(self,qpos_dict,hand_qpos_names=None):
        self.model.validate(qpos_dict)
        return np.array([qpos_dict[n] for n in (hand_qpos_names or self.joint_names)])

    def get_hand_mesh(self,rotation_mat,world_translation,qpos=None,hand_qpos_dict=None,
                      hand_qpos_names=None,without_arm=False):
        if hand_qpos_dict is None:
            if qpos is None:
                raise ValueError('Provide a named qpos dictionary or joint array')
            hand_qpos_dict = self.qpos_to_qpos_dict(qpos,hand_qpos_names)
        self.model.validate(hand_qpos_dict)
        fk = self.model.fk(hand_qpos_dict)
        world = np.eye(4)
        world[:3,:3] = rotation_mat
        world[:3,3] = world_translation
        parts = []
        for name,mesh in self.mesh.items():
            transformed = mesh.copy()
            transformed.apply_transform(world @ fk[name])
            parts.append(transformed)
        mesh = trimesh.util.concatenate(parts)
        return Meshes(verts=[torch.as_tensor(np.array(mesh.vertices),dtype=torch.float32)],
                      faces=[torch.as_tensor(np.array(mesh.faces),dtype=torch.long)])
