"""URDF-derived kinematics and mimic constraints; no Torch/Isaac dependency."""
from pathlib import Path
import hashlib
import json
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from .source import ASSETS, OFFICIAL_URDF, RUNTIME_ASSETS, URDF_COMMIT, verify_official_source
FINGERS = ('index', 'middle', 'ring', 'pinky', 'thumb')
FINGERTIP_LINKS = tuple(f + '_distal' for f in FINGERS)
INDEPENDENT_NAMES = tuple(f + '_' + j for f in FINGERS[:-1]
                          for j in ('mcp_roll', 'mcp_pitch', 'pip')) + (
    'thumb_cmc_yaw', 'thumb_cmc_roll', 'thumb_cmc_pitch', 'thumb_mcp')

def collision_contract(model):
    manifest = json.loads((model.asset_dir/'collision-manifest.json').read_text())
    if manifest.get('source_urdf_sha256') != model.sha256 or manifest.get('method') != 'convex_hull_per_link_v1':
        raise ValueError('Collision assets do not match official source URDF')
    if len(manifest['files']) != 22:
        raise ValueError('Expected 22 generated convex collision meshes')
    for entry in manifest['files']:
        path = model.asset_dir/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Collision asset checksum mismatch: '+str(path))
    return hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()

def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(
            element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t

def resolve_mesh(uri, urdf_path):
    """Resolve relative/file/package paths without depending on working directory."""
    path = Path(urdf_path).resolve()
    if uri.startswith('package://'):
        package, relative = uri[len('package://'):].split('/', 1)
        for parent in path.parents:
            if parent.name == package:
                result = parent / relative
                break
        else:
            raise ValueError('Cannot resolve package URI: ' + uri)
    else:
        result = Path(uri[7:] if uri.startswith('file://') else uri)
        if not result.is_absolute():
            result = path.parent / result
    if not result.is_file():
        raise FileNotFoundError(result)
    return result

class URDFModel:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = ET.parse(self.path).getroot()
        self.joints = {j.get('name'): j for j in self.root.findall('joint')}
        self.names = tuple(n for n,j in self.joints.items() if j.get('type') != 'fixed')
        self.links = tuple(l.get('name') for l in self.root.findall('link'))
        children = {j.find('child').get('link') for j in self.joints.values()}
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError('Expected one URDF root')
        self.base = roots.pop()
        self.limits = {n: (float(self.joints[n].find('limit').get('lower')),
                           float(self.joints[n].find('limit').get('upper'))) for n in self.names}

    def fk(self, qpos):
        if set(qpos) != set(self.names):
            raise ValueError('FK requires exactly the named physical joints')
        result = {self.base: np.eye(4)}
        pending = dict(self.joints)
        while pending:
            progressed = False
            for name, joint in list(pending.items()):
                parent = joint.find('parent').get('link')
                if parent not in result:
                    continue
                transform = origin(joint.find('origin'))
                if joint.get('type') != 'fixed':
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    rotation = np.eye(4)
                    rotation[:3,:3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * qpos[name]).as_matrix()
                    transform = transform @ rotation
                result[joint.find('child').get('link')] = result[parent] @ transform
                del pending[name]
                progressed = True
            if not progressed:
                raise ValueError('Disconnected/cyclic URDF')
        return result

class L20Model(URDFModel):
    def __init__(self, path=OFFICIAL_URDF):
        if Path(path).resolve()==OFFICIAL_URDF.resolve():
            verify_official_source()
        super().__init__(path)
        self.asset_dir = RUNTIME_ASSETS
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        base_inertial = self.root.find("link[@name='base_link']/inertial/origin")
        self.palm_reference = origin(base_inertial)[:3,3]
        self.mimic = {}
        for n,j in self.joints.items():
            m = j.find('mimic')
            if m is not None:
                self.mimic[n] = (m.get('joint'), float(m.get('multiplier','1')), float(m.get('offset','0')))
        if set(self.names) - set(self.mimic) != set(INDEPENDENT_NAMES) or len(self.names) != 21:
            raise ValueError('Not the expected official L20 right-hand contract')
        self.independent_names = INDEPENDENT_NAMES
        bounds = dict(self.limits)
        for child,(parent,m,b) in self.mimic.items():
            if m == 0 or parent in self.mimic:
                raise ValueError('Unsupported mimic graph')
            a,z = sorted(( (self.limits[child][0]-b)/m, (self.limits[child][1]-b)/m ))
            lo,hi = bounds[parent]
            bounds[parent] = (max(lo,a),min(hi,z))
        self.lower = np.array([bounds[n][0] for n in INDEPENDENT_NAMES])
        self.upper = np.array([bounds[n][1] for n in INDEPENDENT_NAMES])
        if np.any(self.lower >= self.upper):
            raise ValueError('Infeasible mimic-adjusted limits')

    def expand(self, independent, clip=False):
        q = np.asarray(independent, dtype=float)
        if q.shape != (16,) or not np.isfinite(q).all():
            raise ValueError('Expected 16 finite independent angles')
        if clip:
            q = np.clip(q,self.lower,self.upper)
        elif np.any(q < self.lower-1e-7) or np.any(q > self.upper+1e-7):
            raise ValueError('Independent angles exceed mimic-adjusted limits')
        full = dict(zip(INDEPENDENT_NAMES,q))
        for child,(parent,m,b) in self.mimic.items():
            full[child] = full[parent]*m+b
        return {n:float(full[n]) for n in self.names}

    def validate(self, full, tolerance=1e-6):
        if set(full) != set(self.names):
            raise ValueError('Unexpected physical joint names')
        for n,v in full.items():
            lo,hi = self.limits[n]
            if not np.isfinite(v) or v < lo-tolerance or v > hi+tolerance:
                raise ValueError('Invalid angle for '+n)
        for child,(parent,m,b) in self.mimic.items():
            if abs(full[child]-(full[parent]*m+b)) > tolerance:
                raise ValueError('Mimic violation: '+child)

    def runtime_urdf(self):
        """All 21 DOFs are controlled explicitly, avoiding importer-specific mimic."""
        root = ET.fromstring(self.path.read_text())
        for j in root.findall('joint'):
            m = j.find('mimic')
            if m is not None:
                j.remove(m)
        # Explicit conservative convex collision assets also used by screening.
        # The upstream base STL is not watertight; never query its interior.
        for link in root.findall('link'):
            for mesh in link.findall('visual/geometry/mesh'):
                source = resolve_mesh(mesh.get('filename'),self.path)
                mesh.set('filename',Path(os.path.relpath(source,self.asset_dir)).as_posix())
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                if mesh is not None:
                    mesh.set('filename','collision/'+link.get('name')+'.STL')
        return ET.tostring(root, encoding='unicode')
