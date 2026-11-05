"""Forward kinematics of the pinned generator MJCF, including LFJ4.

Only the hinge/pos/quat subset actually present in this source asset is accepted.
No angle-copy heuristic is used to define target fingertip positions.
"""
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .model import ASSETS

class ShadowModel:
    def __init__(self):
        self.path = ASSETS/'shadow_hand.xml'
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        root = ET.parse(self.path).getroot()
        if root.find('compiler').get('angle')!='radian':
            raise ValueError('Expected radian MJCF')
        self.body = root.find('worldbody/body')
        self.names = tuple(j.get('name') for j in self.body.findall('.//joint'))
        if len(self.names)!=22:
            raise ValueError('Expected the 22-joint generator model')

    def landmarks(self,qpos):
        if set(qpos)!=set(self.names) or not np.isfinite(list(qpos.values())).all():
            raise ValueError('Expected all 22 finite Shadow joint angles')
        transforms = {}; sites = {}
        def visit(body,parent):
            local = np.eye(4)
            local[:3,3] = np.fromstring(body.get('pos','0 0 0'),sep=' ')
            if 'quat' in body.attrib:
                quat = np.fromstring(body.get('quat'),sep=' ')
                local[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            if any(key in body.attrib for key in ('euler','axisangle','xyaxes','zaxis')):
                raise ValueError('Unsupported source MJCF orientation')
            for joint in body.findall('joint'):
                if joint.get('type','hinge')!='hinge':
                    raise ValueError('Unsupported source joint')
                pivot = np.fromstring(joint.get('pos','0 0 0'),sep=' ')
                axis = np.fromstring(joint.get('axis','0 0 1'),sep=' ')
                rotation = Rotation.from_rotvec(axis/np.linalg.norm(axis)*qpos[joint.get('name')]).as_matrix()
                move = np.eye(4);move[:3,:3]=rotation;move[:3,3]=pivot-rotation@pivot
                local = local @ move
            transform = parent @ local
            transforms[body.get('name')] = transform
            for site in body.findall('site'):
                pos = np.fromstring(site.get('pos','0 0 0'),sep=' ')
                sites[site.get('name')] = transform[:3,:3]@pos+transform[:3,3]
            for child in body.findall('body'):
                visit(child,transform)
        visit(self.body,np.eye(4))
        tips = [sites['robot0:S_'+finger+'tip'] for finger in ('ff','mf','rf','lf','th')]
        bases = [transforms['robot0:'+finger+'proximal'][:3,3] for finger in ('ff','mf','rf','lf','th')]
        return np.array(tips+bases)
