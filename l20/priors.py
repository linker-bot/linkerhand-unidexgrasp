"""Versioned named numeric NPZ records; reject old pickle/dictionary-order data."""
import json
from pathlib import Path
import numpy as np
from .model import L20Model, URDF_COMMIT, INDEPENDENT_NAMES, collision_contract

SCHEMA = 'linkerhand.l20.right.prior.v2'
def save_prior(path, result, scale, plane, source_index, geometry, model=None):
    model = model or L20Model()
    model.validate(result['qpos'])
    scale = float(scale)
    plane = np.asarray(plane,dtype=float)
    if scale <= 0 or not np.isfinite(scale) or plane.shape != (4,) or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-4):
        raise ValueError('Invalid scale or normalized plane')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata = dict(schema=SCHEMA,urdf_commit=URDF_COMMIT,urdf_sha256=model.sha256,
                    collision_sha256=collision_contract(model),
                    frame='object_to_hand_base',units='metre_radian',
                    source_index=int(source_index),tip_error_m=float(result['tip_error_m']),
                    source_object_pose=np.asarray(result.get('source_object_pose',np.eye(4))).tolist(),
                    mapping='shadow_mjcf_landmark_ik_v2',geometry=geometry)
    # Exclusive create avoids silently replacing a prior on rerun.
    with path.open('xb') as stream:
        np.savez_compressed(stream,joint_names=np.array(model.names),
            qpos=np.array([result['qpos'][n] for n in model.names]),
            independent_names=np.array(INDEPENDENT_NAMES),
            independent_qpos=np.array([result['qpos'][n] for n in INDEPENDENT_NAMES]),
            hand_translation=np.asarray(result['translation']),
            hand_rotation=np.asarray(result['rotation']),scale=scale,plane=plane,
            metadata=json.dumps(metadata,sort_keys=True))

def load_prior(path, model=None, require_geometry=True):
    model = model or L20Model()
    with np.load(path,allow_pickle=False) as data:
        try:
            metadata = json.loads(str(data['metadata'].item()))
            names = list(data['joint_names'])
            values = np.asarray(data['qpos'],dtype=float)
            if len(names)!=21 or len(set(names))!=21 or values.shape!=(21,):
                raise ValueError('Invalid named joint array')
            full = dict(zip(names,values))
            rotation = np.asarray(data['hand_rotation'],dtype=float)
            translation = np.asarray(data['hand_translation'],dtype=float)
            scale = float(data['scale'].item())
            plane = np.asarray(data['plane'],dtype=float)
            independent = np.asarray(data['independent_qpos'],dtype=float)
            independent_names = list(data['independent_names'])
        except (KeyError,TypeError) as exc:
            raise ValueError('Regenerate legacy priors using data_for_RL.py') from exc
    if metadata.get('schema')!=SCHEMA or metadata.get('urdf_commit')!=URDF_COMMIT or metadata.get('urdf_sha256')!=model.sha256:
        raise ValueError('Prior/model version mismatch')
    if metadata.get('collision_sha256')!=collision_contract(model):
        raise ValueError('Prior/collision model version mismatch')
    if metadata.get('frame')!='object_to_hand_base' or metadata.get('units')!='metre_radian':
        raise ValueError('Prior frame/units mismatch')
    model.validate(full)
    if independent_names!=list(INDEPENDENT_NAMES) or independent.shape!=(16,) or not np.allclose(independent,[full[n] for n in INDEPENDENT_NAMES],atol=1e-7):
        raise ValueError('Independent/full angles disagree')
    if translation.shape!=(3,) or not np.isfinite(translation).all() or rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1.):
        raise ValueError('Invalid hand transform')
    if not np.isfinite(scale) or scale<=0 or plane.shape!=(4,) or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-4):
        raise ValueError('Invalid scale or plane')
    if require_geometry and metadata.get('geometry',{}).get('passed') is not True:
        raise ValueError('Prior has not passed geometric screening')
    return dict(qpos=full,rotation=rotation,translation=translation,scale=scale,plane=plane,metadata=metadata)
