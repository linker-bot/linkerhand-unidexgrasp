"""Explicit scene/object transforms, with exact recovery for old generator data."""
from pathlib import Path
import numpy as np

def validate_pose(pose):
    pose=np.asarray(pose,dtype=float)
    if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(pose[:3,:3]),1.):
        raise ValueError('Invalid object-to-scene transform')
    return pose

def object_pose(result,index,mesh_root=None):
    if 'object_pose' in result:
        return validate_pose(result['object_pose'][index])
    if mesh_root is None or 'obj_pc' not in result:
        raise ValueError('Missing object_pose; regenerate result.pt or supply source --mesh-root for exact recovery')
    directory=Path(mesh_root)/str(result['object_code'][index])
    poses=np.load(directory/'poses.npy',allow_pickle=False)
    clouds=np.load(directory/'pcs_table.npy',allow_pickle=False,mmap_mode='r')
    observed=np.asarray(result['obj_pc'][index],dtype=float)[:3000]
    scale=float(np.asarray(result['scale'][index]).item())
    matches=[]
    for pose,cloud in zip(poses,clouds):
        pose=np.array(pose,copy=True,dtype=float)
        pose[:2,3]=0 # exactly the original Meshdata / Meshdata_mypc convention
        predicted=scale*(cloud[:3000]@pose[:3,:3].T+pose[:3,3])
        if predicted.shape==observed.shape and np.allclose(predicted,observed,rtol=0,atol=2e-6):
            pose[:3,3]*=scale
            matches.append(validate_pose(pose))
    if not matches or any(not np.allclose(p,matches[0],rtol=0,atol=1e-5) for p in matches[1:]):
        raise ValueError('Cannot uniquely recover object_pose by exact source point correspondence')
    return matches[0]

def to_object_frame(mapped,plane,scene_from_object):
    pose=validate_pose(scene_from_object)
    rotation=pose[:3,:3];translation=pose[:3,3]
    result=dict(mapped)
    result['rotation']=rotation.T@mapped['rotation']
    result['translation']=rotation.T@(mapped['translation']-translation)
    plane=np.asarray(plane,dtype=float)
    if plane.shape!=(4,):
        raise ValueError('Invalid scene plane')
    result['plane']=np.r_[rotation.T@plane[:3],plane[:3]@translation+plane[3]]
    result['source_object_pose']=pose
    return result
