"""Constrained Shadow candidate to official L20 kinematic retargeting.

Landmark fitting is an initialization, not a proof of collision-free grasping.
The historical wrist transform is an initial guess only; target landmarks
come directly from the generator's MJCF. Wrist and 16 target joints are fitted.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .model import L20Model, FINGERTIP_LINKS
from .shadow import ShadowModel

SHADOW_NAMES = tuple('robot0:'+f+j for f,js in (
    ('FFJ',('3','2','1','0')), ('MFJ',('3','2','1','0')),
    ('RFJ',('3','2','1','0')), ('LFJ',('4','3','2','1','0')),
    ('THJ',('4','3','2','1','0'))) for j in js)
OFFICIAL_BASES = ('index_proximal','middle_proximal','ring_proximal','pinky_proximal','thumb_metacarpals')
# Explicit virtual landmarks, in each distal-link frame (metres), not sensors.
TIP_OFFSETS = np.tile(np.array([0.,0.,.02]), (5,1))

def landmarks(model, full, tips, bases):
    fk = model.fk(full)
    tip_points = np.array([fk[n][:3,:3] @ p + fk[n][:3,3] for n,p in zip(tips,TIP_OFFSETS)])
    base_points = np.array([fk[n][:3,3] for n in bases])
    return np.concatenate((tip_points,base_points))

class Retargeter:
    def __init__(self, model=None):
        self.model = model or L20Model()
        self.shadow = ShadowModel()

    def from_shadow(self, shadow, translation, rotation, max_tip_error=.03):
        if not isinstance(shadow,dict):
            shadow = np.asarray(shadow,dtype=float)
            if shadow.shape!=(22,):
                raise ValueError('Expected 22 ShadowHand joint angles')
            shadow = dict(zip(SHADOW_NAMES,shadow))
        translation = np.asarray(translation,dtype=float)
        rotation = np.asarray(rotation,dtype=float)
        if translation.shape != (3,) or rotation.shape != (3,3):
            raise ValueError('Expected translation(3) and rotation(3,3)')
        # Historical transform seeds only the target wrist, never target tips.
        old_rotation = rotation @ np.array([[0,1,0],[-1,0,0],[0,0,1]])
        old_translation = translation + old_rotation @ np.array([0,-.006,-.063])
        target = self.shadow.landmarks(shadow) @ rotation.T + translation
        return self.fit(target, old_translation, old_rotation, max_tip_error)

    def fit(self, target, wrist_translation, wrist_rotation, max_tip_error=.03):
        target = np.asarray(target,dtype=float)
        wrist_translation = np.asarray(wrist_translation,dtype=float)
        wrist_rotation = np.asarray(wrist_rotation,dtype=float)
        if target.shape != (10,3) or not np.isfinite(target).all():
            raise ValueError('Expected ten finite landmark positions')
        if wrist_translation.shape != (3,) or not np.isfinite(wrist_translation).all():
            raise ValueError('Invalid wrist translation')
        if wrist_rotation.shape != (3,3) or not np.allclose(wrist_rotation.T @ wrist_rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(wrist_rotation),1.):
            raise ValueError('Invalid wrist rotation')
        model = self.model
        # A boundary-zero seed can make TRF terminate on its tiny initial step.
        q0 = model.lower + .35*(model.upper-model.lower)
        x0 = np.r_[np.zeros(6),q0]
        lo = np.r_[np.full(3,-.08),np.full(3,-np.pi/2),model.lower]
        hi = np.r_[np.full(3,.08),np.full(3,np.pi/2),model.upper]
        def decode(x):
            r = wrist_rotation @ Rotation.from_rotvec(x[3:6]).as_matrix()
            t = wrist_translation+x[:3]
            return model.expand(x[6:],clip=True),t,r
        def residual(x):
            full,t,r = decode(x)
            points = landmarks(model,full,FINGERTIP_LINKS,OFFICIAL_BASES) @ r.T+t
            errors = points-target
            errors[5:] *= .5
            # Metre-scaled regularization keeps wrist and joints near seed.
            return np.r_[errors.ravel(), .1*x[:3], .002*x[3:6], .0001*(x[6:]-q0)]
        best = least_squares(residual,x0,bounds=(lo,hi),max_nfev=250,
                             ftol=1e-8,xtol=1e-8,gtol=1e-8)
        full,t,r = decode(best.x)
        points = landmarks(model,full,FINGERTIP_LINKS,OFFICIAL_BASES) @ r.T+t
        error = float(np.linalg.norm(points[:5]-target[:5],axis=1).max())
        if not best.success or error > max_tip_error:
            raise ValueError('Retarget did not converge within tip tolerance: %.6f m' % error)
        model.validate(full)
        return dict(qpos=full,translation=t,rotation=r,
                    tip_error_m=error,optimizer_cost=float(best.cost))
