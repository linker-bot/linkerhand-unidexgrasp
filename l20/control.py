"""Explicit named layout and batched affine expansion (NumPy or Torch arrays)."""
import numpy as np
from .model import L20Model, INDEPENDENT_NAMES

NUM_ACTIONS = 6 + len(INDEPENDENT_NAMES)
# q(21), qdot(21), fingertip states(5*13), base pose(6), actions(22),
# object(16), prior delta pose(7), prior delta q(16), visual feature(64).
OBSERVATION_DIM = 21*2 + 5*13 + 6 + NUM_ACTIONS + 16 + 7 + 16 + 64

class JointLayout:
    def __init__(self, imported_names, model=None):
        self.model = model or L20Model()
        if len(set(imported_names)) != 21 or set(imported_names) != set(self.model.names):
            raise ValueError('Importer must expose all 21 official joints, exactly once')
        self.names = tuple(imported_names)
        self.independent_indices = [self.names.index(n) for n in INDEPENDENT_NAMES]
        self.canonical_indices = [self.names.index(n) for n in self.model.names]
        self.matrix = np.zeros((16,21))
        self.offset = np.zeros(21)
        for k,n in enumerate(self.names):
            if n in self.model.mimic:
                parent,m,b = self.model.mimic[n]
                self.matrix[INDEPENDENT_NAMES.index(parent),k] = m
                self.offset[k] = b
            else:
                self.matrix[INDEPENDENT_NAMES.index(n),k] = 1.

    def ordered(self, full):
        self.model.validate(full)
        return np.array([full[n] for n in self.names])

def position_targets(actions, previous, lower, upper, relative, dt, speed, smoothing):
    """Only the last 16 actions operate joints, in both control modes."""
    if actions.shape[-1] != NUM_ACTIONS:
        raise ValueError('L20 actions must have width 22')
    if not 0 <= smoothing <= 1:
        raise ValueError('actionsMovingAverage must be in [0,1]')
    joint_actions = actions[...,6:]
    if relative:
        target = previous + speed*dt*joint_actions
    else:
        target = lower + (joint_actions+1.)*.5*(upper-lower)
        target = smoothing*target + (1.-smoothing)*previous
    # minimum/maximum work for NumPy; Torch uses the same method below.
    if isinstance(target, np.ndarray):
        return np.clip(target,lower,upper)
    import torch
    return torch.maximum(torch.minimum(target,upper),lower)
