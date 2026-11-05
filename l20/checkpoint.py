"""Do not resume a policy trained against a different model/action contract."""
from .model import L20Model, INDEPENDENT_NAMES, collision_contract
from .control import OBSERVATION_DIM, NUM_ACTIONS

def policy_contract(model=None):
    model = model or L20Model()
    return dict(schema='linkerhand.l20.right.policy.v2',urdf_sha256=model.sha256,
                collision_sha256=collision_contract(model),
                independent_names=list(INDEPENDENT_NAMES),physical_names=list(model.names),
                observation_dim=OBSERVATION_DIM,action_dim=NUM_ACTIONS,
                mimic_mode='explicit_position_targets',observation_layout='named_state_v2',
                palm_reward_reference='official_base_link_com')

def pack(state_dict,contract):
    return dict(policy_contract=contract,state_dict=state_dict)

def unpack(payload,contract):
    if not isinstance(payload,dict) or payload.get('policy_contract')!=contract or 'state_dict' not in payload:
        raise ValueError('Checkpoint lacks the matching official L20 v2 contract; start new training')
    return payload['state_dict']
