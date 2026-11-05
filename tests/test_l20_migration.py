import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from l20.model import L20Model, INDEPENDENT_NAMES, OFFICIAL_URDF, resolve_mesh
from l20.control import JointLayout, NUM_ACTIONS, OBSERVATION_DIM, position_targets
from l20.priors import save_prior, load_prior
from l20.retarget import Retargeter, landmarks, FINGERTIP_LINKS, OFFICIAL_BASES
from l20.checkpoint import policy_contract, pack, unpack

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'dexgrasp_policy_l20hand/dexgrasp/tasks/shadow_hand_grasp.py'

def result(model):
    return dict(qpos=model.expand((model.lower+model.upper)*.25),
                translation=np.array([0.,0.,.1]),rotation=np.eye(3),tip_error_m=.001)

def test_official_contract_and_all_mimic_adjusted_bounds(tmp_path):
    m = L20Model()
    assert len(m.names)==21 and len(m.mimic)==5
    for q in (m.lower,m.upper,(m.lower+m.upper)/2):
        full=m.expand(q);m.validate(full)
        for child,(parent,mult,offset) in m.mimic.items():
            assert full[child]==pytest.approx(full[parent]*mult+offset)
    # Verify child limits propagate to parent even if a later revision tightens them.
    import xml.etree.ElementTree as ET
    root=ET.fromstring(OFFICIAL_URDF.read_text())
    root.find("joint[@name='index_dip']/limit").set('upper','1.0')
    path=tmp_path/'tighter.urdf';ET.ElementTree(root).write(path)
    tighter=L20Model(path)
    assert tighter.upper[INDEPENDENT_NAMES.index('index_pip')]==pytest.approx(1./.8917)

def test_importer_order_independence_and_missing_joint_rejection():
    m=L20Model();full=result(m)['qpos'];names=m.names[::-1]
    layout=JointLayout(names,m)
    q=np.array([full[n] for n in INDEPENDENT_NAMES])
    np.testing.assert_allclose(q @ layout.matrix+layout.offset,layout.ordered(full))
    np.testing.assert_allclose(layout.ordered(full)[layout.canonical_indices],[full[n] for n in m.names])
    with pytest.raises(ValueError):JointLayout(names[:-1],m)

@pytest.mark.parametrize('relative',[False,True])
@pytest.mark.parametrize('backend',['numpy','torch'])
def test_control_modes_slice_joints_and_preserve_bounds(relative,backend):
    m=L20Model();a=np.ones((4,NUM_ACTIONS));a[:,:6]=1e6
    lower=m.lower;upper=m.upper;previous=np.tile((lower+upper)/2,(4,1))
    if backend=='torch':
        a,previous,lower,upper=[torch.tensor(x) for x in (a,previous,lower,upper)]
    target=position_targets(a,previous,lower,upper,relative,.02,1.,1.)
    expected=np.clip(np.tile((m.lower+m.upper)/2,(4,1))+.02,m.lower,m.upper) if relative else np.tile(m.upper,(4,1))
    np.testing.assert_allclose(np.asarray(target),expected)

def test_asset_manifest_and_relative_paths():
    import hashlib
    manifest=json.loads(OFFICIAL_URDF.with_name('manifest.json').read_text())
    assert len(manifest['files'])==23
    for record in manifest['files']:
        assert hashlib.sha256((OFFICIAL_URDF.parent/record['path']).read_bytes()).hexdigest()==record['sha256']
    m=L20Model()
    for element in m.root.findall('.//mesh'):
        assert resolve_mesh(element.get('filename'),m.path).is_file()
    assert '<mimic' not in m.runtime_urdf()
    assert '<mimic' in m.path.read_text()

def test_prior_roundtrip_and_version_rejection(tmp_path):
    m=L20Model();p=tmp_path/'p.npz'
    save_prior(p,result(m),.06,[0,0,1,0],5,dict(passed=True),m)
    restored=load_prior(p,m)
    assert restored['qpos']==result(m)['qpos']
    with pytest.raises(FileExistsError):save_prior(p,result(m),.06,[0,0,1,0],5,dict(passed=True),m)
    with np.load(p,allow_pickle=False) as f: fields=dict(f)
    metadata=json.loads(str(fields['metadata'].item()));metadata['urdf_sha256']='wrong'
    fields['metadata']=json.dumps(metadata);np.savez(tmp_path/'bad.npz',**fields)
    with pytest.raises(ValueError,match='version mismatch'):load_prior(tmp_path/'bad.npz',m)

def test_unscreened_and_legacy_data_rejected(tmp_path):
    m=L20Model();p=tmp_path/'p.npz'
    save_prior(p,result(m),.06,[0,0,1,0],0,dict(passed=False),m)
    with pytest.raises(ValueError,match='geometric screening'):load_prior(p,m)
    assert load_prior(p,m,require_geometry=False)['scale']==.06
    np.savez(tmp_path/'old.npz',arr_0={'thumb_joint0':0})
    with pytest.raises(ValueError):load_prior(tmp_path/'old.npz',m)

def test_prior_mimic_violation_rejected(tmp_path):
    m=L20Model();mapped=result(m);mapped['qpos']['index_dip']+=.1
    with pytest.raises(ValueError,match='Mimic'):save_prior(tmp_path/'bad.npz',mapped,.06,[0,0,1,0],0,{},m)

def test_ik_recovers_synthetic_official_landmarks():
    m=L20Model();mapped=result(m)
    target=landmarks(m,mapped['qpos'],FINGERTIP_LINKS,OFFICIAL_BASES)+mapped['translation']
    fit=Retargeter(m).fit(target,mapped['translation'],np.eye(3),max_tip_error=.002)
    assert fit['tip_error_m']<.002;m.validate(fit['qpos'])

def test_checkpoint_requires_exact_model_and_action_contract():
    contract=policy_contract();payload=pack({'weight':[1,2]},contract)
    assert unpack(payload,contract)=={'weight':[1,2]}
    with pytest.raises(ValueError):unpack({'weight':[1,2]},contract)
    changed=dict(contract,action_dim=27)
    with pytest.raises(ValueError):unpack(payload,changed)

def task_method(name,globals_dict):
    # Compile the actual task method, without importing unavailable IsaacGym.
    tree=ast.parse(TASK.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ShadowHandGrasp')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[])
    namespace=dict(globals_dict)
    exec(compile(ast.fix_missing_locations(module),str(TASK),'exec'),namespace)
    return namespace[name]

@pytest.mark.parametrize('relative',[False,True])
def test_real_pre_step_expands_mimic_and_applies_wrist_in_both_modes(relative):
    m=L20Model();layout=JointLayout(m.names[::-1],m);calls={}
    gym=SimpleNamespace(apply_rigid_body_force_tensors=lambda *args:calls.update(force=args[1].clone()),
        set_dof_position_target_tensor_indexed=lambda *args:calls.update(target=args[1].clone()))
    env=SimpleNamespace(reset_buf=torch.zeros(2),reset_goal_buf=torch.zeros(2),num_envs=2,
        device='cpu',get_pose_quat=lambda:None,pose_vec=lambda x:x,
        independent_indices=torch.tensor(layout.independent_indices),prev_targets=torch.zeros(2,21),cur_targets=torch.zeros(2,21),
        independent_lower=torch.tensor(m.lower,dtype=torch.float32),independent_upper=torch.tensor(m.upper,dtype=torch.float32),
        mimic_matrix=torch.tensor(layout.matrix,dtype=torch.float32),mimic_offset=torch.tensor(layout.offset,dtype=torch.float32),
        use_relative_control=relative,dt=.02,shadow_hand_dof_speed_scale=1.,act_moving_average=.5,
        apply_forces=torch.zeros(2,22,3),apply_torque=torch.zeros(2,22,3),hand_body_idx_dict={'palm':0},
        transition_scale=1.,orientation_scale=1.,gym=gym,sim=None,hand_indices=torch.tensor([0,4]))
    method=task_method('pre_physics_step',dict(torch=torch,NUM_ACTIONS=NUM_ACTIONS,position_targets=position_targets,
        gymtorch=SimpleNamespace(unwrap_tensor=lambda x:x),gymapi=SimpleNamespace(ENV_SPACE=0)))
    actions=torch.ones(2,NUM_ACTIONS)*.1;before=actions.clone()
    method(env,actions)
    torch.testing.assert_close(actions,before)
    assert calls['force'][:,0,:].abs().sum()>0
    assert calls['force'][:,1:,:].abs().sum()==0
    for row in calls['target'].numpy():m.validate(dict(zip(layout.names,row)),tolerance=1e-5)

def test_real_observation_fills_238_dimensions_without_mutating_actions():
    m=L20Model();layout=JointLayout(m.names[::-1],m);n=3
    ident=torch.tensor([0.,0.,0.,1.]).repeat(n,1)
    env=SimpleNamespace(get_unpose_quat=lambda:None,num_envs=n,canonical_indices=torch.tensor(layout.canonical_indices),
        shadow_hand_dof_pos=torch.zeros(n,21),shadow_hand_dof_vel=torch.zeros(n,21),
        shadow_hand_dof_lower_limits=torch.tensor([m.limits[x][0] for x in layout.names]),
        shadow_hand_dof_upper_limits=torch.tensor([m.limits[x][1] for x in layout.names]),
        fingertip_state=torch.zeros(n,5,13),right_hand_pos=torch.zeros(n,3),right_hand_rot=ident,
        actions=torch.ones(n,NUM_ACTIONS),object_pos=torch.zeros(n,3),object_rot=ident,
        object_linvel=torch.zeros(n,3),object_angvel=torch.zeros(n,3),goal_pos=torch.ones(n,3),
        delta_target_hand_pos=torch.zeros(n,3),delta_target_hand_rot=ident,delta_qpos=torch.zeros(n,16),
        visual_feat_buf=torch.ones(n,64),vel_obs_scale=.2,obs_buf=torch.full((n,OBSERVATION_DIM),float('nan')),
        asymmetric_obs=True,states_buf=torch.zeros(n,OBSERVATION_DIM),unpose_state=lambda x:x,
        unpose_vec=lambda x:x,unpose_point=lambda x:x,unpose_quat=lambda x:x)
    method=task_method('compute_full_state',dict(torch=torch,OBSERVATION_DIM=OBSERVATION_DIM,
        get_euler_xyz=lambda x:(torch.zeros(n),)*3,unscale=lambda q,l,u:2*(q-l)/(u-l)-1))
    method(env)
    assert OBSERVATION_DIM==238 and torch.isfinite(env.obs_buf).all()
    torch.testing.assert_close(env.obs_buf[:,-64:],torch.full((n,64),.1))
    torch.testing.assert_close(env.states_buf,env.obs_buf)
    torch.testing.assert_close(env.actions,torch.ones(n,NUM_ACTIONS))

def test_real_ppo_network_forward_backward_and_std_contract():
    path=ROOT/'dexgrasp_policy_l20hand/dexgrasp/algorithms/rl/ppo/module.py'
    spec=importlib.util.spec_from_file_location('migration_actor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=dict(pi_hid_sizes=[32,16],vf_hid_sizes=[32,16],activation='elu')
    network=module.ActorCritic((238,),(0,),(22,),.3,cfg)
    obs=torch.randn(4,238);states=torch.zeros(4,0)
    action,logp,value,mean,logstd=network.act(obs,states)
    assert action.shape==(4,22) and value.shape==(4,1)
    evaluated=network.evaluate(obs,states,action)
    torch.testing.assert_close(evaluated[0],logp)
    (evaluated[0].mean()+evaluated[2].mean()).backward()
    assert network.actor[0].weight.grad is not None

def test_export_keeps_all_batches_and_never_overwrites(tmp_path):
    from l20.export import read_result, main
    batch=dict(object_code=['core/test-object'],tta_hand_pose=torch.zeros(1,28),
               scale=torch.tensor([.06]),plane=torch.tensor([[0.,0.,1.,0.]]),object_pose=torch.eye(4).unsqueeze(0))
    source=tmp_path/'result.pt';torch.save([batch,batch],source)
    assert len(read_result(source)['object_code'])==2
    out=tmp_path/'out'
    main(['--input',str(source),'--output',str(out),'--kinematics-only','--max-tip-error','.08'])
    assert len(list(out.glob('core/test-object/*.npz')))==2
    with pytest.raises(FileExistsError):main(['--input',str(source),'--output',str(out),'--kinematics-only','--max-tip-error','.08'])

def test_geometric_screen_detects_penetration_and_separation(tmp_path):
    import trimesh
    from l20.geometry import GeometryScreen
    path=tmp_path/'box.obj';mesh=trimesh.creation.box(extents=(.04,.04,.04));mesh.export(path)
    screen=GeometryScreen(path,1.)
    assert screen.penetration(mesh,np.array([[0.,0.,0.],[1.,1.,1.]]))==pytest.approx(.02)
    assert screen.penetration(mesh,np.array([[1.,1.,1.]]))==0

def test_export_module_import_has_no_io_side_effect(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    p=ROOT/'dexgrasp_generation_forlinker/tests/data_for_RL.py'
    spec=importlib.util.spec_from_file_location('data_export_without_io',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert not list(tmp_path.iterdir())

def test_actual_builder_loads_relative_meshes_and_uses_named_mimic(monkeypatch):
    import sys
    # Only replace PyTorch3D's return container, not mesh loading or FK.
    class Meshes:
        def __init__(self,verts,faces):self.verts=verts;self.faces=faces
    monkeypatch.setitem(sys.modules,'pytorch3d',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'pytorch3d.structures',SimpleNamespace(Meshes=Meshes))
    path=ROOT/'dexgrasp_generation_forlinker/datasets/l20_hand_builder.py'
    spec=importlib.util.spec_from_file_location('migration_builder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    builder=module.L20HandBuilder()
    full=result(builder.model)['qpos']
    ordered=builder.qpos_dict_to_qpos(full)
    assert builder.qpos_to_qpos_dict(ordered)==full
    mesh=builder.get_hand_mesh(np.eye(3),np.zeros(3),hand_qpos_dict=full)
    assert mesh.verts[0].shape[1]==3 and len(mesh.faces[0])>0

def test_actual_create_envs_uses_named_links_pd_drives_and_fixed_table(tmp_path):
    from scipy.spatial.transform import Rotation
    m=L20Model();root=tmp_path/'assets';object_code='core/test-object'
    prior=root/'dataset_l20_right_v2'/object_code/'00000.npz'
    save_prior(prior,result(m),.1,[0,0,1,0],0,dict(passed=True),m)
    feature=root/'meshdatav3_pc_feat'/object_code/'pc_feat_010.npy'
    feature.parent.mkdir(parents=True);np.save(feature,np.ones(64))
    class Vec:
        def __init__(self,x=0,y=0,z=0):self.x=x;self.y=y;self.z=z
        def __add__(self,b):return Vec(self.x+b.x,self.y+b.y,self.z+b.z)
    class Quat:
        def __init__(self):self.x=self.y=self.z=0.;self.w=1.
        def from_euler_zyx(self,*args):return self
    class Transform:
        def __init__(self):self.p=Vec();self.r=Quat()
    class Options:
        def __init__(self):self.fix_base_link=False;self.vhacd_params=SimpleNamespace()
    class Gym:
        def __init__(self):self.actor_count=0;self.tables=[];self.body_names=list(reversed(m.links));self.names=list(reversed(m.names))
        def load_asset(self,sim,root,filename,options):
            if filename.endswith('_isaac.urdf'):
                assert options.default_dof_drive_mode==1 and not options.vhacd_enabled
                return 'hand'
            return 'object'
        def get_asset_rigid_body_count(self,asset):return 22 if asset=='hand' else 1
        def get_asset_rigid_shape_count(self,asset):return self.get_asset_rigid_body_count(asset)
        def get_asset_dof_count(self,asset):return 21 if asset=='hand' else 0
        def get_asset_actuator_count(self,asset):return 0
        def get_asset_dof_name(self,asset,i):return self.names[i]
        def get_asset_dof_properties(self,asset):
            count=self.get_asset_dof_count(asset)
            props=np.zeros(count,dtype=[('lower',float),('upper',float),('driveMode',int),('stiffness',float),('damping',float)])
            for i,n in enumerate(self.names[:count]):props['lower'][i],props['upper'][i]=m.limits[n]
            return props
        def create_sphere(self,*args):return 'goal'
        def create_box(self,sim,x,y,z,options):self.tables.append(options);return 'table'
        def find_asset_rigid_body_index(self,asset,name):return self.body_names.index(name)
        def create_env(self,*args):return object()
        def create_actor(self,*args):self.actor_count+=1;return self.actor_count-1
        def set_actor_dof_properties(self,env,actor,props):assert np.all(props['driveMode']==1) and np.all(props['stiffness']>0)
        def get_actor_index(self,env,actor,domain):return actor
        def get_actor_rigid_body_count(self,env,actor):return 22
        def get_actor_rigid_shape_properties(self,*args):return [SimpleNamespace(friction=0.)]
        def __getattr__(self,name):
            if name.startswith(('set_','begin_','end_')):return lambda *a:None
            raise AttributeError(name)
    gym=Gym();api=SimpleNamespace(AssetOptions=Options,Transform=Transform,Vec3=Vec,Quat=Quat,
        SIM_PHYSX=1,DOF_MODE_POS=1,DOF_MODE_NONE=0,COMPUTE_PER_VERTEX=0,VhacdParams=lambda:SimpleNamespace(),
        DOMAIN_SIM=0,MESH_VISUAL=0)
    env=SimpleNamespace(l20_model=m,cfg={'env':dict(object_code_dict={object_code:[.1]},repose_z=True,
        asset=dict(assetRoot=str(root)),goal_cond=True,random_prior=True,random_time=False)},
        num_envs=2,device='cpu',sim=None,physics_engine=1,gym=gym,aggregate_mode=0,
        fingertips=list(FINGERTIP_LINKS),object_id_buf=torch.zeros(2,dtype=torch.long))
    method=task_method('_create_envs',dict(np=np,os=__import__('os'),osp=__import__('os').path,Path=Path,
        __file__=str(TASK),torch=torch,R=Rotation,load_prior=load_prior,JointLayout=JointLayout,
        plane2euler=lambda plane,axes:(np.zeros(3),np.zeros(3)),
        quat_from_euler_xyz=lambda *v:torch.tensor([0.,0.,0.,1.]),
        to_torch=lambda data,device,dtype=torch.float:torch.tensor(data,device=device,dtype=dtype),gymapi=api))
    method(env,2,1.,1)
    assert all(options.fix_base_link for options in gym.tables)
    assert env.hand_body_idx_dict['palm']==gym.body_names.index('base_link')
    assert env.fingertip_handles.tolist()==[gym.body_names.index(n) for n in FINGERTIP_LINKS]
    prior_q=env.grasp_data[object_code][.1]['target_qpos'][0]
    np.testing.assert_allclose(prior_q.numpy(),[result(m)['qpos'][n] for n in gym.names],atol=1e-7)

def test_frame_transform_preserves_world_hand_and_table():
    from scipy.spatial.transform import Rotation
    from l20.frames import to_object_frame
    m=L20Model();mapped=result(m);pose=np.eye(4)
    pose[:3,:3]=Rotation.from_euler('xyz',[.7,-.2,.4]).as_matrix();pose[:3,3]=[.1,.2,.08]
    transformed=to_object_frame(mapped,[0,0,1,0],pose)
    np.testing.assert_allclose(pose[:3,:3]@transformed['rotation'],mapped['rotation'],atol=1e-12)
    np.testing.assert_allclose(pose[:3,:3]@transformed['translation']+pose[:3,3],mapped['translation'],atol=1e-12)
    p=np.array([.04,.06,.03]);scene=pose[:3,:3]@p+pose[:3,3]
    assert transformed['plane'][:3]@p+transformed['plane'][3]==pytest.approx(scene[2])

def test_exact_object_pose_recovery_rejects_missing_or_ambiguous_sources(tmp_path):
    from l20.frames import object_pose
    root=tmp_path/'core/object';root.mkdir(parents=True)
    pose=np.eye(4);pose[:3,3]=[.2,.3,.8]
    points=np.random.default_rng(7).normal(size=(1,3000,3))
    scale=.1
    observed=(points[0]+[0,0,.8])*scale
    np.save(root/'poses.npy',pose[None]);np.save(root/'pcs_table.npy',points)
    data=dict(object_code=['core/object'],scale=[scale],obj_pc=observed[None])
    recovered=object_pose(data,0,tmp_path)
    np.testing.assert_allclose(recovered[:3,3],[0,0,.08])
    with pytest.raises(ValueError,match='Missing object_pose'):object_pose(data,0)
    data['obj_pc']+=.01
    with pytest.raises(ValueError,match='Cannot uniquely recover'):object_pose(data,0,tmp_path)

def test_shadow_mjcf_zero_pose_matches_known_generator_tip_sites():
    from l20.shadow import ShadowModel
    m=ShadowModel();points=m.landmarks(dict.fromkeys(m.names,0.))
    # palm->FF base (0.033,0,0.095), then segments 0.045+0.025+tip 0.026.
    np.testing.assert_allclose(points[0],[.033,0,.191],atol=1e-9)

@pytest.mark.parametrize('class_name',['Meshdata','Meshdata_mypc'])
def test_generator_dataset_preserves_scaled_object_transform(class_name):
    source=ROOT/'dexgrasp_generation_forlinker/datasets/object_dataset.py'
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    namespace=dict(np=np,torch=torch,pytorch3d=SimpleNamespace(ops=SimpleNamespace(sample_farthest_points=lambda pc,K:(pc[:,:K],None))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
    pose=np.eye(4);pose[:3,3]=[0,0,.8]
    cloud=np.zeros((4000,3));cloud[:3000,0]=.1
    data=namespace['__getitem__'](SimpleNamespace(object_list=[('core/object',cloud,.1,pose)]),0)
    np.testing.assert_allclose(data['object_pose'][:3,3],[0,0,.08])
    np.testing.assert_allclose(data['obj_pc'][0],[.01,0,.08])

def test_reset_expands_velocity_mimic_and_uses_resting_object_height():
    m=L20Model();layout=JointLayout(m.names[::-1],m);n=2
    q=result(m)['qpos'];target=torch.tensor([q[x] for x in layout.names],dtype=torch.float32)
    ident=torch.tensor([0.,0.,0.,1.])
    root=torch.zeros(n*4,13);root[:,6]=1
    object_state=torch.zeros(n,13);object_state[:,6]=1
    data=dict(target_qpos=[target],target_hand_pos=[torch.tensor([.02,.03,.04])],target_hand_rot=[ident],
              object_euler_xy=[torch.zeros(2)],object_init_z=[torch.tensor([.08])])
    env=SimpleNamespace(randomize=False,num_shadow_hand_dofs=21,device='cpu',random_prior=True,
        reset_target_pose=lambda ids:None,object_code_list=['core/object'],object_id_buf=torch.zeros(n,dtype=torch.long),
        object_scale_buf={0:.1,1:.1},grasp_data={'core/object':{.1:data}},target_qpos=torch.zeros(n,21),
        target_hand_pos=torch.zeros(n,3),target_hand_rot=torch.zeros(n,4),object_init_euler_xy=torch.zeros(n,2),object_init_z=torch.zeros(n,1),
        reset_dof_pos_noise=.01,reset_dof_vel_noise=.02,independent_lower=torch.tensor(m.lower,dtype=torch.float32),
        independent_upper=torch.tensor(m.upper,dtype=torch.float32),mimic_matrix=torch.tensor(layout.matrix,dtype=torch.float32),mimic_offset=torch.tensor(layout.offset,dtype=torch.float32),
        shadow_hand_dof_pos=torch.zeros(n,21),shadow_hand_dof_vel=torch.zeros(n,21),prev_targets=torch.zeros(n,21),cur_targets=torch.zeros(n,21),
        hand_indices=torch.tensor([0,4]),object_indices=torch.tensor([1,5]),goal_object_indices=torch.tensor([2,6]),table_indices=torch.tensor([3,7]),
        dof_state=torch.zeros(n,21,2),sim=None,hand_positions=root[:,:3],hand_orientations=root[:,3:7],saved_root_tensor=root.clone(),
        hand_linvels=root[:,7:10],hand_angvels=root[:,10:13],root_state_tensor=root,object_init_state=object_state,
        goal_init_state=object_state.clone(),goal_states=object_state.clone(),goal_displacement=SimpleNamespace(z=.2),
        cfg={'env':dict(handStartClearance=.12)},z_theta=torch.zeros(n),random_time=False,
        progress_buf=torch.zeros(n),reset_buf=torch.ones(n),successes=torch.zeros(n))
    env.gym=SimpleNamespace(set_dof_state_tensor_indexed=lambda *a:None,set_dof_position_target_tensor_indexed=lambda *a:None,set_actor_root_state_tensor_indexed=lambda *a:None)
    namespace=dict(torch=torch,random=__import__('random'),tensor_clamp=lambda x,l,u:torch.maximum(torch.minimum(x,u),l),
        torch_rand_float=lambda a,b,size,device:a+(b-a)*torch.rand(size,device=device),gymtorch=SimpleNamespace(unwrap_tensor=lambda x:x),
        quat_from_euler_xyz=lambda a,b,c:ident.repeat(len(a),1),get_euler_xyz=lambda quat:(torch.zeros(len(quat)),)*3,
        quat_mul=lambda a,b:b,quat_apply=lambda q,v:v)
    method=task_method('reset',namespace);method(env,torch.arange(n),torch.arange(n))
    torch.testing.assert_close(root[env.object_indices,2],torch.full((n,),.68))
    torch.testing.assert_close(root[env.goal_object_indices,2],torch.full((n,),.88))
    torch.testing.assert_close(root[env.hand_indices,2],torch.full((n,),.84))
    for row in env.shadow_hand_dof_pos.numpy():m.validate(dict(zip(layout.names,row)),tolerance=1e-5)
    for child,(parent,mult,_) in m.mimic.items():
        torch.testing.assert_close(env.shadow_hand_dof_vel[:,layout.names.index(child)],env.shadow_hand_dof_vel[:,layout.names.index(parent)]*mult)

def test_shipped_real_priors_have_valid_model_frames_and_geometric_screening():
    paths=list((ROOT/'dexgrasp_policy_l20hand/assets_tiny_l20/dataset_l20_right_v2').rglob('*.npz'))
    assert len(paths)==2
    for path in paths:
        prior=load_prior(path)
        geometry=prior['metadata']['geometry']
        assert geometry['passed'] is True
        assert max(geometry[k] for k in ('object_penetration_m','table_penetration_m','self_penetration_m'))<=.002
        assert geometry['near_contact_fingers']>=2
        assert prior['metadata']['tip_error_m']<=.03
