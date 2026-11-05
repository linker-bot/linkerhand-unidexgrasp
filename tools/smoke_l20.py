"""Run on Linux with the repository's IsaacGym environment, before PPO training."""
import argparse
import json
import os
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--prior-dataset',default='dataset_l20_right_v2')
    parser.add_argument('--num-envs',type=int,default=4)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--sim-device',default='cuda:0')
    parser.add_argument('--rl-device',default='cuda:0')
    parser.add_argument('--relative',action='store_true')
    parser.add_argument('--mimic-tolerance',type=float,default=.1)
    parser.add_argument('--output',type=Path,default=Path('l20-smoke-report.json'))
    options=parser.parse_args()
    output=options.output.resolve()
    assets=options.asset_root.resolve() if options.asset_root else None
    root=Path(__file__).resolve().parents[1]
    policy=root/'dexgrasp_policy_l20hand/dexgrasp'
    # IsaacGym requires its import before Torch.
    import isaacgym
    import torch
    sys.path[:0]=[str(root),str(policy)]
    os.chdir(policy)
    from utils.config import get_args,load_cfg,parse_sim_params,set_seed
    from utils.parse_task import parse_task
    from utils.process_marl import get_AgentIndex
    sys.argv=['smoke_l20','--task','ShadowHandGrasp','--algo','ppo',
        '--num_envs',str(options.num_envs),'--sim_device',options.sim_device,
        '--rl_device',options.rl_device,'--headless','--seed','7']
    args=get_args();cfg,training,_=load_cfg(args)
    cfg['env']['priorDataset']=options.prior_dataset
    cfg['env']['useRelativeControl']=options.relative
    cfg['task']['randomize']=False
    cfg['env']['random_time']=False
    if assets:cfg['env']['asset']['assetRoot']=str(assets)
    set_seed(7)
    sim_params=parse_sim_params(args,cfg,training)
    task,env=parse_task(args,cfg,training,sim_params,get_AgentIndex(cfg))
    report=dict(mode='relative' if options.relative else 'absolute',steps=options.steps,
        policy_contract=task.policy_contract,finite_observations=True,
        mimic_error_max_rad=0.,threshold_rad=options.mimic_tolerance,
        physics='IsaacGym',scope='load/reset/open-close; not a grasp success evaluation')
    try:
        observation=env.reset()
        for step in range(options.steps):
            actions=torch.zeros((env.num_envs,22),device=env.rl_device)
            # Move independent joints through a bounded open/close cycle.
            actions[:,6:]=(.15 if step<options.steps//2 else -.15) if options.relative else (-.8 if step<options.steps//2 else .2)
            observation,_,_,_=env.step(actions)
            report['finite_observations'] &= bool(torch.isfinite(observation).all().item())
            report['mimic_error_max_rad']=max(report['mimic_error_max_rad'],float(task.mimic_error.max().item()))
        report['passed']=report['finite_observations'] and report['mimic_error_max_rad']<=options.mimic_tolerance
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        if not report['passed']:raise SystemExit('Smoke check failed; inspect '+str(output))
    finally:
        if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)

if __name__=='__main__':
    main()
