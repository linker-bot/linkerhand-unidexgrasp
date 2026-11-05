"""Convert trusted UniDexGrasp result.pt candidates to named official priors."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .retarget import Retargeter
from .priors import save_prior
from .frames import object_pose, to_object_frame

def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

def read_result(path):
    import torch
    # result.pt comes from this repository's local generator; never use untrusted pickle files.
    try:
        data = torch.load(str(path),map_location='cpu',weights_only=False)
    except TypeError:  # Torch 1.x
        data = torch.load(str(path),map_location='cpu')
    if isinstance(data,list):
        if not data:
            raise ValueError('Empty generation result')
        # eval.py writes a list of complete batch dictionaries.
        keys = data[0].keys()
        merged = {}
        for key in keys:
            values = [batch[key] for batch in data]
            if hasattr(values[0],'shape'):
                merged[key] = np.concatenate([numpy(v) for v in values],axis=0)
            elif isinstance(values[0],(list,tuple)):
                merged[key] = [item for batch in values for item in batch]
            else:
                raise ValueError('Unsupported batched result field: '+key)
        return merged
    return data

def retarget_record(result,index,retargeter=None,max_tip_error=.03,mesh_root=None):
    pose = numpy(result['tta_hand_pose'][index])
    if pose.shape != (28,) or not np.isfinite(pose).all():
        raise ValueError('Expected translation(3), axis-angle(3), Shadow joints(22)')
    scene_result = (retargeter or Retargeter()).from_shadow(
        pose[6:],pose[:3],Rotation.from_rotvec(pose[3:6]).as_matrix(),max_tip_error)
    return to_object_frame(scene_result,numpy(result['plane'][index]),object_pose(result,index,mesh_root))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mesh-root',type=Path,help='DFCData/meshes directory containing core/...')
    parser.add_argument('--object-prefix',default='',help='Destination namespace, e.g. core for bundled bare object codes')
    parser.add_argument('--kinematics-only',action='store_true',help='Export unscreened candidates; policy rejects these by default')
    parser.add_argument('--max-tip-error',type=float,default=.03)
    parser.add_argument('--no-refine',action='store_true',help='Screen the IK candidate without local collision refinement')
    args = parser.parse_args(argv)
    if not args.kinematics_only and args.mesh_root is None:
        parser.error('--mesh-root is required for geometric screening')
    result = read_result(args.input)
    retargeter = Retargeter()
    report = dict(input=str(args.input),accepted=0,rejected=[],total=len(result['object_code']))
    screens = {}
    args.output.mkdir(parents=True,exist_ok=True)
    root = args.output.resolve()
    for index,source_code in enumerate(result['object_code']):
        source_code = str(source_code)
        relative = Path(source_code)
        if Path(args.object_prefix).is_absolute() or '..' in Path(args.object_prefix).parts:
            raise ValueError('Invalid object prefix')
        code = str(Path(args.object_prefix)/relative)
        print('Processing %d/%d: %s' % (index+1,report['total'],code),file=sys.stderr,flush=True)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid object code: '+code)
        try:
            mapped = retarget_record(result,index,retargeter,args.max_tip_error,args.mesh_root)
            scale = float(numpy(result['scale'][index]).item())
            plane = mapped['plane']
            geometry = dict(passed=False,method='not_screened')
            if not args.kinematics_only:
                from .geometry import GeometryScreen, screen_record
                key = (code,scale)
                if key not in screens:
                    screens[key] = GeometryScreen(args.mesh_root/relative/'coacd/decomposed.obj',scale,retargeter.model)
                    # Full training sets contain thousands of objects; do not
                    # retain one mesh/r-tree collection for every object/scale.
                    if len(screens)>4:
                        del screens[next(iter(screens))]
                mapped,geometry = screen_record(mapped,screens[key],not args.no_refine,args.max_tip_error)
                if not geometry['passed']:
                    report['rejected'].append(dict(index=index,object_code=code,reason='geometry',details=geometry))
                    continue
            destination = root/code/('%08d.npz' % index)
            if root not in destination.resolve().parents:
                raise ValueError('Output path escapes dataset root')
            save_prior(destination,mapped,scale,plane,index,geometry,retargeter.model)
            report['accepted'] += 1
        except (ValueError,FileNotFoundError) as exc:
            report['rejected'].append(dict(index=index,object_code=code,reason=str(exc)))
    (args.output/'export-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['accepted']==0:
        raise SystemExit('No accepted priors; inspect export-report.json')

if __name__=='__main__':
    main()
