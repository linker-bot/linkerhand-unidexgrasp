import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .export import read_result, retarget_record, numpy
from .priors import load_prior
from .frames import object_pose

def main(argv=None):
    parser = argparse.ArgumentParser(description='Preview the same official pose used by export')
    parser.add_argument('--exp_dir',type=Path,default=Path('eval'))
    parser.add_argument('--num',type=int,default=0)
    parser.add_argument('--prior',type=Path,help='Display a saved v2 prior instead of recomputing')
    parser.add_argument('--mesh-root',type=Path,help='Source DFCData/meshes for recovery of old object poses')
    parser.add_argument('--show-shadow',action='store_true')
    parser.add_argument('--canonical_frame',type=int,choices=(0,1),default=0)
    parser.add_argument('--html',type=Path)
    args = parser.parse_args(argv)
    import plotly.graph_objects as go
    generation = Path(__file__).resolve().parents[1]/'dexgrasp_generation_forlinker'
    sys.path.insert(0,str(generation))
    from datasets.l20_hand_builder import L20HandBuilder
    result = read_result(args.exp_dir/'result.pt')
    mapped = load_prior(args.prior,require_geometry=False) if args.prior else retarget_record(result,args.num,mesh_root=args.mesh_root)
    if args.prior and mapped['metadata']['source_index']!=args.num:
        raise ValueError('--num does not match the saved prior source_index')
    if not args.prior and args.mesh_root:
        from .geometry import GeometryScreen, screen_record
        screen=GeometryScreen(args.mesh_root/str(result['object_code'][args.num])/'coacd/decomposed.obj',float(numpy(result['scale'][args.num]).item()))
        mapped,geometry=screen_record(mapped,screen)
        print('Geometric screening passed:',geometry['passed'])
    pose = object_pose(result,args.num,args.mesh_root)
    rotation = pose[:3,:3] @ mapped['rotation']
    translation = pose[:3,:3] @ mapped['translation'] + pose[:3,3]
    frame = numpy(result['sampled_rotation'][args.num]).T if args.canonical_frame else np.eye(3)
    hand = L20HandBuilder().get_hand_mesh(frame @ rotation,frame @ translation,hand_qpos_dict=mapped['qpos'])
    def plot(mesh,color):
        vertices = mesh.verts_list()[0].numpy(); faces = mesh.faces_list()[0].numpy()
        return go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
                         i=faces[:,0],j=faces[:,1],k=faces[:,2],color=color)
    points = numpy(result['obj_pc'][args.num]) @ frame.T
    plots = [go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',marker=dict(size=2)),plot(hand,'lightblue')]
    if args.show_shadow:
        import os
        from datasets.shadow_hand_builder import ShadowHandBuilder
        pose = numpy(result['tta_hand_pose'][args.num])
        previous = Path.cwd()
        try:
            os.chdir(generation)
            shadow = ShadowHandBuilder().get_hand_mesh(frame @ Rotation.from_rotvec(pose[3:6]).as_matrix(),frame @ pose[:3],qpos=pose[6:])
        finally:
            os.chdir(previous)
        plots.append(plot(shadow,'lightgrey'))
    figure = go.Figure(plots)
    figure.update_layout(scene_aspectmode='data',title='Official L20 right hand · shared export mapping')
    if args.html:
        figure.write_html(str(args.html),include_plotlyjs=True)
    else:
        figure.show()

if __name__=='__main__':
    main()
