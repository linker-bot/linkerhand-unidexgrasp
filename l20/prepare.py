"""Verify pinned assets and generate an importer-independent 21-DOF URDF."""
import hashlib
import json
from .model import L20Model, resolve_mesh, collision_contract
from .source import verify_official_source

def main(rebuild=False):
    verify_official_source(check_revision=True)
    import trimesh
    model = L20Model()
    for mesh in model.root.findall('.//mesh'):
        resolve_mesh(mesh.get('filename'),model.path)
    runtime = model.asset_dir/'linkerhand_l20_right_isaac.urdf'
    if not rebuild and (model.asset_dir/'collision-manifest.json').is_file():
        collision_contract(model)
        expected = model.runtime_urdf()
        if not runtime.is_file() or runtime.read_text()!=expected:
            runtime.write_text(expected)
        print('Verified upstream submodule and reused the prepared collision assets')
        return
    directory = model.asset_dir/'collision'
    directory.mkdir(parents=True,exist_ok=True)
    generated = []
    for link in model.root.findall('link'):
        for collision in link.findall('collision'):
            element = collision.find('geometry/mesh')
            if element is None:
                continue
            mesh = trimesh.load(str(resolve_mesh(element.get('filename'),model.path)),force='mesh',process=True)
            hull = mesh.convex_hull
            path = directory/(link.get('name')+'.STL')
            hull.export(str(path),file_type='stl')
            generated.append(dict(path=str(path.relative_to(model.asset_dir)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    path = runtime
    path.write_text(model.runtime_urdf())
    (model.asset_dir/'collision-manifest.json').write_text(json.dumps(
        dict(method='convex_hull_per_link_v1',source_urdf_sha256=model.sha256,
             trimesh_version=trimesh.__version__,files=generated),indent=2)+'\n')
    print('Verified official URDF and meshes; generated '+str(path))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild',action='store_true',help='Regenerate hulls; existing prior/checkpoint collision hashes may change')
    main(parser.parse_args().rebuild)
