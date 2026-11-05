"""Verify pinned assets and generate an importer-independent 21-DOF URDF."""
import hashlib
import json
from .model import L20Model, URDF_COMMIT, resolve_mesh, collision_contract

def main(rebuild=False):
    import trimesh
    model = L20Model()
    manifest = json.loads(model.path.with_name('manifest.json').read_text())
    if manifest['commit']!=URDF_COMMIT:
        raise ValueError('Asset manifest commit mismatch')
    for record in manifest['files']:
        path = model.path.parent/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Asset checksum mismatch: '+str(path))
    for mesh in model.root.findall('.//mesh'):
        resolve_mesh(mesh.get('filename'),model.path)
    runtime = model.path.with_name('linkerhand_l20_right_isaac.urdf')
    if not rebuild and model.path.with_name('collision-manifest.json').is_file():
        collision_contract(model)
        if runtime.is_file() and runtime.read_text()==model.runtime_urdf():
            print('Verified and reused the pinned prepared collision assets')
            return
    directory = model.path.parent/'collision'
    directory.mkdir(exist_ok=True)
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
            generated.append(dict(path=str(path.relative_to(model.path.parent)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    path = model.path.with_name('linkerhand_l20_right_isaac.urdf')
    path.write_text(model.runtime_urdf())
    model.path.with_name('collision-manifest.json').write_text(json.dumps(
        dict(method='convex_hull_per_link_v1',source_urdf_sha256=model.sha256,
             trimesh_version=trimesh.__version__,files=generated),indent=2)+'\n')
    print('Verified official URDF and meshes; generated '+str(path))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild',action='store_true',help='Regenerate hulls; existing prior/checkpoint collision hashes may change')
    main(parser.parse_args().rebuild)
