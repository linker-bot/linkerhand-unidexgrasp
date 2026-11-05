"""Deterministic sampled geometric screening, not force-closure validation."""
import hashlib
import numpy as np
from .model import L20Model, origin, collision_contract

def sample_vertices(mesh, count=256):
    points = np.concatenate((mesh.vertices,mesh.triangles_center))
    return points[np.linspace(0,len(points)-1,min(count,len(points)),dtype=int)]

class GeometryScreen:
    def __init__(self, object_path, scale, model=None, max_penetration=.002,
                 max_contact_distance=.015, min_contact_fingers=2):
        import trimesh
        self.trimesh = trimesh
        self.model = model or L20Model()
        self.collision_sha256 = collision_contract(self.model)
        self.object_path = object_path
        self.scale = float(scale)
        # CoACD OBJ components can share coordinates. Welding across hulls breaks
        # topology, so preserve indices and treat disconnected solids as a union.
        self.object = trimesh.load(str(object_path),force='mesh',process=False)
        self.object.apply_scale(scale)
        if not self.object.is_watertight:
            raise ValueError('Geometric screening requires a watertight object mesh')
        self.object_solids = list(self.object.split(only_watertight=False))
        if not self.object_solids or any(not part.is_watertight for part in self.object_solids):
            raise ValueError('Invalid decomposed object solids')
        self.object_points = sample_vertices(self.object,512)
        self.max_penetration = max_penetration
        self.max_contact_distance = max_contact_distance
        self.min_contact_fingers = min_contact_fingers
        self.links = {}
        for link in self.model.root.findall('link'):
            collision = link.find('collision')
            element = collision.find('geometry/mesh') if collision is not None else None
            if element is None:
                continue
            # Load precisely the same prepared collision file as IsaacGym.
            path = self.model.path.parent/'collision'/(link.get('name')+'.STL')
            mesh = trimesh.load(str(path),force='mesh',process=True)
            mesh.apply_scale(np.fromstring(element.get('scale','1 1 1'),sep=' '))
            mesh.apply_transform(origin(collision.find('origin')))
            if not mesh.is_watertight:
                raise ValueError('Non-watertight collision mesh: '+link.get('name'))
            self.links[link.get('name')] = mesh
        # Ignore pairs within two joints; CAD joint housings intentionally overlap.
        graph = {n:set() for n in self.model.links}
        for j in self.model.joints.values():
            a,b = j.find('parent').get('link'),j.find('child').get('link')
            graph[a].add(b); graph[b].add(a)
        self.ignore = {}
        for a in graph:
            near = {a} | graph[a]
            self.ignore[a] = near | set().union(*(graph[b] for b in graph[a]))

    def penetration(self, solid, points):
        lower,upper = solid.bounds
        inside_box = ((points>=lower)&(points<=upper)).all(axis=1)
        candidates = points[inside_box]
        if not len(candidates):
            return 0.
        contained = candidates[solid.contains(candidates)]
        if not len(contained):
            return 0.
        return float(self.trimesh.proximity.closest_point(solid,contained)[1].max())

    def check(self, result, plane):
        model = self.model
        model.validate(result['qpos'])
        fk = model.fk(result['qpos'])
        world = np.eye(4)
        world[:3,:3] = result['rotation']; world[:3,3] = result['translation']
        meshes = {}
        points = {}
        for n,mesh in self.links.items():
            meshes[n] = mesh.copy()
            meshes[n].apply_transform(world @ fk[n])
            points[n] = sample_vertices(meshes[n])
        plane = np.asarray(plane)
        table_pen = max(0.,-min(float((p @ plane[:3]+plane[3]).min()) for p in points.values()))
        object_pen = 0.
        contacts = 0
        for n,mesh in meshes.items():
            object_pen = max(object_pen,max(self.penetration(part,points[n]) for part in self.object_solids),self.penetration(mesh,self.object_points))
            if n.endswith('_distal'):
                distance = self.trimesh.proximity.closest_point(self.object,points[n])[1]
                contacts += int(float(distance.min()) <= self.max_contact_distance)
        self_pen = 0.
        names = list(meshes)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if b in self.ignore[a]:
                    continue
                if np.any(meshes[a].bounds[1]<meshes[b].bounds[0]) or np.any(meshes[b].bounds[1]<meshes[a].bounds[0]):
                    continue
                self_pen = max(self_pen,self.penetration(meshes[a],points[b]),self.penetration(meshes[b],points[a]))
        return dict(passed=max(object_pen,table_pen,self_pen)<=self.max_penetration and contacts>=self.min_contact_fingers,
                    method='sampled_bidirectional_mesh_v1',collision_model='convex_hull_per_link_v1',collision_sha256=self.collision_sha256,points_per_link=256,
                    object_penetration_m=object_pen,table_penetration_m=table_pen,
                    self_penetration_m=self_pen,near_contact_fingers=contacts,
                    max_penetration_m=self.max_penetration,max_contact_distance_m=self.max_contact_distance,
                    min_contact_fingers=self.min_contact_fingers,object_scale=self.scale,
                    object_mesh_sha256=hashlib.sha256(open(self.object_path,'rb').read()).hexdigest())

    def refine(self,result,plane,max_nfev=40):
        """Bounded local collision/contact refinement; final mesh screening decides.

        Convex halfspaces are fast optimization surrogates. The returned candidate
        is never considered valid until check() passes on the prepared meshes.
        """
        from scipy.optimize import least_squares
        from scipy.spatial import ConvexHull, cKDTree
        from scipy.spatial.transform import Rotation
        from .model import INDEPENDENT_NAMES, FINGERTIP_LINKS
        from .retarget import TIP_OFFSETS
        model=self.model
        names=list(self.links)
        local={n:sample_vertices(mesh,64) for n,mesh in self.links.items()}
        equations={n:ConvexHull(mesh.vertices).equations for n,mesh in self.links.items()}
        objects=[ConvexHull(mesh.vertices).equations for mesh in self.object_solids]
        bounds={id(equations[n]):mesh.bounds for n,mesh in self.links.items()}
        bounds.update({id(eq):mesh.bounds for eq,mesh in zip(objects,self.object_solids)})
        object_points=sample_vertices(self.object,256)
        nearest=cKDTree(sample_vertices(self.object,4096))
        pairs=[(a,b) for i,a in enumerate(names) for b in names[i+1:] if b not in self.ignore[a]]
        q0=np.array([result['qpos'][n] for n in INDEPENDENT_NAMES])
        lo=np.r_[[-.025]*3,[-.3]*3,np.maximum(model.lower,q0-.35)]
        hi=np.r_[[.025]*3,[.3]*3,np.minimum(model.upper,q0+.35)]
        x0=np.r_[np.zeros(6),np.clip(q0,lo[6:]+1e-8,hi[6:]-1e-8)]
        def decode(x):
            mapped=dict(result)
            mapped['qpos']=model.expand(x[6:],clip=True)
            mapped['translation']=result['translation']+x[:3]
            mapped['rotation']=result['rotation']@Rotation.from_rotvec(x[3:6]).as_matrix()
            return mapped
        def state(mapped):
            fk=model.fk(mapped['qpos']);r=mapped['rotation'];t=mapped['translation']
            transforms={n:(r@fk[n][:3,:3],r@fk[n][:3,3]+t) for n in names}
            points={n:local[n]@transforms[n][0].T+transforms[n][1] for n in names}
            return transforms,points
        initial_transforms,initial=state(result)
        def inside(eq,points):
            low,high=bounds[id(eq)]
            points=points[((points>=low)&(points<=high)).all(axis=1)]
            if not len(points):return 0.
            return max(0.,float((-np.max(points@eq[:,:3].T+eq[:,3],axis=1)).max()))
        def residual(x):
            mapped=decode(x);transforms,points=state(mapped)
            object_pen=[];table_pen=[];self_pen=[]
            for n in names:
                r,t=transforms[n]
                penetration=max(inside(eq,points[n]) for eq in objects)
                penetration=max(penetration,inside(equations[n],(object_points-t)@r))
                object_pen.append(max(0.,penetration-.0005))
                table_pen.append(max(0.,-float((points[n]@plane[:3]+plane[3]).min())-.0005))
            for a,b in pairs:
                ra,ta=transforms[a];rb,tb=transforms[b]
                penetration=max(inside(equations[a],(points[b]-ta)@ra),inside(equations[b],(points[a]-tb)@rb))
                self_pen.append(max(0.,penetration-.0005))
            # Attract the thumb and the closest other finger toward the object.
            distances=[float(nearest.query(points[n])[0].min()) for n in FINGERTIP_LINKS]
            contact=[max(0.,distances[-1]-.006),max(0.,min(distances[:-1])-.006)]
            drift=np.concatenate([(points[n]-initial[n]).ravel() for n in FINGERTIP_LINKS])
            return np.r_[.05*drift, .05*x[:3], .001*x[3:6],.0001*(x[6:]-q0),
                         8*np.array(object_pen),8*np.array(table_pen),6*np.array(self_pen),contact]
        best=least_squares(residual,x0,bounds=(lo,hi),max_nfev=max_nfev,
                           ftol=1e-6,xtol=1e-6,gtol=1e-6)
        refined=decode(best.x)
        final_transforms,points=state(refined)
        max_drift=max(float(np.linalg.norm(points[n]-initial[n],axis=1).max()) for n in FINGERTIP_LINKS)
        # Bound the original landmark error using the exact virtual tip points,
        # rather than treating a sampled surface displacement as a tip bound.
        tip_drift=max(float(np.linalg.norm(
            final_transforms[n][0]@offset+final_transforms[n][1] -
            initial_transforms[n][0]@offset-initial_transforms[n][1]))
            for n,offset in zip(FINGERTIP_LINKS,TIP_OFFSETS))
        refined['tip_error_m']=float(result['tip_error_m'])+tip_drift
        refined['refinement']=dict(method='bounded_convex_contact_v1',success=bool(best.success),
                                   max_distal_point_drift_m=max_drift,max_virtual_tip_drift_m=tip_drift,nfev=int(best.nfev))
        return refined

def screen_record(mapped,screen,refine=True,max_tip_error=.03):
    geometry=screen.check(mapped,mapped['plane'])
    if not geometry['passed'] and refine:
        initial=geometry
        mapped=screen.refine(mapped,mapped['plane'])
        geometry=screen.check(mapped,mapped['plane'])
        geometry['initial_screen']=initial
        geometry['refinement']=mapped['refinement']
    if mapped['tip_error_m']>max_tip_error:
        geometry['passed']=False
        geometry['tip_error_exceeds_tolerance']=True
    return mapped,geometry
