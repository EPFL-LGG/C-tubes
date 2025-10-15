import torch, numpy as np
from Ctubes.misc_utils import write_polyline_to_obj, write_mesh_to_obj
import matplotlib.pyplot as plt
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from Ctubes.geometry_utils import compute_plane_normals, compute_tangents, angle_around_axis, rotate_about_axis, triangulate_polygon, compute_quad_nonplanarity
from Ctubes.tube_generation import compute_ctube, compute_ctube_topology, compute_ctube_topology_tri, compute_unrolled_strips
from Ctubes.plot_utils import plot_tube, plot_tube_3d, plot_unrolled_strips

class Directrix:
    """
    A 3D curve represented as a natural cubic spline defined by K control points.

    Attributes:
        Q (torch.Tensor): Control points of the spline, shape (K, 3).
        M (int): Number of discretization points along the curve.
        closed_curve (bool): Whether the curve is closed.
        X (torch.Tensor): Discretized points along the curve, shape (M, 3).
        tangents (torch.Tensor): Tangent vectors at each discretized point, shape (M, 3).
    """
    def __init__(self, Q, M, cache_tangents=False, symmetry_transforms=None):
        self.Q = Q
        self.K = Q.shape[0]
        self.M = M
        self.closed_curve = self.first_last_points_match()
        self.cache_tangents = cache_tangents
        self.symmetry_transforms = symmetry_transforms

        self.X = self.sample_vertices_with_symmetry()[0:self.M]
        if self.cache_tangents:
            self.tangents = compute_tangents(self.X)
    
    def first_last_points_match(self):
        return torch.allclose(self.Q[0], self.Q[-1])

    def has_symmetry(self):
        return self.symmetry_transforms is not None and len(self.symmetry_transforms) > 0

    def set_control_points(self, Q):
        assert Q.shape[0] == self.K, "The number of control points ({}) must match the original number ({})".format(Q.shape[0], self.K)
        self.Q = Q
        if self.closed_curve:
            assert self.first_last_points_match(), "The first and last control points must match for a closed curve."
        self.X = self.sample_vertices_with_symmetry()[0:self.M]
        if self.cache_tangents:
            self.tangents = compute_tangents(self.X)

    def sample_vertices(self):
        "Sample the spline defined by the control points with self.M points."
        ts_cp = torch.linspace(0.0, 1.0, self.K)
        spline_coeffs = natural_cubic_spline_coeffs(ts_cp, self.Q, close_spline=self.closed_curve)
        spline = NaturalCubicSpline(spline_coeffs)
        ts_disc = torch.linspace(0.0, 1.0, self.M)
        X = spline.evaluate(ts_disc)
        return X
    
    def sample_vertices_with_symmetry(self):
        if not self.has_symmetry():
            return self.sample_vertices()
        else:
            n_symm = len(self.symmetry_transforms)
            K_symm = (self.K - 1) * (n_symm + 1) + 1
            M_symm = (self.M - 1) * (n_symm + 1) + 1

            cps_symm = torch.zeros(K_symm, 3)
            cps_symm[0:self.K] = self.Q
            for si, symmetry_transform in enumerate(self.symmetry_transforms):
                cps_new = symmetry_transform(self.Q)
                assert torch.allclose(cps_new[0], cps_symm[(si + 1) * (self.K - 1)], atol=1e-10), "Control points {} and {} not matching after applying symmetry transformation #{}.".format(cps_new[0], cps_symm[(si + 1) * (self.K - 1)], si)
                cps_symm[(si + 1) * (self.K - 1):(si + 2) * (self.K - 1) + 1] = cps_new

            closed_curve_symm = torch.allclose(cps_symm[0], cps_symm[-1])  # check if the control points closed up after applying the last symmetry transformation

            # Sample the spline defined by the control points with M points
            ts_cp_symm = torch.linspace(0.0, 1.0, K_symm)
            spline_coeffs_symm = natural_cubic_spline_coeffs(ts_cp_symm, cps_symm, close_spline=closed_curve_symm)
            spline_symm = NaturalCubicSpline(spline_coeffs_symm)
            ts_disc_symm = torch.linspace(0.0, 1.0, M_symm)
            X_symm = spline_symm.evaluate(ts_disc_symm)
            return X_symm

    def get_control_points(self):
        return self.Q

    def get_vertices(self):
        return self.X
    
    def get_vertices_with_symmetry(self):
        return self.sample_vertices_with_symmetry()
    
    def get_tangents(self):
        if self.cache_tangents:
            return self.tangents
        else:
            return compute_tangents(self.X)
        
    def aabb_diagonal_length(self):
        "Axis-aligned bounding box diagonal length of the curve."
        min_coords = torch.min(self.X, dim=0).values
        max_coords = torch.max(self.X, dim=0).values
        diag_length = torch.linalg.norm(max_coords - min_coords)
        return diag_length

    def clone(self):
        """Return a deep copy of the Directrix instance."""
        Q_clone = self.Q.clone()
        symmetry_transforms_clone = self.symmetry_transforms[:] if self.symmetry_transforms is not None else None
        return Directrix(
            Q_clone,
            self.M,
            cache_tangents=self.cache_tangents,
            symmetry_transforms=symmetry_transforms_clone
        )
    

class Generatrix:
    """
    A closed 3D polygon representing the cross-section of the tube.
    The polygon is defined in a local 2D coordinate system and then mapped to 3D
    using a specified origin, normal vector, and rotation angle (theta).

    Attributes:
        G0_2d (torch.Tensor): Vertices of the polygon in the xy plane, shape (N, 2).
        origin_3d (torch.Tensor): 3D point corresponding to the origin (0, 0) in the xy plane, shape (3,).
        normal_3d (torch.Tensor): Normal vector of the polygon in 3D, corresponding to the z-axis in 2D, shape (3,).
        theta (torch.Tensor): Rotation angle about the z-axis in radians.
        init_scale (torch.Tensor): Initial scaling factor for the cross-section.
        G0 (torch.Tensor): Vertices of the polygon in 3D, shape (N, 3).
        N (int): Number of vertices in the polygon.
    """
    def __init__(self, G0_2d, origin_3d, normal_3d, theta=torch.tensor(0.0), init_scale=torch.tensor(1.0)):
        self.G0_2d = G0_2d
        self.N = G0_2d.shape[0]
        self.origin_3d = origin_3d
        self.normal_3d = normal_3d
        self.theta = theta
        self.init_scale = init_scale
        self.G0 = self.map_2d_to_3d(G0_2d, origin_3d, normal_3d, theta, init_scale)  # map the 2D polygon to 3D

    def map_2d_to_3d(self, G0_2d, origin_3d, normal_3d, theta, init_scale):
        if torch.allclose(normal_3d, torch.tensor([0.0, 0.0, 1.0]), atol=1e-10):
            rot_xyz_to_frame = torch.eye(3, dtype=normal_3d.dtype)  # no rotation needed
        else:
            e1 = torch.cross(normal_3d, torch.tensor([0.0, 0.0, 1.0]), dim=0)
            e1 = e1 / torch.linalg.norm(e1)
            e2 = torch.cross(normal_3d, e1, dim=0)
            rot_xyz_to_frame = torch.stack([e1, e2, normal_3d], dim=1)

        G0_2d_scaled = torch.zeros(size=(self.N, 3))
        G0_2d_scaled[:, :2] = G0_2d * init_scale

        G0_2d_rotated = rotate_about_axis(G0_2d_scaled, torch.tensor([0.0, 0.0, 1.0]), theta)
        G0 = origin_3d.unsqueeze(0) + G0_2d_rotated @ rot_xyz_to_frame.T
        return G0
    
    def set_theta(self, theta):
        self.theta = theta
        self.G0 = self.map_2d_to_3d(self.G0_2d, self.origin_3d, self.normal_3d, theta, self.init_scale)

    def get_theta(self):
        return self.theta

    def set_init_scale(self, init_scale):
        self.init_scale = init_scale
        self.G0 = self.map_2d_to_3d(self.G0_2d, self.origin_3d, self.normal_3d, self.theta, init_scale)

    def get_init_scale(self):
        return self.init_scale

    def get_vertices(self):
        return self.G0

    def aabb_diagonal_length(self):
        "Axis-aligned bounding box diagonal length of the cross-section."
        min_coords = torch.min(self.G0_2d, dim=0).values
        max_coords = torch.max(self.G0_2d, dim=0).values
        diag_length = torch.linalg.norm(max_coords - min_coords)
        return diag_length

class CTube:
    """
    A C-tube defined by a directrix polyline and a generatrix polygon.
    Optionally, the class constructor can take:
        1) custom plane normals; by default, they are computed as the averaged vertex tangents of the directrix
        2) an apex locating function; by default, it is set to 1 everywhere (cylindrical tube)

    Attributes:
        directrix (Directrix): The centerline curve of the tube.
        generatrix (Generatrix): The cross-section polygon of the tube.
        plane_normals (torch.Tensor): Normals of the planes containing the cross-sections, shape (M, 3) or (M, N, 3).
        apex_loc_func (torch.Tensor): Apex locating function, shape (M-1,).
        dihedral_angles (torch.Tensor): Dihedral angles between adjacent cross-sections, shape (M-1, N).
        plane_normal_kind (str): Method to compute plane normals if not provided.
        symmetry_transforms (list): List of symmetry transformations applied to the tube.
        M (int): Number of discretization points along the directrix.
        N (int): Number of vertices in the cross-section polygon.
    """
    def __init__(self, directrix, generatrix, plane_normals=None, apex_loc_func=None, dihedral_angles=None, plane_normal_kind=None, symmetry_transforms=None):

        self.directrix = directrix
        self.generatrix = generatrix
        self.M = directrix.M
        self.K = directrix.K
        self.N = generatrix.N

        # Default values
        if plane_normals is None:
            plane_normals = compute_plane_normals(directrix.X, kind='bisecting', closed_curve=directrix.closed_curve)
        if apex_loc_func is None:
            apex_loc_func = torch.ones(self.M - 1)

        self.apex_loc_func = apex_loc_func
        self.plane_normals = plane_normals
        self.dihedral_angles = dihedral_angles
        self.plane_normal_kind = plane_normal_kind

        assert symmetry_transforms == directrix.symmetry_transforms, "The symmetry transforms of the tube must match those of the directrix."
        self.symmetry_transforms = symmetry_transforms

        self.directrix_ref = self.directrix.clone()  # reference curve for feature preservation objectives

        if dihedral_angles is not None:
            assert dihedral_angles.shape == (self.M-1, self.N), "The dihedral angles must have shape (M-1, N)."
            assert apex_loc_func is None, "Using dihedral angles or apex locating function is mutually exclusive."
            assert not self.has_symmetry(), "Dihedral angles are currently not supported for symmetric tubes."

        self.validate_plane_normals(plane_normals)
        if plane_normals.ndimension() == 3:
            self.planar_profile = False
            assert dihedral_angles is None, "Using dihedral angles is mutually exclusive with a list of plane normals."
            assert not self.has_symmetry(), "Using a list of plane normals is currently not supported for symmetric tubes."
            assert plane_normal_kind is None, "Using a list of plane normals is mutually exclusive with specifying a plane normal kind."
        elif plane_normals.ndimension() == 2:
            self.planar_profile = True

        if plane_normal_kind is not None:
            self.update_plane_normals(kind=plane_normal_kind)
            assert torch.allclose(plane_normals, self.plane_normals), "Using {} plane normals, but the provided plane normals do not match. Which one should be used?".format(plane_normal_kind)

        self.ctube_vertices = self.compute_vertices()
        self.ctube_vertices_ref = self.ctube_vertices.clone()  # reference swept vertices for feature preservation objectives

    def validate_plane_normals(self, plane_normals):
        if isinstance(plane_normals, torch.Tensor):
            if plane_normals.ndimension() == 2:
                assert plane_normals.shape == (self.M, 3), "The plane normals must have shape (M, 3) but have shape {}.".format(plane_normals.shape)
            elif plane_normals.ndimension() == 3:
                assert plane_normals.shape == (self.M, self.N, 3), "The plane normals must have shape (M, N, 3) but have shape {}.".format(plane_normals.shape)
            else:
                raise ValueError("The plane normals must have shape (M, 3) or (M, N, 3) but have shape {}.".format(plane_normals.shape))
        else:
            raise ValueError("plane_normals must be a tensor.")

    def validate_plane_normals_with_symmetry(self, plane_normals):
        n0_transformed = self.symmetry_transforms[0](plane_normals[0].reshape(1, 3)).reshape(3)
        assert torch.allclose(n0_transformed, plane_normals[-1]), "The first plane normal transformed by the first symmetry transformation must match the last plane normal."

    def has_symmetry(self):
        return self.symmetry_transforms is not None and len(self.symmetry_transforms) > 0
    
    def has_planar_profile(self):
        return self.planar_profile
    
    def use_dihedral_angles(self):
        return self.dihedral_angles is not None
    
    def reference_length(self):
        '''Reference length of the spine curve as it was initialized (directrix_ref).'''
        return torch.sum(torch.linalg.norm(self.directrix_ref.X[1:] - self.directrix_ref.X[:-1], dim=1))
    
    def M_with_symmetry(self):
        if not self.has_symmetry():
            return self.M
        return (self.M - 1) * (len(self.symmetry_transforms) + 1) + 1

    def get_polyline(self):
        return self.directrix.get_vertices()
        
    def get_polyline_with_symmetry(self):
        return self.directrix.get_vertices_with_symmetry()
    
    def compute_vertices(self):
        "Compute the swept vertices of the tube"
        ctube_vertices = compute_ctube(self.directrix.X, self.generatrix.G0, plane_normals=self.plane_normals, apex_loc_func=self.apex_loc_func, dihedral_angles=self.dihedral_angles, closed_curve=self.directrix.closed_curve)
        return ctube_vertices

    def compute_vertices_with_symmetry(self):
        if not self.has_symmetry():
            return self.compute_vertices()
        else:
            assert not self.use_dihedral_angles(), "Dihedral angles are currently not supported for symmetric tubes."
            cps_symm, plane_normals_symm, apex_loc_func_symm = self.compute_dofs_with_symmetry()
            directrix_symm = self.get_polyline_with_symmetry()

            closed_curve_symm = torch.allclose(cps_symm[0], cps_symm[-1])  # check if the control points closed up after applying the last symmetry transformation
            ctube_vertices_symm = compute_ctube(directrix_symm, self.generatrix.G0, plane_normals=plane_normals_symm, apex_loc_func=apex_loc_func_symm, closed_curve=closed_curve_symm)
            return ctube_vertices_symm
        
    def compute_dofs_with_symmetry(self):
        'Return symmetrized instances of the tube parameters cps_symm, plane_normals_symm, apex_loc_func_symm.'
        if not self.has_symmetry():
            return self.cps, self.plane_normals, self.apex_loc_func
        
        cps = self.get_control_points()
        apex_loc_func = self.get_apex_loc_func()
        plane_normals = self.get_plane_normals()

        n_symm = len(self.symmetry_transforms)
        K_symm = (self.K - 1) * (n_symm + 1) + 1
        M_symm = (self.M - 1) * (n_symm + 1) + 1

        cps_symm = torch.zeros(K_symm, 3)
        apex_loc_func_symm = torch.zeros(M_symm - 1)
        plane_normals_symm = torch.zeros(M_symm, 3)

        cps_symm[0:self.K] = cps
        apex_loc_func_symm[0:self.M - 1] = apex_loc_func
        plane_normals_symm[0:self.M] = plane_normals
        for si, symmetry_transform in enumerate(self.symmetry_transforms):

            cps_new = symmetry_transform(cps)

            det = torch.linalg.det(symmetry_transform(torch.eye(3)))  # +1 for rotations, -1 for reflections
            plane_normals_new = det * symmetry_transform(plane_normals)
                    
            cps_symm[(si + 1) * (self.K - 1):(si + 2) * (self.K - 1) + 1] = cps_new
            plane_normals_symm[(si + 1) * (self.M - 1):(si + 2) * (self.M - 1) + 1] = plane_normals_new
            apex_loc_func_symm[(si + 1) * (self.M - 1):(si + 2) * (self.M - 1)] = apex_loc_func
        
        return cps_symm, plane_normals_symm, apex_loc_func_symm

    def compute_aabb(self):
        aabb_min = torch.min(self.ctube_vertices.reshape(-1, 3), dim=0).values
        aabb_max = torch.max(self.ctube_vertices.reshape(-1, 3), dim=0).values
        return aabb_min, aabb_max
    
    def aabb_diagonal_length(self):
        aabb_min, aabb_max = self.compute_aabb()
        return torch.linalg.norm(aabb_max - aabb_min)
    
    def generatrix_aabb_diagonal_length(self):
        aabb_min = torch.min(self.generatrix.reshape(-1, 2), dim=0).values
        aabb_max = torch.max(self.generatrix.reshape(-1, 2), dim=0).values
        return torch.linalg.norm(aabb_max - aabb_min)

    def compute_dihedral_angles(self):
        if self.use_dihedral_angles():
            return self.dihedral_angles
        else:  # if, e.g., we are using the apex locating function parameterization then dihedral angles are not readily available but can be computed
            dihedral_angles = torch.zeros((self.M-1, self.N))
            for i in range(self.M-1):
                n_cs = self.plane_normals[i]
                for csi in range(self.N):
                    csip1 = (csi + 1) % self.N
                    e0_quad = self.ctube_vertices[i, csip1] - self.ctube_vertices[i, csi]
                    e0_quad /= torch.linalg.norm(e0_quad)
                    e1_quad = self.ctube_vertices[i+1, csi] - self.ctube_vertices[i, csi]
                    e1_quad /= torch.linalg.norm(e1_quad)
                    n_quad = torch.cross(e0_quad, e1_quad, dim=0)
                    n_quad /= torch.linalg.norm(n_quad)
                    dihedral_angles[i, csi] = angle_around_axis(e0_quad, n_cs, n_quad)
            return dihedral_angles

    def set_control_points(self, Q):
        self.directrix.set_control_points(Q)
        self.generatrix.origin_3d = self.directrix.X[0]
        self.generatrix.normal_3d = self.directrix.get_tangents()[0]
        if self.plane_normal_kind is not None:
            self.update_plane_normals(kind=self.plane_normal_kind)

    def set_theta(self, theta):
        self.generatrix.set_theta(theta)

    def set_apex_loc_func(self, apex_loc_func):
        assert not self.use_dihedral_angles(), "Using apex locating function is mutually exclusive with dihedral angles."
        self.apex_loc_func = apex_loc_func

    def set_plane_normals(self, plane_normals):
        if self.plane_normal_kind is not None:
            self.update_plane_normals(kind=self.plane_normal_kind)
            assert torch.allclose(plane_normals, self.plane_normals), "Using {} plane normals, but the provided plane normals do not match. Which one should be used?".format(self.plane_normal_kind)
        else:
            if self.has_symmetry():
                self.validate_plane_normals_with_symmetry(plane_normals)
            else:
                self.validate_plane_normals(plane_normals)
            self.plane_normals = plane_normals
    
    def set_init_scale(self, init_scale):
        self.generatrix.set_init_scale(init_scale)

    def set_dihedral_angles(self, dihedral_angles):
        assert self.apex_loc_func is None, "Using dihedral angles is mutually exclusive with apex locating function."
        self.dihedral_angles = dihedral_angles

    def update_ctube_vertices(self):
        self.ctube_vertices = self.compute_vertices()

    def get_control_points(self):
        return self.directrix.get_control_points()
    
    def get_theta(self):
        return self.generatrix.get_theta()

    def get_apex_loc_func(self):
        return self.apex_loc_func
    
    def get_plane_normals(self):
        return self.plane_normals
    
    def get_plane_normals_with_symmetry(self):
        if not self.has_symmetry():
            return self.plane_normals
        else:
            plane_normals_symm = self.plane_normals
            for si, symmetry_transform in enumerate(self.symmetry_transforms):
                plane_normals_new = symmetry_transform(self.plane_normals)
                plane_normals_symm[(si + 1) * (self.M - 1):(si + 2) * (self.M - 1) + 1] = plane_normals_new
            return plane_normals_symm
    
    def get_init_scale(self):
        return self.generatrix.get_init_scale()

    def get_dihedral_angles(self):
        assert self.use_dihedral_angles(), "Dihedral angles are not available. Use compute_dihedral_angles() to compute them for the current tube."
        return self.dihedral_angles

    def get_geometry(self):

        # Topology
        quads, end_cap_poly = compute_ctube_topology(self.M, self.N)

        # 3D Mesh and 2D unrolled strips
        V, F = self.get_mesh(triangulate=False, with_symmetry=True, end_caps=True, split_strips=True, global_face_indices=False)
        ctube_vertices = self.compute_vertices_with_symmetry() if self.has_symmetry() else self.compute_vertices()
        V_strips, F_strips = compute_unrolled_strips(ctube_vertices, triangulate=False, global_face_indices=False, y_offset_total=None, y_offset_per_strip=None, selected_strips=None, axes_first_edges=None, points_first_nodes=None)

        # Convert arrays and tensors to lists
        V = [v.tolist() for v in V]
        F = [f.tolist() for f in F]
        V_strips = [v.tolist() for v in V_strips]
        F_strips = [f.tolist() for f in F_strips]

        if isinstance(self.plane_normals, list):
            plane_normals_list = [n.detach().numpy().tolist() for n in self.plane_normals]
        elif isinstance(self.plane_normals, torch.Tensor):
            plane_normals_list = self.plane_normals.detach().numpy().tolist()

        geo_dict = {
            # Opt vars
            'cps': self.directrix.Q.detach().numpy().tolist(),
            'theta': self.generatrix.theta.item(),
            'init_scale': 1.0 if self.generatrix.init_scale is None else self.generatrix.init_scale.item(), # 1 by default
            'apex_loc_func': self.apex_loc_func.detach().numpy().tolist(),
            'plane_normals': plane_normals_list,

            # Geometry
            'generatrix': self.generatrix.G0.detach().numpy().tolist(),
            'directrix': self.get_polyline_with_symmetry().detach().numpy().tolist(),  # '_with_symmetry' falls back to the base curve if no symmetry is present
            'ctube_vertices': self.compute_vertices_with_symmetry().detach().numpy().tolist(),  # '_with_symmetry' falls back to the base curve if no symmetry is present
            'planar_profile': self.planar_profile,
            'plane_normal_kind': self.plane_normal_kind,
            'quad_nonplanarity': compute_quad_nonplanarity(self.ctube_vertices).detach().numpy().tolist(),

            # Topology
            'K': self.K,
            'M': self.M,
            'N': self.N,
            'quads': quads.detach().numpy().tolist(),
            'end_cap_poly': end_cap_poly.detach().numpy().tolist(),
            'closed_curve': self.directrix.closed_curve,

            # Targets
            'directrix_ref': self.directrix_ref.X.detach().numpy().tolist(),
            'ctube_vertices_ref': self.ctube_vertices_ref.detach().numpy().tolist(),

            # 3D mesh, with individual strips and end caps
            'mesh': {
                'vertices': V[:-2],  # N lists of 2*M vertices; each vertex will appear twice in this field, once for each strip
                'quads': F[:-2],  # quads indices, numbered locally for each strip (i.e. the first vertex of the second strip is again 0, not 2*M+1, etc.)
                'end_cap_vertices': V[-2:],  # end cap vertices; these are yet another copy of the vertices of the first and last cross-sections
                'end_cap_tris': F[-2:],  # end cap triangles indices, numbered locally for each end cap
            },

            # 2D unrolled strips
            'strips': {
                'vertices': V_strips, # N lists of 2*M vertices
                'faces': F_strips, # face indices, numbered locally for each strip (i.e. the first vertex of the second strip is again 0, not 2*M+1, etc.)
            }
        }
        return geo_dict
    
    def save_json(self, file_path):
        import json
        geo_dict = self.get_geometry()
        with open(file_path, 'w') as f:
            json.dump(geo_dict, f, indent=4)
    
    def get_mesh(self, triangulate=False, with_symmetry=True, end_caps=False, split_strips=False, global_face_indices=True):
        '''Get the mesh of the tube as a list of vertices and faces. 
        If triangulate is True, the faces are triangulated. 
        If end_caps is True, the end caps are included as separate lists of faces. 
        If split_strips is True, the vertices are split into separate lists for each strip. 
        If global_face_indices is True, the face indices are numbered globally for the entire mesh (i.e. the numbering does not restart for each strip).
        '''
        if self.has_symmetry() and with_symmetry:
            ctube_vertices = self.compute_vertices_with_symmetry()
        else:
            ctube_vertices = self.compute_vertices()
        M = ctube_vertices.shape[0]   # accounts for possible symmetry
        N = ctube_vertices.shape[1]

        if split_strips:  # all vertices appear twice, once for each strip
            assert not triangulate, "Triangulation is not supported for split strips."
            strips = []
            faces = []
            face_offset = 0
            for i in range(N):
                # Vertices along a strip are odered in a zig-zag pattern s.t. each of the long edges of the strip only contains either even or odd indices
                strip = torch.stack((ctube_vertices[:, i, :], ctube_vertices[:, (i+1) % N, :]), dim=1).flatten().reshape(-1, 3)
                face = torch.stack((
                    torch.arange(0, 2*M-2, 2), 
                    torch.arange(1, 2*M-1, 2), 
                    torch.arange(3, 2*M  , 2),
                    torch.arange(2, 2*M-1, 2),
                ), dim=1) + face_offset
                strips.append(strip)
                faces.append(face)
                face_offset += 2*M if global_face_indices else 0
            V_list = [strip.detach().numpy() for strip in strips]
            F_list = [face.detach().numpy() for face in faces]
            if end_caps:
                end_cap_offset = 2*M*N if global_face_indices else 0
                for ci in [0, M-1]:
                    V = ctube_vertices[ci]
                    F = triangulate_polygon(V) + end_cap_offset
                    V_list.append(V.detach().numpy())
                    F_list.append(F.detach().numpy())
                    end_cap_offset += N if global_face_indices else 0
            return V_list, F_list  # list of vertices and faces, one tensor for each strip
            
        else:  # welded mesh, no vertex duplication
            assert global_face_indices, "Global face indices are the only option possible for welded meshes."
            V = torch.cat([pts.reshape(-1, 3) for pts in ctube_vertices], dim=0).detach().numpy()
            if triangulate:
                F, F_caps = compute_ctube_topology_tri(M, N)
                if end_caps:
                    global_cap_indices_0 = F_caps[0]
                    global_cap_indices_1 = F_caps[1]
                    F_caps_tri = []
                    for j in range(0, N - 2):
                        F_caps_tri.append(torch.stack((global_cap_indices_0[0], global_cap_indices_0[j+1], global_cap_indices_0[j+2])))
                    for j in range(0, N - 2):
                        F_caps_tri.append(torch.stack((global_cap_indices_1[0], global_cap_indices_1[j+2], global_cap_indices_1[j+1])))
                    F_caps_tri = torch.cat(F_caps_tri).reshape(-1, 3)
                    F = torch.cat([F, F_caps_tri], dim=0)
            else:
                if end_caps:
                    raise NotImplementedError("End caps are not supported without triangulation for welded meshes.")
                F, F_caps = compute_ctube_topology(M, N)
            F = F.detach().numpy()
            return V, F  # tensor of vertices and faces

    def save_obj(self, file_path, triangulate=False, with_symmetry=True, end_caps=False, save_directrix=False, save_generatrix=False, split_strips=False, max_quads_per_strip=None, skip_quad_indices=None, object_name='tube'):
        V, F = self.get_mesh(triangulate=triangulate, with_symmetry=with_symmetry, end_caps=end_caps, split_strips=split_strips)

        def limit_quads(f, max_quads_per_strip, skip_indices=None):
            if skip_indices is not None and len(skip_indices) > 0:
                mask = torch.ones(f.shape[0], dtype=torch.bool)
                mask[skip_indices] = False
                f = f[mask]
            if max_quads_per_strip is not None:
                f = f[:max_quads_per_strip]
            return f

        if split_strips:
            assert isinstance(V, list) and isinstance(F, list) and len(V) == len(F), "The number of sets of vertices and faces must match."
            n_strips = len(V) if not end_caps else len(V) - 2
            # Ensure skip_quad_indices is a list of lists of length n_strips
            if skip_quad_indices is not None:
                assert isinstance(skip_quad_indices, list) and len(skip_quad_indices) == n_strips, "skip_quad_indices must be a list of N lists, N = number of strips"
                # Ensure each element is a list or None; if empty list, treat as None
                skip_quad_indices = [skips if (skips is not None and len(skips) > 0) else None for skips in skip_quad_indices]
                for skips in skip_quad_indices:
                    assert skips is None or isinstance(skips, list), "Each element in skip_quad_indices must be a list or None"
            else:
                skip_quad_indices = [None] * n_strips

            # Write each strip as a separate object in the .obj file
            for i in range(n_strips):
                v = V[i]
                skips = skip_quad_indices[i] if skip_quad_indices is not None else None
                f = limit_quads(F[i], max_quads_per_strip, skips)
                v_np = torch.as_tensor(v).detach().numpy()
                f_np = torch.as_tensor(f).detach().numpy()
                strip_name = f"strip_{i}"
                write_mesh_to_obj(file_path, v_np, f_np, mode='a' if i > 0 else 'w', object_name=strip_name)
            # Write end caps as separate objects if requested
            if end_caps:
                for cap_idx, cap_name in zip([-2, -1], ["end_cap_0", "end_cap_1"]):
                    v_cap = torch.as_tensor(V[cap_idx]).detach().numpy()
                    f_cap = torch.as_tensor(F[cap_idx] + n_vertices_tube_mesh).detach().numpy()
                    write_mesh_to_obj(file_path, v_cap, f_cap, mode='a', object_name=cap_name)
                    n_vertices_tube_mesh += v_cap.shape[0]
        else:
            assert isinstance(V, np.ndarray) and isinstance(F, np.ndarray), "The vertices and faces must be numpy arrays."
            if skip_quad_indices is not None and len(skip_quad_indices) > 0:
                mask = np.ones(F.shape[0], dtype=bool)
                mask[skip_quad_indices] = False
                F = F[mask]
            if max_quads_per_strip is not None:
                F = F[:max_quads_per_strip]
            write_mesh_to_obj(file_path, V, F, mode='w', object_name=object_name)
            n_vertices_tube_mesh = V.shape[0]

        if save_directrix:
            directrix_name = object_name + "_directrix" if object_name is not None else None
            V_dir = self.get_polyline_with_symmetry().detach().numpy() if with_symmetry else self.get_polyline().detach().numpy()
            closed_curve = self.directrix.closed_curve if not with_symmetry else False
            write_polyline_to_obj(file_path, V_dir, write_edges=True, edge_idx_offset=n_vertices_tube_mesh, mode='a', object_name=directrix_name, closed_curve=closed_curve)
            n_vertices_tube_mesh += V_dir.shape[0]

        if save_generatrix:
            generatrix_name = object_name + "_generatrix" if object_name is not None else None
            V_gen = self.compute_vertices()[0].detach().numpy()
            write_polyline_to_obj(file_path, V_gen, write_edges=True, edge_idx_offset=n_vertices_tube_mesh, mode='a', object_name=generatrix_name, closed_curve=True)
            n_vertices_tube_mesh += V_gen.shape[0]

    def save_strips_obj(self, file_path, triangulate=False, with_symmetry=True, y_offset_total=None, y_offset_per_strip=None, selected_strips=None, axes_first_edges=None, points_first_nodes=None, one_file_per_strip=False, delta_z=None, max_quads_per_strip=None, skip_quad_indices=None):
        if with_symmetry:
            ctube_vertices = self.compute_vertices_with_symmetry()
        else:
            ctube_vertices = self.compute_vertices()
        V, F = compute_unrolled_strips(
            ctube_vertices,
            triangulate=triangulate,
            global_face_indices=True,
            y_offset_total=y_offset_total,
            y_offset_per_strip=y_offset_per_strip,
            selected_strips=selected_strips,
            axes_first_edges=axes_first_edges,
            points_first_nodes=points_first_nodes
        )

        def apply_delta_z(v, delta_z):
            n = v.shape[0]
            if n < 4 or delta_z is None:
                return v
            n_pairs = n // 2
            dzs = torch.linspace(0, delta_z, n_pairs)
            v_new = v.clone()
            for i in range(n_pairs):
                v_new[2*i, 2] += dzs[i]
                v_new[2*i+1, 2] += dzs[i]
            return v_new

        def limit_quads(f, max_quads_per_strip, skip_indices=None):
            if skip_indices is not None and len(skip_indices) > 0:
                mask = torch.ones(f.shape[0], dtype=torch.bool)
                mask[skip_indices] = False
                f = f[mask]
            if max_quads_per_strip is not None:
                f = f[:max_quads_per_strip]
            return f

        n_strips = len(V)
        if skip_quad_indices is not None:
            assert isinstance(skip_quad_indices, list) and len(skip_quad_indices) == n_strips, "skip_quad_indices must be a list of N lists, N = number of strips"
            for skips in skip_quad_indices:
                assert skips is None or isinstance(skips, list), "Each element in skip_quad_indices must be a list or None"

        if one_file_per_strip:
            import os
            base, ext = os.path.splitext(file_path)
            iter_file_name = os.path.basename(base)
            base_dir = os.path.dirname(base)
            for i, (v, f) in enumerate(zip(V, F)):
                strip_file = f"{base_dir}/unrolled_strip_{i}_{iter_file_name}{ext}"
                v_mod = v
                if delta_z is not None:
                    v_mod = apply_delta_z(v_mod, delta_z)
                skip_indices = skip_quad_indices[i] if skip_quad_indices is not None else None
                f_mod = limit_quads(f, max_quads_per_strip, skip_indices)
                v_np = v_mod.reshape(-1, 3).detach().numpy()
                f_np = f_mod.detach().numpy()
                if f_np.min() != 0:
                    f_np = f_np - f_np.min()
                write_mesh_to_obj(strip_file, v_np, f_np)
        else:
            V_mod = []
            F_mod = []
            for i, (v, f) in enumerate(zip(V, F)):
                if delta_z is not None:
                    v = apply_delta_z(v, delta_z)
                V_mod.append(v)
                skip_indices = skip_quad_indices[i] if skip_quad_indices is not None else None
                f_mod = limit_quads(f, max_quads_per_strip, skip_indices)
                F_mod.append(f_mod)
            F_cat = torch.cat([f for f in F_mod], dim=0).detach().numpy()
            V_cat = torch.cat([v.reshape(-1, 3) for v in V_mod], dim=0).detach().numpy()
            write_mesh_to_obj(file_path, V_cat, F_cat)
        
    def update_plane_normals(self, kind):
        assert self.plane_normal_kind is not None and kind == self.plane_normal_kind, "plane_normal_kind ({}) must be set to compute the plane normals, and it must match the provided kind ({}).".format(self.plane_normal_kind, kind)
        if not self.has_symmetry():
            self.plane_normals = compute_plane_normals(self.directrix.X, kind=self.plane_normal_kind, closed_curve=self.directrix.closed_curve)
        else:
            directrix_symm = self.get_polyline_with_symmetry()
            closed_curve_symm = torch.allclose(directrix_symm[0], directrix_symm[-1])  # check if the control points closed up after applying the last symmetry transformation
            plane_normals_symm = compute_plane_normals(directrix_symm, kind=self.plane_normal_kind, closed_curve=closed_curve_symm)
            self.plane_normals = plane_normals_symm[0:self.M]

    def plot(self, fig=None, ax=None, save_path=None, xlim=None, ylim=None):
        return plot_tube(self.directrix.X, self.ctube_vertices, fig=fig, ax=ax, save_path=save_path, xlim=xlim, ylim=ylim)

    def plot_3d(self, fig=None, ax=None, save_path=None, xlim=None, ylim=None, zlim=None, expand_axes=0.3, plot_curve_shadows=True, plot_planes=True, plot_origin=True, plot_Q=True, tube_color=None, with_symmetry=True):
        if self.has_symmetry() and with_symmetry:
            directrix = self.get_polyline_with_symmetry()
            ctube_vertices = self.compute_vertices_with_symmetry()
            Q = self.compute_dofs_with_symmetry()[0] if plot_Q else None
        else:
            directrix = self.directrix.X
            ctube_vertices = self.ctube_vertices
            Q = self.directrix.get_control_points() if plot_Q else None
        return plot_tube_3d(directrix, ctube_vertices, cps=Q, fig=fig, ax=ax, save_path=save_path, xlim=xlim, ylim=ylim, zlim=zlim, expand_axes=expand_axes, plot_curve_shadows=plot_curve_shadows, plot_planes=plot_planes, plot_origin=plot_origin, tube_color=tube_color)

    def plot_unrolled_strips(self, fig=None, ax=None, save_path=None, xlim=None, ylim=None, y_offset_total=None, y_offset_per_strip=None, selected_strips=None, axes_first_edges=None, points_first_nodes=None, strip_colors=None, with_symmetry=True):
        if self.has_symmetry() and with_symmetry:
            ctube_vertices = self.compute_vertices_with_symmetry()
        else:
            ctube_vertices = self.compute_vertices()
        return plot_unrolled_strips(ctube_vertices, fig=fig, ax=ax, save_path=save_path, xlim=xlim, ylim=ylim, y_offset_total=y_offset_total, y_offset_per_strip=y_offset_per_strip, selected_strips=selected_strips, axes_first_edges=axes_first_edges, points_first_nodes=points_first_nodes, strip_colors=strip_colors)



